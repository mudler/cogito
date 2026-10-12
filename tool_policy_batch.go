package cogito

import (
	"fmt"
	"reflect"

	"github.com/sashabaranov/go-openai"
)

// ownPolicyChoice copies execution data without coercing json.Number. Policy
// callbacks may retain their inputs and Modified values, but never own history.
func ownPolicyChoice(choice *ToolChoice) (*ToolChoice, error) {
	v, err := cloneCompletionValue(reflect.ValueOf(choice), 0)
	if err != nil {
		return nil, err
	}
	return v.Interface().(*ToolChoice), nil
}

// richAdjustment requests replacement of the whole closed batch, not a call edit.
type richAdjustment string

func (a richAdjustment) Error() string {
	return "Tool batch superseded; resubmit all required calls: " + string(a)
}

type adjustmentChainKey struct{}

type richPolicyResult struct {
	text   string
	status ToolStatus
	tool   ToolDefinitionInterface
}

// executeRichPolicyBatch preflights the entire batch before any side effect.
// Only nonstream rich selections use this boundary: even interruption persists
// the original assistant and exactly one ordered result for each received call.
func executeRichPolicyBatch(f, selected Fragment, messages []Message, choices []*ToolChoice, tools Tools, o *Options) (Fragment, bool, error) {
	original := messages[len(messages)-1].Portable.ToolCalls
	results := make([]richPolicyResult, len(original))
	execution := make([]*ToolChoice, len(original))
	var terminal error
	sink := false
	for i, choice := range choices {
		if terminal != nil {
			break
		}
		if err := o.context.Err(); err != nil {
			terminal = err
			break
		}
		if o.sinkState && choice.Name == o.sinkStateTool.Tool().Function.Name {
			sink = true
			results[i].text = "Sink state selected; tool not executed"
			continue
		}
		if checkForLoop(f.Status.PastActions, choice, o.loopDetectionSteps) {
			terminal = ErrLoopDetected
			break
		}
		owned, err := ownPolicyChoice(choice)
		if err != nil {
			terminal = err
			break
		}
		if o.toolCallCallback != nil {
			// Use the existing session snapshot contract: histories are owned;
			// execution capabilities and application result data remain opaque.
			snapshot := cloneAgentFragment(f)
			for _, statuses := range [][]ToolStatus{snapshot.Status.ToolResults, snapshot.Status.PastActions} {
				for j := range statuses {
					choice, err := ownPolicyChoice(&statuses[j].ToolArguments)
					if err != nil {
						terminal = err
						break
					}
					statuses[j].ToolArguments = *choice
				}
			}
			if terminal != nil {
				break
			}
			decision := o.toolCallCallback(owned, &SessionState{ToolChoice: owned, Fragment: snapshot})
			if err := o.context.Err(); err != nil {
				terminal = err
				break
			}
			if owned.ID != original[i].ID || (decision.Modified != nil && decision.Modified.ID != original[i].ID) {
				terminal = fmt.Errorf("provider-backed tool policy cannot change the original call ID")
				break
			}
			if !decision.Approved {
				terminal = ErrToolCallCallbackInterrupted
				break
			}
			if decision.Skip {
				results[i].text = "Tool call skipped by user"
				continue
			}
			if decision.Modified != nil {
				owned = decision.Modified
			} else if decision.Adjustment != "" {
				terminal = richAdjustment(decision.Adjustment)
				break
			}
		}
		// Discard callback edits to diagnostic provenance; detach retained maps.
		detached := *owned
		detached.OriginalCall = choice.OriginalCall
		execution[i], err = ownPolicyChoice(&detached)
		if err != nil {
			terminal = err
			break
		}
	}
	if terminal == nil {
		terminal = o.context.Err()
	}
	if terminal == nil {
		run := func(i int) richPolicyResult {
			tc := execution[i]
			tool := tools.Find(tc.Name)
			if tool == nil {
				return richPolicyResult{text: fmt.Sprintf("Error: tool %s not found", tc.Name)}
			}
			r := richPolicyResult{tool: tool}
			var data any
			for attempt := 0; attempt < o.maxAttempts; attempt++ {
				if o.context.Err() != nil {
					break
				}
				var err error
				r.text, data, err = tool.Execute(tc.Arguments)
				r.status.Executed = true
				if err == nil {
					break
				}
				r.text = fmt.Sprintf("Error running tool: %v", err)
			}
			r.status.Result, r.status.ResultData = r.text, data
			r.status.ToolArguments, r.status.Name = *tc, tc.Name
			return r
		}
		if o.parallelToolExecution {
			// Validate lookups before starting parallel work. Completion order must not
			// affect replay order. All started work is joined before closing the batch.
			for _, tc := range execution {
				if tc != nil && tools.Find(tc.Name) == nil {
					terminal = fmt.Errorf("tool %s not found", tc.Name)
					break
				}
			}
			if terminal == nil {
				type indexed struct {
					i      int
					result richPolicyResult
				}
				ch := make(chan indexed, len(execution))
				count := 0
				for i, tc := range execution {
					if tc != nil {
						count++
						go func(i int) { ch <- indexed{i, run(i)} }(i)
					}
				}
				for n := 0; n < count; n++ {
					r := <-ch
					results[r.i] = r.result
				}
				terminal = o.context.Err()
			}
		} else {
			for i, tc := range execution {
				if tc == nil {
					continue
				}
				if err := o.context.Err(); err != nil {
					terminal = err
					break
				}
				if tools.Find(tc.Name) == nil {
					terminal = fmt.Errorf("tool %s not found", tc.Name)
					break
				}
				results[i] = run(i)
				if err := o.context.Err(); err != nil {
					terminal = err
					break
				}
			}
		}
	}
	f = f.AppendMessages(messages...)
	f.Status.LastUsage = selected.Status.LastUsage
	for i, call := range original {
		r := results[i]
		if r.text == "" && !r.status.Executed {
			r.text = "Tool not executed"
			if terminal != nil {
				r.text += ": " + terminal.Error()
			}
		}
		if _, superseded := terminal.(richAdjustment); superseded && r.text != "Tool call skipped by user" {
			r.text = "Tool not executed: " + terminal.Error()
		}
		f = f.appendPortableMessage(openai.ChatCompletionMessage{Role: "tool", ToolCallID: call.ID, Name: call.Function.Name, Content: r.text})
		if !r.status.Executed {
			continue
		}
		f = appendToolImages(f, r.status, o.toolImageForwarding, r.status.Name)
		f.Status.ToolsCalled = append(f.Status.ToolsCalled, r.tool)
		f.Status.ToolResults = append(f.Status.ToolResults, r.status)
		f.Status.PastActions = append(f.Status.PastActions, r.status)
		if o.toolCallResultCallback != nil {
			o.toolCallResultCallback(r.status)
		}
	}
	if adjustment, ok := terminal.(richAdjustment); ok {
		f = f.AddMessage(UserMessageRole, "The previous tool batch was superseded without execution (explicit skips remain skipped). Select a complete replacement batch; all calls require fresh approval. Adjustment feedback: "+string(adjustment))
		return f, false, terminal
	}
	f.Status.Iterations++
	return f, sink, terminal
}

// unexecutedAdjustmentBatch is only used after correlation IDs have been validated.
func unexecutedAdjustmentBatch(original Message, err error) []Message {
	closed := CloneMessages([]Message{original})
	for _, call := range original.Portable.ToolCalls {
		closed = append(closed, Message{Portable: openai.ChatCompletionMessage{Role: "tool", ToolCallID: call.ID, Name: call.Function.Name, Content: "Tool not executed: " + err.Error()}})
	}
	return closed
}
