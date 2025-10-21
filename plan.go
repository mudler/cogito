package cogito

import (
	"errors"
	"fmt"

	"github.com/mudler/cogito/pkg/xlog"
	"github.com/mudler/cogito/prompt"
	"github.com/mudler/cogito/structures"
	"github.com/sashabaranov/go-openai"
)

type PlanStatus struct {
	Plan  structures.Plan
	Tools []ToolStatus
}

var (
	ErrGoalNotAchieved error = errors.New("goal not achieved")
)

// ExtractPlan extracts a plan from a conversation
// To override the prompt, define a PromptPlanType, PromptReEvaluatePlanType and PromptSubtaskExtractionType
func ExtractPlan(llm LLM, f Fragment, goal *structures.Goal, opts ...Option) (*structures.Plan, error) {
	o := defaultOptions()
	o.Apply(opts...)

	// First we ask the LLM to organize subtasks
	prompter := o.prompts.GetPrompt(prompt.PromptPlanType)

	toolDefs := o.tools.Definitions()
	planOptions := struct {
		Context              string
		AdditionalContext    string
		Goal                 *structures.Goal
		Tools                []*openai.FunctionDefinition
		FeedbackConversation string
	}{
		Context: f.String(),
		Goal:    goal,
		Tools:   toolDefs,
	}
	if o.deepContext && f.ParentFragment != nil {
		planOptions.AdditionalContext = f.ParentFragment.AllFragmentsStrings()
	}

	var feedbackConv *Fragment
	if o.feedbackCallback != nil {
		feedbackConv = o.feedbackCallback()
		planOptions.FeedbackConversation = feedbackConv.String()
	}

	prompt, err := prompter.Render(planOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	return applyPlanFromPrompt(llm, o, prompt, feedbackConv)
}

// ExtractPlan extracts a plan from a conversation
// to override the prompt, define a PromptReEvaluatePlanType and PromptSubtaskExtractionType
func ReEvaluatePlan(llm LLM, f, subtaskFragment Fragment, goal *structures.Goal, toolStatuses []ToolStatus, subtask string, opts ...Option) (*structures.Plan, error) {
	o := defaultOptions()
	o.Apply(opts...)

	// First we ask the LLM to organize subtasks
	prompter := o.prompts.GetPrompt(prompt.PromptReEvaluatePlanType)

	toolDefs := o.tools.Definitions()
	planOptions := struct {
		Context              string
		AdditionalContext    string
		Subtask              string
		SubtaskConversation  string
		Goal                 string
		Tools                []*openai.FunctionDefinition
		PastActionHistory    []ToolStatus
		FeedbackConversation string
	}{
		Context:             f.String(),
		Goal:                goal.Goal,
		Subtask:             subtask,
		Tools:               toolDefs,
		PastActionHistory:   toolStatuses,
		SubtaskConversation: subtaskFragment.String(),
	}
	if o.deepContext && f.ParentFragment != nil {
		planOptions.AdditionalContext = f.ParentFragment.AllFragmentsStrings()
	}

	var feedbackConv *Fragment
	if o.feedbackCallback != nil {
		feedbackConv = o.feedbackCallback()
		planOptions.FeedbackConversation = feedbackConv.String()
	}

	prompt, err := prompter.Render(planOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	return applyPlanFromPrompt(llm, o, prompt, feedbackConv)
}

func applyPlanFromPrompt(llm LLM, o *Options, planPrompt string, feedbackConv *Fragment) (*structures.Plan, error) {
	multimedias := []Multimedia{}
	if feedbackConv != nil {
		multimedias = feedbackConv.Multimedia
	}
	planConv := NewEmptyFragment().AddMessage("user", planPrompt, multimedias...)
	reasoningPlan, err := llm.Ask(o.context, planConv)
	if err != nil {
		return nil, fmt.Errorf("failed to ask LLM for plan identification: %w", err)
	}

	identifiedPlan := reasoningPlan.LastMessage()

	structure, plan := structures.StructurePlan()

	prompter := o.prompts.GetPrompt(prompt.PromptSubtaskExtractionType)

	planOptions := struct {
		Context string
	}{
		Context: identifiedPlan.Content,
	}

	prompt, err := prompter.Render(planOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	planConv = NewEmptyFragment().AddMessage("user", prompt)

	err = planConv.ExtractStructure(o.context, llm, structure)
	if err != nil {
		return nil, fmt.Errorf("failed to extract structure: %w", err)
	}

	plan.Description = identifiedPlan.Content
	return plan, err
}

// ExecutePlan Executes an already-defined plan with a set of options.
// To override its prompt, configure PromptPlanExecutionType, PromptPlanType, PromptReEvaluatePlanType and PromptSubtaskExtractionType
func ExecutePlan(llm LLM, conv Fragment, plan *structures.Plan, goal *structures.Goal, opts ...Option) (Fragment, error) {
	o := defaultOptions()
	o.Apply(opts...)

	if len(plan.Subtasks) == 0 {
		return NewEmptyFragment(), fmt.Errorf("no subtasks found in plan")
	}

	xlog.Debug("Executing plan for conversation", "length", len(conv.Messages), "plan", plan.Description, "subtasks", plan.Subtasks)

	var toolStatuses []ToolStatus
	index := 0
	attempts := 1
	for {
		subtask := plan.Subtasks[index]

		xlog.Debug("Executing subtask", "goal", goal.Goal, "subtask", subtask)

		prompter := o.prompts.GetPrompt(prompt.PromptPlanExecutionType)

		subtaskOption := struct {
			Goal    string
			Subtask string
		}{
			Goal:    goal.Goal,
			Subtask: subtask,
		}

		prompt, err := prompter.Render(subtaskOption)
		if err != nil {
			return NewEmptyFragment(), fmt.Errorf("failed to render tool reasoner prompt: %w", err)
		}

		subtaskConv := NewEmptyFragment().AddMessage("user", prompt)

		subtaskConvResult, err := ExecuteTools(llm, subtaskConv, opts...)
		if err != nil {
			return conv, err
		}

		conv.Messages = append(conv.Messages, subtaskConvResult.LastAssistantAndToolMessages()...)
		conv.Status.Iterations = conv.Status.Iterations + 1
		conv.Status.ToolsCalled = append(conv.Status.ToolsCalled, subtaskConvResult.Status.ToolsCalled...)
		conv.Status.ToolResults = append(conv.Status.ToolResults, subtaskConvResult.Status.ToolResults...)
		toolStatuses = append(toolStatuses, subtaskConvResult.Status.ToolResults...)

		boolean, err := IsGoalAchieved(llm, subtaskConvResult, nil, opts...)
		if err != nil {
			return conv, err
		}

		toolStatuses := []ToolStatus{}
		for i := range conv.Status.ToolsCalled {
			toolStatuses = append(toolStatuses, conv.Status.ToolResults[i])
		}

		if !boolean.Boolean {
			if attempts >= o.maxAttempts {
				if !o.planReEvaluator {
					return conv, ErrGoalNotAchieved
				}
				xlog.Debug("All attempts failed, re-evaluating plan")
				plan, err = ReEvaluatePlan(llm, conv, subtaskConv, goal, toolStatuses, subtask, opts...)
				if err != nil {
					return conv, err
				}

				// Start again
				index = 0
				attempts = 1
			} else {
				xlog.Debug("Attempt failed to achieve goal, retrying")
				attempts++
			}
		} else {
			xlog.Debug("Goal correctly achieved")
			attempts = 1 // reset attempts
			if len(plan.Subtasks)-1 > index {
				index++
			} else if !(o.infiniteExecution) {
				break
			}
		}
	}

	conv.Status.Plans = append(conv.Status.Plans, PlanStatus{
		Plan:  *plan,
		Tools: toolStatuses,
	})

	return conv, nil
}
