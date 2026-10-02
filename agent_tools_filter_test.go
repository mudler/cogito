package cogito

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/sashabaranov/go-openai"
)

func TestAgentToolsSelection(t *testing.T) {
	cases := []struct {
		name    string
		names   []string
		enabled bool
		want    []string
	}{
		{"nil", nil, true, agentToolNames},
		{"empty", []string{}, true, nil},
		{"subset", []string{"get_agent_result", "spawn_agent", "spawn_agent"}, true, []string{"spawn_agent", "get_agent_result"}},
		{"unknown", []string{"unknown", "SPAWN_AGENT"}, true, nil},
		{"gate", agentToolNames, false, nil},
	}
	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			o := defaultOptions()
			o.Apply(WithAgentTools(tt.names))
			if tt.enabled {
				o.Apply(EnableAgentSpawning)
			}
			got := Tools(prepareAgentTools(o, nil)).Names()
			if len(got) != len(tt.want) || (len(got) > 0 && !reflect.DeepEqual(got, tt.want)) {
				t.Fatalf("got %v want %v", got, tt.want)
			}
		})
	}
}

func TestAgentToolsCopiesAndReconstruction(t *testing.T) {
	names := []string{"check_agent"}
	opt := WithAgentTools(names)
	names[0] = "send_agent_message"
	o := defaultOptions()
	o.Apply(opt)
	o.agentTools[0] = "spawn_agent"
	other := defaultOptions()
	other.Apply(opt)
	if !reflect.DeepEqual(other.agentTools, []string{"check_agent"}) {
		t.Fatal(other.agentTools)
	}
	for _, names := range [][]string{nil, {}, {"check_agent"}} {
		o := defaultOptions()
		o.Apply(WithAgentTools(names))
		rebuilt := defaultOptions()
		rebuilt.Apply(convertOptionsToFunctions(o)...)
		if !reflect.DeepEqual(o.agentTools, rebuilt.agentTools) {
			t.Fatalf("lost selection: %#v %#v", o.agentTools, rebuilt.agentTools)
		}
	}
}

func TestAgentToolsSchemaParityAndInheritedTools(t *testing.T) {
	for _, names := range [][]string{nil, {}, {"spawn_agent", "check_agent", "get_agent_result"}, {"unknown"}} {
		t.Run("selection", func(t *testing.T) {
			opts := []Option{EnableAgentSpawning, WithAgentTools(names), WithTools(newNamedTool("ordinary"))}
			// Inherited definitions must not restore an explicitly excluded tool.
			if names != nil {
				for _, name := range agentToolNames {
					opts = append(opts, WithTools(newNamedTool(name)))
				}
			}
			p, e := &captureLLM{}, &captureLLM{}
			f := NewEmptyFragment().AddMessage("user", "hi")
			if err := Prefill(context.Background(), p, f, opts...); err != nil {
				t.Fatal(err)
			}
			if _, err := ExecuteTools(e, f, opts...); err != nil {
				t.Fatal(err)
			}
			pr, _ := p.request(0)
			er, _ := e.request(0)
			if !reflect.DeepEqual(pr.Tools, er.Tools) {
				t.Fatal("schema mismatch")
			}
			counts := map[string]int{}
			for _, tool := range pr.Tools {
				counts[tool.Function.Name]++
			}
			if counts["ordinary"] != 1 {
				t.Fatal(counts)
			}
			for _, name := range agentToolNames {
				want := 0
				if names == nil || contains(names, name) {
					want = 1
				}
				if counts[name] != want {
					t.Fatalf("%s count %d want %d", name, counts[name], want)
				}
			}
		})
	}
}

func TestAgentToolsDisabledRequestCannotRun(t *testing.T) {
	calls := 0
	forbidden := NewToolDefinition(&filterCountingTool{calls: &calls}, struct{}{}, "send_agent_message", "forbidden")
	reply := chatToolCall("disabled", `{}`, openai.FinishReasonToolCalls)
	reply.Message.ToolCalls[0].Function.Name = "send_agent_message"
	llm := &scriptedChatLLM{replies: []openai.ChatCompletionChoice{reply}, usages: []LLMUsage{{}}}
	_, _ = ExecuteTools(llm, NewEmptyFragment().AddMessage("user", "send"), EnableAgentSpawning, WithAgentTools([]string{"check_agent"}), WithTools(forbidden), WithMaxRetries(1))
	if len(llm.requests) == 0 {
		t.Fatal("no model request")
	}
	if calls != 0 {
		t.Fatal("excluded tool executed")
	}
}

type filterCountingTool struct{ calls *int }

func (r *filterCountingTool) Run(struct{}) (string, any, error) {
	*r.calls++
	return "called", nil, nil
}

func TestAgentToolsChildLeakage(t *testing.T) {
	for _, requested := range [][]string{nil, {"ordinary", "send_agent_message"}} {
		for _, definition := range []bool{false, true} {
			llm := &captureLLM{}
			o := defaultOptions()
			o.Apply(EnableAgentSpawning, WithAgentTools([]string{"spawn_agent"}), WithTools(newNamedTool("ordinary"), newNamedTool("send_agent_message")))
			args := SpawnAgentArgs{Task: "child", Tools: requested}
			if definition {
				o.Apply(WithAgentDefinitions([]AgentDefinition{{Name: "worker", Tools: requested}}...))
				args.AgentType = "worker"
				args.Tools = nil
			}
			tools := Tools(prepareAgentTools(o, llm))
			spawn := tools.Find("spawn_agent")
			_, _, err := spawn.Execute(filterArgs(args))
			if err != nil {
				t.Fatal(err)
			}
			got := llm.toolNames(0)
			if !contains(got, "ordinary") {
				t.Fatal(got)
			}
			for _, name := range agentToolNames {
				if contains(got, name) {
					t.Fatalf("child leaked %s", name)
				}
			}
		}
	}
}

func filterArgs(args SpawnAgentArgs) map[string]any {
	b, _ := json.Marshal(args)
	var m map[string]any
	_ = json.Unmarshal(b, &m)
	return m
}

func TestAgentToolsLegacyChildSemantics(t *testing.T) {
	for _, requested := range [][]string{nil, {"ordinary", "send_agent_message"}} {
		llm := &captureLLM{}
		o := defaultOptions()
		o.Apply(EnableAgentSpawning, WithTools(newNamedTool("ordinary"), newNamedTool("send_agent_message")))
		spawn := Tools(prepareAgentTools(o, llm)).Find("spawn_agent")
		if _, _, err := spawn.Execute(filterArgs(SpawnAgentArgs{Task: "child", Tools: requested})); err != nil {
			t.Fatal(err)
		}
		got := llm.toolNames(0)
		if !contains(got, "ordinary") || contains(got, "spawn_agent") {
			t.Fatal(got)
		}
		if contains(got, "send_agent_message") != (requested != nil) {
			t.Fatalf("legacy explicit inheritance changed: %v", got)
		}
	}
}

func TestAgentToolsReconstructedExclusions(t *testing.T) {
	o := defaultOptions()
	o.Apply(WithAgentTools([]string{}), WithTools(newNamedTool("send_agent_message"), newNamedTool("ordinary")))
	llm := &captureLLM{}
	if _, err := ExecuteTools(llm, NewEmptyFragment().AddMessage("user", "hi"), convertOptionsToFunctions(o)...); err != nil {
		t.Fatal(err)
	}
	got := llm.toolNames(0)
	if contains(got, "send_agent_message") || !contains(got, "ordinary") {
		t.Fatal(got)
	}
}
