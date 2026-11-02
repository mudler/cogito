package cogito

import (
	"encoding/json"
	"testing"

)

func TestSinkStateDisabledByDefault(t *testing.T) {
	if defaultOptions().sinkState {
		t.Fatal("sink state must be disabled by default")
	}
	o := defaultOptions()
	o.Apply(EnableSinkState)
	if !o.sinkState {
		t.Fatal("EnableSinkState did not enable the sink state")
	}
	o.Apply(DisableSinkState)
	if o.sinkState {
		t.Fatal("DisableSinkState did not disable the sink state")
	}
}

func TestDefaultRequestDoesNotAdvertiseReply(t *testing.T) {
	llm := &captureLLM{}
	f := NewEmptyFragment().AddMessage("user", "hi")
	if _, err := ExecuteTools(llm, f, WithTools(newNamedTool("ordinary"))); err != nil {
		t.Fatal(err)
	}
	for _, name := range llm.toolNames(0) {
		if name == "reply" {
			t.Fatalf("default request advertised the reply sink tool: %v", llm.toolNames(0))
		}
	}
}

func TestDefaultSinkStateToolHasObjectSchema(t *testing.T) {
	b, err := json.Marshal((&defaultSinkStateTool{}).Tool().Function.Parameters)
	if err != nil {
		t.Fatal(err)
	}
	var schema map[string]any
	if err := json.Unmarshal(b, &schema); err != nil || schema == nil {
		t.Fatalf("reply parameters must be a JSON object, got %s", b)
	}
	if schema["type"] != "object" {
		t.Fatalf("reply schema type = %v, want object", schema["type"])
	}
}

func TestSubAgentsInheritSinkState(t *testing.T) {
	for _, tc := range []struct {
		name string
		opts []Option
		want bool
	}{
		{"default", nil, false},
		{"disabled", []Option{EnableSinkState, DisableSinkState}, false},
		{"enabled", []Option{EnableSinkState}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			parent := defaultOptions()
			parent.Apply(append([]Option{EnableAgentSpawning}, tc.opts...)...)
			var spawn *spawnAgentRunner
			for _, td := range prepareAgentTools(parent, &captureLLM{}) {
				if def, ok := td.(*ToolDefinition[SpawnAgentArgs]); ok {
					spawn, _ = def.ToolRunner.(*spawnAgentRunner)
				}
			}
			if spawn == nil {
				t.Fatal("spawn_agent runner not found")
			}
			child := defaultOptions()
			child.Apply(spawn.parentOpts...)
			if child.sinkState != tc.want {
				t.Fatalf("sub-agent sinkState = %v, want %v", child.sinkState, tc.want)
			}
		})
	}
}

func TestPlanOptionRebuildKeepsSinkState(t *testing.T) {
	custom := newNamedTool("custom_reply")
	for _, enabled := range []bool{false, true} {
		o := defaultOptions()
		if enabled {
			o.Apply(WithSinkState(custom))
		}
		rebuilt := defaultOptions()
		rebuilt.Apply(convertOptionsToFunctions(o)...)
		if rebuilt.sinkState != enabled {
			t.Fatalf("rebuilt sinkState = %v, want %v", rebuilt.sinkState, enabled)
		}
		if enabled && rebuilt.sinkStateTool != custom {
			t.Fatal("rebuilt options lost the custom sink tool")
		}
	}
}

