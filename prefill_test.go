package cogito

import "testing"

func TestPrepareAgentToolsDisabled(t *testing.T) {
	o := defaultOptions()
	if got := prepareAgentTools(o, nil); got != nil {
		t.Fatalf("spawning disabled: want nil, got %d tools", len(got))
	}
}

func TestPrepareAgentToolsReturnsFourAndInitializes(t *testing.T) {
	o := defaultOptions()
	o.Apply(EnableAgentSpawning)
	tools := prepareAgentTools(o, nil)
	if len(tools) != 4 {
		t.Fatalf("want 4 agent tools, got %d", len(tools))
	}
	if o.agentManager == nil {
		t.Fatal("agentManager must be auto-created")
	}
	if o.messageInjectionChan == nil {
		t.Fatal("messageInjectionChan must be auto-created")
	}
	names := map[string]bool{}
	for _, tl := range tools {
		names[tl.Tool().Function.Name] = true
	}
	for _, want := range []string{"spawn_agent", "check_agent", "get_agent_result", "send_agent_message"} {
		if !names[want] {
			t.Fatalf("missing agent tool %q (got %v)", want, names)
		}
	}
}
