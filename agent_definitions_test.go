package cogito

import "testing"

func TestWithAgentDefinitionsStoresDefs(t *testing.T) {
	defs := []AgentDefinition{
		{Name: "explore", Description: "read-only exploration",
			SystemPrompt: "You explore.", Tools: []string{"echo"},
			Model: "small-model", Temperature: 0.2,
			Iterations: 20, MaxAttempts: 2, MaxRetries: 2},
	}
	o := defaultOptions()
	o.Apply(WithAgentDefinitions(defs...))
	if len(o.agentDefinitions) != 1 || o.agentDefinitions[0].Name != "explore" {
		t.Fatalf("agent definitions not stored: %+v", o.agentDefinitions)
	}
}

func TestFindAgentDefinition(t *testing.T) {
	defs := []AgentDefinition{{Name: "plan"}, {Name: "explore"}}
	if d := findAgentDefinition(defs, "explore"); d == nil || d.Name != "explore" {
		t.Fatalf("expected to find explore, got %+v", d)
	}
	if d := findAgentDefinition(defs, "missing"); d != nil {
		t.Fatalf("expected nil for missing type, got %+v", d)
	}
}
