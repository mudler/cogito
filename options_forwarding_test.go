package cogito

import "testing"

func TestWithToolImageForwarding(t *testing.T) {
	o := defaultOptions()
	if o.toolImageForwarding {
		t.Fatalf("expected default false")
	}
	WithToolImageForwarding(true)(o)
	if !o.toolImageForwarding {
		t.Fatalf("expected true after WithToolImageForwarding(true)")
	}
}

// Inspect the captured options without spawning agents or making LLM calls.
func TestConversationManipulatorOptionForwarding(t *testing.T) {
	o := defaultOptions()
	if o.conversationManipulator != nil {
		t.Fatal("unexpected default callback")
	}
	calls := 0
	o.Apply(EnableAgentSpawning, WithConversationManipulator(func(messages []Message) ([]Message, error) {
		calls++
		return messages, nil
	}))
	tools := prepareAgentTools(o, nil)
	checked := 0
	for _, tool := range tools {
		var opts []Option
		switch definition := tool.(type) {
		case *ToolDefinition[SpawnAgentArgs]:
			opts = definition.ToolRunner.(*spawnAgentRunner).parentOpts
		case *ToolDefinition[SendAgentMessageArgs]:
			opts = definition.ToolRunner.(*sendAgentMessageRunner).subOpts
		default:
			continue
		}
		child := defaultOptions()
		child.Apply(opts...)
		if child.conversationManipulator == nil {
			t.Fatalf("%s lost callback", tool.Tool().Function.Name)
		}
		if _, err := buildToolSelectionConversation(child, NewEmptyFragment(), nil, nil); err != nil {
			t.Fatal(err)
		}
		checked++
	}
	if checked != 2 || calls != 2 {
		t.Fatalf("checked %d runners, invoked %d callbacks", checked, calls)
	}
	o.Apply(WithConversationManipulator(nil))
	if o.conversationManipulator != nil {
		t.Fatal("nil option did not clear callback")
	}
}
