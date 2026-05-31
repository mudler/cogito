package cogito

import "testing"

func TestSessionStateHasAgentID(t *testing.T) {
	s := SessionState{AgentID: "abc-123"}
	if s.AgentID != "abc-123" {
		t.Fatalf("expected AgentID to round-trip, got %q", s.AgentID)
	}
}
