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
