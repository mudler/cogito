package cogito

import "testing"

func TestParseStreamedToolArgs_Defensive(t *testing.T) {
	cases := []struct {
		name, raw, key string
		val            any
		wantErr        bool
	}{
		{"plain valid", `{"query":"x"}`, "query", "x", false},
		{"concatenated duplicate", `{"query":"x"}{"query":"x"}`, "query", "x", false},
		{"concatenated different -> first", `{"city":"NYC"}{"city":"LA"}`, "city", "NYC", false},
		{"malformed", `{"query":`, "", nil, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out := make(map[string]any)
			err := parseStreamedToolArgs(c.raw, &out)
			if c.wantErr {
				if err == nil {
					t.Fatalf("want err")
				}
				return
			}
			if err != nil {
				t.Fatalf("err: %v", err)
			}
			if out[c.key] != c.val {
				t.Fatalf("got %v want %v", out[c.key], c.val)
			}
		})
	}
}
