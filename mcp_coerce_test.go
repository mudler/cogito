package cogito

import (
	"encoding/json"
	"testing"

	"github.com/tmc/langchaingo/jsonschema"
)

// TestCoerceNullableTypes pins the workaround for
// modelcontextprotocol/go-sdk v1.4+ schemas that emit
// "type": ["null", "X"] for nullable Go fields like []string slices.
// Without coercion, tools with such fields silently drop out of the
// agent's tool list because langchaingo/jsonschema.Definition.Type
// is a single string and the JSON-array form fails to unmarshal.
func TestCoerceNullableTypes(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "nullable array stays as array",
			in: `{"tags": {
				"type": ["null", "array"],
				"items": {"type": "string"}
			}}`,
			want: "array",
		},
		{
			name: "nullable string stays as string",
			in: `{"name": {
				"type": ["null", "string"]
			}}`,
			want: "string",
		},
		{
			name: "single string type passes through",
			in: `{"name": {
				"type": "string"
			}}`,
			want: "string",
		},
		{
			name: "nested properties are coerced too",
			in: `{"obj": {
				"type": "object",
				"properties": {
					"inner": {"type": ["null", "string"]}
				}
			}}`,
			want: "object",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var props map[string]any
			if err := json.Unmarshal([]byte(tc.in), &props); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			coerceNullableTypes(props)

			// The whole point: after coercion, this round-trips into
			// jsonschema.Definition without error.
			b, err := json.Marshal(props)
			if err != nil {
				t.Fatalf("re-marshal: %v", err)
			}
			out := map[string]jsonschema.Definition{}
			if err := json.Unmarshal(b, &out); err != nil {
				t.Fatalf("post-coerce unmarshal into jsonschema.Definition: %v", err)
			}

			// Verify the top-level field's type is the expected single string.
			for _, def := range out {
				if string(def.Type) != tc.want {
					t.Errorf("got type=%q, want %q", def.Type, tc.want)
				}
			}
			// Nested case: assert the inner "inner" field is also coerced.
			if tc.name == "nested properties are coerced too" {
				obj := out["obj"]
				inner := obj.Properties["inner"]
				if string(inner.Type) != "string" {
					t.Errorf("nested inner type=%q, want string", inner.Type)
				}
			}
		})
	}
}
