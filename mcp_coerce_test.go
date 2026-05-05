package cogito

import (
	"encoding/json"
	"testing"

	"github.com/tmc/langchaingo/jsonschema"
)

// containsTypeArray returns true if any nested node in v still carries
// a JSON-array `type` (i.e. coerce missed it). Used by tests to assert
// the coerce reaches every corner of a schema.
func containsTypeArray(v any) bool {
	switch n := v.(type) {
	case map[string]any:
		if _, ok := n["type"].([]any); ok {
			return true
		}
		for _, child := range n {
			if containsTypeArray(child) {
				return true
			}
		}
	case []any:
		for _, child := range n {
			if containsTypeArray(child) {
				return true
			}
		}
	}
	return false
}

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
		// langchaingoCompatible reports whether the test's schema
		// shape is one langchaingo's Definition struct can represent.
		// Tuple-style `items: [...]` and other JSON Schema 2020-12
		// constructs aren't representable but we still want coerce
		// to walk into them; for those, skip the round-trip and
		// keep just the deep-walk assertion.
		langchaingoCompatible bool
	}{
		{
			name:                  "nullable array stays as array",
			in:                    `{"tags": {"type": ["null", "array"], "items": {"type": "string"}}}`,
			want:                  "array",
			langchaingoCompatible: true,
		},
		{
			name:                  "nullable string stays as string",
			in:                    `{"name": {"type": ["null", "string"]}}`,
			want:                  "string",
			langchaingoCompatible: true,
		},
		{
			name:                  "single string type passes through",
			in:                    `{"name": {"type": "string"}}`,
			want:                  "string",
			langchaingoCompatible: true,
		},
		{
			name: "nested properties are coerced too",
			in: `{"obj": {
				"type": "object",
				"properties": {"inner": {"type": ["null", "string"]}}
			}}`,
			want:                  "object",
			langchaingoCompatible: true,
		},
		{
			name: "oneOf member with nullable type",
			in: `{"v": {"oneOf": [
				{"type": ["null", "string"]},
				{"type": ["null", "integer"]}
			]}}`,
			// langchaingo's Definition has no oneOf — round-trip
			// drops the field but coerce should still reach inside.
		},
		{
			name: "anyOf member with nullable type",
			in:   `{"v": {"anyOf": [{"type": ["null", "boolean"]}]}}`,
		},
		{
			name: "allOf member with nullable type",
			in: `{"v": {"allOf": [
				{"type": ["null", "object"], "properties": {"x": {"type": ["null", "string"]}}}
			]}}`,
		},
		{
			name: "items as schema array (tuple)",
			in: `{"v": {"type": "array", "items": [
				{"type": ["null", "string"]},
				{"type": ["null", "integer"]}
			]}}`,
			// langchaingo's Items is a single *Definition, not an
			// array — the tuple shape isn't representable. Coerce
			// still has to walk into it though.
		},
		{
			name:                  "additionalProperties schema",
			in:                    `{"v": {"type": "object", "additionalProperties": {"type": ["null", "string"]}}}`,
			want:                  "object",
			langchaingoCompatible: true,
		},
		{
			name: "$defs nested",
			in: `{"v": {
				"type": "object",
				"$defs": {"named": {"type": ["null", "string"]}}
			}}`,
			want:                  "object",
			langchaingoCompatible: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var props map[string]any
			if err := json.Unmarshal([]byte(tc.in), &props); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			coerceNullableTypes(props)

			// Universal assertion: deep walk leaves ZERO JSON-array
			// type nodes anywhere — properties, items, oneOf, anyOf,
			// allOf, additionalProperties, $defs. This is the
			// failure mode that silently dropped tools in prod;
			// the recursion has to reach every corner.
			if containsTypeArray(props) {
				dump, _ := json.MarshalIndent(props, "", "  ")
				t.Errorf("residual JSON-array type in coerced schema:\n%s", string(dump))
			}

			// langchaingo round-trip — only when the schema shape
			// is representable. JSON-Schema 2020-12 has constructs
			// (oneOf, tuple items, etc.) langchaingo can't model.
			if !tc.langchaingoCompatible {
				return
			}
			b, err := json.Marshal(props)
			if err != nil {
				t.Fatalf("re-marshal: %v", err)
			}
			out := map[string]jsonschema.Definition{}
			if err := json.Unmarshal(b, &out); err != nil {
				t.Fatalf("post-coerce unmarshal into jsonschema.Definition: %v", err)
			}
			if tc.want != "" {
				for _, def := range out {
					if string(def.Type) != tc.want {
						t.Errorf("got type=%q, want %q", def.Type, tc.want)
					}
				}
			}
			// Nested-properties case: assert the inner field too.
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
