package cogito

import (
	"encoding/json"
	"testing"

	"github.com/tmc/langchaingo/jsonschema"
)

// containsBoolSchema reports whether any schema-valued slot in v is still a
// JSON boolean (i.e. the normalization missed it). Mirrors containsTypeArray.
func containsBoolSchema(v any) bool {
	// Keywords whose value IS a schema, so a bool there is a boolean schema.
	// "properties"/"$defs" etc. are bags of schemas and handled by recursion.
	single := []string{"items", "additionalProperties", "contains", "not",
		"if", "then", "else", "propertyNames"}
	switch n := v.(type) {
	case map[string]any:
		for _, k := range single {
			if _, ok := n[k].(bool); ok {
				return true
			}
		}
		for _, child := range n {
			if containsBoolSchema(child) {
				return true
			}
		}
	case []any:
		for _, child := range n {
			if containsBoolSchema(child) {
				return true
			}
		}
	}
	return false
}

// JSON Schema 2020-12 lets any schema be the boolean `true` ("allow anything")
// or `false` ("allow nothing"), and google/jsonschema-go emits exactly that for
// a Go `any` — an empty schema marshals as `true`. langchaingo's
// jsonschema.Definition models every nested schema as a *Definition, so a bool
// fails the unmarshal and mcpToolsFromTransport drops the ENTIRE tool with a
// single log line. The server is healthy and its tool simply never reaches the
// model. Same failure class as the type-array coercion next door.
func TestCoerceBooleanSchemas(t *testing.T) {
	cases := []struct {
		name string
		in   string
	}{
		{
			// The real-world shape that surfaced this: Go [][]any (rows of
			// cells) infers `items: true` for the innermost element.
			name: "items true nested in array of arrays",
			in:   `{"rows":{"type":"array","items":{"type":"array","items":true}}}`,
		},
		{
			name: "additionalProperties false",
			in:   `{"cfg":{"type":"object","additionalProperties":false}}`,
		},
		{
			name: "bool schema inside properties",
			in:   `{"outer":{"type":"object","properties":{"anything":true}}}`,
		},
		{
			name: "bool schema in composition keyword",
			in:   `{"x":{"anyOf":[true,{"type":"string"}]}}`,
		},
		{
			name: "bool schema in $defs and prefixItems",
			in:   `{"x":{"$defs":{"any":true},"prefixItems":[true,{"type":"string"}]}}`,
		},
		{
			name: "false schema is preserved as allow-nothing",
			in:   `{"never":{"type":"array","items":false}}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var props map[string]any
			if err := json.Unmarshal([]byte(tc.in), &props); err != nil {
				t.Fatalf("bad test fixture: %v", err)
			}

			coerceNullableTypes(props)

			if containsBoolSchema(props) {
				got, _ := json.Marshal(props)
				t.Errorf("a boolean schema survived normalization: %s", got)
			}

			// The point of the exercise: cogito must be able to convert the
			// result into langchaingo Definitions, or it drops the tool.
			dat, err := json.Marshal(props)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			defs := map[string]jsonschema.Definition{}
			if err := json.Unmarshal(dat, &defs); err != nil {
				t.Errorf("tool would be DROPPED — properties do not convert to langchaingo Definitions: %v\ngot: %s", err, dat)
			}
		})
	}
}

// `true` means "anything allowed" and `false` means "nothing allowed". The
// rewrite must preserve that distinction rather than flattening both to an
// empty schema, which would silently widen a deliberately-closed schema.
func TestBooleanSchemaSemanticsPreserved(t *testing.T) {
	var props map[string]any
	if err := json.Unmarshal([]byte(`{"a":{"items":true},"b":{"items":false}}`), &props); err != nil {
		t.Fatal(err)
	}
	coerceNullableTypes(props)

	a := props["a"].(map[string]any)["items"]
	if m, ok := a.(map[string]any); !ok || len(m) != 0 {
		t.Errorf(`items:true must become the empty schema {} (allow anything), got %#v`, a)
	}

	b := props["b"].(map[string]any)["items"]
	m, ok := b.(map[string]any)
	if !ok {
		t.Fatalf(`items:false must become an object, got %#v`, b)
	}
	if _, hasNot := m["not"]; !hasNot {
		t.Errorf(`items:false must become {"not":{}} (allow nothing), got %#v`, m)
	}
}

// Normalization must not disturb a schema that was already convertible.
func TestNormalizationLeavesPlainSchemasAlone(t *testing.T) {
	const in = `{"name":{"type":"string","description":"a name"},"tags":{"type":"array","items":{"type":"string"}}}`
	var props map[string]any
	if err := json.Unmarshal([]byte(in), &props); err != nil {
		t.Fatal(err)
	}
	coerceNullableTypes(props)

	got, err := json.Marshal(props)
	if err != nil {
		t.Fatal(err)
	}
	var want, have any
	_ = json.Unmarshal([]byte(in), &want)
	_ = json.Unmarshal(got, &have)
	if string(got) == "" || !jsonEqual(want, have) {
		t.Errorf("plain schema was modified:\n want %s\n  got %s", in, got)
	}
}

func jsonEqual(a, b any) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return string(x) == string(y)
}
