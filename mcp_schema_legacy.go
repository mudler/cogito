package cogito

// CoerceNullableTypes applies the legacy lossy schema normalization.
//
// Deprecated: MCP discovery now preserves the complete input schema and does
// not need this workaround. Retained only for existing callers.
func CoerceNullableTypes(props map[string]any) { coerceNullableTypes(props) }

// coerceNullableTypes recursively walks a property bag and normalizes the
// JSON-Schema 2020-12 constructs that langchaingo/jsonschema.Definition
// cannot represent. Both rewrites exist for the same reason: Definition is
// the struct the old discovery pipeline used. This helper is no longer
// called by discovery because it changes the server's schema semantics.
//
//  1. "type": ["null", "X"] becomes "type": "X". Definition.Type is a single
//     string. Picks the first non-null member; falls back to the first
//     member if all are null.
//  2. A boolean schema becomes its object equivalent: `true` (allow
//     anything) becomes {}, `false` (allow nothing) becomes {"not": {}}.
//     2020-12 permits a boolean wherever a schema is allowed, and
//     google/jsonschema-go emits exactly that for a Go `any` — an empty
//     schema marshals as `true`, so a [][]any field yields "items": true.
//     Definition models nested schemas as *Definition, so a bare bool fails
//     the unmarshal.
//
// Recurses into every nested schema location: properties, items,
// oneOf/anyOf/allOf members, prefixItems, $defs/definitions,
// additionalProperties, patternProperties, contains, not, if/then/else,
// propertyNames.
func coerceNullableTypes(props map[string]any) {
	if props == nil {
		return
	}
	// Range with the key so a property that is ITSELF a boolean schema can be
	// replaced in place; the value-only loop could not rewrite it.
	for name, raw := range props {
		if b, ok := raw.(bool); ok {
			props[name] = boolSchema(b)
			continue
		}
		coerceSchema(raw)
	}
}

// boolSchema returns the object form of a JSON-Schema boolean schema,
// preserving its meaning: `true` allows anything, `false` allows nothing.
// Collapsing both to {} would silently widen a deliberately-closed schema.
func boolSchema(allow bool) map[string]any {
	if allow {
		return map[string]any{}
	}
	return map[string]any{"not": map[string]any{}}
}

// schemaValuedKeys are the keywords whose value is a single schema, so a
// boolean there is a boolean schema rather than an ordinary flag.
var schemaValuedKeys = []string{
	"items", "additionalProperties", "contains", "not",
	"if", "then", "else", "propertyNames",
}

// schemaListKeys are the keywords whose value is an array of schemas.
var schemaListKeys = []string{"oneOf", "anyOf", "allOf", "prefixItems", "items"}

// normalizeBoolSchemas replaces every boolean schema directly under obj with
// its object equivalent. Nested bags (properties, $defs, …) are reached by
// coerceNullableTypes, which performs the same replacement for their members.
func normalizeBoolSchemas(obj map[string]any) {
	for _, key := range schemaValuedKeys {
		if b, ok := obj[key].(bool); ok {
			obj[key] = boolSchema(b)
		}
	}
	for _, key := range schemaListKeys {
		arr, ok := obj[key].([]any)
		if !ok {
			continue
		}
		for i, member := range arr {
			if b, ok := member.(bool); ok {
				arr[i] = boolSchema(b)
			}
		}
	}
}

// coerceSchema applies the type-array → string and boolean-schema →
// object rewrites to a single schema node, then recurses into every
// nested schema location.
func coerceSchema(node any) {
	obj, ok := node.(map[string]any)
	if !ok {
		return
	}

	// Run first, so the objects it creates are recursed into below like any
	// other nested schema.
	normalizeBoolSchemas(obj)

	if t, ok := obj["type"].([]any); ok {
		pick := ""
		for _, m := range t {
			s, ok := m.(string)
			if !ok || s == "null" {
				continue
			}
			pick = s
			break
		}
		if pick == "" && len(t) > 0 {
			if s, ok := t[0].(string); ok {
				pick = s
			}
		}
		if pick != "" {
			obj["type"] = pick
		}
	}

	// Object properties — map of name → schema.
	if nested, ok := obj["properties"].(map[string]any); ok {
		coerceNullableTypes(nested)
	}
	// patternProperties — same shape as properties, just regex-keyed.
	if nested, ok := obj["patternProperties"].(map[string]any); ok {
		coerceNullableTypes(nested)
	}
	// $defs / definitions — JSON-Schema named schema bag.
	if nested, ok := obj["$defs"].(map[string]any); ok {
		coerceNullableTypes(nested)
	}
	if nested, ok := obj["definitions"].(map[string]any); ok {
		coerceNullableTypes(nested)
	}
	// Single nested schema fields.
	coerceSchema(obj["items"])
	coerceSchema(obj["additionalProperties"])
	coerceSchema(obj["contains"])
	coerceSchema(obj["not"])
	coerceSchema(obj["if"])
	coerceSchema(obj["then"])
	coerceSchema(obj["else"])
	coerceSchema(obj["propertyNames"])
	// Composition keywords — arrays of schemas.
	for _, key := range []string{"oneOf", "anyOf", "allOf"} {
		arr, ok := obj[key].([]any)
		if !ok {
			continue
		}
		for _, member := range arr {
			coerceSchema(member)
		}
	}
	// "items" can also be an array (tuple validation in older drafts).
	if arr, ok := obj["items"].([]any); ok {
		for _, member := range arr {
			coerceSchema(member)
		}
	}
	// "prefixItems" (2020-12 tuple).
	if arr, ok := obj["prefixItems"].([]any); ok {
		for _, member := range arr {
			coerceSchema(member)
		}
	}
}
