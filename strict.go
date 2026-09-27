package cogito

import (
	"encoding/json"
	"sort"

	"github.com/sashabaranov/go-openai"
)

// strictTool advertises inner with "strict": true and a schema that follows
// the rules strict mode needs:
//
//   - every object sets additionalProperties to false;
//   - every property is listed in required, so a property that was optional
//     becomes nullable instead, and the model sends null to leave it out.
//
// Execute removes those nulls again before it calls inner, so inner sees an
// omitted argument, as it would without strict mode. A schema that strict
// mode cannot express (an object whose additionalProperties is a schema or
// true, oneOf, a non-object top level) is sent unchanged and not strict.
type strictTool struct {
	inner ToolDefinitionInterface
}

// strictTools wraps each tool in strictTool, once.
func strictTools(tools Tools) Tools {
	out := make(Tools, len(tools))
	for i, t := range tools {
		if _, done := t.(strictTool); done {
			out[i] = t
			continue
		}
		out[i] = strictTool{inner: t}
	}
	return out
}

func (t strictTool) Tool() openai.Tool {
	tool := t.inner.Tool()
	if tool.Function == nil || tool.Function.Parameters == nil {
		return tool
	}
	raw, err := json.Marshal(tool.Function.Parameters)
	if err != nil {
		return tool
	}
	var schema map[string]any
	if err := json.Unmarshal(raw, &schema); err != nil {
		return tool
	}
	if schema["type"] != "object" || !strictSchema(schema) {
		return tool
	}
	fn := *tool.Function
	fn.Parameters = schema
	fn.Strict = true
	tool.Function = &fn
	return tool
}

func (t strictTool) Execute(args map[string]any) (string, any, error) {
	return t.inner.Execute(dropNulls(args))
}

// strictSchema rewrites schema in place to follow strict mode's rules and
// reports whether it could. On false the caller must discard schema.
func strictSchema(schema map[string]any) bool {
	if _, ok := schema["oneOf"]; ok {
		return false
	}
	if isObjectSchema(schema) {
		switch ap := schema["additionalProperties"].(type) {
		case nil:
		case bool:
			if ap {
				return false
			}
		default:
			return false // a map type: strict mode cannot describe it
		}
		schema["additionalProperties"] = false

		props, _ := schema["properties"].(map[string]any)
		if props == nil {
			props = map[string]any{}
			schema["properties"] = props
		}
		required := map[string]bool{}
		if rs, ok := schema["required"].([]any); ok {
			for _, r := range rs {
				if s, ok := r.(string); ok {
					required[s] = true
				}
			}
		}
		names := make([]string, 0, len(props))
		for name, p := range props {
			ps, ok := p.(map[string]any)
			if !ok {
				return false // a boolean schema
			}
			if !strictSchema(ps) {
				return false
			}
			if !required[name] {
				props[name] = nullable(ps)
			}
			names = append(names, name)
		}
		sort.Strings(names)
		req := make([]any, len(names))
		for i, n := range names {
			req[i] = n
		}
		schema["required"] = req
	}
	if items, ok := schema["items"].(map[string]any); ok && !strictSchema(items) {
		return false
	}
	for _, key := range []string{"anyOf", "allOf"} {
		if list, ok := schema[key].([]any); ok {
			for _, s := range list {
				if m, ok := s.(map[string]any); ok && !strictSchema(m) {
					return false
				}
			}
		}
	}
	for _, key := range []string{"$defs", "definitions"} {
		if defs, ok := schema[key].(map[string]any); ok {
			for _, d := range defs {
				if m, ok := d.(map[string]any); ok && !strictSchema(m) {
					return false
				}
			}
		}
	}
	return true
}

func isObjectSchema(schema map[string]any) bool {
	if schema["type"] == "object" {
		return true
	}
	if ts, ok := schema["type"].([]any); ok {
		for _, t := range ts {
			if t == "object" {
				return true
			}
		}
	}
	return false
}

// nullable returns schema extended to also accept null.
func nullable(schema map[string]any) map[string]any {
	switch t := schema["type"].(type) {
	case string:
		schema["type"] = []any{t, "null"}
	case []any:
		for _, x := range t {
			if x == "null" {
				return schema
			}
		}
		schema["type"] = append(t, "null")
	default:
		return map[string]any{"anyOf": []any{schema, map[string]any{"type": "null"}}}
	}
	if enum, ok := schema["enum"].([]any); ok {
		schema["enum"] = append(enum, nil)
	}
	return schema
}

// dropNulls returns args without the null values strict mode makes the
// model send for an argument it leaves out, in nested objects too.
func dropNulls(args map[string]any) map[string]any {
	if args == nil {
		return nil
	}
	out := make(map[string]any, len(args))
	for k, v := range args {
		switch x := v.(type) {
		case nil:
			continue
		case map[string]any:
			out[k] = dropNulls(x)
		default:
			out[k] = v
		}
	}
	return out
}
