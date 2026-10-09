package cogito

import (
	"context"
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// Test the discovery boundary, not a hand-built approximation of Parameters.
func TestMCPDiscoveryPreservesInputSchema(t *testing.T) {
	cases := map[string]string{
		"nullable enum and defaults":         `{"type":"object","properties":{"status":{"anyOf":[{"enum":["pending","shipped","delivered","cancelled"],"type":"string"},{"type":"null"}],"default":null},"limit":{"default":20,"type":"integer"}}}`,
		"root references and compositions":   `{"type":"object","$schema":"https://json-schema.org/draft/2020-12/schema","$defs":{"args":{"type":"object","properties":{"id":{"type":"integer","minimum":1}}}},"$ref":"#/$defs/args","allOf":[{"required":["id"]}],"anyOf":[{"required":["id"]},false],"oneOf":[{"required":["id"]}],"unevaluatedProperties":false,"title":"Arguments","x-extension":{"enabled":false}}`,
		"nullable types and boolean schemas": `{"type":"object","properties":{"tags":{"type":["null","array"],"items":{"type":["string","null"]}},"rows":{"type":"array","items":{"type":"array","items":true}},"never":false,"anything":true,"closed":{"type":"object","additionalProperties":false},"flag":{"type":"boolean","default":false},"tuple":{"prefixItems":[true,false],"items":false}},"additionalProperties":false}`,
		"nested constraints":                 `{"type":"object","properties":{"config":{"type":"object","patternProperties":{"^x":{"type":["string","null"]}},"additionalProperties":{"type":"integer"},"dependentSchemas":{"x":{"required":["y"]}},"propertyNames":{"minLength":1}},"values":{"type":"array","contains":{"const":0},"minContains":1,"maxItems":10,"uniqueItems":true,"items":{"anyOf":[{"type":"integer"},false]}}},"if":{"required":["config"]},"then":{"required":["values"]},"else":{"not":{"required":["values"]}},"required":[]}`,
		"no arguments absent properties":     `{"type":"object"}`,
		"no arguments explicit properties":   `{"type":"object","properties":{},"additionalProperties":false}`,
	}
	for name, schema := range cases {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			impl := &mcp.Implementation{Name: "schema-test", Version: "1"}
			server := mcp.NewServer(impl, nil)
			server.AddTool(&mcp.Tool{Name: "example", Description: "Example tool", InputSchema: json.RawMessage(schema)}, func(context.Context, *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
				return &mcp.CallToolResult{}, nil
			})
			st, ct := mcp.NewInMemoryTransports()
			ss, err := server.Connect(ctx, st, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer ss.Close()
			session, err := mcp.NewClient(impl, nil).Connect(ctx, ct, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer session.Close()
			tools, err := mcpToolsFromTransport(ctx, session, nil)
			if err != nil {
				t.Fatal(err)
			}
			if len(tools) != 1 {
				t.Fatalf("discovered %d tools, want 1", len(tools))
			}
			tool := tools[0].Tool()
			if tool.Function.Name != "example" || tool.Function.Description != "Example tool" {
				t.Fatalf("tool metadata changed: %+v", tool)
			}
			got, err := json.Marshal(tool.Function.Parameters)
			if err != nil {
				t.Fatal(err)
			}
			var wantValue, gotValue any
			if err := json.Unmarshal([]byte(schema), &wantValue); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(got, &gotValue); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(wantValue, gotValue) {
				t.Fatalf("schema changed during discovery:\nwant %s\n got %s", schema, got)
			}
		})
	}
}
