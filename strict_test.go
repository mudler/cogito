package cogito

import (
	"context"
	"encoding/json"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/sashabaranov/go-openai"
)

type shellArgs struct {
	Script  string `json:"script" jsonschema:"the script to run"`
	Timeout int    `json:"timeout,omitempty" jsonschema:"seconds"`
}

// startTypedMCP serves one tool whose schema the go-sdk infers from
// shellArgs, as nib's bash tool does: script required, timeout optional,
// additionalProperties false. The server validates every call against it.
func startTypedMCP() (*mcpsdk.ClientSession, func()) {
	impl := &mcpsdk.Implementation{Name: "typed", Version: "0.0.1"}
	srv := mcpsdk.NewServer(impl, nil)
	mcpsdk.AddTool(srv, &mcpsdk.Tool{Name: "bash", Description: "run a script"},
		func(_ context.Context, _ *mcpsdk.CallToolRequest, in shellArgs) (*mcpsdk.CallToolResult, any, error) {
			return &mcpsdk.CallToolResult{Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "ran " + in.Script}}}, nil, nil
		})
	srvT, clientT := mcpsdk.NewInMemoryTransports()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	go func() { _ = srv.Run(ctx, srvT) }()
	sess, err := mcpsdk.NewClient(impl, nil).Connect(ctx, clientT, nil)
	Expect(err).ToNot(HaveOccurred())
	return sess, func() { _ = sess.Close(); cancel() }
}

func paramsOf(t openai.Tool) map[string]any {
	raw, err := json.Marshal(t.Function.Parameters)
	Expect(err).ToNot(HaveOccurred())
	var m map[string]any
	Expect(json.Unmarshal(raw, &m)).To(Succeed())
	return m
}

var _ = Describe("strict tool schemas", func() {
	var (
		sess     *mcpsdk.ClientSession
		teardown func()
	)
	AfterEach(func() {
		if teardown != nil {
			teardown()
			teardown = nil
		}
	})

	It("advertises an MCP tool with the additionalProperties its server declares", func() {
		sess, teardown = startTypedMCP()
		tools, err := mcpToolsFromTransport(context.Background(), sess, nil)
		Expect(err).ToNot(HaveOccurred())
		Expect(tools).To(HaveLen(1))

		p := paramsOf(tools[0].Tool())
		Expect(p).To(HaveKeyWithValue("additionalProperties", false))
		Expect(p["required"]).To(ConsistOf("script"))
		Expect(tools[0].Tool().Function.Strict).To(BeFalse(), "strict is opt-in")
	})

	It("sends a strict schema and drops the nulls it makes the model send", func() {
		sess, teardown = startTypedMCP()
		tools, err := mcpToolsFromTransport(context.Background(), sess, nil)
		Expect(err).ToNot(HaveOccurred())
		tool := strictTools(tools)[0]

		def := tool.Tool()
		Expect(def.Function.Strict).To(BeTrue())
		p := paramsOf(def)
		Expect(p).To(HaveKeyWithValue("additionalProperties", false))
		Expect(p["required"]).To(ConsistOf("script", "timeout"))
		timeout := p["properties"].(map[string]any)["timeout"].(map[string]any)
		Expect(timeout["type"]).To(ConsistOf("integer", "null"))

		// The server rejects "timeout": null (it wants an integer), so this
		// only succeeds if the wrapper leaves the null out.
		out, _, err := tool.Execute(map[string]any{"script": "ls", "timeout": nil})
		Expect(err).ToNot(HaveOccurred())
		Expect(out).To(ContainSubstring("ran ls"))
	})

	It("wraps the tools usableTools returns only when the option is set", func() {
		sess, teardown = startTypedMCP()
		plain, _, _, err := usableTools(nil, NewEmptyFragment(), WithMCPs(sess))
		Expect(err).ToNot(HaveOccurred())
		Expect(plain[0].Tool().Function.Strict).To(BeFalse())

		strict, _, _, err := usableTools(nil, NewEmptyFragment(), WithMCPs(sess), EnableStrictToolSchemas)
		Expect(err).ToNot(HaveOccurred())
		Expect(strict[0].Tool().Function.Strict).To(BeTrue())
		Expect(strictTools(strict)[0]).To(Equal(strict[0]), "wrapping twice is a no-op")
	})
})

var _ = Describe("strictSchema", func() {
	parse := func(s string) map[string]any {
		var m map[string]any
		Expect(json.Unmarshal([]byte(s), &m)).To(Succeed())
		return m
	}

	It("makes nested optional properties nullable and closes every object", func() {
		m := parse(`{"type":"object","properties":{
			"mode":{"type":"string","enum":["a","b"]},
			"opts":{"type":"object","properties":{"deep":{"type":"boolean"}}},
			"list":{"type":"array","items":{"type":"object","properties":{"k":{"type":"string"}},"required":["k"]}}
		},"required":["opts"]}`)
		Expect(strictSchema(m)).To(BeTrue())
		Expect(m["required"]).To(ConsistOf("mode", "opts", "list"))

		props := m["properties"].(map[string]any)
		mode := props["mode"].(map[string]any)
		Expect(mode["type"]).To(ConsistOf("string", "null"))
		Expect(mode["enum"]).To(ContainElement(BeNil()))

		opts := props["opts"].(map[string]any)
		Expect(opts["type"]).To(Equal("object"), "a required property stays non-null")
		Expect(opts).To(HaveKeyWithValue("additionalProperties", false))
		deep := opts["properties"].(map[string]any)["deep"].(map[string]any)
		Expect(deep["type"]).To(ConsistOf("boolean", "null"))

		item := props["list"].(map[string]any)["items"].(map[string]any)
		Expect(item).To(HaveKeyWithValue("additionalProperties", false))
		Expect(item["required"]).To(ConsistOf("k"))
	})

	It("wraps an optional property without a type in anyOf with null", func() {
		m := parse(`{"type":"object","properties":{"v":{"anyOf":[{"type":"string"},{"type":"integer"}]}}}`)
		Expect(strictSchema(m)).To(BeTrue())
		v := m["properties"].(map[string]any)["v"].(map[string]any)
		Expect(v["anyOf"]).To(HaveLen(2))
		Expect(v["anyOf"].([]any)[1]).To(Equal(map[string]any{"type": "null"}))
	})

	DescribeTable("refuses what strict mode cannot express",
		func(schema string) {
			Expect(strictSchema(parse(schema))).To(BeFalse())
		},
		Entry("a map-typed object", `{"type":"object","additionalProperties":{"type":"string"}}`),
		Entry("open objects", `{"type":"object","additionalProperties":true}`),
		Entry("oneOf", `{"type":"object","properties":{"x":{"oneOf":[{"type":"string"}]}}}`),
	)

	It("leaves such a tool unchanged and not strict", func() {
		inner := &fakeParamsTool{params: map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "string"}}}
		def := strictTool{inner: inner}.Tool()
		Expect(def.Function.Strict).To(BeFalse())
		Expect(paramsOf(def)["additionalProperties"]).To(Equal(map[string]any{"type": "string"}))
	})
})

type fakeParamsTool struct{ params any }

func (f *fakeParamsTool) Tool() openai.Tool {
	return openai.Tool{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "f", Parameters: f.params}}
}
func (f *fakeParamsTool) Execute(map[string]any) (string, any, error) { return "", nil, nil }
