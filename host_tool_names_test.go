package cogito

import (
	"context"
	"sort"
	"sync/atomic"
	"testing"
	"time"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/sashabaranov/go-openai"
)

// hostToolNames is every name a host realistically supplies that cogito's
// built-ins could collide with: the four agent tools, plus names an embedding
// host defines itself and that cogito must leave alone.
var hostToolNames = append(append([]string{}, agentToolNames...), "wait_for", "wait_agents", "ordinary")

// hostMCP serves one no-op tool per name and counts the calls each receives.
func hostMCP(t *testing.T, names ...string) (*mcpsdk.ClientSession, map[string]*atomic.Int32) {
	t.Helper()
	impl := &mcpsdk.Implementation{Name: "host", Version: "0.0.1"}
	srv := mcpsdk.NewServer(impl, nil)
	calls := map[string]*atomic.Int32{}
	for _, name := range names {
		n := &atomic.Int32{}
		calls[name] = n
		mcpsdk.AddTool(srv, &mcpsdk.Tool{Name: name, Description: name + " from the host"},
			func(_ context.Context, _ *mcpsdk.CallToolRequest, _ map[string]any) (*mcpsdk.CallToolResult, map[string]any, error) {
				n.Add(1)
				return &mcpsdk.CallToolResult{Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: "host ok"}}}, nil, nil
			})
	}
	srvT, clientT := mcpsdk.NewInMemoryTransports()
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	go func() { _ = srv.Run(ctx, srvT) }()
	sess, err := mcpsdk.NewClient(impl, nil).Connect(ctx, clientT, nil)
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = sess.Close(); cancel() })
	return sess, calls
}

// callThenStop makes the model call one tool and then answer in text.
func callThenStop(name string) *scriptedChatLLM {
	call := openai.ChatCompletionChoice{FinishReason: openai.FinishReasonToolCalls, Message: openai.ChatCompletionMessage{
		Role: "assistant",
		ToolCalls: []openai.ToolCall{{ID: "c1", Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{Name: name, Arguments: `{}`}}},
	}}
	done := openai.ChatCompletionChoice{FinishReason: openai.FinishReasonStop, Message: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}}
	return &scriptedChatLLM{replies: []openai.ChatCompletionChoice{call, done}, usages: []LLMUsage{{}, {}}}
}

func requestToolCounts(req openai.ChatCompletionRequest) map[string]int {
	counts := map[string]int{}
	for _, tool := range req.Tools {
		counts[tool.Function.Name]++
	}
	return counts
}

func assertOneEach(t *testing.T, req openai.ChatCompletionRequest, want ...string) {
	t.Helper()
	counts := requestToolCounts(req)
	for _, name := range want {
		if counts[name] != 1 {
			t.Fatalf("tool %q offered %d times, want 1 (all: %v)", name, counts[name], counts)
		}
	}
	for name, n := range counts {
		if n != 1 {
			t.Fatalf("tool %q offered %d times (all: %v)", name, n, counts)
		}
	}
}

func TestHostMCPToolKeepsItsNameAndIsCallable(t *testing.T) {
	selections := map[string][]Option{
		"agent tools off":               nil,
		"agent tools on":                {EnableAgentSpawning},
		"agent tools on, none selected": {EnableAgentSpawning, WithAgentTools([]string{})},
		"agent tools on, all selected":  {EnableAgentSpawning, WithAgentTools(agentToolNames)},
		"selection set, spawning off":   {WithAgentTools([]string{})},
	}
	for label, extra := range selections {
		for _, name := range agentToolNames {
			t.Run(label+"/"+name, func(t *testing.T) {
				sess, calls := hostMCP(t, name)
				llm := callThenStop(name)
				opts := append([]Option{WithMCPs(sess)}, extra...)
				if _, err := ExecuteTools(llm, NewEmptyFragment().AddMessage("user", "go"), opts...); err != nil {
					t.Fatal(err)
				}
				if len(llm.requests) == 0 {
					t.Fatal("no model request")
				}
				assertOneEach(t, llm.requests[0], name)
				if got := calls[name].Load(); got != 1 {
					t.Fatalf("host tool %q received %d calls, want 1", name, got)
				}
			})
		}
	}
}

func TestHostToolOverridesBuiltinOfTheSameName(t *testing.T) {
	for _, name := range agentToolNames {
		t.Run(name, func(t *testing.T) {
			var ran atomic.Int32
			host := NewToolDefinition(&countingRunner{n: &ran}, struct{}{}, name, "host "+name)
			llm := callThenStop(name)
			_, err := ExecuteTools(llm, NewEmptyFragment().AddMessage("user", "go"), EnableAgentSpawning, WithTools(host))
			if err != nil {
				t.Fatal(err)
			}
			assertOneEach(t, llm.requests[0], name)
			for _, tool := range llm.requests[0].Tools {
				if tool.Function.Name == name && tool.Function.Description != "host "+name {
					t.Fatalf("model was offered the built-in %q, not the host tool", name)
				}
			}
			if ran.Load() != 1 {
				t.Fatalf("host tool ran %d times, want 1", ran.Load())
			}
		})
	}
}

type countingRunner struct{ n *atomic.Int32 }

func (r *countingRunner) Run(struct{}) (string, any, error) { r.n.Add(1); return "ok", nil, nil }

func TestNoDuplicateToolNamesAcrossTheFullSet(t *testing.T) {
	// The host supplies every name through MCP and again through WithTools, with
	// the built-ins on. The model must still see each name exactly once.
	sess, _ := hostMCP(t, hostToolNames...)
	var tools []ToolDefinitionInterface
	for _, name := range hostToolNames {
		tools = append(tools, newNamedTool(name))
	}
	llm := &captureLLM{}
	_, err := ExecuteTools(llm, NewEmptyFragment().AddMessage("user", "go"), WithMCPs(sess), EnableAgentSpawning, WithTools(tools...))
	if err != nil {
		t.Fatal(err)
	}
	req, ok := llm.request(0)
	if !ok {
		t.Fatal("no model request")
	}
	assertOneEach(t, req, hostToolNames...)
	// And the same through the prefill path.
	pre := &captureLLM{}
	if err := Prefill(context.Background(), pre, NewEmptyFragment().AddMessage("user", "go"), WithMCPs(sess), EnableAgentSpawning, WithTools(tools...)); err != nil {
		t.Fatal(err)
	}
	preq, _ := pre.request(0)
	assertOneEach(t, preq, hostToolNames...)
}

func TestUnrelatedNamesAreUnchanged(t *testing.T) {
	sess, calls := hostMCP(t, "wait_for", "ordinary")
	llm := callThenStop("wait_for")
	_, err := ExecuteTools(llm, NewEmptyFragment().AddMessage("user", "go"), WithMCPs(sess), EnableAgentSpawning)
	if err != nil {
		t.Fatal(err)
	}
	var got []string
	for name := range requestToolCounts(llm.requests[0]) {
		got = append(got, name)
	}
	sort.Strings(got)
	want := append(append([]string{}, agentToolNames...), "ordinary", "wait_for")
	sort.Strings(want)
	if len(got) != len(want) {
		t.Fatalf("got %v want %v", got, want)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("got %v want %v", got, want)
		}
	}
	assertOneEach(t, llm.requests[0], want...)
	if calls["wait_for"].Load() != 1 {
		t.Fatal("wait_for did not reach the host")
	}
}

func TestDedupeKeepsFirstOfEqualProvenance(t *testing.T) {
	a, b := newNamedTool("same"), newNamedTool("same")
	got := dedupeToolsByName(Tools{a, newNamedTool("other"), b})
	if len(got) != 2 || got[0] != a {
		t.Fatalf("got %v", got.Names())
	}
}

func TestSubAgentsInheritHostToolsNamedLikeBuiltins(t *testing.T) {
	host := newNamedTool("spawn_agent")
	builtin := newBuiltinToolDefinition(&countingRunner{n: &atomic.Int32{}}, struct{}{}, "check_agent", "builtin")
	got := FilterToolsForSubAgent(Tools{host, builtin}, nil)
	if len(got) != 1 || got[0] != host {
		t.Fatalf("got %v", got.Names())
	}
}
