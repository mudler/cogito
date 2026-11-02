package cogito

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/sashabaranov/go-openai"
)

// Use an actual SDK server and transport so metadata must survive the wire.
func invocationMCP(t *testing.T, handler mcp.ToolHandler) Tools {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	server := mcp.NewServer(&mcp.Implementation{Name: "identity-test", Version: "1"}, nil)
	server.AddTool(&mcp.Tool{Name: "memory", InputSchema: map[string]any{"type": "object", "additionalProperties": true}}, handler)
	st, ct := mcp.NewInMemoryTransports()
	go func() { _ = server.Run(ctx, st) }()
	session, err := mcp.NewClient(&mcp.Implementation{Name: "test", Version: "1"}, nil).Connect(ctx, ct, nil)
	if err != nil {
		cancel()
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = session.Close(); cancel() })
	tools, err := mcpToolsFromTransport(ctx, session, nil)
	if err != nil {
		t.Fatal(err)
	}
	return Tools(tools)
}

type invocationLLM struct {
	lifecycleLLM
	calls []openai.ToolCall
}

func (l *invocationLLM) CreateChatCompletion(_ context.Context, req openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
	l.requests = append(l.requests, req)
	msg := openai.ChatCompletionMessage{Role: "assistant", Content: "done"}
	if len(l.requests) == 1 {
		msg.ToolCalls = l.calls
	}
	return LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: msg}}}}, LLMUsage{}, nil
}
func invocationCall(id, args string) openai.ToolCall {
	return openai.ToolCall{ID: id, Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "memory", Arguments: args}}
}

func TestMCPInvocationRetry(t *testing.T) {
	var mu sync.Mutex
	var ids []any
	tools := invocationMCP(t, func(_ context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		mu.Lock()
		defer mu.Unlock()
		ids = append(ids, req.Params.Meta["cogito.tool_call_id"])
		// Simulate a host write that succeeds but whose response is lost.
		if len(ids) == 1 {
			return nil, fmt.Errorf("response lost after write")
		}
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "saved"}}}, nil
	})
	llm := &invocationLLM{calls: []openai.ToolCall{invocationCall("call-1", `{"cogito.tool_call_id":"forged"}`)}}
	opts := append(lifecycleOptions(tools...), WithMaxAttempts(2))
	if _, err := ExecuteTools(llm, NewEmptyFragment(), opts...); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if !reflect.DeepEqual(ids, []any{"call-1", "call-1"}) {
		t.Fatalf("wire invocation IDs: %v", ids)
	}
}

func TestMCPInvocationConcurrentSameTool(t *testing.T) {
	var mu sync.Mutex
	got := map[string]string{}
	arrived := make(chan struct{}, 2)
	release := make(chan struct{})
	tools := invocationMCP(t, func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		mu.Lock()
		got[string(req.Params.Arguments)] = fmt.Sprint(req.Params.Meta["cogito.tool_call_id"])
		mu.Unlock()
		arrived <- struct{}{}
		select {
		case <-release:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return &mcp.CallToolResult{}, nil
	})
	llm := &invocationLLM{calls: []openai.ToolCall{invocationCall("call-a", `{"value":"a"}`), invocationCall("call-b", `{"value":"b"}`)}}
	done := make(chan error, 1)
	go func() {
		_, err := ExecuteTools(llm, NewEmptyFragment(), append(lifecycleOptions(tools...), EnableParallelToolExecution)...)
		done <- err
	}()
	for range 2 {
		select {
		case <-arrived:
		case <-time.After(3 * time.Second):
			close(release)
			t.Fatal("calls did not overlap")
		}
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	want := map[string]string{`{"value":"a"}`: "call-a", `{"value":"b"}`: "call-b"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("wire IDs/arguments: %v", got)
	}
}

// Independent executions share the same discovered tool and MCP session. A
// cancelled call must neither borrow discovery's context nor affect its sibling.
func TestMCPInvocationCancellationNoIDSteal(t *testing.T) {
	var mu sync.Mutex
	got := map[string]string{}
	arrived := make(chan string, 2)
	stopped := make(chan struct{})
	release := make(chan struct{})
	var once sync.Once
	unblock := func() { once.Do(func() { close(release) }) }
	defer unblock()
	tools := invocationMCP(t, func(ctx context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		args := string(req.Params.Arguments)
		mu.Lock()
		got[args] = fmt.Sprint(req.Params.Meta[MCPToolCallIDMetaKey])
		mu.Unlock()
		arrived <- args
		if args == `{"value":"cancel"}` {
			<-ctx.Done()
			close(stopped)
			return nil, ctx.Err()
		}
		select {
		case <-release:
			return &mcp.CallToolResult{}, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	run := func(ctx context.Context, id, args string) <-chan error {
		done := make(chan error, 1)
		go func() {
			llm := &invocationLLM{calls: []openai.ToolCall{invocationCall(id, args)}}
			_, err := ExecuteTools(llm, NewEmptyFragment(), append(lifecycleOptions(tools...), WithContext(ctx), WithMaxAttempts(2))...)
			done <- err
		}()
		return done
	}
	cancelled := run(ctx, "call-cancel", `{"value":"cancel"}`)
	live := run(context.Background(), "call-live", `{"value":"live"}`)
	for range 2 {
		select {
		case <-arrived:
		case <-time.After(3 * time.Second):
			t.Fatal("calls did not overlap")
		}
	}
	cancel()
	select {
	case <-cancelled:
	case <-time.After(2 * time.Second):
		t.Fatal("execution ignored cancellation")
	}
	lifecycleWait(t, stopped)
	select {
	case err := <-live:
		t.Fatalf("sibling ended before release: %v", err)
	default:
	}
	unblock()
	select {
	case err := <-live:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("sibling did not finish")
	}
	mu.Lock()
	defer mu.Unlock()
	want := map[string]string{`{"value":"cancel"}`: "call-cancel", `{"value":"live"}`: "call-live"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("wire IDs/arguments: %v", got)
	}
	select {
	case args := <-arrived:
		t.Fatalf("unexpected retry after cancellation: %s", args)
	default:
	}
}

func TestMCPInvocationLegacyExecute(t *testing.T) {
	var gotID any
	var gotArgs string
	tools := invocationMCP(t, func(_ context.Context, req *mcp.CallToolRequest) (*mcp.CallToolResult, error) {
		gotID = req.Params.Meta[MCPToolCallIDMetaKey]
		gotArgs = string(req.Params.Arguments)
		return &mcp.CallToolResult{Content: []mcp.Content{&mcp.TextContent{Text: "legacy"}}}, nil
	})
	result, _, err := tools[0].Execute(map[string]any{"value": "legacy"})
	if err != nil || result != "legacy" {
		t.Fatalf("legacy Execute: %q, %v", result, err)
	}
	if gotID != nil || gotArgs != `{"value":"legacy"}` {
		t.Fatalf("legacy wire ID/arguments: %v / %s", gotID, gotArgs)
	}
	if id := ToolCallID(context.Background()); id != "" {
		t.Fatalf("ID leaked outside execution: %q", id)
	}
}

func TestToolInvocationLegacyExecute(t *testing.T) {
	calls := 0
	tool := lifecycleTool{"memory", func() (string, any, error) { calls++; return "legacy", nil, nil }}
	if _, ok := any(tool).(ContextToolDefinitionInterface); ok {
		t.Fatal("fixture must implement only legacy Execute")
	}
	llm := &invocationLLM{calls: []openai.ToolCall{invocationCall("call-legacy", `{}`)}}
	if _, err := ExecuteTools(llm, NewEmptyFragment(), lifecycleOptions(tool)...); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Fatalf("legacy executions: %d", calls)
	}
}
