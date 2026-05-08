package cogito

import (
	"context"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// startInMemoryMCP spins up an MCP server with the named no-op tools,
// connects an in-memory client, and returns the connected session
// along with a teardown function.
func startInMemoryMCP(toolNames ...string) (*mcpsdk.ClientSession, func()) {
	impl := &mcpsdk.Implementation{Name: "stub", Version: "0.0.1"}
	srv := mcpsdk.NewServer(impl, nil)
	for _, name := range toolNames {
		name := name
		mcpsdk.AddTool(
			srv,
			&mcpsdk.Tool{Name: name, Description: name + " (stub)"},
			func(_ context.Context, _ *mcpsdk.CallToolRequest, _ map[string]any) (*mcpsdk.CallToolResult, map[string]any, error) {
				return &mcpsdk.CallToolResult{}, nil, nil
			},
		)
	}

	srvT, clientT := mcpsdk.NewInMemoryTransports()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)

	go func() {
		_ = srv.Run(ctx, srvT)
	}()

	client := mcpsdk.NewClient(impl, nil)
	sess, err := client.Connect(ctx, clientT, nil)
	Expect(err).ToNot(HaveOccurred())

	teardown := func() {
		_ = sess.Close()
		cancel()
	}
	return sess, teardown
}

var _ = Describe("MCPToolFilter", func() {
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

	It("drops tools the filter rejects from the discovered set", func() {
		sess, teardown = startInMemoryMCP("list_issues", "delete_issue")

		keep := map[string]bool{"list_issues": true} // delete_issue absent → drop
		called := map[string]int{}
		filter := func(s *mcpsdk.ClientSession, tool string) bool {
			Expect(s).To(Equal(sess), "filter must receive the session it discovers from")
			called[tool]++
			return keep[tool]
		}

		tools, err := mcpToolsFromTransport(context.Background(), sess, filter)
		Expect(err).ToNot(HaveOccurred())
		Expect(tools).To(HaveLen(1))

		mt, ok := tools[0].(*mcpTool)
		Expect(ok).To(BeTrue())
		Expect(mt.name).To(Equal("list_issues"))
		Expect(called["list_issues"]).To(BeNumerically(">", 0))
		Expect(called["delete_issue"]).To(BeNumerically(">", 0))
	})

	It("treats a nil filter as always-allow (default Options state)", func() {
		sess, teardown = startInMemoryMCP("alpha", "beta")
		tools, err := mcpToolsFromTransport(context.Background(), sess, nil)
		Expect(err).ToNot(HaveOccurred())
		Expect(tools).To(HaveLen(2))
	})

	It("yields an empty slice (not nil-error) when every tool is rejected", func() {
		sess, teardown = startInMemoryMCP("x", "y", "z")
		tools, err := mcpToolsFromTransport(
			context.Background(),
			sess,
			func(*mcpsdk.ClientSession, string) bool { return false },
		)
		Expect(err).ToNot(HaveOccurred())
		Expect(tools).To(BeEmpty())
	})
})
