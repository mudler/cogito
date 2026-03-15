package cogito

// StreamEventType identifies the kind of streaming event.
type StreamEventType string

const (
	StreamEventReasoning  StreamEventType = "reasoning"    // LLM thinking delta
	StreamEventContent    StreamEventType = "content"      // answer text delta
	StreamEventToolCall   StreamEventType = "tool_call"    // tool selected + args
	StreamEventToolResult StreamEventType = "tool_result"  // tool execution result
	StreamEventStatus     StreamEventType = "status"       // status message
	StreamEventDone       StreamEventType = "done"         // stream complete
	StreamEventError      StreamEventType = "error"        // error
)

// StreamEvent represents a single streaming event from the LLM or tool pipeline.
type StreamEvent struct {
	Type       StreamEventType
	Content    string   // text delta (reasoning/content)
	ToolName   string   // for tool_call/tool_result
	ToolArgs   string   // accumulated JSON args
	ToolResult string   // tool result text
	Error      error    // populated on error
	Usage      LLMUsage // populated on done
}

// StreamCallback is a function that receives streaming events.
type StreamCallback func(StreamEvent)
