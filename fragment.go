package cogito

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/mudler/cogito/structures"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

type MessageRole string

const (
	AssistantMessageRole MessageRole = "assistant"
	UserMessageRole      MessageRole = "user"
	ToolMessageRole      MessageRole = "tool"
	SystemMessageRole    MessageRole = "system"
)

func (m MessageRole) String() string {
	return string(m)
}

type InjectedMessage struct {
	Message   openai.ChatCompletionMessage
	Iteration int // Iteration number when message was injected
}

type Status struct {
	LastUsage        LLMUsage // Track token usage from the last LLM call
	CumulativeUsage  LLMUsage // Sum of token usage across every LLM call in the run
	Iterations       int
	ToolsCalled      Tools
	ToolResults      []ToolStatus
	Plans            []PlanStatus
	PastActions      []ToolStatus         // Track past actions for loop detections
	ReasoningLog     []string             // Track reasoning for each iteration
	TODOs            *structures.TODOList // TODO tracking for iterative execution
	TODOIteration    int                  // Current TODO iteration
	TODOPhase        string               // Current phase: "work" or "review"
	InjectedMessages []InjectedMessage    // Track successfully injected messages with timing
}

type Fragment struct {
	// ConversationMessages owns lossless envelopes; nil selects legacy Messages.
	ConversationMessages []Message
	Messages             []openai.ChatCompletionMessage
	ParentFragment       *Fragment
	Status               *Status
	Multimedia           []Multimedia
	PendingNativeParts   []NativePart // transient: audio/video for the current turn (send-once)
}

// GetMessages is a display/legacy-only projection, not a lossless replay API.
// It returns the chat completion messages from this fragment,
// automatically prepending a force-text-reply system message if tool calls are detected.
// This ensures LLMs provide natural language responses instead of JSON tool syntax
// when Ask() is called after ExecuteTools().
func (f Fragment) GetMessages() []openai.ChatCompletionMessage {

	// TODO: this is kinda of brittle because LLM interface implementers needs to call this methods to get the messages,
	// but we don't enforce it - worse is that we change the user messages and they might not expect that when calling GetMessages().
	// We should move away from this, and have a more explicit way to get the messages.

	messages := PortableMessages(WrapMessages(f.Messages))

	// Check if conversation contains tool calls
	hasToolCalls := false
	for _, msg := range messages {
		if len(msg.ToolCalls) > 0 {
			hasToolCalls = true
			break
		}
	}

	// If tool calls detected, prepend instruction for text-only reply
	// This prevents the LLM from outputting JSON tool syntax like:
	// [{"index":0,"type":"function","function":{"name":"tool","arguments":"..."}}]
	if hasToolCalls {
		// Reply to the user without using any tools or function calls. Just reply with the message.
		forceTextReply := "Provide a natural language response to the user. Do not use any tools or function calls in your reply."

		messages = append([]openai.ChatCompletionMessage{
			{
				Role:    "system",
				Content: forceTextReply,
			},
		}, messages...)
	}

	return normalizeSystemMessages(messages)
}

func NewEmptyFragment() Fragment {
	return Fragment{
		Status: &Status{
			PastActions:  []ToolStatus{},
			ReasoningLog: []string{},
			ToolsCalled:  Tools{},
			ToolResults:  []ToolStatus{},
			LastUsage:    LLMUsage{},
		},
	}
}

func NewFragment(messages ...openai.ChatCompletionMessage) Fragment {
	return Fragment{
		Messages: messages,
		Status: &Status{
			PastActions:  []ToolStatus{},
			ReasoningLog: []string{},
			ToolsCalled:  Tools{},
			ToolResults:  []ToolStatus{},
			LastUsage:    LLMUsage{},
		},
	}
}

type MediaKind int

const (
	MediaImage MediaKind = iota
	MediaAudio
	MediaVideo
)

// Multimedia is unchanged for backward compatibility (image via URL()).
type Multimedia interface {
	URL() string
}

// TypedMultimedia is the opt-in richer form carrying a media kind and payload.
// AddMessage type-asserts for it; a plain Multimedia is treated as MediaImage.
type TypedMultimedia interface {
	Multimedia
	MediaKind() MediaKind
	Data() string   // raw base64 (no "data:" prefix), used for input_audio
	Format() string // audio container, e.g. "wav"; "" for image/video
}

// NativePart is a cogito-owned audio/video part awaiting send-once serialization
// by the LocalAI client (go-openai's ChatMessagePart cannot represent them).
type NativePart struct {
	Kind   MediaKind
	URL    string // data: URI (video)
	Data   string // base64 (audio)
	Format string // "wav" (audio)
}

// NativePartsAware is implemented by clients that can serialize NativeParts
// (the LocalAI client). Fragment->request sites type-assert for it and set the
// current turn's parts before issuing a request.
type NativePartsAware interface {
	SetPendingNativeParts(parts []NativePart)
}

func (r Fragment) AddMessage(role MessageRole, content string, mm ...Multimedia) Fragment {
	chatCompletionMessage := openai.ChatCompletionMessage{
		Role: role.String(),
	}

	// Separate image parts (which persist via go-openai MultiContent) from
	// audio/video parts (which cannot live in a ChatMessagePart and are held
	// transiently for send-once serialization by the LocalAI client).
	r.Multimedia = slices.Clone(r.Multimedia)
	r.PendingNativeParts = slices.Clone(r.PendingNativeParts)
	var imageParts []Multimedia
	for _, m := range mm {
		r.Multimedia = append(r.Multimedia, m)
		kind := MediaImage
		if tm, ok := m.(TypedMultimedia); ok {
			kind = tm.MediaKind()
		}
		switch kind {
		case MediaAudio, MediaVideo:
			np := NativePart{Kind: kind, URL: m.URL()}
			if tm, ok := m.(TypedMultimedia); ok {
				np.Data = tm.Data()
				np.Format = tm.Format()
			}
			r.PendingNativeParts = append(r.PendingNativeParts, np)
		default:
			imageParts = append(imageParts, m)
		}
	}

	if len(imageParts) > 0 {
		multiContent := []openai.ChatMessagePart{
			{Text: content, Type: openai.ChatMessagePartTypeText},
		}
		for _, img := range imageParts {
			multiContent = append(multiContent, openai.ChatMessagePart{
				Type:     openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{URL: img.URL()},
			})
		}
		chatCompletionMessage.MultiContent = multiContent
	} else {
		chatCompletionMessage.Content = content
	}

	return r.appendPortableMessage(chatCompletionMessage)
}

// AddToolMessage adds a tool result message with the specified tool_call_id
func (r Fragment) AddToolMessage(content, toolCallID string) Fragment {
	chatCompletionMessage := openai.ChatCompletionMessage{
		Role:       "tool",
		Content:    content,
		ToolCallID: toolCallID,
	}

	return r.appendPortableMessage(chatCompletionMessage)
}

func (r Fragment) AddStartMessage(role MessageRole, content string, mm ...Multimedia) Fragment {
	message := Message{Portable: openai.ChatCompletionMessage{Role: role.String(), Content: content}}
	if r.ConversationMessages == nil {
		r.Messages = append(PortableMessages([]Message{message}), PortableMessages(WrapMessages(r.Messages))...)
		return r
	}
	// Prepend independently so an existing projection mismatch stays invalid.
	r.ConversationMessages = append([]Message{message}, CloneMessages(r.ConversationMessages)...)
	if r.Messages != nil {
		r.Messages = append(PortableMessages([]Message{message}), PortableMessages(WrapMessages(r.Messages))...)
	}
	return r
}

func (r Fragment) Extract(ctx context.Context, llm LLM, obj any) error {
	schema, err := jsonschema.GenerateSchemaForType(obj)
	if err != nil {
		return fmt.Errorf("failed to generate schema for type: %w", err)
	}

	return r.ExtractStructure(ctx, llm, structures.Structure{
		Schema: *schema,
		Object: &obj,
	})
}

// ExtractStructure extracts a structure from the result using the provided JSON schema definition
// and unmarshals it into the provided destination
func (r Fragment) ExtractStructure(ctx context.Context, llm LLM, s structures.Structure) error {
	toolName := "json"
	messages, err := r.Conversation()
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	decision := openai.ChatCompletionRequest{
		Tools: []openai.Tool{
			{
				Type: openai.ToolTypeFunction,
				Function: &openai.FunctionDefinition{
					Strict:     true,
					Name:       toolName,
					Parameters: s.Schema,
				},
			},
		},

		ToolChoice: openai.ToolChoice{
			Type:     openai.ToolTypeFunction,
			Function: openai.ToolFunction{Name: toolName},
		},
	}

	resp, usage, err := CompleteMessages(ctx, llm, CompletionRequest{Options: decision, Messages: messages})
	if err != nil {
		return err
	}

	if err := ctx.Err(); err != nil {
		return err
	}
	if r.Status != nil {
		r.Status.LastUsage = usage
	}

	if len(resp.Choices) != 1 {
		return fmt.Errorf("no choices: %d", len(resp.Choices))
	}

	msg := resp.Choices[0].Message.Portable

	if len(msg.ToolCalls) == 0 {
		return fmt.Errorf("no tool calls: %d", len(msg.ToolCalls))
	}

	return json.Unmarshal([]byte(msg.ToolCalls[0].Function.Arguments), s.Object)
}

type ToolChoice struct {
	// OriginalCall is an owned copy of the received call, before execution
	// changes. It is diagnostic data, not a replay key; the assistant message
	// remains authoritative. Synthetic choices may leave it nil.
	OriginalCall *openai.ToolCall `json:"original_call,omitempty"`
	Name         string           `json:"name"`
	// Arguments are execution-only. Parsed JSON numbers use json.Number.
	Arguments map[string]any `json:"arguments"`
	ID        string         `json:"id"`
	Reasoning string         `json:"reasoning"`
}

// ToolCallDecision represents the decision made by a tool call callback
// It allows the callback to approve, reject, provide adjustment feedback, or directly modify the tool choice
type ToolCallDecision struct {
	// Approved: true to proceed with the tool call, false to interrupt execution
	Approved bool

	// Adjustment: feedback string for the LLM to interpret and adjust the tool call
	// Empty string means no adjustment needed. If provided, the LLM will re-evaluate
	// the tool call based on this feedback.
	Adjustment string

	// Modified: directly modified tool choice that takes precedence over Adjustment
	// If set, this tool choice is used directly without re-querying the LLM
	// This allows programmatic modification of tool arguments
	Modified *ToolChoice

	// Skip: skip this tool call but continue execution (alternative to Approved: false)
	// When true, the tool call is skipped and execution continues
	Skip bool
}

// SelectTool allows the LLM to select a tool from the fragment of conversation
func (f Fragment) SelectTool(ctx context.Context, llm LLM, availableTools Tools, forceTool string) (Fragment, *ToolChoice, error) {
	messages, err := f.Conversation()
	if err != nil {
		return Fragment{}, nil, err
	}
	if err := ctx.Err(); err != nil {
		return Fragment{}, nil, err
	}
	decision := openai.ChatCompletionRequest{
		Tools: availableTools.ToOpenAI(),
	}

	if forceTool != "" {
		decision.ToolChoice = openai.ToolChoice{
			Type:     openai.ToolTypeFunction,
			Function: openai.ToolFunction{Name: forceTool},
		}
	}

	resp, usage, err := CompleteMessages(ctx, llm, CompletionRequest{Options: decision, Messages: messages})
	if err != nil {
		return Fragment{}, nil, err
	}

	if err := ctx.Err(); err != nil {
		return Fragment{}, nil, err
	}
	if f.Status != nil {
		f.Status.LastUsage = usage
	}

	if len(resp.Choices) != 1 {
		return Fragment{}, nil, fmt.Errorf("no choices: %d", len(resp.Choices))
	}

	if len(resp.Choices[0].Message.Portable.ToolCalls) == 0 {
		if f.ConversationMessages == nil && resp.Choices[0].Message.Provider == nil {
			return Fragment{}, nil, nil
		}
		return f.AppendMessages(resp.Choices[0].Message), nil, nil
	}

	toolCall := resp.Choices[0].Message.Portable.ToolCalls[0]
	choices, bad := parseToolCalls([]openai.ToolCall{toolCall}, string(resp.Choices[0].FinishReason))
	if bad != nil {
		return Fragment{}, nil, fmt.Errorf("failed to parse tool call arguments: %w", bad.err)
	}
	f = f.AppendMessages(resp.Choices[0].Message)
	return f, choices[0], nil
}

func (f Fragment) String() string {
	var str strings.Builder
	for _, msg := range f.Messages {
		str.WriteString(fmt.Sprintf("%s: %s\n", msg.Role, msg.Content))
		if len(msg.ToolCalls) > 0 {
			for _, tool := range msg.ToolCalls {
				str.WriteString(fmt.Sprintf("  Tool call: %s(%s)\n", tool.Function.Name, tool.Function.Arguments))
			}
		}
	}

	return str.String()
}

// AllFragmentsStrings walks through all the fragment parents to retrieve all the conversations and represent that as a string
// This is particularly useful if chaining different fragments and want to still feed the conversation
// as a context to the LLM.
func (f Fragment) AllFragmentsStrings() string {
	if f.ParentFragment == nil {
		return f.String()
	}
	return f.String() + "\n\n" + f.ParentFragment.AllFragmentsStrings()
}

func (f Fragment) AddLastMessage(f2 Fragment) Fragment {
	history, err := f2.Conversation()
	if err != nil {
		// Do not select a last message from an inconsistent source: the two
		// sequences may disagree about which envelope it is. Retain the whole
		// invalid source alongside the destination for the next boundary.
		projection := PortableMessages(WrapMessages(f.Messages))
		f = f.AppendMessages(f2.ConversationMessages...)
		if f2.Messages == nil {
			f.Messages = nil
		} else {
			f.Messages = append(projection, PortableMessages(WrapMessages(f2.Messages))...)
			// Independent mismatches must not accidentally cancel each other.
			if _, check := f.Conversation(); check == nil {
				f.Messages = nil
			}
		}
		return f
	}
	if len(history) == 0 {
		return f
	}
	last := history[len(history)-1]
	if f.ConversationMessages == nil && last.Provider == nil {
		return f.appendPortableMessage(last.Portable)
	}
	return f.AppendMessages(last)
}

func (f Fragment) LastMessage() *openai.ChatCompletionMessage {
	if len(f.Messages) == 0 {
		return nil
	}
	return &f.Messages[len(f.Messages)-1]
}

func (f Fragment) LastAssistantAndToolMessages() []openai.ChatCompletionMessage {

	lastMessages := []openai.ChatCompletionMessage{}
	found := false
	for i := len(f.Messages) - 1; i >= 0; i-- {

		if f.Messages[i].Role == AssistantMessageRole.String() || f.Messages[i].Role == ToolMessageRole.String() {
			found = true
			lastMessages = append([]openai.ChatCompletionMessage{f.Messages[i]}, lastMessages...)
		}

		if found && (f.Messages[i].Role != AssistantMessageRole.String() && f.Messages[i].Role != ToolMessageRole.String()) {
			break
		}
	}

	return lastMessages
}
