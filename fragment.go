package cogito

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"

	"github.com/mudler/cogito/pkg/xlog"
	"github.com/mudler/cogito/structures"
	"github.com/sashabaranov/go-openai"
)

type Status struct {
	Iterations   int
	ToolsCalled  Tools
	ToolResults  []ToolStatus
	Plans        []PlanStatus
	PastActions  []ToolStatus // Track past actions for loop detection
	ReasoningLog []string     // Track reasoning for each iteration
}

type Fragment struct {
	Messages       []openai.ChatCompletionMessage
	ParentFragment *Fragment
	Status         *Status
	Multimedia     []Multimedia
}

// Messages returns the chat completion messages from this fragment,
// automatically prepending a force-text-reply system message if tool calls are detected.
// This ensures LLMs provide natural language responses instead of JSON tool syntax
// when Ask() is called after ExecuteTools().
func (f Fragment) GetMessages() []openai.ChatCompletionMessage {

	// TODO: this is kinda of brittle because LLM interface implementers needs to call this methods to get the messages,
	// but we don't enforce it - worse is that we change the user messages and they might not expect that when calling GetMessages().
	// We should move away from this, and have a more explicit way to get the messages.

	messages := f.Messages

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

	return messages
}

func NewEmptyFragment() Fragment {
	return Fragment{
		Status: &Status{
			PastActions:  []ToolStatus{},
			ReasoningLog: []string{},
			ToolsCalled:  Tools{},
			ToolResults:  []ToolStatus{},
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
		},
	}
}

// TODO: Video, Audio, Image input
type Multimedia interface {
	URL() string
}

func (r Fragment) AddMessage(role, content string, mm ...Multimedia) Fragment {
	chatCompletionMessage := openai.ChatCompletionMessage{
		Role: role,
	}

	if len(mm) > 0 {
		multiContent := []openai.ChatMessagePart{
			{
				Text: content,
				Type: openai.ChatMessagePartTypeText,
			},
		}

		for _, img := range mm {
			r.Multimedia = append(r.Multimedia, img)
			multiContent = append(multiContent, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{
					URL: img.URL(),
				},
			})
		}
		chatCompletionMessage.MultiContent = multiContent
	} else {
		chatCompletionMessage.Content = content
	}

	r.Messages = append(r.Messages, chatCompletionMessage)

	return r
}

func (r Fragment) AddStartMessage(role, content string, mm ...Multimedia) Fragment {
	r.Messages = append([]openai.ChatCompletionMessage{
		{
			Role:    role,
			Content: content,
		},
	}, r.Messages...)
	return r
}

// ExtractStructure extracts a structure from the result using the provided JSON schema definition
// and unmarshals it into the provided destination
func (r Fragment) ExtractStructure(ctx context.Context, llm LLM, s structures.Structure) error {
	toolName := "json"
	messages := slices.Clone(r.Messages)

	decision := openai.ChatCompletionRequest{
		Messages: messages,
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

	resp, err := llm.CreateChatCompletion(ctx, decision)
	if err != nil {
		return err
	}

	if len(resp.Choices) != 1 {
		return fmt.Errorf("no choices: %d", len(resp.Choices))
	}

	msg := resp.Choices[0].Message

	if len(msg.ToolCalls) == 0 {
		return fmt.Errorf("no tool calls: %d", len(msg.ToolCalls))
	}

	return json.Unmarshal([]byte(msg.ToolCalls[0].Function.Arguments), s.Object)
}

type ToolChoice struct {
	Name      string
	Arguments map[string]any
}

// SelectTool allows the LLM to select a tool from the fragment of conversation
func (f Fragment) SelectTool(ctx context.Context, llm LLM, availableTools Tools, forceTool string) (Fragment, *ToolChoice, error) {
	messages := slices.Clone(f.Messages)
	decision := openai.ChatCompletionRequest{
		Messages: messages,
		Tools:    availableTools.ToOpenAI(),
	}

	if forceTool != "" {
		decision.ToolChoice = openai.ToolChoice{
			Type:     openai.ToolTypeFunction,
			Function: openai.ToolFunction{Name: forceTool},
		}
	}

	resp, err := llm.CreateChatCompletion(ctx, decision)
	if err != nil {
		return Fragment{}, nil, err
	}

	if len(resp.Choices) != 1 {
		return Fragment{}, nil, fmt.Errorf("no choices: %d", len(resp.Choices))
	}

	if len(resp.Choices[0].Message.ToolCalls) == 0 {
		xlog.Debug("LLM did not select any tool", "response", resp.Choices[0].Message)
		return Fragment{}, nil, nil
	}

	toolCall := resp.Choices[0].Message.ToolCalls[0]
	arguments := make(map[string]any)

	if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments); err != nil {
		return Fragment{}, nil, fmt.Errorf("failed to parse tool call arguments: %w", err)
	}

	f.Messages = append(f.Messages, openai.ChatCompletionMessage{
		Role: "assistant",
		ToolCalls: []openai.ToolCall{
			{
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      toolCall.Function.Name,
					Arguments: toolCall.Function.Arguments,
				},
			},
		},
	})

	return f, &ToolChoice{Name: toolCall.Function.Name, Arguments: arguments}, nil
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
	if len(f2.Messages) > 0 {
		f.Messages = append(f.Messages, f2.Messages[len(f2.Messages)-1])
	}
	return f
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

		if f.Messages[i].Role == "assistant" || f.Messages[i].Role == "tool" {
			found = true
			lastMessages = append([]openai.ChatCompletionMessage{f.Messages[i]}, lastMessages...)
		}

		if found && (f.Messages[i].Role != "assistant" && f.Messages[i].Role != "tool") {
			break
		}
	}

	return lastMessages
}
