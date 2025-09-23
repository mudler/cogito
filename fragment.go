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
	Iterations  int
	ToolsCalled Tools
}

type Fragment struct {
	Messages       []openai.ChatCompletionMessage
	ParentFragment *Fragment
	Status         Status
	Multimedia     []Multimedia
}

func NewEmptyFragment() Fragment {
	return Fragment{}
}

func NewFragment(messages ...openai.ChatCompletionMessage) Fragment {
	return Fragment{
		Messages: messages,
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
func (r Fragment) ExtractStructure(ctx context.Context, llm *LLM, s structures.Structure) error {
	toolName := "json"
	messages := slices.Clone(r.Messages)

	decision := openai.ChatCompletionRequest{

		Model:    llm.model,
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

	resp, err := llm.client.CreateChatCompletion(ctx, decision)
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

type ToolArguments struct {
	Name      string
	Arguments map[string]any
}

// SelectTool allows the LLM to select a tool from the fragment of conversation
func (f Fragment) SelectTool(ctx context.Context, llm *LLM, availableTools Tools, forceTool string) (Fragment, *ToolArguments, error) {
	messages := slices.Clone(f.Messages)
	decision := openai.ChatCompletionRequest{
		Model:    llm.model,
		Messages: messages,
		Tools:    availableTools.ToOpenAI(),
	}

	if forceTool != "" {
		decision.ToolChoice = openai.ToolChoice{
			Type:     openai.ToolTypeFunction,
			Function: openai.ToolFunction{Name: forceTool},
		}
	}

	resp, err := llm.client.CreateChatCompletion(ctx, decision)
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

	return f, &ToolArguments{Name: toolCall.Function.Name, Arguments: arguments}, nil
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

func (f Fragment) LastAssistantMessages() []openai.ChatCompletionMessage {

	lastMessages := []openai.ChatCompletionMessage{}
	found := false
	for i := len(f.Messages) - 1; i >= 0; i-- {

		if f.Messages[i].Role == "assistant" {
			found = true
			lastMessages = append([]openai.ChatCompletionMessage{f.Messages[i]}, lastMessages...)
		}

		if found && f.Messages[i].Role != "assistant" {
			break
		}
	}

	return lastMessages
}
