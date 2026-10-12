package cogito

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/sashabaranov/go-openai"
	"reflect"
	"strings"
)

// CompletionRequest separates model options from the owned conversation.
// Options.Messages must be nil, even when Messages is empty.
type CompletionRequest struct {
	Options  openai.ChatCompletionRequest
	Messages []Message
}

// CompletionChoice retains the authoritative message for a response choice.
type CompletionChoice struct {
	Index        int
	Message      Message
	FinishReason openai.FinishReason
}

// CompletionReply pairs authoritative choices with their legacy projection.
type CompletionReply struct {
	Response LLMReply
	Choices  []CompletionChoice
}

// MessageLLM is the optional non-streaming, lossless completion capability.
type MessageLLM interface {
	LLM
	CreateMessageCompletion(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error)
}

// ErrProviderMessagesUnsupported reports an opaque message sent to a legacy LLM.
type ErrProviderMessagesUnsupported struct{ Index int }

func (e *ErrProviderMessagesUnsupported) Error() string {
	return fmt.Sprintf("provider messages unsupported by LLM (message %d)", e.Index)
}

// CompleteMessages dispatches a lossless completion, or a safe legacy fallback.
func CompleteMessages(ctx context.Context, llm LLM, request CompletionRequest) (CompletionReply, LLMUsage, error) {
	if request.Options.Messages != nil {
		return CompletionReply{}, LLMUsage{}, fmt.Errorf("completion options Messages must be nil")
	}
	rich, capable := llm.(MessageLLM)
	if !capable {
		for i, message := range request.Messages {
			if message.Provider != nil {
				return CompletionReply{}, LLMUsage{}, &ErrProviderMessagesUnsupported{Index: i}
			}
		}
	}
	options, err := cloneCompletionValue(reflect.ValueOf(request.Options), 0)
	if err != nil {
		return CompletionReply{}, LLMUsage{}, fmt.Errorf("clone completion options: %w", err)
	}
	request.Options = options.Interface().(openai.ChatCompletionRequest)
	request.Messages = CloneMessages(request.Messages)
	var reply CompletionReply
	var usage LLMUsage
	if capable {
		reply, usage, err = rich.CreateMessageCompletion(ctx, request)
	} else {
		request.Options.Messages = legacyTransportMessages(PortableMessages(request.Messages))
		reply.Response, usage, err = llm.CreateChatCompletion(ctx, request.Options)
		choices := reply.Response.ChatCompletionResponse.Choices
		if choices != nil {
			reply.Choices = make([]CompletionChoice, len(choices))
			for i, choice := range choices {
				reply.Choices[i] = CompletionChoice{Index: choice.Index, Message: Message{Portable: choice.Message}, FinishReason: choice.FinishReason}
			}
		}
	}
	// An errored or malformed response is not a committed conversation message.
	if err != nil {
		return CompletionReply{}, usage, err
	}
	if err = validateCompletionReply(reply); err != nil {
		return CompletionReply{}, usage, err
	}
	// Headers are an unexported embedded SDK field with an explicit copy API.
	headers := reply.Response.ChatCompletionResponse.Header().Clone()
	reply.Response.ChatCompletionResponse.SetHeader(nil)
	owned, err := cloneCompletionValue(reflect.ValueOf(reply), 0)
	if err != nil {
		return CompletionReply{}, usage, fmt.Errorf("clone completion reply: %w", err)
	}
	reply = owned.Interface().(CompletionReply)
	reply.Response.ChatCompletionResponse.SetHeader(headers)
	return reply, usage, nil
}

// legacyTransportMessages owns the wire projection. Legacy templates may parse
// historical arguments as JSON; durable history and rich replay remain raw.
func legacyTransportMessages(messages []openai.ChatCompletionMessage) []openai.ChatCompletionMessage {
	messages = PortableMessages(WrapMessages(messages))
	for i := range messages {
		message := &messages[i]
		if message.Role != openai.ChatMessageRoleAssistant {
			continue
		}
		for j := range message.ToolCalls {
			call := &message.ToolCalls[j]
			if strings.TrimSpace(call.Function.Arguments) != "" && !json.Valid([]byte(call.Function.Arguments)) {
				call.Function.Arguments = "{}"
			}
		}
	}
	return messages
}

// Choices use the same order, indexes, finish reasons and exact portable values
// as the compatibility projection. Indexes need not be contiguous, but must be
// nonnegative and unique. Provider scope/version validation is adapter-owned.
func validateCompletionReply(reply CompletionReply) error {
	projection := reply.Response.ChatCompletionResponse.Choices
	if len(reply.Choices) != len(projection) || (reply.Choices == nil) != (projection == nil) {
		return fmt.Errorf("completion choice projection length or nilness mismatch")
	}
	seen := make(map[int]bool, len(reply.Choices))
	for i, choice := range reply.Choices {
		if choice.Index < 0 || seen[choice.Index] {
			return fmt.Errorf("invalid completion choice index %d", choice.Index)
		}
		seen[choice.Index] = true
		if choice.Index != projection[i].Index || choice.FinishReason != projection[i].FinishReason || !reflect.DeepEqual(choice.Message.Portable, projection[i].Message) {
			return fmt.Errorf("completion choice %d projection mismatch", choice.Index)
		}
	}
	return nil
}

// cloneCompletionValue handles the SDK's nested options, including typed schema
// structs and maps held in interface fields, without JSON coercion or loss of
// nil/empty distinctions. Non-data values, private nonzero fields, and excessively
// deep/cyclic data are rejected rather than shared with a backend. Each branch
// is copied independently so authoritative messages never alias projections.
func cloneCompletionValue(value reflect.Value, depth int) (reflect.Value, error) {
	if depth > 100 {
		return reflect.Value{}, fmt.Errorf("completion data is cyclic or exceeds depth 100")
	}
	switch value.Kind() {
	case reflect.Interface, reflect.Pointer:
		if value.IsNil() {
			return reflect.Zero(value.Type()), nil
		}
		child, err := cloneCompletionValue(value.Elem(), depth+1)
		if err != nil {
			return reflect.Value{}, err
		}
		result := reflect.New(value.Type()).Elem()
		if value.Kind() == reflect.Pointer {
			result.Set(reflect.New(value.Type().Elem()))
			result.Elem().Set(child)
		} else {
			result.Set(child)
		}
		return result, nil
	case reflect.Slice, reflect.Array:
		if value.Kind() == reflect.Slice && value.IsNil() {
			return reflect.Zero(value.Type()), nil
		}
		result := reflect.New(value.Type()).Elem()
		if value.Kind() == reflect.Slice {
			result = reflect.MakeSlice(value.Type(), value.Len(), value.Len())
		}
		for i := 0; i < value.Len(); i++ {
			child, err := cloneCompletionValue(value.Index(i), depth+1)
			if err != nil {
				return reflect.Value{}, err
			}
			result.Index(i).Set(child)
		}
		return result, nil
	case reflect.Map:
		if value.IsNil() {
			return reflect.Zero(value.Type()), nil
		}
		result := reflect.MakeMapWithSize(value.Type(), value.Len())
		iter := value.MapRange()
		for iter.Next() {
			key, err := cloneCompletionValue(iter.Key(), depth+1)
			if err != nil {
				return reflect.Value{}, err
			}
			child, err := cloneCompletionValue(iter.Value(), depth+1)
			if err != nil {
				return reflect.Value{}, err
			}
			result.SetMapIndex(key, child)
		}
		return result, nil
	case reflect.Struct:
		result := reflect.New(value.Type()).Elem()
		for i := 0; i < value.NumField(); i++ {
			if value.Type().Field(i).PkgPath != "" {
				if !value.Field(i).IsZero() {
					return reflect.Value{}, fmt.Errorf("cannot own private field %s.%s", value.Type(), value.Type().Field(i).Name)
				}
				continue
			}
			child, err := cloneCompletionValue(value.Field(i), depth+1)
			if err != nil {
				return reflect.Value{}, err
			}
			result.Field(i).Set(child)
		}
		return result, nil
	case reflect.Chan, reflect.Func, reflect.UnsafePointer:
		return reflect.Value{}, fmt.Errorf("unsupported completion data type %s", value.Type())
	default:
		return value, nil
	}
}
