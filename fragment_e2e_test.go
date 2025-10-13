package cogito_test

import (
	"context"
	"encoding/json"
	"strings"

	. "github.com/mudler/cogito"
	"github.com/mudler/cogito/structures"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
	"github.com/sashabaranov/go-openai/jsonschema"
)

type GeatWeatherTool struct {
	status *ToolStatus
}

type MultimediaImage struct {
	url string
}

func (m MultimediaImage) URL() string {
	return m.url
}

func (s *GeatWeatherTool) Status() *ToolStatus {
	if s.status == nil {
		s.status = &ToolStatus{}
	}
	return s.status
}

func (s *GeatWeatherTool) Run(args map[string]any) (string, error) {

	return "", nil
}

func (s *GeatWeatherTool) Tool() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        "get_weather",
			Description: "Get the weather",
			Parameters: jsonschema.Definition{
				Type: jsonschema.Object,
				Properties: map[string]jsonschema.Definition{
					"city": {
						Type:        jsonschema.String,
						Description: "The city to look-up the weather for",
					},
				},
				Required: []string{"city"},
			},
		},
	}
}

var _ = Describe("Fragment test", func() {
	Context("Basic operations", func() {
		It("Should add messages", func() {
			fragment := Fragment{}
			fragment = fragment.AddMessage("user", "Hello")
			fragment = fragment.AddMessage("assistant", "Hi!")
			fragment = fragment.AddStartMessage("system", "You are a helpful assistant.")

			Expect(len(fragment.Messages)).To(Equal(3))
			Expect(fragment.Messages[0].Role).To(Equal("system"))
			Expect(fragment.Messages[1].Role).To(Equal("user"))
			Expect(fragment.Messages[2].Role).To(Equal("assistant"))
		})

		It("Should extract assistant messages", func() {
			fragment := Fragment{}
			fragment = fragment.AddMessage("user", "Hello")
			fragment = fragment.AddMessage("assistant", "Hi!")
			fragment = fragment.AddStartMessage("system", "You are a helpful assistant.")
			fragment = fragment.AddMessage("assistant", "Byee!")

			Expect(len(fragment.Messages)).To(Equal(4))
			Expect(len(fragment.LastAssistantAndToolMessages())).To(Equal(2))
			Expect(fragment.LastAssistantAndToolMessages()[0].Content).To(Equal("Hi!"))
			Expect(fragment.LastAssistantAndToolMessages()[1].Content).To(Equal("Byee!"))
			conv := NewEmptyFragment()
			conv.Messages = append(conv.Messages, fragment.LastAssistantAndToolMessages()...)

			Expect(conv.Messages[0].Content).To(Equal("Hi!"))
			Expect(conv.Messages[1].Content).To(Equal("Byee!"))
		})

		It("Should return all parent strings", func() {
			fragment := NewEmptyFragment().AddMessage("zeepod", "baltazar")
			fragmentParent := NewEmptyFragment().AddMessage("foo", "bar")
			fragmentGrandFather := NewEmptyFragment().AddMessage("anakin", "skywalker")

			fragment.ParentFragment = &fragmentParent
			fragmentParent.ParentFragment = &fragmentGrandFather

			parentContext := fragment.AllFragmentsStrings()
			Expect(parentContext).To(ContainSubstring("foo"))
			Expect(parentContext).To(ContainSubstring("anakin"))
			Expect(parentContext).To(ContainSubstring("baltazar"))

			parentContext = fragmentParent.AllFragmentsStrings()
			Expect(parentContext).To(ContainSubstring("foo"))
			Expect(parentContext).To(ContainSubstring("anakin"))
			Expect(parentContext).ToNot(ContainSubstring("baltazar"))

			parentContext = fragmentGrandFather.AllFragmentsStrings()
			Expect(parentContext).ToNot(ContainSubstring("foo"))
			Expect(parentContext).To(ContainSubstring("anakin"))
			Expect(parentContext).ToNot(ContainSubstring("baltazar"))
		})

		It("should add multimedia", func() {
			fragment := NewEmptyFragment().AddMessage("user", "Hello", MultimediaImage{
				url: "https://example.com/image.png",
			})
			Expect(fragment.Multimedia).To(HaveLen(1))
			Expect(fragment.Multimedia[0].URL()).To(Equal("https://example.com/image.png"))
			Expect(fragment.Messages[0].MultiContent).To(HaveLen(2))
			Expect(fragment.Messages[0].MultiContent[0].Type).To(Equal(openai.ChatMessagePartTypeText))
			Expect(fragment.Messages[0].MultiContent[0].Text).To(Equal("Hello"))
			Expect(fragment.Messages[0].MultiContent[1].Type).To(Equal(openai.ChatMessagePartTypeImageURL))
			Expect(fragment.Messages[0].MultiContent[1].ImageURL.URL).To(Equal("https://example.com/image.png"))
		})
	})
})

var _ = Describe("Result test", Label("e2e"), func() {
	Context("A simple pipeline", func() {
		It("should extract a structure", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)

			fragment := NewEmptyFragment().AddMessage("user", "Write a short poem about the sea in less than 20 words.")

			result, err := defaultLLM.Ask(context.TODO(), fragment)

			Expect(err).ToNot(HaveOccurred())

			subjectObject := struct {
				Subject string `json:"subject"`
			}{}
			err = result.ExtractStructure(context.TODO(), defaultLLM,
				structures.Structure{
					Schema: jsonschema.Definition{
						Type: jsonschema.Object,
						Properties: map[string]jsonschema.Definition{
							"subject": {
								Type:        jsonschema.String,
								Description: "The subject of the poem",
							},
						},
						Required: []string{"subject"},
					},
					Object: &subjectObject,
				},
			)

			Expect(err).ToNot(HaveOccurred())
			Expect(subjectObject.Subject).ToNot(BeEmpty())
			Expect(strings.ToLower(subjectObject.Subject)).To(ContainSubstring("sea"))
		})

		It("should select a tool", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)

			fragment := NewFragment(openai.ChatCompletionMessage{
				Role:    "user",
				Content: "What's the weather today in San Francisco?",
			})

			newFragment, result, err := fragment.SelectTool(context.TODO(), defaultLLM, Tools{
				&GeatWeatherTool{},
			}, "")

			Expect(err).ToNot(HaveOccurred())
			Expect(result.Name).To(Equal("get_weather"))

			Expect(result.Arguments).To(HaveKey("city"))
			Expect(strings.ToLower(result.Arguments["city"].(string))).To(Equal("san francisco"))

			json, err := json.Marshal(result.Arguments)
			Expect(err).ToNot(HaveOccurred())

			Expect(newFragment.Messages[len(newFragment.Messages)-1].ToolCalls).To(HaveExactElements(openai.ToolCall{
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_weather",
					Arguments: string(json),
				},
			}))
		})
	})
})
