package cogito_test

import (
	. "github.com/teslashibe/cogito"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/sashabaranov/go-openai"
)

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
