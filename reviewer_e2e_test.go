package cogito_test

import (
	"context"

	. "github.com/mudler/cogito"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("cogito test", Label("e2e"), func() {
	Context("A simple refinement", func() {
		It("is able to refine a content", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)

			conv := NewEmptyFragment().AddMessage("user", "Explain how a combustion engine works in less than 100 words.")

			result, err := defaultLLM.Ask(context.TODO(), conv)

			Expect(err).ToNot(HaveOccurred())

			f, err := ContentReview(defaultLLM, result, WithIterations(1))
			Expect(err).ToNot(HaveOccurred())
			Expect(f.String()).ToNot(BeEmpty())
		})

		It("is able to refine a content with a search tool", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)

			conv := NewEmptyFragment().AddMessage("user", "What are the latest news today?")

			result, err := defaultLLM.Ask(context.TODO(), conv)

			Expect(err).ToNot(HaveOccurred())
			Expect(result.String()).ToNot(BeEmpty())

			searchTool := &SearchTool{}
			f, err := ContentReview(defaultLLM, result, WithMaxAttempts(1), WithIterations(2), WithTools(
				searchTool,
			))
			Expect(err).ToNot(HaveOccurred())
			Expect(f.String()).ToNot(BeEmpty())
			Expect(f.Status.Iterations).To(BeNumerically(">=", 2))
			Expect(len(f.Status.ToolsCalled)).To(BeNumerically(">=", 2))
			Expect(f.Status.ToolsCalled[0].Tool().Function.Name).To(Equal("search"))
			Expect(searchTool.searchedQuery).ToNot(BeEmpty())
		})

		It("is able to refine a content with a search tool", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)

			conv := NewEmptyFragment().AddMessage("user", "What are the latest news today?")

			searchTool := &SearchTool{}
			f, err := ContentReview(defaultLLM, conv, WithIterations(2), WithTools(searchTool))
			Expect(err).ToNot(HaveOccurred())
			Expect(f.String()).ToNot(BeEmpty())
			Expect(f.Status.Iterations).To(Equal(2))
			Expect(len(f.Status.ToolsCalled)).To(BeNumerically(">=", 2))
			Expect(f.Status.ToolsCalled[0].Tool().Function.Name).To(Equal("search"))
			Expect(f.Status.ToolResults[0].Executed).To(BeTrue())
			Expect(searchTool.searchedQuery).ToNot(BeEmpty())
		})
	})
})
