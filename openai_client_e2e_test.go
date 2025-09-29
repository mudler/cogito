package cogito_test

import (
	"context"

	. "github.com/mudler/cogito"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Client test", Label("e2e"), func() {
	Context("A simple pipeline", func() {
		It("should ask to the LLM", func() {
			defaultLLM := NewOpenAILLM(defaultModel, "", apiEndpoint)

			conv := NewEmptyFragment().AddMessage("user", "Hi!")

			result, err := defaultLLM.Ask(context.TODO(), conv)
			Expect(err).ToNot(HaveOccurred())
			Expect(result.String()).ToNot(BeEmpty())
		})
	})
})
