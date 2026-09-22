package cogito

import "strings"

// contextOverflowMarkers are fragments of the errors backends return when a
// request does not fit the model's context window:
//
//   - llama.cpp / LocalAI: "request (9739 tokens) exceeds the available context size (8192 tokens), try increasing it"
//   - OpenAI / vLLM: "This model's maximum context length is 8192 tokens. However, ..."
//   - OpenAI error code: "context_length_exceeded"
//   - Anthropic: "prompt is too long: 210000 tokens > 200000 maximum"
var contextOverflowMarkers = []string{
	"exceeds the available context size",
	"maximum context length",
	"context_length_exceeded",
	"prompt is too long",
}

// isContextOverflowError reports whether err says the request did not fit the
// model's context window. Such a request fails the same way on every attempt,
// so the decision loops return it at once instead of retrying with backoff.
func isContextOverflowError(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	for _, marker := range contextOverflowMarkers {
		if strings.Contains(msg, marker) {
			return true
		}
	}
	return false
}
