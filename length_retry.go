package cogito

import (
	"fmt"

	"github.com/mudler/xlog"
	"github.com/sashabaranov/go-openai"
)

// maxLengthRetryOutputTokens bounds the output cap a length retry asks for.
// Doubling a 16K default gives 32K, which fits the context of any model that
// reasons at length; going past 64K mostly buys a request the backend rejects
// as exceeding the context, since cogito does not know the model's window.
const maxLengthRetryOutputTokens = 65536

// lengthRetryOutputCap returns the output cap to retry a decision with after
// it was truncated by finish_reason=length before producing any content or
// tool call, or 0 when a retry cannot raise the cap.
//
// A truncated attempt fails identically at the same cap, but a reasoning model
// that spent the whole budget thinking often finishes with more room. cogito
// usually does not set a cap on the request: the client applies its own (see
// SetMaxTokens in clients/) only when the request carries none. The completion
// tokens the truncated attempt reported are that cap, as it ran into it, so
// they are the base to double. Without a cap on the request or a usage report
// there is nothing to double, and the caller fails fast as before.
func lengthRetryOutputCap(req openai.ChatCompletionRequest, usage LLMUsage) int {
	current := req.MaxCompletionTokens
	if current == 0 {
		current = req.MaxTokens
	}
	if current == 0 {
		current = usage.CompletionTokens
	}
	if current <= 0 || current >= maxLengthRetryOutputTokens {
		return 0
	}
	return min(current*2, maxLengthRetryOutputTokens)
}

// setOutputCap puts the cap on the request, where every bundled client takes
// it over its own default. It keeps the field the request already uses and
// otherwise uses max_tokens, the field the clients themselves fill in.
func setOutputCap(req *openai.ChatCompletionRequest, n int) {
	if req.MaxCompletionTokens > 0 {
		req.MaxCompletionTokens = n
		return
	}
	req.MaxTokens = n
}

// lengthTruncatedError is surfaced to users verbatim, so it says what to change.
// retriedCap is the cap of the retry that also truncated, or 0 without a retry.
func lengthTruncatedError(what string, retriedCap int) error {
	retried := ""
	if retriedCap > 0 {
		retried = fmt.Sprintf(", also after a retry with max tokens %d", retriedCap)
	}
	return fmt.Errorf("%s truncated before producing content (finish_reason=length%s): the model exhausted its output-token budget, likely on reasoning - raise max tokens/context or reduce prompt size (e.g. a large image)", what, retried)
}

// lengthRetry holds the one length retry a decision loop allows.
type lengthRetry struct {
	raised int // the raised cap, 0 until the retry is spent
}

// prepare raises req's output cap for the retry after a length truncation, or
// returns the error to surface when the retry is spent or cannot raise the cap.
// what names the call in the error ("decision", "streaming decision").
func (l *lengthRetry) prepare(req *openai.ChatCompletionRequest, usage LLMUsage, what string) error {
	if l.raised != 0 {
		return lengthTruncatedError(what, l.raised)
	}
	n := lengthRetryOutputCap(*req, usage)
	if n == 0 {
		return lengthTruncatedError(what, 0)
	}
	xlog.Warn("Decision truncated by length before producing content, retrying with a larger output cap", "call", what, "maxTokens", n)
	setOutputCap(req, n)
	l.raised = n
	return nil
}
