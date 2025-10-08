package cogito

import (
	"errors"
	"fmt"

	"github.com/mudler/cogito/pkg/xlog"
	"github.com/mudler/cogito/prompt"
)

// ContentReview refines an LLM response until for a fixed number of iterations or if the LLM doesn't find anymore gaps
func ContentReview(llm LLM, originalFragment Fragment, opts ...Option) (Fragment, error) {
	o := defaultOptions()
	o.Apply(opts...)

	gaps := []string{}

	f := originalFragment

	refinedMessage := ""
	// Iterative refinement loop
	for i := range o.maxIterations {
		var err error
		originalFragment.Status.Iterations = i + 1

		xlog.Debug("Refined message", "refinedMessage", refinedMessage, "iteration", i+1)

		if len(o.tools) > 0 {
			f, err = ExecuteTools(llm, f, append([]Option{WithGaps(gaps...)}, opts...)...)
			if err != nil && !errors.Is(err, ErrNoToolSelected) {
				return Fragment{}, fmt.Errorf("failed to execute tools in iteration %d: %w", i+1, err)
			}

			originalFragment.Status.ToolsCalled = f.Status.ToolsCalled
			originalFragment.Status.ToolResults = f.Status.ToolResults
		}

		// Analyze knowledge gaps
		gaps, err = ExtractKnowledgeGaps(llm, f, opts...)
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to analyze gaps in iteration %d: %w", i+1, err)
		}

		// If no gaps found, we're done
		if len(gaps) == 0 {
			xlog.Debug("No gaps found, stop!")
			break
		}

		xlog.Debug("Knowledge gaps identified", "iteration", i+1, "gaps", gaps)

		// Generate improved content based on gaps
		improvedContent, err := improveContent(llm, f, refinedMessage, gaps, o)
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to improve content in iteration %d: %w", i+1, err)
		}
		refinedMessage = improvedContent.LastMessage().Content
		o.statusCallback(improvedContent.LastMessage().Content)
		xlog.Debug("Improved content generated", "iteration", i+1)
	}

	return originalFragment.AddMessage("assistant", refinedMessage), nil
}

func improveContent(llm LLM, f Fragment, refinedMessage string, gaps []string, o *Options) (Fragment, error) {
	prompter := o.prompts.GetPrompt(prompt.ContentImproverType)

	renderOptions := struct {
		Context           string
		AdditionalContext string
		Gaps              []string
		RefinedMessage    string
	}{
		Context:        f.String(),
		Gaps:           gaps,
		RefinedMessage: refinedMessage,
	}

	if f.ParentFragment != nil {
		if o.deepContext {
			renderOptions.AdditionalContext = f.ParentFragment.AllFragmentsStrings()
		} else {
			renderOptions.AdditionalContext = f.ParentFragment.String()
		}
	}

	p, err := prompter.Render(renderOptions)
	if err != nil {
		return Fragment{}, fmt.Errorf("failed to render content improver prompt: %w", err)
	}

	newFragment := NewEmptyFragment().
		AddMessage("user", p)

	xlog.Debug("Improving content", "prompt", p)

	newFragment.ParentFragment = f.ParentFragment

	return llm.Ask(o.context, newFragment)
}
