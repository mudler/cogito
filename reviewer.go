package cogito

import (
	"errors"
	"fmt"

	"github.com/mudler/cogito/pkg/xlog"
	"github.com/mudler/cogito/prompt"
)

// ContentReview refines an LLM response until for a fixed number of iterations or if the LLM doesn't find anymore gaps
func ContentReview(llm *LLM, originalFragment Fragment, opts ...Option) (Fragment, error) {
	o := defaultOptions()
	o.Apply(opts...)

	gaps := []string{}

	f := originalFragment

	refinedMessage := ""
	// Iterative refinement loop
	for i := range o.MaxIterations {
		var err error

		xlog.Debug("Refined message", "refinedMessage", refinedMessage, "iteration", i+1)

		if len(o.Tools) > 0 {
			f, err = ExecuteTools(llm, f, append([]Option{WithGaps(gaps...)}, opts...)...)
			if err != nil && !errors.Is(err, ErrNoToolSelected) {
				return Fragment{}, fmt.Errorf("failed to execute tools in iteration %d: %w", i+1, err)
			}
		}

		// Analyze knowledge gaps
		gaps, err = ExtractKnowledgeGaps(llm, f, opts...)
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to analyze gaps in iteration %d: %w", i+1, err)
		}

		xlog.Debug("Knowledge gaps identified", "iteration", i+1, "gaps", gaps)

		// Generate improved content based on gaps
		improvedContent, err := improveContent(llm, f.AddMessage("assistant", refinedMessage), gaps, o)
		if err != nil {
			return Fragment{}, fmt.Errorf("failed to improve content in iteration %d: %w", i+1, err)
		}

		o.StatusCallback(improvedContent.LastMessage().Content)
		xlog.Debug("Improved content generated", "iteration", i+1)

		// Update fragment and status
		originalFragment.Status.ToolsCalled = f.Status.ToolsCalled
		originalFragment.Status.Iterations = i + 1
		refinedMessage = improvedContent.LastMessage().Content

		xlog.Debug("Refinement iteration completed", "iteration", i+1, "gaps_found", len(gaps))

		// If no gaps found, we're done
		if len(gaps) == 0 {
			xlog.Debug("No gaps found, stop!")
			break
		}
	}

	return originalFragment.AddMessage("assistant", refinedMessage), nil
}

func improveContent(llm *LLM, f Fragment, gaps []string, o *Options) (Fragment, error) {
	prompter := o.Prompts.GetPrompt(prompt.ContentImproverType)

	renderOptions := struct {
		Context           string
		AdditionalContext string
		Gaps              []string
	}{
		Context: f.String(),
		Gaps:    gaps,
	}

	if f.ParentFragment != nil {
		if o.DeepContext {
			renderOptions.AdditionalContext = f.ParentFragment.AllFragmentsStrings()
		} else {
			renderOptions.AdditionalContext = f.ParentFragment.String()
		}
	}

	p, err := prompter.Render(renderOptions)
	if err != nil {
		return Fragment{}, fmt.Errorf("failed to render content improver prompt: %w", err)
	}

	systemPrompt, err := prompt.PromptExpertGapWriter.Render(struct{}{})
	if err != nil {
		return Fragment{}, fmt.Errorf("failed to render system prompt: %w", err)
	}

	newFragment := NewEmptyFragment().
		AddMessage("system", systemPrompt).
		AddMessage("user", p)

	xlog.Debug("Improving content", "prompt", p)

	newFragment.ParentFragment = f.ParentFragment

	return llm.Ask(o.Context, newFragment)
}
