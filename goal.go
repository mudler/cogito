package cogito

import (
	"fmt"

	"github.com/mudler/cogito/prompt"
	"github.com/mudler/cogito/structures"
	"github.com/mudler/xlog"
)

// ExtractGoal extracts a goal from a conversation
func ExtractGoal(llm LLM, f Fragment, opts ...Option) (*structures.Goal, error) {
	o := defaultOptions()
	o.Apply(opts...)

	if err := validatePlanReviewSource(llm, f, o); err != nil {
		return nil, err
	}

	// First we ask the LLM if there is a goal from the conversation
	prompter := o.prompts.GetPrompt(prompt.PromptIdentifyGoalType)

	goalIdentifierOptions := struct {
		Context           string
		AdditionalContext string
	}{
		Context: f.String(),
	}
	if o.deepContext && f.ParentFragment != nil {
		goalIdentifierOptions.AdditionalContext = f.ParentFragment.AllFragmentsStrings()
	}

	prompt, err := prompter.Render(goalIdentifierOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	history, err := extractionHistory(llm, f)
	if err != nil {
		return nil, err
	}
	goalConv, err := extractionConversation(o, f, history, "user", prompt)
	if err != nil {
		return nil, err
	}

	reasoningGoal, err := askWithStreaming(o.context, llm, goalConv, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to ask LLM for goal identification: %w", err)
	}

	identifiedGoal := reasoningGoal.LastMessage()

	structure, goal := structures.StructureGoal()

	history, err = extractionHistory(llm, reasoningGoal)
	if err != nil {
		return nil, err
	}
	goalConv, err = extractionConversation(o, reasoningGoal, history, "user", identifiedGoal.Content)
	if err != nil {
		return nil, err
	}

	err = goalConv.ExtractStructure(o.context, llm, structure)
	if err != nil {
		return nil, fmt.Errorf("failed to extract boolean structure: %w", err)
	}

	return goal, nil
}

// IsGoalAchieved checks if a goal has been achieved
func IsGoalAchieved(llm LLM, f Fragment, goal *structures.Goal, opts ...Option) (*structures.Boolean, error) {
	o := defaultOptions()
	o.Apply(opts...)

	if err := validatePlanReviewSource(llm, f, o); err != nil {
		return nil, err
	}

	// First we ask the LLM if there is a goal from the conversation
	prompter := o.prompts.GetPrompt(prompt.PromptGoalAchievedType)

	goalAchievedOpts := struct {
		Context              string
		AdditionalContext    string
		Goal                 string
		FeedbackConversation string
	}{
		Context: f.String(),
	}
	if goal != nil {
		goalAchievedOpts.Goal = goal.Goal
	}
	if o.deepContext && f.ParentFragment != nil {
		goalAchievedOpts.AdditionalContext = f.ParentFragment.AllFragmentsStrings()
	}
	var feedbackConv *Fragment
	if o.feedbackCallback != nil {
		feedbackConv = o.feedbackCallback()
		if feedbackConv != nil {
			if err := validatePlanReviewSource(llm, *feedbackConv, o); err != nil {
				return nil, err
			}
			goalAchievedOpts.FeedbackConversation = feedbackConv.String()
		}
	}

	prompt, err := prompter.Render(goalAchievedOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	multimedias := []Multimedia{}
	if feedbackConv != nil {
		multimedias = feedbackConv.Multimedia
	}
	history, err := extractionHistory(llm, f)
	if err != nil {
		return nil, err
	}
	goalAchievedConv := NewEmptyFragment()
	if f.ConversationMessages != nil {
		goalAchievedConv = goalAchievedConv.WithConversation(history)
	}
	// Keep feedback media on the synthesized prompt, never provider blocks.
	goalAchievedConv = goalAchievedConv.AddMessage("user", prompt, multimedias...)
	reasoningGoal, err := askPlanReview(llm, goalAchievedConv, o)
	if err != nil {
		return nil, fmt.Errorf("failed to ask LLM for goal identification: %w", err)
	}

	// Rich reasoning is retained atomically by ExtractBoolean. Legacy Ask
	// results still use its historical prompt-only extraction behavior.
	boolConv := reasoningGoal

	xlog.Debug("Check if goal is achieved in current conversation", "reasoning", reasoningGoal.LastMessage().Content)

	/// XXX: ExtractBoolean seems to be really brittle
	return ExtractBoolean(llm, boolConv, opts...)
}
