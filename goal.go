package cogito

import (
	"fmt"

	"github.com/mudler/cogito/pkg/xlog"
	"github.com/mudler/cogito/prompt"
	"github.com/mudler/cogito/structures"
)

// ExtractGoal extracts a goal from a conversation
func ExtractGoal(llm LLM, f Fragment, opts ...Option) (*structures.Goal, error) {
	o := defaultOptions()
	o.Apply(opts...)

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

	goalConv := NewEmptyFragment().AddMessage("user", prompt)

	reasoningGoal, err := llm.Ask(o.context, goalConv)
	if err != nil {
		return nil, fmt.Errorf("failed to ask LLM for goal identification: %w", err)
	}

	identifiedGoal := reasoningGoal.LastMessage()

	structure, goal := structures.StructureGoal()

	goalConv = NewEmptyFragment().AddMessage("user", identifiedGoal.Content)

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
		goalAchievedOpts.FeedbackConversation = feedbackConv.String()
	}

	prompt, err := prompter.Render(goalAchievedOpts)
	if err != nil {
		return nil, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	multimedias := []Multimedia{}
	if feedbackConv != nil {
		multimedias = feedbackConv.Multimedia
	}
	goalAchievedConv := NewEmptyFragment().AddMessage("user", prompt, multimedias...)

	reasoningGoal, err := llm.Ask(o.context, goalAchievedConv)
	if err != nil {
		return nil, fmt.Errorf("failed to ask LLM for goal identification: %w", err)
	}

	boolConv := NewEmptyFragment().AddMessage("user", reasoningGoal.LastMessage().Content)

	xlog.Debug("Check if goal is achieved in current conversation", "reasoning", reasoningGoal.LastMessage().Content)

	return ExtractBoolean(llm, boolConv, opts...)
}
