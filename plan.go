package cogito

import (
	"errors"
	"fmt"
	"os"
	"time"

	"github.com/mudler/cogito/prompt"
	"github.com/mudler/cogito/structures"
	"github.com/mudler/xlog"
	"github.com/sashabaranov/go-openai"
)

type PlanStatus struct {
	Plan  structures.Plan
	Tools []ToolStatus
}

var (
	ErrGoalNotAchieved error = errors.New("goal not achieved")
)

// ExtractPlan extracts a plan from a conversation
// To override the prompt, define a PromptPlanType, PromptReEvaluatePlanType and PromptSubtaskExtractionType
func ExtractPlan(llm LLM, f Fragment, goal *structures.Goal, opts ...Option) (*structures.Plan, error) {
	o := defaultOptions()
	o.Apply(opts...)

	// First we ask the LLM to organize subtasks
	prompter := o.prompts.GetPrompt(prompt.PromptPlanType)

	toolDefs := o.tools.Definitions()
	planOptions := struct {
		Context              string
		AdditionalContext    string
		Goal                 *structures.Goal
		Tools                []*openai.FunctionDefinition
		FeedbackConversation string
	}{
		Context: f.String(),
		Goal:    goal,
		Tools:   toolDefs,
	}
	if o.deepContext && f.ParentFragment != nil {
		planOptions.AdditionalContext = f.ParentFragment.AllFragmentsStrings()
	}

	var feedbackConv *Fragment
	if o.feedbackCallback != nil {
		feedbackConv = o.feedbackCallback()
		planOptions.FeedbackConversation = feedbackConv.String()
	}

	prompt, err := prompter.Render(planOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	return applyPlanFromPrompt(llm, o, prompt, feedbackConv)
}

// ExtractPlan extracts a plan from a conversation
// to override the prompt, define a PromptReEvaluatePlanType and PromptSubtaskExtractionType
func ReEvaluatePlan(llm LLM, f, subtaskFragment Fragment, goal *structures.Goal, toolStatuses []ToolStatus, subtask string, opts ...Option) (*structures.Plan, error) {
	o := defaultOptions()
	o.Apply(opts...)

	// First we ask the LLM to organize subtasks
	prompter := o.prompts.GetPrompt(prompt.PromptReEvaluatePlanType)

	toolDefs := o.tools.Definitions()
	planOptions := struct {
		Context              string
		AdditionalContext    string
		Subtask              string
		SubtaskConversation  string
		Goal                 string
		Tools                []*openai.FunctionDefinition
		PastActionHistory    []ToolStatus
		FeedbackConversation string
	}{
		Context:             f.String(),
		Goal:                goal.Goal,
		Subtask:             subtask,
		Tools:               toolDefs,
		PastActionHistory:   toolStatuses,
		SubtaskConversation: subtaskFragment.String(),
	}
	if o.deepContext && f.ParentFragment != nil {
		planOptions.AdditionalContext = f.ParentFragment.AllFragmentsStrings()
	}

	var feedbackConv *Fragment
	if o.feedbackCallback != nil {
		feedbackConv = o.feedbackCallback()
		planOptions.FeedbackConversation = feedbackConv.String()
	}

	prompt, err := prompter.Render(planOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	return applyPlanFromPrompt(llm, o, prompt, feedbackConv)
}

func applyPlanFromPrompt(llm LLM, o *Options, planPrompt string, feedbackConv *Fragment) (*structures.Plan, error) {
	multimedias := []Multimedia{}
	if feedbackConv != nil {
		multimedias = feedbackConv.Multimedia
	}
	planConv := NewEmptyFragment().AddMessage("user", planPrompt, multimedias...)
	reasoningPlan, err := llm.Ask(o.context, planConv)
	if err != nil {
		return nil, fmt.Errorf("failed to ask LLM for plan identification: %w", err)
	}

	identifiedPlan := reasoningPlan.LastMessage()

	structure, plan := structures.StructurePlan()

	prompter := o.prompts.GetPrompt(prompt.PromptSubtaskExtractionType)

	planOptions := struct {
		Context string
	}{
		Context: identifiedPlan.Content,
	}

	prompt, err := prompter.Render(planOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to render tool reasoner prompt: %w", err)
	}

	planConv = NewEmptyFragment().AddMessage("user", prompt)

	err = planConv.ExtractStructure(o.context, llm, structure)
	if err != nil {
		return nil, fmt.Errorf("failed to extract structure: %w", err)
	}

	plan.Description = identifiedPlan.Content
	return plan, err
}

// ExtractTODOs generates a TODO list from plan subtasks using the LLM
func ExtractTODOs(llm LLM, plan *structures.Plan, goal *structures.Goal, opts ...Option) (*structures.TODOList, error) {
	o := defaultOptions()
	o.Apply(opts...)

	prompter := o.prompts.GetPrompt(prompt.PromptTODOGenerationType)

	todoOptions := struct {
		Goal *structures.Goal
		Plan *structures.Plan
	}{
		Goal: goal,
		Plan: plan,
	}

	promptStr, err := prompter.Render(todoOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to render TODO generation prompt: %w", err)
	}

	todoConv := NewEmptyFragment().AddMessage("user", promptStr)
	reasoningTodo, err := llm.Ask(o.context, todoConv)
	if err != nil {
		return nil, fmt.Errorf("failed to ask LLM for TODO generation: %w", err)
	}

	identifiedTodo := reasoningTodo.LastMessage()

	structure, todoList := structures.StructureTODO()

	todoConv = NewEmptyFragment().AddMessage("user", identifiedTodo.Content)

	err = todoConv.ExtractStructure(o.context, llm, structure)
	if err != nil {
		return nil, fmt.Errorf("failed to extract TODO structure: %w", err)
	}

	// Initialize markdown representation
	todoList.ToMarkdown()
	todoList.LastUpdated = time.Now()

	return todoList, nil
}

// ExecutePlan Executes an already-defined plan with a set of options.
// To override its prompt, configure PromptPlanExecutionType, PromptPlanType, PromptReEvaluatePlanType and PromptSubtaskExtractionType
func ExecutePlan(llm LLM, conv Fragment, plan *structures.Plan, goal *structures.Goal, opts ...Option) (Fragment, error) {
	o := defaultOptions()
	o.Apply(opts...)

	if len(plan.Subtasks) == 0 {
		return NewEmptyFragment(), fmt.Errorf("no subtasks found in plan")
	}

	// Check if Planning with TODOs is enabled (judge LLM must be set)
	if len(o.reviewerLLMs) > 0 {
		// Generate TODOs from plan if not provided
		if o.todos == nil {
			todoList, err := ExtractTODOs(llm, plan, goal, opts...)
			if err != nil {
				return NewEmptyFragment(), fmt.Errorf("failed to extract TODOs: %w", err)
			}
			o.todos = todoList
		}

		// Load TODOs from file if persistence path is set
		if o.todoPersistencePath != "" {
			if err := loadTODOsFromFile(o.todoPersistencePath, o.todos); err != nil {
				xlog.Debug("Failed to load TODOs from file, using provided/generated list", "error", err)
			}
		}

		return executePlanWithTODOs(llm, o.reviewerLLMs, conv, plan, goal, o)
	}

	xlog.Debug("Executing plan for conversation", "length", len(conv.Messages), "plan", plan.Description, "subtasks", plan.Subtasks)

	var toolStatuses []ToolStatus

	conversation := &conv

	defer func(conversation *Fragment) {
		conversation.Status.Plans = append(conversation.Status.Plans, PlanStatus{
			Plan:  *plan,
			Tools: toolStatuses,
		})
	}(conversation)

	index := 0
	attempts := 1
	for {
		subtask := plan.Subtasks[index]

		xlog.Debug("Executing subtask", "goal", goal.Goal, "subtask", subtask)

		prompter := o.prompts.GetPrompt(prompt.PromptPlanExecutionType)

		subtaskOption := struct {
			Goal    string
			Subtask string
		}{
			Goal:    goal.Goal,
			Subtask: subtask,
		}

		prompt, err := prompter.Render(subtaskOption)
		if err != nil {
			return NewEmptyFragment(), fmt.Errorf("failed to render tool reasoner prompt: %w", err)
		}

		subtaskConv := NewEmptyFragment().AddMessage("user", prompt)

		subtaskConvResult, err := ExecuteTools(llm, subtaskConv, opts...)
		if err != nil {
			return *conversation, err
		}

		conversation.Messages = append(conversation.Messages, subtaskConvResult.LastAssistantAndToolMessages()...)
		conversation.Status.Iterations = conversation.Status.Iterations + 1
		conversation.Status.ToolsCalled = append(conversation.Status.ToolsCalled, subtaskConvResult.Status.ToolsCalled...)
		conversation.Status.ToolResults = append(conversation.Status.ToolResults, subtaskConvResult.Status.ToolResults...)
		toolStatuses = append(toolStatuses, subtaskConvResult.Status.ToolResults...)

		boolean, err := IsGoalAchieved(llm, subtaskConvResult, nil, opts...)
		if err != nil {
			return *conversation, err
		}

		toolStatuses := []ToolStatus{}
		for i := range conversation.Status.ToolsCalled {
			toolStatuses = append(toolStatuses, conversation.Status.ToolResults[i])
		}

		if !boolean.Boolean {
			if attempts >= o.maxAttempts {
				if !o.planReEvaluator {
					return *conversation, ErrGoalNotAchieved
				}
				xlog.Debug("All attempts failed, re-evaluating plan")
				plan, err = ReEvaluatePlan(llm, *conversation, subtaskConv, goal, toolStatuses, subtask, opts...)
				if err != nil {
					return *conversation, err
				}

				// Start again
				index = 0
				attempts = 1
			} else {
				xlog.Debug("Attempt failed to achieve goal, retrying")
				attempts++
			}
		} else {
			xlog.Debug("Goal correctly achieved")
			attempts = 1 // reset attempts
			if len(plan.Subtasks)-1 > index {
				index++
			} else if !(o.infiniteExecution) {
				break
			}
		}
	}

	return *conversation, nil
}

// executePlanWithTODOs executes a plan with Planning with TODOs
func executePlanWithTODOs(workerLLM LLM, reviewerLLMs []LLM, conv Fragment, plan *structures.Plan, goal *structures.Goal, o *Options) (Fragment, error) {
	if len(plan.Subtasks) == 0 {
		return NewEmptyFragment(), fmt.Errorf("no subtasks found in plan")
	}

	xlog.Debug("Executing plan with TODOs", "plan", plan.Description, "subtasks", plan.Subtasks, "maxIterations", o.maxIterations)

	conversation := &conv
	if conversation.Status == nil {
		conversation.Status = &Status{}
	}
	conversation.Status.TODOs = o.todos

	var toolStatuses []ToolStatus
	var previousFeedback string

	// Outer loop: TODO iterations
	for todoIteration := 1; todoIteration <= o.maxIterations; todoIteration++ {
		conversation.Status.TODOIteration = todoIteration
		xlog.Debug("Starting TODO iteration", "iteration", todoIteration, "maxIterations", o.maxIterations)

		// Inner loop: execute plan subtasks
		index := 0
		attempts := 1
		for index < len(plan.Subtasks) {

			subtask := plan.Subtasks[index]
			xlog.Debug("Executing subtask", "goal", goal.Goal, "subtask", subtask, "todoIteration", todoIteration)

			// WORK PHASE
			conversation.Status.TODOPhase = "work"
			workResult, err := executeWorkPhase(workerLLM, o.todos, goal, subtask, previousFeedback, o)
			if err != nil {
				return *conversation, fmt.Errorf("work phase failed: %w", err)
			}

			// Update TODOs from work result
			o.todos, err = updateTODOsFromWork(workerLLM, workResult, o.todos, o)
			if err != nil {
				xlog.Debug("Failed to update TODOs from work", "error", err)
			}

			// REVIEW PHASE
			conversation.Status.TODOPhase = "review"
			reviewResult, goalCompleted, err := executeReviewPhase(reviewerLLMs, workResult, goal, o.todos, o)
			if err != nil {
				return *conversation, fmt.Errorf("review phase failed: %w", err)
			}

			// Extract feedback from review
			previousFeedback = extractFeedbackFromReview(reviewResult)

			// Update TODOs from feedback
			o.todos, err = updateTODOsFromFeedback(reviewResult, o.todos, o.todoPersistencePath)
			if err != nil {
				xlog.Debug("Failed to update TODOs from feedback", "error", err)
			}

			// Save TODOs to file if persistence path is set
			if o.todoPersistencePath != "" {
				if err := saveTODOsToFile(o.todoPersistencePath, o.todos); err != nil {
					xlog.Debug("Failed to save TODOs to file", "error", err)
				}
			}

			conversation.Status.TODOs = o.todos
			conversation.Messages = append(conversation.Messages, workResult.LastAssistantAndToolMessages()...)
			conversation.Status.Iterations = conversation.Status.Iterations + 1
			conversation.Status.ToolsCalled = append(conversation.Status.ToolsCalled, workResult.Status.ToolsCalled...)
			conversation.Status.ToolResults = append(conversation.Status.ToolResults, workResult.Status.ToolResults...)
			toolStatuses = append(toolStatuses, workResult.Status.ToolResults...)

			if goalCompleted {
				xlog.Debug("Goal execution completed", "subtask", subtask)
				attempts = 1
				if len(plan.Subtasks)-1 > index {
					index++
				} else {
					// All subtasks completed
					conversation.Status.Plans = append(conversation.Status.Plans, PlanStatus{
						Plan:  *plan,
						Tools: toolStatuses,
					})
					return *conversation, nil
				}
			} else {
				// Goal execution incomplete: needs rework
				if attempts >= o.maxAttempts {
					if !o.planReEvaluator {
						return *conversation, ErrGoalNotAchieved
					}
					xlog.Debug("All attempts failed, re-evaluating plan")
					// Create a fresh conversation for re-evaluation (fresh context)
					reEvalConv := NewEmptyFragment()
					reEvalConv.Status = conversation.Status
					plan, err = ReEvaluatePlan(workerLLM, reEvalConv, workResult, goal, toolStatuses, subtask)
					if err != nil {
						return *conversation, err
					}
					// Start again with fresh context
					index = 0
					attempts = 1
				} else {
					xlog.Debug("Attempt failed to achieve goal, retrying with feedback", "attempts", attempts)
					attempts++
					// Continue with same subtask but with updated feedback
				}
			}
		}
	}

	conversation.Status.Plans = append(conversation.Status.Plans, PlanStatus{
		Plan:  *plan,
		Tools: toolStatuses,
	})

	return *conversation, nil
}

// executeWorkPhase executes the work phase with fresh context including TODOs, goal, and feedback
func executeWorkPhase(workerLLM LLM, todoList *structures.TODOList, goal *structures.Goal, subtask string, previousFeedback string, o *Options) (Fragment, error) {
	prompter := o.prompts.GetPrompt(prompt.PromptTODOWorkType)

	// Ensure markdown is up to date
	todoMarkdown := todoList.ToMarkdown()

	workOptions := struct {
		Goal             string
		Subtask          string
		TODOMarkdown     string
		PreviousFeedback string
	}{
		Goal:             goal.Goal,
		Subtask:          subtask,
		TODOMarkdown:     todoMarkdown,
		PreviousFeedback: previousFeedback,
	}

	promptStr, err := prompter.Render(workOptions)
	if err != nil {
		return NewEmptyFragment(), fmt.Errorf("failed to render work phase prompt: %w", err)
	}

	// Create fresh fragment with work context
	workFragment := NewEmptyFragment().AddMessage("user", promptStr)

	// Execute tools with the work fragment
	// Convert Options struct to Option functions for ExecuteTools
	opts := convertOptionsToFunctions(o)
	workResult, err := ExecuteTools(workerLLM, workFragment, opts...)
	if err != nil {
		return NewEmptyFragment(), fmt.Errorf("failed to execute tools in work phase: %w", err)
	}

	return workResult, nil
}

// executeReviewPhase executes the review phase using the judge LLM
func executeReviewPhase(reviewerLLMs []LLM, workFragment Fragment, goal *structures.Goal, todoList *structures.TODOList, o *Options) (Fragment, bool, error) {
	prompter := o.prompts.GetPrompt(prompt.PromptTODOReviewType)

	todoMarkdown := todoList.ToMarkdown()

	// Get work results as string
	workResults := workFragment.String()

	reviewOptions := struct {
		Goal         string
		WorkResults  string
		TODOMarkdown string
	}{
		Goal:         goal.Goal,
		WorkResults:  workResults,
		TODOMarkdown: todoMarkdown,
	}

	promptStr, err := prompter.Render(reviewOptions)
	if err != nil {
		return NewEmptyFragment(), false, fmt.Errorf("failed to render review phase prompt: %w", err)
	}

	// Create review fragment
	reviewFragment := NewEmptyFragment().AddMessage("user", promptStr)

	// Use IsGoalAchieved to determine if goal execution is completed
	opts := convertOptionsToFunctions(o)

	reviews := []struct {
		boolean      *structures.Boolean
		reviewResult Fragment
	}{}

	for _, reviewerLLM := range reviewerLLMs {

		boolean, err := IsGoalAchieved(reviewerLLM, reviewFragment, goal, opts...)
		if err != nil {
			return NewEmptyFragment(), false, fmt.Errorf("failed to check if goal achieved: %w", err)
		}

		// Get the reasoning from the review
		reviewResult, err := reviewerLLM.Ask(o.context, reviewFragment)
		if err != nil {
			return NewEmptyFragment(), false, fmt.Errorf("failed to get review result: %w", err)
		}

		reviews = append(reviews, struct {
			boolean      *structures.Boolean
			reviewResult Fragment
		}{boolean, reviewResult})
	}

	boolean := &structures.Boolean{Boolean: false}
	var reviewResult Fragment
	var positiveReview Fragment
	var negativeReview Fragment

	// Majority vote
	// Count the number of true booleans
	trueCount := 0
	for _, review := range reviews {
		if review.boolean.Boolean {
			trueCount++
			positiveReview = review.reviewResult
		} else {
			negativeReview = review.reviewResult
		}
	}

	// If the number of true booleans is greater than the number of false booleans, set the boolean to true
	if trueCount > len(reviews)/2 {
		boolean.Boolean = true
		reviewResult = positiveReview
	} else {
		reviewResult = negativeReview
	}

	goalCompleted := boolean.Boolean
	return reviewResult, goalCompleted, nil
}

// updateTODOsFromWork extracts TODO updates from work results
func updateTODOsFromWork(workerLLM LLM, workFragment Fragment, todoList *structures.TODOList, o *Options) (*structures.TODOList, error) {
	// Try to extract TODO updates from the work fragment
	prompter := o.prompts.GetPrompt(prompt.PromptTODOTrackingType)

	todoMarkdown := todoList.ToMarkdown()

	trackingOptions := struct {
		Context      string
		TODOMarkdown string
	}{
		Context:      workFragment.String(),
		TODOMarkdown: todoMarkdown,
	}

	promptStr, err := prompter.Render(trackingOptions)
	if err != nil {
		return todoList, fmt.Errorf("failed to render TODO tracking prompt: %w", err)
	}

	trackingConv := NewEmptyFragment().AddMessage("user", promptStr)
	structure, updatedTodoList := structures.StructureTODO()

	// We use the worker LLM here to extract the structure. Maybe we should use the reviewer LLM instead?
	// TODO: Implement a better way to select the LLM to use for extraction?
	err = trackingConv.ExtractStructure(o.context, workerLLM, structure)
	if err != nil {
		// If extraction fails, return original list
		xlog.Debug("Failed to extract TODO updates from work", "error", err)
		return todoList, nil
	}

	// Update markdown
	updatedTodoList.ToMarkdown()
	updatedTodoList.LastUpdated = time.Now()

	return updatedTodoList, nil
}

// updateTODOsFromFeedback updates TODOs based on review feedback
func updateTODOsFromFeedback(reviewFragment Fragment, todoList *structures.TODOList, persistencePath string) (*structures.TODOList, error) {
	// Extract feedback from review
	feedback := extractFeedbackFromReview(reviewFragment)

	// For now, we'll add feedback to incomplete TODOs
	// In a more sophisticated implementation, we could parse the feedback
	// and update specific TODOs
	incomplete := todoList.GetIncompleteTODOs()
	if len(incomplete) > 0 && feedback != "" {
		// Add feedback to the first incomplete TODO
		todoList.AddFeedback(incomplete[0].ID, feedback)
	}

	todoList.LastUpdated = time.Now()
	todoList.ToMarkdown()

	return todoList, nil
}

// extractFeedbackFromReview extracts feedback text from review fragment
func extractFeedbackFromReview(reviewFragment Fragment) string {
	if len(reviewFragment.Messages) == 0 {
		return ""
	}

	lastMessage := reviewFragment.LastMessage()
	if lastMessage == nil {
		return ""
	}

	// Return the content of the last message as feedback
	return lastMessage.Content
}

// saveTODOsToFile saves TODO list to a file
func saveTODOsToFile(path string, todoList *structures.TODOList) error {
	if todoList == nil {
		return fmt.Errorf("TODO list is nil")
	}

	data, err := todoList.ToJSON()
	if err != nil {
		return fmt.Errorf("failed to serialize TODOs: %w", err)
	}

	err = os.WriteFile(path, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write TODOs to file: %w", err)
	}

	return nil
}

// convertOptionsToFunctions converts an Options struct to a slice of Option functions
// This extracts the important options to pass to functions that expect []Option
func convertOptionsToFunctions(o *Options) []Option {
	var opts []Option

	// Preserve tools
	if len(o.tools) > 0 {
		opts = append(opts, WithTools(o.tools...))
	}

	// Preserve context
	if o.context != nil {
		opts = append(opts, WithContext(o.context))
	}

	// Preserve other important options
	if o.toolReasoner {
		opts = append(opts, EnableToolReasoner)
	}

	if o.deepContext {
		opts = append(opts, EnableDeepContext)
	}
	if o.planReEvaluator {
		opts = append(opts, EnableAutoPlanReEvaluator)
	}
	if o.strictGuidelines {
		opts = append(opts, EnableStrictGuidelines)
	}
	if len(o.guidelines) > 0 {
		opts = append(opts, WithGuidelines(o.guidelines...))
	}
	if len(o.mcpSessions) > 0 {
		opts = append(opts, WithMCPs(o.mcpSessions...))
	}
	if o.maxAttempts > 0 {
		opts = append(opts, WithMaxAttempts(o.maxAttempts))
	}
	if o.statusCallback != nil {
		opts = append(opts, WithStatusCallback(o.statusCallback))
	}
	if o.reasoningCallback != nil {
		opts = append(opts, WithReasoningCallback(o.reasoningCallback))
	}
	if o.feedbackCallback != nil {
		opts = append(opts, WithFeedbackCallback(o.feedbackCallback))
	}
	if o.toolCallCallback != nil {
		opts = append(opts, WithToolCallBack(o.toolCallCallback))
	}
	if o.maxAdjustmentAttempts > 0 {
		opts = append(opts, WithMaxAdjustmentAttempts(o.maxAdjustmentAttempts))
	}
	if o.forceReasoning {
		opts = append(opts, WithForceReasoning())
	}
	if o.maxRetries > 0 {
		opts = append(opts, WithMaxRetries(o.maxRetries))
	}
	if o.loopDetectionSteps > 0 {
		opts = append(opts, WithLoopDetection(o.loopDetectionSteps))
	}
	if len(o.gaps) > 0 {
		opts = append(opts, WithGaps(o.gaps...))
	}
	if o.prompts != nil {
		for promptType, p := range o.prompts {
			if staticPrompt, ok := p.(prompt.StaticPrompt); ok {
				opts = append(opts, WithPrompt(promptType, staticPrompt))
			}
		}
	}

	return opts
}

// loadTODOsFromFile loads TODO list from a file
func loadTODOsFromFile(path string, todoList *structures.TODOList) error {
	if todoList == nil {
		return fmt.Errorf("TODO list is nil")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			// File doesn't exist yet, that's okay
			return nil
		}
		return fmt.Errorf("failed to read TODOs from file: %w", err)
	}

	err = todoList.FromJSON(data)
	if err != nil {
		return fmt.Errorf("failed to deserialize TODOs: %w", err)
	}

	// Update markdown
	todoList.ToMarkdown()

	return nil
}
