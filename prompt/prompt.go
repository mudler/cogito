package prompt

type PromptType uint

const (
	GapAnalysisType                PromptType = iota
	ContentImproverType            PromptType = iota
	ToolReasonerType               PromptType = iota
	PromptBooleanType              PromptType = iota
	PromptIdentifyGoalType         PromptType = iota
	PromptGoalAchievedType         PromptType = iota
	PromptPlanType                 PromptType = iota
	PromptReEvaluatePlanType       PromptType = iota
	PromptSubtaskExtractionType    PromptType = iota
	PromptPlanExecutionType        PromptType = iota
	PromptGuidelinesType           PromptType = iota
	PromptGuidelinesExtractionType PromptType = iota
	PromptPlanDecisionType         PromptType = iota
	PromptToolReEvaluationType     PromptType = iota
	PromptParameterReasoningType   PromptType = iota
	PromptTODOGenerationType       PromptType = iota
	PromptTODOWorkType             PromptType = iota
	PromptTODOReviewType           PromptType = iota
	PromptTODOTrackingType         PromptType = iota
)

var (
	defaultPromptMap PromptMap = map[PromptType]Prompt{
		GapAnalysisType:                PromptGapsAnalysis,
		ContentImproverType:            PromptContentImprover,
		ToolReasonerType:               PromptToolReasoner,
		PromptBooleanType:              PromptExtractBoolean,
		PromptIdentifyGoalType:         PromptIdentifyGoal,
		PromptGoalAchievedType:         PromptGoalAchieved,
		PromptPlanType:                 PromptPlan,
		PromptReEvaluatePlanType:       PromptReEvaluatePlan,
		PromptSubtaskExtractionType:    PromptSubtaskExtraction,
		PromptPlanExecutionType:        PromptPlanExecution,
		PromptGuidelinesType:           PromptGuidelines,
		PromptGuidelinesExtractionType: PromptGuidelinesExtraction,
		PromptPlanDecisionType:         DecideIfPlanningIsNeeded,
		PromptToolReEvaluationType:     PromptToolReEvaluation,
		PromptParameterReasoningType:   PromptParameterReasoning,
		PromptTODOGenerationType:       PromptTODOGeneration,
		PromptTODOWorkType:             PromptTODOWork,
		PromptTODOReviewType:           PromptTODOReview,
		PromptTODOTrackingType:         PromptTODOTracking,
	}

	PromptGuidelinesExtraction = NewPrompt("What guidelines should be applied? return only the numbers of the guidelines by using the json tool with a list of integers corresponding to the guidelines.")

	PromptGuidelines = NewPrompt(`You are an AI assistant that needs to understand if any of the guidelines should be applied to the conversation.

Guidelines:
{{ range $index, $guideline := .Guidelines }}
{{add1 $index}}. {{$guideline.Condition}} (Suggested action: {{$guideline.Action}})
{{- end }}

Conversation:
{{.Context}}

{{ if ne .AdditionalContext "" }}
Additional Context:
{{.AdditionalContext}}
{{ end }}

Identify if any of the guidelines should be applied to the conversation.
If so, return the relevant guidelines with the numbers of the guidelines.

If no guideline should be applied, just say so and why.
`)

	PromptPlanExecution = NewPrompt(`You are an AI assistant that is executing a goal and a subtask.
	
Goal: {{.Goal}}
	
Subtask: {{.Subtask}}
	
`)

	PromptSubtaskExtraction = NewPrompt(`You are an AI assistant that extract subtasks from a plan to achieve a specific goal.
Context: 

{{.Context}}

Use the "json" tool to return a list of detailed subtasks to execute from the given context. 
Each subtask should contain a description of what to do, for instance "do a research about guinea pigs". Be as much descriptive as possible`)

	PromptPlan = NewPrompt(`You are an AI assistant that breaks down a goal into a series of actionable steps (subtasks).

Goal: {{.Goal.Goal}}

Context:
{{.Context}}

{{if ne .AdditionalContext ""}}
AdditionalContext:
{{.AdditionalContext}}
{{end}}

{{if ne .FeedbackConversation ""}}
Feedback Context:
{{.FeedbackConversation}}
{{end}}


Available tools:
{{ range $index, $tool := .Tools }}
- Tool name: "{{$tool.Name}}" 
  Tool description: {{$tool.Description}}
  Tool arguments: {{$tool.Parameters | toJson}}
{{ end }}

Based on the goal, context, and available tools, create a detailed plan with clear and actionable steps (subtasks) to achieve the goal.
If a tool is relevant to a subtask, mention it explicitly in the step description and how should be used.`)

	PromptReEvaluatePlan = NewPrompt(`You are an AI assistant that re-evaluates a plan after a subtask executio to achieve a specific goal, 
and breaks down into a series of actionable steps (subtasks).

Overall Goal: {{.Goal}}

Overall Context:
{{.Context}}

{{if ne .AdditionalContext ""}}
AdditionalContext:
{{.AdditionalContext}}
{{end}}

{{if ne .FeedbackConversation ""}}
Feedback Context:
{{.FeedbackConversation}}
{{end}}

Subtask: {{.Subtask}}

Subtask action and result:
{{.SubtaskConversation}}

Available tools:
{{ range $index, $tool := .Tools }}
- Tool name: "{{$tool.Name}}" 
  Tool description: {{$tool.Description}}
  Tool arguments: {{$tool.Parameters | toJson}}
{{ end }}

Tools already called:
{{ range $index, $tool := .PastActionHistory }}
- Tool name: "{{$tool.Name}}" 
  Tool result: {{$tool.Result}}
  Tool arguments: {{$tool.ToolArguments | toJson}}
{{ end }}

Based on the overall goal, the overall context, the subtask and the subtask result and available tools, re-evaluate a more effective plan with clear and actionable steps (subtasks) to achieve the goal.
If a tool is relevant to a subtask, mention it explicitly in the step description and how should be used.`)

	PromptGoalAchieved = NewPrompt(`You are an AI assistant that determines if a goal has been achieved based on the provided conversation.

{{if ne .Goal ""}}
Overall Goal: {{.Goal}}
{{end}}

Conversation:
{{.Context}}

{{if ne .AdditionalContext ""}}
Additional Context:
{{.AdditionalContext}}
{{end}}

{{if ne .FeedbackConversation ""}}
Feedback Context:
{{.FeedbackConversation}}
{{end}}

Identify from the context if the goal has been achieved, answer with yes or no and justify your answer with a reasoning.`)

	PromptIdentifyGoal = NewPrompt(
		`Analyze the following text and the context to identify the goal.
Context:
{{.Context}}

{{if ne .AdditionalContext ""}}
AdditionalContext:
{{.AdditionalContext}}
{{end}}
`,
	)
	PromptGapsAnalysis = NewPrompt(
		`Analyze the following conversation and the context to identify knowledge gaps or areas that need further coverage or improvement in the assistant response.
Conversation:
{{.Text}}

{{if ne .Context ""}}
Context:
{{.Context}}
{{end}}

Identify specific gaps that would make the assistant response more comprehensive and accurate.
Focus on concrete, actionable improvements by considering the provided context if any.`,
	)

	PromptContentImprover = NewPrompt(`Improve the reply of the assistant (or suggest one if not present) in the conversation and try to address the knowledge gaps considering the provided context or tools results.

Current conversation:
{{.Context}}

{{if ne .AdditionalContext ""}}
Additional Context:
{{.AdditionalContext}}
{{end}}

Identified Gaps to Address:
{{ range $index, $gap := .Gaps }}
- {{$gap}}
{{ end }}

{{if ne .RefinedMessage ""}}
Current assistant response:
{{.RefinedMessage}}
{{else}}
No current assistant response provided. You have to write the assistant response from scratch.
{{end}}

Please rewrite the assistant response to cover these gaps while maintaining the original style and quality. 
Make it more comprehensive and accurate by leveraging the additional context.`)

	PromptToolReasoner = NewPrompt(`You are an AI assistant, based on the following context, you have to decide if to use a tool to better answer or if it's not required answer directly.

Context:
{{.Context}}

{{ if ne .AdditionalContext "" }}
Additional context
{{.AdditionalContext}}
{{end}}

{{ range $index, $guideline := .Guidelines }}
Guideline {{add1 $index}}: {{$guideline.Condition}} (Suggested action: {{$guideline.Action}}) ( Suggested Tools to use: {{$guideline.Tools | toJson}} )
{{ end }}

Available tools:
{{ range $index, $tool := .Tools }}
- Tool name: "{{$tool.Name}}" 
  Tool description: {{$tool.Description}}
  Tool arguments: {{$tool.Parameters | toJson}}
{{ end }}


Based on the context, evaluate if you need to use a tool to better answer the question or you can answer directly.
If you decide to use a tool justify with a reasoning your answer and explain why and how to use the tool to answer more in detail.`)

	PromptExtractBoolean = NewPrompt(`You are an AI assistant that extracts booleans (yes or no) from a context.

Context:
{{.Context}}

You will use the "json" tool with the option "extract_boolean" set to either yes or no.
Reply with the appropriate boolean extraction tool with yes or no, based on the context. 
If the context speaks about, let's say doing something, you will replay with yes, or a no otherwise.`)

	DecideIfPlanningIsNeeded = NewPrompt(`You are an AI assistant that decides if planning and executing subtasks in sequence is needed from a conversation.

Conversation: 
{{.Context}}

{{if ne .AdditionalContext ""}}
AdditionalContext:
{{.AdditionalContext}}
{{end}}

Available tools:
{{ range $index, $tool := .Tools }}
- Tool name: "{{$tool.Name}}" 
  Tool description: {{$tool.Description}}
  Tool arguments: {{$tool.Parameters | toJson}}
{{ end }}

Based on the conversation, context, and available tools, decide if planning and executing subtasks in sequence is needed.
Keep in mind that Planning will later involve in breaking down the problem into a set of subtasks that require running tools in sequence and evaluating their results.
If you think planning is needed, reply with yes, otherwise reply with no.`)

	PromptToolReEvaluation = NewPrompt(`You are an AI assistant re-evaluating the conversation after tool executions.

Your task is to:
1. Review all tool execution results
2. Assess if the goal has been achieved
3. Determine if additional actions are needed
4. Decide on the next best course of action

Context:
{{.Context}}

{{ if ne .AdditionalContext "" }}
Additional Context:
{{.AdditionalContext}}
{{ end }}

Previous Tool Executions:
{{if .PreviousTools}}
{{ range $index, $tool := .PreviousTools }}
Tool {{add1 $index}}: {{$tool.Name}}
Arguments: {{$tool.ToolArguments | toJson}}
Result: {{$tool.Result}}
{{ end }}
{{else}}
No previous tool executions.
{{end}}

{{ range $index, $guideline := .Guidelines }}
Guideline {{add1 $index }}: If {{$guideline.Condition}} then {{$guideline.Action}} ( Suggested Tools to use: {{$guideline.Tools | toJson}} )
{{ end }}

Available Tools:
{{ range $index, $tool := .Tools }}
- Tool name: "{{$tool.Name}}" 
  Tool description: {{$tool.Description}}
  Tool parameters: {{$tool.Parameters | toJson}}
{{ end }}

Based on all tool execution results and current context:
1. Has the goal been achieved, or do we need to take additional actions?
2. If more actions are needed, which tool(s) should we use next and why?
3. If the goal is achieved, we can conclude and provide a final response.

Analyze the situation considering all previous tool executions and provide your reasoning about what to do next.`)

	PromptParameterReasoning = NewPrompt(`You are tasked with generating the optimal parameters for the tool "{{.ToolName}}". The tool requires the following parameters:
{{.Parameters}}

Your task is to:
1. Generate the best possible values for each required parameter
2. If the parameter requires code, provide complete, working code
3. If the parameter requires text or documentation, provide comprehensive, well-structured content
4. Ensure all parameters are complete and ready to be used

Focus on quality and completeness. Do not explain your reasoning or analyze the tool's purpose - just provide the best possible parameter values.`)

	PromptTODOGeneration = NewPrompt(`You are an AI assistant that converts plan subtasks into a structured TODO list.

Goal: {{.Goal.Goal}}

Plan Description: {{.Plan.Description}}

Plan Subtasks:
{{ range $index, $subtask := .Plan.Subtasks }}
{{add1 $index}}. {{$subtask}}
{{ end }}

Convert each subtask into a TODO item. Each TODO should have:
- A unique ID
- A clear description based on the subtask
- Completed status set to false (all TODOs start incomplete)

Use the "json" tool to return a structured TODO list with all subtasks as incomplete TODOs.`)

	PromptTODOWork = NewPrompt(`You are working on a task. Here is the context:

**Overall Goal:**
{{.Goal}}

**Current Subtask:**
{{.Subtask}}

**TODO List (Current Progress):**
{{.TODOMarkdown}}

{{if ne .PreviousFeedback ""}}
**Feedback from Previous Review:**
{{.PreviousFeedback}}
{{end}}

Execute the current subtask, updating the TODO list as you complete items. Mark TODOs as complete when you finish working on them.`)

	PromptTODOReview = NewPrompt(`You are reviewing work that has been completed. Here is the context:

**Overall Goal:**
{{.Goal}}

**Work Completed:**
{{.WorkResults}}

**Current TODO List:**
{{.TODOMarkdown}}

Review the work and determine if the goal has been achieved. Consider:
1. Have all necessary TODOs been completed?
2. Is the work quality sufficient?
3. Does the work meet the goal requirements?

Provide feedback on what needs to be improved if the goal is not yet achieved.`)

	PromptTODOTracking = NewPrompt(`Extract TODO updates from the following conversation. Identify which TODOs have been completed and any new TODOs that should be added.

Conversation:
{{.Context}}

Current TODO List:
{{.TODOMarkdown}}

Use the "json" tool to return an updated TODO list with:
- Completed TODOs marked as completed
- Any new TODOs that were identified
- Updated feedback for TODOs if provided`)
)
