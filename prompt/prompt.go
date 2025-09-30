package prompt

type PromptType uint

const (
	GapAnalysisType             PromptType = iota
	ContentImproverType         PromptType = iota
	ToolSelectorType            PromptType = iota
	ToolReasonerType            PromptType = iota
	PromptBooleanType           PromptType = iota
	PromptIdentifyGoalType      PromptType = iota
	PromptGoalAchievedType      PromptType = iota
	PromptPlanType              PromptType = iota
	PromptReEvaluatePlanType    PromptType = iota
	PromptSubtaskExtractionType PromptType = iota
	PromptPlanExecutionType     PromptType = iota
)

var (
	defaultPromptMap PromptMap = map[PromptType]Prompt{
		GapAnalysisType:             PromptGapsAnalysis,
		ContentImproverType:         PromptContentImprover,
		ToolSelectorType:            PromptToolSelector,
		ToolReasonerType:            PromptToolReasoner,
		PromptBooleanType:           PromptExtractBoolean,
		PromptIdentifyGoalType:      PromptIdentifyGoal,
		PromptGoalAchievedType:      PromptGoalAchieved,
		PromptPlanType:              PromptPlan,
		PromptReEvaluatePlanType:    PromptReEvaluatePlan,
		PromptSubtaskExtractionType: PromptSubtaskExtraction,
		PromptPlanExecutionType:     PromptPlanExecution,
	}

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

Available tools:
{{ range $index, $tool := .Tools }}
- Tool name: "{{$tool.Name}}" 
  Tool description: {{$tool.Description}}
  Tool arguments: {{$tool.Parameters | toJson}}
{{ end }}


Based on the context, evaluate if you need to use a tool to better answer the question or you can answer directly.
If you decide to use a tool justify with a reasoning your answer and explain why and how to use the tool to answer more in detail.`)

	PromptToolSelector = NewPrompt(`You are an AI assistant that needs to decide if to use a tool in a conversation.

Based on the conversationn and the available tools, if needed, select the most appropriate tool to use with a clear and detailed description on why it should be used, and with what parameters. 
If not necessary, you will not choose any tool.

Guidelines:
- Choose the tool that best matches the task requirements, if no tool is necessary, just reply without selecting any tool
- Provide appropriate parameters for the selected tool
- If multiple tools could work, choose the most appropriate one

Context:
{{.Context}}

{{if .Gaps}}
Identified Gaps to Address:
{{ range $index, $gap := .Gaps }}
- {{$gap}}
{{ end }}
{{ end }}

{{ if ne .AdditionalContext "" }}
Additional context
{{.AdditionalContext}}
{{end}}

Available tools:
{{ range $index, $tool := .Tools }}
- Tool name: "{{$tool.Name}}" 
  Tool description: {{$tool.Description}}
  Tool arguments: {{$tool.Parameters | toJson}}
{{ end }}`)

	PromptExtractBoolean = NewPrompt(`You are an AI assistant that extracts booleans (yes or no) from a context.

Context:
{{.Context}}

You will use the "json" tool with the option "extract_boolean" set to either yes or no.
Reply with the appropriate boolean extraction tool with yes or no, based on the context. 
If the context speaks about, let's say doing something, you will replay with yes, or a no otherwise.`)
)
