<img width="300"  align="left" alt="Gemini_Generated_Image_jbv0xajbv0xajbv0" src="https://github.com/user-attachments/assets/ab243414-4452-4f03-b266-f9326ad2c8d1" />

**Cogito** is a powerful Go library for building intelligent, co-operative agentic software and LLM-powered workflows, focusing on improving results for small, open source language models that scales to any LLM.

🧪 **Tested on Small Models** ! Our test suite runs on 0.6b Qwen (not fine-tuned), proving effectiveness even with minimal resources.

> 📝 **Working on Official Paper**  
> I am currently working on the official academic/white paper for Cogito. The paper will provide detailed theoretical foundations, experimental results, and comprehensive analysis of the framework's capabilities.

## 🏗️ Architecture

Cogito is the result of building [LocalAI](https://github.com/mudler/LocalAI), [LocalAGI](https://github.com/mudler/LocalAGI) and [LocalOperator](https://github.com/mudler/LocalOperator) (yet to be released).

Cogito uses an internal pipeline to first make the LLM reason about a specific task, forcing the model to reason and later extracts with BNF grammars exact data structures from the LLM. This is applied to every primitive exposed by the framework.

It provides a comprehensive framework for creating conversational AI systems with advanced reasoning, tool execution, goal-oriented planning, iterative content refinement capabilities, and seamless integration with external tools also via the Model Context Protocol (MCP).

> 🔧 **Composable Primitives**  
> Cogito primitives can be combined to form more complex pipelines, enabling sophisticated AI workflows.

## 🚀 Quick Start

### Installation

```bash
go get github.com/mudler/cogito
```

### Basic Usage

```go
package main

import (
    "context"
    "fmt"
    "github.com/mudler/cogito"
)

func main() {
    // Create an LLM client
    llm := cogito.NewOpenAILLM("your-model", "api-key", "https://api.openai.com")
    
    // Create a conversation fragment
    fragment := cogito.NewEmptyFragment().
        AddMessage("user", "Tell me about artificial intelligence")
    
    // Get a response
    newFragment, err := llm.Ask(context.Background(), fragment)
    if err != nil {
        panic(err)
    }
    
    fmt.Println(newFragment.LastMessage().Content)
}
```

### Using Tools

#### Creating Custom Tools

To create a custom tool, implement the `Tool[T]` interface:

```go
type MyToolArgs struct {
    Param string `json:"param" description:"A parameter"`
}

type MyTool struct{}

// Implement the Tool interface
func (t *MyTool) Run(args MyToolArgs) (string, error) {
    // Your tool logic here
    return fmt.Sprintf("Processed: %s", args.Param), nil
}

// Create a ToolDefinition using NewToolDefinition helper
myTool := cogito.NewToolDefinition(
    &MyTool{},
    MyToolArgs{},
    "my_tool",
    "A custom tool",
)
```


Tools in Cogito are added by calling `NewToolDefinition` on your tool, which automatically generates `openai.Tool` via the `Tool()` method. Tools are then passed by to `cogito.WithTools`:

```go
// Define tool argument types
type WeatherArgs struct {
    City string `json:"city" description:"The city to get weather for"`
}

type SearchArgs struct {
    Query string `json:"query" description:"The search query"`
}

// Create tool definitions - these automatically generate openai.Tool
weatherTool := cogito.NewToolDefinition(
    &WeatherTool{},
    WeatherArgs{},
    "get_weather",
    "Get the current weather for a city",
)

searchTool := cogito.NewToolDefinition(
    &SearchTool{},
    SearchArgs{},
    "search",
    "Search for information",
)

// Create a fragment with user input
fragment := cogito.NewFragment(openai.ChatCompletionMessage{
    Role:    "user",
    Content: "What's the weather in San Francisco?",
})

// Execute with tools - you can pass multiple tools with different types
result, err := cogito.ExecuteTools(llm, fragment, 
    cogito.WithTools(weatherTool, searchTool))
if err != nil {
    panic(err)
}

// result.Status.ToolsCalled will contain all the tools being called
```

#### Tool Call Callbacks and Adjustments

Cogito allows you to intercept and adjust tool calls before they are executed. This enables interactive workflows where users can review, approve, modify, or directly edit tool calls.

**ToolCallDecision Struct:**

The callback returns a `ToolCallDecision` struct that provides fine-grained control:

```go
type ToolCallDecision struct {
    Approved   bool         // true to proceed, false to interrupt
    Adjustment string       // Feedback for LLM to interpret (empty = no adjustment)
    Modified   *ToolChoice  // Direct modification (takes precedence over Adjustment)
    Skip       bool         // Skip this tool call but continue execution
}
```

**Basic Tool Call Callback:**

```go
result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithTools(searchTool),
    cogito.WithToolCallBack(func(tool *cogito.ToolChoice, state *cogito.SessionState) cogito.ToolCallDecision {
        fmt.Printf("Agent wants to run: %s with args: %v\n", tool.Name, tool.Arguments)
        return cogito.ToolCallDecision{Approved: true} // Proceed without adjustment
    }))
```

**Interactive Tool Call Approval with Adjustments:**

```go
import (
    "bufio"
    "encoding/json"
    "os"
    "strings"
)

result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithTools(searchTool),
    cogito.WithMaxAdjustmentAttempts(3), // Limit adjustment attempts
    cogito.WithToolCallBack(func(tool *cogito.ToolChoice, state *cogito.SessionState) cogito.ToolCallDecision {
        args, _ := json.Marshal(tool.Arguments)
        fmt.Printf("The agent wants to run the tool %s with arguments: %s\n", tool.Name, string(args))
        fmt.Println("Do you want to run the tool? (y/n/adjust)")
        
        reader := bufio.NewReader(os.Stdin)
        text, _ := reader.ReadString('\n')
        text = strings.TrimSpace(text)
        
        switch text {
        case "y":
            return cogito.ToolCallDecision{Approved: true}
        case "n":
            return cogito.ToolCallDecision{Approved: false}
        default:
            // Provide adjustment feedback - the LLM will re-evaluate the tool call
            return cogito.ToolCallDecision{
                Approved:   true,
                Adjustment: text,
            }
        }
    }))
```

**Direct Tool Modification:**

You can directly modify tool arguments without relying on LLM interpretation:

```go
result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithTools(searchTool),
    cogito.WithToolCallBack(func(tool *cogito.ToolChoice, state *cogito.SessionState) cogito.ToolCallDecision {
        // Validate and fix arguments directly
        if tool.Name == "search" {
            query, ok := tool.Arguments["query"].(string)
            if !ok || len(query) < 3 {
                // Directly modify instead of asking LLM
                fixed := *tool
                fixed.Arguments = map[string]any{
                    "query": "default search query",
                }
                return cogito.ToolCallDecision{
                    Approved: true,
                    Modified: &fixed,
                }
            }
        }
        return cogito.ToolCallDecision{Approved: true}
    }))
```

**Skipping Tool Calls:**

You can skip a tool call while continuing execution (useful for conditional tool execution):

```go
result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithTools(searchTool),
    cogito.WithToolCallBack(func(tool *cogito.ToolChoice, state *cogito.SessionState) cogito.ToolCallDecision {
        // Skip certain tools based on conditions
        if tool.Name == "search" && someCondition {
            return cogito.ToolCallDecision{
                Approved: true,
                Skip:     true, // Skip this tool but continue execution
            }
        }
        return cogito.ToolCallDecision{Approved: true}
    }))
```

When a tool is skipped:
- The tool call is added to the conversation fragment
- A message indicating the tool was skipped is added
- Execution continues to the next iteration
- This is different from `Approved: false` which interrupts execution entirely

**Session State and Resuming Execution:**

The `SessionState` contains the current tool choice and fragment, allowing you to save and resume execution:

```go
var savedState *cogito.SessionState

result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithTools(searchTool),
    cogito.WithToolCallBack(func(tool *cogito.ToolChoice, state *cogito.SessionState) cogito.ToolCallDecision {
        // Save the state for later resumption
        savedState = state
        
        // Interrupt execution
        return cogito.ToolCallDecision{Approved: false}
    }))

// Later, resume execution from the saved state
if savedState != nil {
    resumedFragment, err := savedState.Resume(llm, cogito.WithTools(searchTool))
    if err != nil {
        panic(err)
    }
    // Continue with resumedFragment
}
```

**Starting with a Specific Tool Choice:**

You can pre-select a tool to start execution with:

```go
initialTool := &cogito.ToolChoice{
    Name: "search",
    Arguments: map[string]any{
        "query": "artificial intelligence",
    },
}

result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithTools(searchTool),
    cogito.WithStartWithAction(initialTool))
```

**Forcing Reasoning:**

Enable forced reasoning to ensure the LLM provides detailed reasoning for tool selections:

```go
result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithTools(searchTool),
    cogito.WithForceReasoning())
```

**Error Handling:**

When a tool call callback interrupts execution, Cogito returns `cogito.ErrToolCallCallbackInterrupted`:

```go
result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithTools(searchTool),
    cogito.WithToolCallBack(func(tool *cogito.ToolChoice, state *cogito.SessionState) cogito.ToolCallDecision {
        return cogito.ToolCallDecision{Approved: false} // Interrupt
    }))

if err != nil {
    if errors.Is(err, cogito.ErrToolCallCallbackInterrupted) {
        fmt.Println("Execution was interrupted by tool call callback")
    }
}
```

**Notes:**
- The callback receives both the `ToolChoice` and `SessionState` for full context
- `Approved: false` interrupts execution entirely
- `Approved: true, Skip: true` skips the tool call but continues execution (useful for conditional execution)
- `Adjustment` (non-empty) triggers an adjustment loop where the LLM re-evaluates the tool call
- `Modified` (non-nil) directly uses the modified tool choice without re-querying the LLM
- When a tool is skipped, it's added to the conversation with a "skipped" message, preserving history
- The adjustment loop has a maximum attempt limit (default: 5, configurable via `WithMaxAdjustmentAttempts`)
- `SessionState` can be serialized to JSON for persistence
- The adjustment prompt has been improved to provide better guidance to the LLM

#### Configuring Sink State

When the LLM determines that no tool is needed to respond to the user, Cogito uses a "sink state" tool to handle the response. By default, Cogito uses a built-in `reply` tool, but you can customize or disable this behavior.

**Disable Sink State:**

```go
// Disable sink state entirely - the LLM will return an error if no tool is selected
result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithTools(weatherTool, searchTool),
    cogito.DisableSinkState)
```

**Custom Sink State Tool:**

```go
// Define a custom sink state tool
type CustomReplyArgs struct {
    Reasoning string `json:"reasoning" description:"The reasoning for the reply"`
}

type CustomReplyTool struct{}

func (t *CustomReplyTool) Run(args CustomReplyArgs) (string, error) {
    // Custom logic to process the reasoning and generate a response
    return fmt.Sprintf("Based on: %s", args.Reasoning), nil
}

// Create a custom sink state tool
customSinkTool := cogito.NewToolDefinition(
    &CustomReplyTool{},
    CustomReplyArgs{},
    "custom_reply",
    "Custom tool for handling responses when no other tool is needed",
)

// Use the custom sink state tool
result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithTools(weatherTool, searchTool),
    cogito.WithSinkState(customSinkTool))
```

**Notes:**
- The sink state tool is enabled by default with a built-in `reply` tool
- When enabled, the sink state tool appears as an option in the tool selection enum
- The sink state tool receives a `reasoning` parameter containing the LLM's reasoning about why no tool is needed
- Custom sink state tools must accept a `reasoning` parameter in their arguments

#### Field Annotations for Tool Arguments

Cogito supports several struct field annotations to control how tool arguments are defined in the generated JSON schema:

**Available Annotations:**

- `json:"field_name"` - **Required**. Defines the JSON field name for the parameter.
- `description:"text"` - Provides a description for the field that helps the LLM understand what the parameter is for.
- `enum:"value1,value2,value3"` - Restricts the field to a specific set of allowed values (comma-separated).
- `required:"false"` - Makes the field optional. By default, all fields are required unless marked with `required:"false"`.

**Examples:**

```go
// Basic required field with description
type BasicArgs struct {
    Query string `json:"query" description:"The search query"`
}

// Optional field
type OptionalArgs struct {
    Query string `json:"query" required:"false" description:"Optional search query"`
    Limit int    `json:"limit" required:"false" description:"Maximum number of results"`
}

// Field with enum values
type EnumArgs struct {
    Action string `json:"action" enum:"create,read,update,delete" description:"The action to perform"`
}

// Field with enum and description
type WeatherArgs struct {
    City        string `json:"city" description:"The city name"`
    Unit        string `json:"unit" enum:"celsius,fahrenheit" description:"Temperature unit"`
    Format      string `json:"format" enum:"short,detailed" required:"false" description:"Output format"`
}

// Complete example with multiple field types
type AdvancedSearchArgs struct {
    // Required field with description
    Query string `json:"query" description:"The search query"`
    
    // Optional field with enum
    SortBy string `json:"sort_by" enum:"relevance,date,popularity" required:"false" description:"Sort order"`
    
    // Optional numeric field
    Limit int `json:"limit" required:"false" description:"Maximum number of results"`
    
    // Optional boolean field
    IncludeImages bool `json:"include_images" required:"false" description:"Include images in results"`
}

// Create tool with advanced arguments
searchTool := cogito.NewToolDefinition(
    &AdvancedSearchTool{},
    AdvancedSearchArgs{},
    "advanced_search",
    "Advanced search with sorting and filtering options",
)
```

**Notes:**
- Fields without `required:"false"` are automatically marked as required in the JSON schema
- Enum values are case-sensitive and should match exactly what you expect in `Run()`
- The `json` tag is required for all fields that should be included in the tool schema
- Descriptions help the LLM understand the purpose of each parameter, leading to better tool calls

Alternatively, you can implement `ToolDefinitionInterface` directly if you prefer more control:

```go
type CustomTool struct{}

func (t *CustomTool) Tool() openai.Tool {
    return openai.Tool{
        Type: openai.ToolTypeFunction,
        Function: &openai.FunctionDefinition{
            Name:        "custom_tool",
            Description: "A custom tool",
            Parameters: jsonschema.Definition{
                // Define your schema
            },
        },
    }
}

func (t *CustomTool) Execute(args map[string]any) (string, error) {
    // Your execution logic
    return "result", nil
}
```

### Guidelines for Intelligent Tool Selection

Guidelines provide a powerful way to define conditional rules for tool usage. The LLM intelligently selects which guidelines are relevant based on the conversation context, enabling dynamic and context-aware tool selection.

```go
// Create tool definitions
searchTool := cogito.NewToolDefinition(
    &SearchTool{},
    SearchArgs{},
    "search",
    "Search for information",
)

weatherTool := cogito.NewToolDefinition(
    &WeatherTool{},
    WeatherArgs{},
    "get_weather",
    "Get weather information",
)

// Define guidelines with conditions and associated tools
guidelines := cogito.Guidelines{
    cogito.Guideline{
        Condition: "User asks about information or facts",
        Action:    "Use the search tool to find information",
        Tools: cogito.Tools{
            searchTool,
        },
    },
    cogito.Guideline{
        Condition: "User asks for the weather in a city",
        Action:    "Use the weather tool to find the weather",
        Tools: cogito.Tools{
            weatherTool,
        },
    },
}

// Get relevant guidelines for the current conversation
fragment := cogito.NewEmptyFragment().
    AddMessage("user", "When was Isaac Asimov born?")

// Execute tools with guidelines
result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithGuidelines(guidelines),
    cogito.EnableStrictGuidelines) // Only use tools from relevant guidelines
if err != nil {
    panic(err)
}
```

#### Automatic Tool Guidance with EnableGuidedTools

The `EnableGuidedTools` option enables intelligent filtering of tools through guidance, even when explicit guidelines aren't provided for all tools. This feature automatically creates "virtual guidelines" from tool descriptions, allowing the LLM to intelligently filter and select tools based on their descriptions.

**When No Guidelines Exist:**

When `EnableGuidedTools` is enabled and no guidelines are provided, Cogito automatically creates virtual guidelines for all tools using their descriptions as the condition. This allows the LLM to filter tools based on relevance to the conversation context.

```go
// Create tools without any guidelines
searchTool := cogito.NewToolDefinition(
    &SearchTool{},
    SearchArgs{},
    "search",
    "A search engine to find information about a topic",
)

weatherTool := cogito.NewToolDefinition(
    &WeatherTool{},
    WeatherArgs{},
    "get_weather",
    "Get weather information for a specific city",
)

// EnableGuidedTools creates virtual guidelines from tool descriptions
result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithTools(searchTool, weatherTool),
    cogito.EnableGuidedTools) // Creates virtual guidelines from descriptions
```

**When Guidelines Exist:**

When guidelines are provided but some tools aren't included in any guideline, `EnableGuidedTools` creates virtual guidelines only for those "unguided" tools. This allows you to have explicit guidelines for some tools while automatically handling others.

```go
// Define guidelines for search tool only
guidelines := cogito.Guidelines{
    cogito.Guideline{
        Condition: "User asks about information or facts",
        Action:    "Use the search tool to find information",
        Tools: cogito.Tools{
            searchTool,
        },
    },
}

// weatherTool is not in any guideline - EnableGuidedTools will create
// a virtual guideline for it using its description
result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithGuidelines(guidelines...),
    cogito.WithTools(weatherTool),
    cogito.EnableGuidedTools) // Creates virtual guidelines for unguided tools
```

**How It Works:**

1. **No Guidelines Scenario**: When no guidelines exist and `EnableGuidedTools` is enabled:
   - Virtual guidelines are created for ALL tools
   - Each tool's description becomes the condition/guidance in the template
   - The LLM filters tools based on how well their descriptions match the conversation context

2. **With Guidelines Scenario**: When guidelines exist and `EnableGuidedTools` is enabled:
   - Only tools NOT in any guideline get virtual guidelines
   - Virtual guidelines use the format: "The task requires: [tool description]"
   - Real guidelines and virtual guidelines are merged and filtered together

**Benefits:**

- **Automatic Tool Filtering**: No need to create guidelines for every tool
- **Better Tool Selection**: LLM can intelligently filter tools based on descriptions
- **Flexible Configuration**: Mix explicit guidelines with automatic guidance
- **Reduced Configuration**: Especially useful when you have many tools

**Notes:**

- Tool descriptions should be meaningful and descriptive for best results
- Virtual guidelines follow the same filtering process as real guidelines
- When no guidelines exist, tool descriptions serve as both condition and guidance in the template
- The feature adds LLM call overhead for filtering virtual guidelines

### Goal-Oriented Planning

```go
// Extract a goal from conversation
goal, err := cogito.ExtractGoal(llm, fragment)
if err != nil {
    panic(err)
}

// Create tool definition
searchTool := cogito.NewToolDefinition(
    &SearchTool{},
    SearchArgs{},
    "search",
    "Search for information",
)

// Create a plan to achieve the goal
plan, err := cogito.ExtractPlan(llm, fragment, goal, 
    cogito.WithTools(searchTool))
if err != nil {
    panic(err)
}

// Execute the plan
result, err := cogito.ExecutePlan(llm, fragment, plan, goal,
    cogito.WithTools(searchTool))
if err != nil {
    panic(err)
}
```

### Planning with TODOs

Planning with TODOs addresses context accumulation by starting each iteration with fresh context while persisting TODOs and feedback between iterations. This pattern uses separate worker and judge models: the worker executes tasks, and one or more judge LLMs review the work to determine if goal execution is completed or needs rework.

**How it works:**

1. **Work Phase**: Worker model executes tasks with fresh context that includes:
   - Overall goal
   - TODO list with progress (markdown checkboxes)
   - Previous feedback from review phase
2. **Review Phase**: One or more judge LLMs review the work and decide if goal execution is completed or incomplete (needs rework). When multiple reviewers are provided, a majority vote determines the final decision.
3. **Persistence**: TODOs and feedback persist between iterations; conversation history is cleared
4. **Iteration**: Loop continues until goal execution is completed or max iterations

**Automatic TODO Generation (Recommended):**

TODOs are automatically generated from plan subtasks when `WithReviewerLLM()` (judge LLM) is provided:

```go
workerLLM := cogito.NewOpenAILLM("worker-model", "key", "url")
judgeLLM := cogito.NewOpenAILLM("judge-model", "key", "url")

goal, _ := cogito.ExtractGoal(workerLLM, fragment)
plan, _ := cogito.ExtractPlan(workerLLM, fragment, goal)

// Execute plan with Planning with TODOs enabled
// TODOs are automatically generated from plan subtasks
result, err := cogito.ExecutePlan(
    workerLLM,  // worker LLM
    fragment,
    plan,
    goal,
    cogito.WithTools(searchTool, writeTool),
    cogito.WithIterations(5),  // TODO iterations
    cogito.WithReviewerLLM(judgeLLM),  // Provide judge LLM for review (enables Planning with TODOs)
    cogito.WithTODOPersistence("./todos.json"),  // Optional: file persistence
)
```

**Manual TODO List:**

You can also provide a manual TODO list:

```go
import "github.com/mudler/cogito/structures"

// Initialize TODO list manually
todoList := &structures.TODOList{
    TODOs: []structures.TODO{
        {ID: "1", Description: "Research topic", Completed: false},
        {ID: "2", Description: "Write draft", Completed: false},
    },
}

// Execute plan with manually provided TODOs
result, err := cogito.ExecutePlan(
    workerLLM,
    fragment,
    plan,
    goal,
    cogito.WithTools(searchTool, writeTool),
    cogito.WithIterations(5),
    cogito.WithTODOs(todoList),   // Manually provide TODO list
    cogito.WithReviewerLLM(judgeLLM),  // Provide judge LLM for review
    cogito.WithTODOPersistence("./todos.json"),
)
```

**When to use Planning with TODOs:**

- When dealing with complex, multi-step tasks that may require multiple iterations
- When you want to prevent context accumulation from failed attempts
- When you need separate models for work and review phases (worker and judge)
- When you want to track progress explicitly with checkboxes

**Multiple Reviewer LLMs (Majority Voting):**

You can provide multiple reviewer LLMs for more robust decision-making. When multiple reviewers are provided, Cogito uses majority voting to determine if the goal has been achieved:

```go
workerLLM := cogito.NewOpenAILLM("worker-model", "key", "url")
judgeLLM1 := cogito.NewOpenAILLM("judge-model-1", "key", "url")
judgeLLM2 := cogito.NewOpenAILLM("judge-model-2", "key", "url")
judgeLLM3 := cogito.NewOpenAILLM("judge-model-3", "key", "url")

goal, _ := cogito.ExtractGoal(workerLLM, fragment)
plan, _ := cogito.ExtractPlan(workerLLM, fragment, goal)

// Execute plan with multiple reviewers
// The goal is considered achieved if more than half of reviewers agree
result, err := cogito.ExecutePlan(
    workerLLM,
    fragment,
    plan,
    goal,
    cogito.WithTools(searchTool, writeTool),
    cogito.WithIterations(5),
    cogito.WithReviewerLLM(judgeLLM1, judgeLLM2, judgeLLM3),  // Multiple reviewers
    cogito.WithTODOPersistence("./todos.json"),
)
```

**How Majority Voting Works:**

- Each reviewer LLM independently evaluates whether the goal has been achieved
- If more than half of the reviewers determine the goal is achieved, the overall decision is positive
- The review result from the majority is used as the final feedback
- This approach provides more reliable and consistent goal achievement detection, especially when using smaller or less reliable models

**File-based Persistence:**

TODOs can be persisted to a file and loaded between sessions:

```go
result, err := cogito.ExecutePlan(
    workerLLM,
    fragment,
    plan,
    goal,
    cogito.WithReviewerLLM(judgeLLM),  // Single reviewer (or multiple)
    cogito.WithTODOPersistence("./todos.json"),  // Save/load TODOs from file
)
```

The TODO file is automatically saved after each iteration and loaded at the start of execution.

### Content Refinement

```go
// Create tool definition
searchTool := cogito.NewToolDefinition(
    &SearchTool{},
    SearchArgs{},
    "search",
    "Search for information",
)

// Refine content through iterative improvement
refined, err := cogito.ContentReview(llm, fragment,
    cogito.WithIterations(3),
    cogito.WithTools(searchTool))
if err != nil {
    panic(err)
}
```



### Iterative Content Improvement

An example on how to iteratively improve content by using two separate models:

```go
llm := cogito.NewOpenAILLM("your-model", "api-key", "https://api.openai.com")
reviewerLLM := cogito.NewOpenAILLM("your-reviewer-model", "api-key", "https://api.openai.com")

// Create content to review
initial := cogito.NewEmptyFragment().
    AddMessage("user", "Write about climate change")

response, _ := llm.Ask(ctx, initial)

// Create tool definitions
searchTool := cogito.NewToolDefinition(
    &SearchTool{},
    SearchArgs{},
    "search",
    "Search for information",
)

factCheckTool := cogito.NewToolDefinition(
    &FactCheckTool{},
    FactCheckArgs{},
    "fact_check",
    "Verify facts",
)

// Iteratively improve with tool support
improvedResponse, _ := cogito.ContentReview(reviewerLLM, response,
    cogito.WithIterations(3),
    cogito.WithTools(searchTool, factCheckTool),
    cogito.EnableToolReasoner)
```


### Model Context Protocol (MCP) Integration

Cogito supports the Model Context Protocol (MCP) for seamless integration with external tools and services. MCP allows you to connect to remote tool providers and use their capabilities directly within your Cogito workflows.

```go
import (
    "github.com/modelcontextprotocol/go-sdk/mcp"
)

// Create MCP client sessions
command := exec.Command("docker", "run", "-i", "--rm", "ghcr.io/mudler/mcps/weather:master")
transport := &mcp.CommandTransport{ Command: command }

client := mcp.NewClient(&mcp.Implementation{Name: "test", Version: "v1.0.0"}, nil)
mcpSession, _ := client.Connect(context.Background(), transport, nil)

// Use MCP tools in your workflows
result, _ := cogito.ExecuteTools(llm, fragment,
    cogito.WithMCPs(mcpSession))

```

#### MCP with Guidelines

```go
// Define guidelines that include MCP tools
guidelines := cogito.Guidelines{
    cogito.Guideline{
        Condition: "User asks about information or facts",
        Action:    "Use the MCP search tool to find information",
    },
}

// Execute with MCP tools and guidelines
result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithMCPs(searchSession),
    cogito.WithGuidelines(guidelines),
    cogito.EnableStrictGuidelines)
```

### Custom Prompts

```go
customPrompt := cogito.NewPrompt(`Your custom prompt template with {{.Context}}`)

result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithPrompt(cogito.ToolReasonerType, customPrompt))
```

## 🎮 Examples

### Interactive Chat Bot

```bash
# Run the example chat application
make example-chat
```

This starts an interactive chat session with tool support including web search capabilities.

### Custom Tool Implementation

See `examples/internal/search/search.go` for a complete example of implementing a DuckDuckGo search tool.

## 🧪 Testing

The library includes comprehensive test coverage using Ginkgo and Gomega. Tests use containerized LocalAI for integration testing.

### Running Tests

```bash
# Run all tests
make test

# Run with specific log level
LOG_LEVEL=debug make test

# Run with custom arguments
GINKGO_ARGS="--focus=Fragment" make test
```

## 📄 License

Ettore Di Giacinto 2025-now. Cogito is released under the Apache 2.0 License.

## 📚 Citation

If you use Cogito in your research or academic work, please cite our paper:

```bibtex
@article{cogito2025,
  title={Cogito: A Framework for Building Intelligent Agentic Software with LLM-Powered Workflows},
  author={Ettore Di Giacinto <mudler@localai.io>},
  journal={https://github.com/mudler/cogito},
  year={2025},
  note={}
}
```
