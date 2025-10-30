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
