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

First, implement a tool by creating a type that implements the `Tool` interface:

```go
import (
    "fmt"
    "github.com/mudler/cogito"
)

type WeatherTool struct{}

// Tool implements the Tool interface
func (w *WeatherTool) Run(args map[string]any) (string, error) {
    city := args["city"].(string)
    // Your tool implementation here
    return fmt.Sprintf("Weather in %s: Sunny, 72°F", city), nil
}
```

Then, create a `ToolDefinition` wrapping your tool. You can use either a map (JSON schema format) or a struct for `InputArguments`:

```go
// Using map format (JSON schema)
weatherTool := &cogito.ToolDefinition{
    ToolRunner: &WeatherTool{},
    Name:       "get_weather",
    InputArguments: map[string]interface{}{
        "description": "Get the current weather for a city",
        "type":        "object",
        "properties": map[string]interface{}{
            "city": map[string]interface{}{
                "type":        "string",
                "description": "The city name",
            },
        },
        "required": []string{"city"},
    },
}

// Or using a struct (schema is auto-generated)
weatherTool := &cogito.ToolDefinition{
    ToolRunner: &WeatherTool{},
    Name:       "get_weather",
    InputArguments: struct {
        City string `json:"city" jsonschema:"description=The city name"`
    }{},
}

// Create a fragment with user input
fragment := cogito.NewEmptyFragment().
    AddMessage("user", "What's the weather in San Francisco?")

// Execute with tools
result, err := cogito.ExecuteTools(llm, fragment, 
    cogito.WithTools(weatherTool))
if err != nil {
    panic(err)
}

// result.Status.ToolsCalled will contain all the tools being called
```

### Guidelines for Intelligent Tool Selection

Guidelines provide a powerful way to define conditional rules for tool usage. The LLM intelligently selects which guidelines are relevant based on the conversation context, enabling dynamic and context-aware tool selection.

```go
// Define tools using ToolDefinition
searchTool := &cogito.ToolDefinition{
    ToolRunner: &SearchTool{},
    Name:       "search",
    InputArguments: map[string]interface{}{
        "description": "Search for information",
        "type":        "object",
        "properties": map[string]interface{}{
            "query": map[string]interface{}{
                "type":        "string",
                "description": "The search query",
            },
        },
        "required": []string{"query"},
    },
}

weatherTool := &cogito.ToolDefinition{
    ToolRunner: &WeatherTool{},
    Name:       "get_weather",
    InputArguments: map[string]interface{}{
        "description": "Get the weather for a city",
        "type":        "object",
        "properties": map[string]interface{}{
            "city": map[string]interface{}{
                "type":        "string",
                "description": "The city name",
            },
        },
        "required": []string{"city"},
    },
}

// Define guidelines with conditions and associated tools
guidelines := cogito.Guidelines{
    cogito.Guideline{
        Condition: "User asks about information or facts",
        Action:    "Use the search tool to find information",
        Tools:     cogito.Tools{searchTool},
    },
    cogito.Guideline{
        Condition: "User asks for the weather in a city",
        Action:    "Use the weather tool to find the weather",
        Tools:     cogito.Tools{weatherTool},
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
// Define your search tool
searchTool := &cogito.ToolDefinition{
    ToolRunner: &SearchTool{},
    Name:       "search",
    InputArguments: map[string]interface{}{
        "description": "Search for information",
        "type":        "object",
        "properties": map[string]interface{}{
            "query": map[string]interface{}{
                "type":        "string",
                "description": "The search query",
            },
        },
        "required": []string{"query"},
    },
}

// Extract a goal from conversation
goal, err := cogito.ExtractGoal(llm, fragment)
if err != nil {
    panic(err)
}

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
// Define your search tool
searchTool := &cogito.ToolDefinition{
    ToolRunner: &SearchTool{},
    Name:       "search",
    InputArguments: map[string]interface{}{
        "description": "Search for information",
        "type":        "object",
        "properties": map[string]interface{}{
            "query": map[string]interface{}{
                "type":        "string",
                "description": "The search query",
            },
        },
        "required": []string{"query"},
    },
}

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

// Define your tools
searchTool := &cogito.ToolDefinition{
    ToolRunner: &SearchTool{},
    Name:       "search",
    InputArguments: map[string]interface{}{
        "description": "Search for information",
        "type":        "object",
        "properties": map[string]interface{}{
            "query": map[string]interface{}{
                "type":        "string",
                "description": "The search query",
            },
        },
        "required": []string{"query"},
    },
}

factCheckTool := &cogito.ToolDefinition{
    ToolRunner: &FactCheckTool{},
    Name:       "fact_check",
    InputArguments: map[string]interface{}{
        "description": "Verify facts in content",
        "type":        "object",
        "properties": map[string]interface{}{
            "claim": map[string]interface{}{
                "type":        "string",
                "description": "The claim to verify",
            },
        },
        "required": []string{"claim"},
    },
}

// Create content to review
initial := cogito.NewEmptyFragment().
    AddMessage("user", "Write about climate change")

response, _ := llm.Ask(ctx, initial)

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

To implement a custom tool, follow these steps:

1. **Implement the `Tool` interface**: Create a type with a `Run` method:
   ```go
   type SearchTool struct{}
   
   func (s *SearchTool) Run(args map[string]any) (string, error) {
       query, ok := args["query"].(string)
       if !ok {
           return "", fmt.Errorf("query parameter is required")
       }
       // Your implementation here
       return "Search results...", nil
   }
   ```

2. **Create a `ToolDefinition`**: Wrap your tool with metadata:
   ```go
   searchTool := &cogito.ToolDefinition{
       ToolRunner: &SearchTool{},
       Name:       "search",
       InputArguments: map[string]interface{}{
           "description": "Search the web for information",
           "type":        "object",
           "properties": map[string]interface{}{
               "query": map[string]interface{}{
                   "type":        "string",
                   "description": "The search query",
               },
           },
           "required": []string{"query"},
       },
   }
   ```

3. **Use the tool**: Pass it to any function that accepts tools:
   ```go
   result, err := cogito.ExecuteTools(llm, fragment,
       cogito.WithTools(searchTool))
   ```

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
