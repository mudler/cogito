<img width="300"  align="left" alt="Gemini_Generated_Image_jbv0xajbv0xajbv0" src="https://github.com/user-attachments/assets/ab243414-4452-4f03-b266-f9326ad2c8d1" />

**Cogito** is a powerful Go library for building intelligent, co-operative agentic software and LLM-powered workflows, focusing on improving results for small, open source language models that scales to any LLM.

🧪 **Tested on Small Models** ! Our test suite runs on 0.6b Qwen (not fine-tuned), proving effectiveness even with minimal resources.

> 📝 **Working on Official Paper**  
> I am currently working on the official academic/white paper for Cogito. The paper will provide detailed theoretical foundations, experimental results, and comprehensive analysis of the framework's capabilities.

## 🏗️ Architecture

Cogito is the result of building [LocalAI](https://github.com/mudler/LocalAI), [LocalAGI](https://github.com/mudler/LocalAGI) and [LocalOperator](https://github.com/mudler/LocalOperator) (yet to be released).

Cogito uses an internal pipeline to first make the LLM reason about a specific task, forcing the model to reason and later extracts with BNF grammars exact data structures from the LLM. This is applied to every primitive exposed by the framework.

It provides a comprehensive framework for creating conversational AI systems with advanced reasoning, tool execution, goal-oriented planning, and iterative content refinement capabilities.

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

```go
// Create a fragment with user input
fragment := cogito.NewFragment(openai.ChatCompletionMessage{
    Role:    "user",
    Content: "What's the weather in San Francisco?",
})

// Execute with tools
result, err := cogito.ExecuteTools(llm, fragment, 
    cogito.WithTools(&WeatherTool{}))
if err != nil {
    panic(err)
}

// result.Status.ToolsCalled will contain all the tools being called
```

### Guidelines for Intelligent Tool Selection

Guidelines provide a powerful way to define conditional rules for tool usage. The LLM intelligently selects which guidelines are relevant based on the conversation context, enabling dynamic and context-aware tool selection.

```go
// Define guidelines with conditions and associated tools
guidelines := cogito.Guidelines{
    cogito.Guideline{
        Condition: "User asks about information or facts",
        Action:    "Use the search tool to find information",
        Tools: cogito.Tools{
            &SearchTool{},
        },
    },
    cogito.Guideline{
        Condition: "User asks for the weather in a city",
        Action:    "Use the weather tool to find the weather",
        Tools: cogito.Tools{
            &WeatherTool{},
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

// Create a plan to achieve the goal
plan, err := cogito.ExtractPlan(llm, fragment, goal, 
    cogito.WithTools(&SearchTool{}))
if err != nil {
    panic(err)
}

// Execute the plan
result, err := cogito.ExecutePlan(llm, fragment, plan, goal,
    cogito.WithTools(&SearchTool{}))
if err != nil {
    panic(err)
}
```

### Content Refinement

```go
// Refine content through iterative improvement
refined, err := cogito.ContentReview(llm, fragment,
    cogito.WithIterations(3),
    cogito.WithTools(&SearchTool{}))
if err != nil {
    panic(err)
}
```



### Iterative Content Improvement

An example on how to iteratively improve content by using two separate models:

```go
// Create initial content
initial := cogito.NewEmptyFragment().
    AddMessage("user", "Write about climate change")

llm := cogito.NewOpenAILLM("your-model", "api-key", "https://api.openai.com")
response, err := llm.Ask(ctx, initial)

reviewerLLM := cogito.NewOpenAILLM("your-reviewer-model", "api-key", "https://api.openai.com")
// Iteratively improve with tool support
improved, err := cogito.ContentReview(reviewerLLM, response,
    cogito.WithIterations(3),
    cogito.WithTools(&SearchTool{}, &FactCheckTool{}),
    cogito.EnableToolReasoner)
```


### Custom Prompts

```go
customPrompt := cogito.NewPrompt(`Your custom prompt template with {{.Context}}`)

result, err := cogito.ExecuteTools(llm, fragment,
    cogito.WithPrompt(cogito.ToolSelectorType, customPrompt))
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
