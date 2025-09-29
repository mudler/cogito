package mock

import (
	. "github.com/mudler/cogito"
	"github.com/sashabaranov/go-openai"
)

// MockTool implements the Tool interface for testing
type MockTool struct {
	name        string
	description string
	runResults  []string
	runError    error
	runIndex    int
	status      *ToolStatus
}

func NewMockTool(name, description string) *MockTool {
	return &MockTool{
		name:        name,
		description: description,
		status:      &ToolStatus{},
	}
}

func (m *MockTool) Tool() openai.Tool {
	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        m.name,
			Description: m.description,
			// We don't need parameters in this mock
			// Will show up as null
		},
	}
}

func (m *MockTool) Status() *ToolStatus {
	return m.status
}

func (m *MockTool) Run(args map[string]any) (string, error) {
	if m.runError != nil {
		return "", m.runError
	}
	defer func() {
		m.runIndex++
	}()
	return m.runResults[m.runIndex], nil
}

func (m *MockTool) SetRunResult(result string) {
	m.runResults = append(m.runResults, result)
}

func (m *MockTool) SetRunError(err error) {
	m.runError = err
}
