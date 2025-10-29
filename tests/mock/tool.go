package mock

import (
	. "github.com/mudler/cogito"
)

// MockTool implements the Tool interface for testing
type MockTool struct {
	name        string
	description string
	runResults  []string
	runError    error
	runIndex    int
	status      *ToolStatus
	toolDef     *ToolDefinition
}

func NewMockTool(name, description string) *ToolDefinition {
	mockTool := &MockTool{
		name:        name,
		description: description,
		status:      &ToolStatus{},
	}
	toolDef := &ToolDefinition{
		ToolRunner:  mockTool,
		Name:        name,
		Description: description,
		InputArguments: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	}
	mockTool.toolDef = toolDef
	return toolDef
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

// GetMockTool extracts the MockTool from a ToolDefinition (if it contains one)
func GetMockTool(toolDef *ToolDefinition) *MockTool {
	if mockTool, ok := toolDef.ToolRunner.(*MockTool); ok {
		return mockTool
	}
	return nil
}

// SetRunResult sets the result for a mock tool within a ToolDefinition
func SetRunResult(toolDef *ToolDefinition, result string) {
	if mockTool := GetMockTool(toolDef); mockTool != nil {
		mockTool.SetRunResult(result)
	}
}

// SetRunError sets an error for a mock tool within a ToolDefinition
func SetRunError(toolDef *ToolDefinition, err error) {
	if mockTool := GetMockTool(toolDef); mockTool != nil {
		mockTool.SetRunError(err)
	}
}
