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
	toolDef     *ToolDefinition[map[string]any]
}

func NewMockTool(name, description string) ToolDefinitionInterface {
	mockTool := &MockTool{
		name:        name,
		description: description,
		status:      &ToolStatus{},
	}
	toolDef := &ToolDefinition[map[string]any]{
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

func (m *MockTool) Run(args map[string]any) (string, any, error) {
	if m.runError != nil {
		return "", nil, m.runError
	}
	defer func() {
		m.runIndex++
	}()
	return m.runResults[m.runIndex], nil, nil
}

func (m *MockTool) NewArgs() *map[string]any {
	args := make(map[string]any)
	return &args
}

func (m *MockTool) SetRunResult(result string) {
	m.runResults = append(m.runResults, result)
}

func (m *MockTool) SetRunError(err error) {
	m.runError = err
}

// GetMockTool extracts the MockTool from a ToolDef (if it contains one)
func GetMockTool(toolDef ToolDefinitionInterface) *MockTool {
	if toolDefT, ok := toolDef.(*ToolDefinition[map[string]any]); ok {
		if mockTool, ok := toolDefT.ToolRunner.(*MockTool); ok {
			return mockTool
		}
	}
	return nil
}

// SetRunResult sets the result for a mock tool within a ToolDef
func SetRunResult(toolDef ToolDefinitionInterface, result string) {
	if mockTool := GetMockTool(toolDef); mockTool != nil {
		mockTool.SetRunResult(result)
	}
}

// SetRunError sets an error for a mock tool within a ToolDef
func SetRunError(toolDef ToolDefinitionInterface, err error) {
	if mockTool := GetMockTool(toolDef); mockTool != nil {
		mockTool.SetRunError(err)
	}
}
