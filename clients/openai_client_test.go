package clients

import "testing"

func TestNewOpenAILLMWithOptionsSetsTemperature(t *testing.T) {
	llm := NewOpenAILLMWithOptions("m", "k", "http://localhost", OpenAIOptions{Temperature: 0.7})
	if llm.temperature != 0.7 {
		t.Fatalf("expected temperature 0.7, got %v", llm.temperature)
	}
}

func TestNewOpenAILLMDefaultsTemperatureZeroMeansUnset(t *testing.T) {
	llm := NewOpenAILLM("m", "k", "http://localhost")
	if llm.temperature != 0 {
		t.Fatalf("expected default temperature 0 (unset), got %v", llm.temperature)
	}
}
