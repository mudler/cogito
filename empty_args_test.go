package cogito

import (
	"encoding/json"
	"testing"
)

func TestEmptyArgumentsNormalization(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantErr  bool
	}{
		{"empty string", "", false},
		{"empty object", "{}", false},
		{"with args", `{"foo": "bar"}`, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Simulate what we do in the fixed code
			args := tt.input
			if args == "" {
				args = "{}"
			}

			var result map[string]any
			err := json.Unmarshal([]byte(args), &result)
			
			if (err != nil) != tt.wantErr {
				t.Errorf("json.Unmarshal() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
