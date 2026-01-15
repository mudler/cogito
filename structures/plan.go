package structures

import "github.com/sashabaranov/go-openai/jsonschema"

type Plan struct {
	Subtasks    []string `json:"subtasks"`
	Description string   `json:"description"`
}

func StructurePlan() (Structure, *Plan) {
	return structureType[Plan](jsonschema.Definition{
		Type:                 jsonschema.Object,
		AdditionalProperties: false,
		Properties: map[string]jsonschema.Definition{
			"description": {
				Type:        jsonschema.String,
				Description: "Detailed description of the plan to achieve the goal",
			},
			"subtasks": {
				Type:        jsonschema.Array,
				Items:       &jsonschema.Definition{Type: jsonschema.String},
				Description: "List of detailed subtasks which compose the plan",
			},
		},
		Required: []string{"description", "subtasks"},
	})
}
