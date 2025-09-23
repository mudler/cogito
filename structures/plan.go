package structures

import "github.com/sashabaranov/go-openai/jsonschema"

type Plan struct {
	Subtasks []string `json:"subtasks"`
}

func StructurePlan() (Structure, *Plan) {
	return structureType[Plan](jsonschema.Definition{
		Type: jsonschema.Object,
		Properties: map[string]jsonschema.Definition{
			"subtasks": {
				Type:        jsonschema.Array,
				Items:       &jsonschema.Definition{Type: jsonschema.String},
				Description: "List of detailed subtasks which compose the plan",
			},
		},
		Required: []string{"subtasks"},
	})
}
