package structures

import "github.com/sashabaranov/go-openai/jsonschema"

type Boolean struct {
	Boolean bool `json:"extract_boolean"`
}

func StructureBoolean() (Structure, *Boolean) {
	return structureType[Boolean](
		jsonschema.Definition{
			Type: jsonschema.Object,
			Properties: map[string]jsonschema.Definition{
				"extract_boolean": {
					Type:        jsonschema.Boolean,
					Description: "Yes/no answer",
				},
			},
			Required: []string{"extract_boolean"},
		})
}
