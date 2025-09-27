package structures

import "github.com/sashabaranov/go-openai/jsonschema"

type Goal struct {
	Goal string `json:"goal"`
}

func StructureGoal() (Structure, *Goal) {
	return structureType[Goal](
		jsonschema.Definition{
			Type:                 jsonschema.Object,
			AdditionalProperties: false,
			Properties: map[string]jsonschema.Definition{
				"goal": {
					Type:        jsonschema.String,
					Description: "The final goal to reach",
				},
			},
			Required: []string{"goal"},
		})
}
