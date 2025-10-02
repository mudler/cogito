package structures

import "github.com/sashabaranov/go-openai/jsonschema"

type Guidelines struct {
	Guidelines []int `json:"guidelines"`
}

func StructureGuidelines() (Structure, *Guidelines) {
	return structureType[Guidelines](
		jsonschema.Definition{
			Type:                 jsonschema.Object,
			AdditionalProperties: false,
			Properties: map[string]jsonschema.Definition{
				"guidelines": {
					Type:        jsonschema.Array,
					Items:       &jsonschema.Definition{Type: jsonschema.Integer},
					Description: "List of guidelines",
				},
			},
			Required: []string{"guidelines"},
		})
}
