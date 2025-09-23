package structures

import "github.com/sashabaranov/go-openai/jsonschema"

type Gaps struct {
	Gaps []string `json:"gaps"`
}

func StructureGaps() (Structure, *Gaps) {
	return structureType[Gaps](
		jsonschema.Definition{
			Type: jsonschema.Object,
			Properties: map[string]jsonschema.Definition{
				"gaps": {
					Type:        jsonschema.Array,
					Items:       &jsonschema.Definition{Type: jsonschema.String},
					Description: "List of gaps in the content",
				},
			},
			Required: []string{"gaps"},
		})
}
