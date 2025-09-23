package structures

import "github.com/sashabaranov/go-openai/jsonschema"

type Structure struct {
	Schema jsonschema.Definition
	Object any
}

func structureType[T any](definition jsonschema.Definition) (Structure, *T) {
	var t T
	return Structure{definition, &t}, &t
}
