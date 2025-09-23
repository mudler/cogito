package prompt

import (
	"bytes"
	"text/template"

	"github.com/Masterminds/sprig/v3"
)

type StaticPrompt struct {
	template string
}

type Prompt interface {
	Render(data any) (string, error)
}

func NewPrompt(template string) StaticPrompt {
	return StaticPrompt{template: template}
}

func (p StaticPrompt) Render(data any) (string, error) {

	b := bytes.NewBuffer([]byte{})

	tmpl, err := template.New("prompt").Funcs(sprig.FuncMap()).Parse(p.template)
	if err != nil {
		return "", err
	}

	err = tmpl.Execute(b, data)

	return b.String(), err
}

type PromptMap map[PromptType]Prompt

func (p PromptMap) GetPrompt(t PromptType) Prompt {
	prompter, exists := p[t]
	if !exists {
		return defaultPromptMap[t]
	}

	return prompter
}
