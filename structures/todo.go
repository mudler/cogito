package structures

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai/jsonschema"
)

type TODO struct {
	ID          string         `json:"id"`
	Description string         `json:"description"`
	Completed   bool           `json:"completed"`
	Feedback    string         `json:"feedback,omitempty"`
	Metadata    map[string]any `json:"metadata,omitempty"`
}

type TODOList struct {
	TODOs       []TODO    `json:"todos"`
	Markdown    string    `json:"markdown"`
	LastUpdated time.Time `json:"last_updated"`
}

func StructureTODO() (Structure, *TODOList) {
	return structureType[TODOList](jsonschema.Definition{
		Type:                 jsonschema.Object,
		AdditionalProperties: false,
		Properties: map[string]jsonschema.Definition{
			"todos": {
				Type:        jsonschema.Array,
				Description: "List of TODO items",
				Items: &jsonschema.Definition{
					Type: jsonschema.Object,
					Properties: map[string]jsonschema.Definition{
						"id": {
							Type:        jsonschema.String,
							Description: "Unique identifier for the TODO",
						},
						"description": {
							Type:        jsonschema.String,
							Description: "Description of the TODO item",
						},
						"completed": {
							Type:        jsonschema.Boolean,
							Description: "Whether the TODO is completed",
						},
						"feedback": {
							Type:        jsonschema.String,
							Description: "Optional feedback for this TODO",
						},
						"metadata": {
							Type:        jsonschema.Object,
							Description: "Optional metadata for this TODO",
						},
					},
					Required: []string{"id", "description", "completed"},
				},
			},
		},
		Required: []string{"todos"},
	})
}

// ToMarkdown generates markdown representation with checkboxes
func (tl *TODOList) ToMarkdown() string {
	var builder strings.Builder
	for _, todo := range tl.TODOs {
		if todo.Completed {
			builder.WriteString(fmt.Sprintf("- [x] %s\n", todo.Description))
		} else {
			builder.WriteString(fmt.Sprintf("- [ ] %s\n", todo.Description))
		}
		if todo.Feedback != "" {
			builder.WriteString(fmt.Sprintf("  Feedback: %s\n", todo.Feedback))
		}
	}
	tl.Markdown = builder.String()
	return tl.Markdown
}

// FromMarkdown parses markdown checkboxes into TODO list
func (tl *TODOList) FromMarkdown(markdown string) error {
	lines := strings.Split(markdown, "\n")
	tl.TODOs = []TODO{}

	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Check for checkbox pattern: - [x] or - [ ]
		if strings.HasPrefix(line, "- [") {
			completed := strings.Contains(line, "- [x]") || strings.Contains(line, "- [X]")
			description := ""

			if completed {
				description = strings.TrimPrefix(line, "- [x]")
				description = strings.TrimPrefix(description, "- [X]")
			} else {
				description = strings.TrimPrefix(line, "- [ ]")
			}
			description = strings.TrimSpace(description)

			// Check if next line is feedback
			feedback := ""
			if i+1 < len(lines) {
				nextLine := strings.TrimSpace(lines[i+1])
				if strings.HasPrefix(nextLine, "Feedback:") {
					feedback = strings.TrimPrefix(nextLine, "Feedback:")
					feedback = strings.TrimSpace(feedback)
				}
			}

			todo := TODO{
				ID:          fmt.Sprintf("%d", len(tl.TODOs)+1),
				Description: description,
				Completed:   completed,
				Feedback:    feedback,
			}
			tl.TODOs = append(tl.TODOs, todo)
		}
	}

	tl.Markdown = markdown
	tl.LastUpdated = time.Now()
	return nil
}

// ToJSON serializes TODOList to JSON
func (tl *TODOList) ToJSON() ([]byte, error) {
	return json.Marshal(tl)
}

// FromJSON deserializes JSON to TODOList
func (tl *TODOList) FromJSON(data []byte) error {
	return json.Unmarshal(data, tl)
}

// MarkComplete marks a TODO as completed by ID
func (tl *TODOList) MarkComplete(id string) {
	for i := range tl.TODOs {
		if tl.TODOs[i].ID == id {
			tl.TODOs[i].Completed = true
			tl.LastUpdated = time.Now()
			tl.ToMarkdown()
			return
		}
	}
}

// AddFeedback adds feedback to a TODO by ID
func (tl *TODOList) AddFeedback(id, feedback string) {
	for i := range tl.TODOs {
		if tl.TODOs[i].ID == id {
			tl.TODOs[i].Feedback = feedback
			tl.LastUpdated = time.Now()
			tl.ToMarkdown()
			return
		}
	}
}

// GetIncompleteTODOs returns all incomplete TODOs
func (tl *TODOList) GetIncompleteTODOs() []TODO {
	var incomplete []TODO
	for _, todo := range tl.TODOs {
		if !todo.Completed {
			incomplete = append(incomplete, todo)
		}
	}
	return incomplete
}

// GetCompletedTODOs returns all completed TODOs
func (tl *TODOList) GetCompletedTODOs() []TODO {
	var completed []TODO
	for _, todo := range tl.TODOs {
		if todo.Completed {
			completed = append(completed, todo)
		}
	}
	return completed
}
