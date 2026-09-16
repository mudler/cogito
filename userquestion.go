package cogito

import (
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"
)

// UserQuestionToolName is the name of the built-in tool that WithUserQuestions
// injects into the tool set.
const UserQuestionToolName = "ask_user"

var (
	// ErrQuestionNotFound is returned by QuestionRegistry.Answer for an id that
	// is unknown or was already answered.
	ErrQuestionNotFound = errors.New("cogito: question not found or already answered")
	// ErrInvalidAnswer is returned (wrapped) when an answer does not fit the
	// question: empty, a label outside Options, or free text where none is
	// allowed.
	ErrInvalidAnswer = errors.New("cogito: invalid answer")
	// ErrQuestionCancelled is returned by a UserQuestionHandler whose context
	// ended before the user answered.
	ErrQuestionCancelled = errors.New("cogito: question cancelled before it was answered")
)

// UserQuestion is a structured question the model asks through the built-in
// ask_user tool. ID and AskedAt are set by cogito before the handler runs;
// AgentID names the sub-agent that asked, or is empty for the root agent.
type UserQuestion struct {
	ID            string    `json:"id"`
	AgentID       string    `json:"agent_id"`
	Question      string    `json:"question"`
	Options       []string  `json:"options"`
	AllowFreeText bool      `json:"allow_free_text"`
	AskedAt       time.Time `json:"asked_at"`
}

// UserAnswer is the user's reply to a UserQuestion: labels picked from
// Options (several are allowed), free text, or both.
type UserAnswer struct {
	Selected []string `json:"selected"`
	Text     string   `json:"text"`
}

// String renders the answer as the text the model receives as the tool result.
func (a UserAnswer) String() string {
	var b strings.Builder
	if len(a.Selected) > 0 {
		b.WriteString("Selected: ")
		b.WriteString(strings.Join(a.Selected, ", "))
	}
	if strings.TrimSpace(a.Text) != "" {
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString("Answer: ")
		b.WriteString(a.Text)
	}
	return b.String()
}

// Validate reports whether a is an acceptable answer to q. QuestionRegistry
// calls it from Answer; embedders can call it first to turn a bad answer into
// a validation error before touching the registry.
func (q UserQuestion) Validate(a UserAnswer) error {
	hasText := strings.TrimSpace(a.Text) != ""
	if len(a.Selected) == 0 && !hasText {
		return fmt.Errorf("%w: empty answer", ErrInvalidAnswer)
	}
	if len(a.Selected) > 0 && len(q.Options) == 0 {
		return fmt.Errorf("%w: the question offers no options to select from", ErrInvalidAnswer)
	}
	for _, s := range a.Selected {
		if !slices.Contains(q.Options, s) {
			return fmt.Errorf("%w: %q is not one of the options", ErrInvalidAnswer, s)
		}
	}
	if hasText && len(q.Options) > 0 && !q.AllowFreeText {
		return fmt.Errorf("%w: free text is not allowed for this question", ErrInvalidAnswer)
	}
	return nil
}
