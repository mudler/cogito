package cogito

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/mudler/xlog"
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

// UserQuestionHandler delivers a question to the user and blocks until an
// answer arrives or ctx ends. It runs inside the ask_user tool call, on the
// agent loop's goroutine, so the answer becomes the tool result and no extra
// model call is spent. Return ErrQuestionCancelled when ctx ends first.
type UserQuestionHandler func(ctx context.Context, q UserQuestion) (UserAnswer, error)

type pendingQuestion struct {
	q  UserQuestion
	ch chan UserAnswer // buffered(1): Answer never blocks on the handler
}

// QuestionRegistry is a ready-made UserQuestionHandler for embedders that
// surface questions asynchronously (a web UI, a chat connector). Handle parks
// the question and fires the notifier so the embedder can show it; Answer
// releases the blocked tool call; Pending lists what is still unanswered so a
// reconnecting client can re-render its cards. Share one registry per agent.
type QuestionRegistry struct {
	mu      sync.Mutex
	pending map[string]*pendingQuestion
	notify  func(UserQuestion)
}

// NewQuestionRegistry builds a registry. onQuestion, if non-nil, is called
// synchronously from Handle with every new question; keep it quick (emit an
// event, enqueue) because it runs on the agent loop's goroutine.
func NewQuestionRegistry(onQuestion func(UserQuestion)) *QuestionRegistry {
	return &QuestionRegistry{pending: map[string]*pendingQuestion{}, notify: onQuestion}
}

// Handle implements UserQuestionHandler: it registers q, notifies, and blocks
// until Answer delivers a reply or ctx ends (ErrQuestionCancelled).
func (r *QuestionRegistry) Handle(ctx context.Context, q UserQuestion) (UserAnswer, error) {
	if ctx.Err() != nil {
		return UserAnswer{}, ErrQuestionCancelled
	}

	q = cloneUserQuestion(q)
	p := &pendingQuestion{q: q, ch: make(chan UserAnswer, 1)}
	r.mu.Lock()
	if ctx.Err() != nil {
		r.mu.Unlock()
		return UserAnswer{}, ErrQuestionCancelled
	}
	if _, exists := r.pending[q.ID]; exists {
		r.mu.Unlock()
		return UserAnswer{}, fmt.Errorf("cogito: question %q is already pending", q.ID)
	}
	r.pending[q.ID] = p
	r.mu.Unlock()

	if r.notify != nil {
		r.notify(cloneUserQuestion(q))
	}

	select {
	case a := <-p.ch:
		return a, nil
	case <-ctx.Done():
		r.mu.Lock()
		if current, ok := r.pending[q.ID]; ok && current == p {
			delete(r.pending, q.ID)
		}
		r.mu.Unlock()

		// An answer that raced the cancellation is still an answer.
		select {
		case a := <-p.ch:
			return a, nil
		default:
		}
		return UserAnswer{}, ErrQuestionCancelled
	}
}

// Pending returns the unanswered questions, oldest first. The returned values
// are snapshots and may be modified by the caller.
func (r *QuestionRegistry) Pending() []UserQuestion {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]UserQuestion, 0, len(r.pending))
	for _, p := range r.pending {
		out = append(out, cloneUserQuestion(p.q))
	}
	sort.Slice(out, func(i, j int) bool { return out[i].AskedAt.Before(out[j].AskedAt) })
	return out
}

// Answer validates a against the pending question id and delivers it to the
// blocked Handle. It returns ErrQuestionNotFound for an unknown or already
// answered id and a wrapped ErrInvalidAnswer for an answer that does not fit;
// in both cases nothing changes.
func (r *QuestionRegistry) Answer(id string, a UserAnswer) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	p, ok := r.pending[id]
	if !ok {
		return ErrQuestionNotFound
	}
	if err := p.q.Validate(a); err != nil {
		return err
	}
	a.Selected = slices.Clone(a.Selected)
	delete(r.pending, id)
	p.ch <- a
	return nil
}

func cloneUserQuestion(q UserQuestion) UserQuestion {
	q.Options = slices.Clone(q.Options)
	return q
}

// withoutUserQuestions drops the handler; spawnAgentRunner uses it for
// children whose tool allow-list leaves ask_user out.
func withoutUserQuestions() Option {
	return func(o *Options) { o.userQuestionHandler = nil }
}

// AskUserArgs are the arguments of the built-in ask_user tool.
type AskUserArgs struct {
	Question      string   `json:"question" description:"The question to ask the user. One question at a time, specific and concise."`
	Options       []string `json:"options" description:"Optional short answer choices for the user to pick from. Leave empty to ask for a free-text answer."`
	AllowFreeText bool     `json:"allow_free_text" description:"When options are given, also accept a free-text answer."`
}

type askUserRunner struct {
	ctx     context.Context
	handler UserQuestionHandler
	agentID string
}

// Run blocks until the user answers. Problems are returned as result text
// with a nil error, like the other built-ins (spawnAgentRunner.Run), because
// a non-nil error makes ExecuteTools retry the tool up to maxAttempts times,
// which here would ask the same question again.
func (r *askUserRunner) Run(args AskUserArgs) (string, any, error) {
	if strings.TrimSpace(args.Question) == "" {
		return "Error: question must not be empty", nil, nil
	}
	q := UserQuestion{
		ID:            uuid.New().String(),
		AgentID:       r.agentID,
		Question:      args.Question,
		Options:       args.Options,
		AllowFreeText: args.AllowFreeText || len(args.Options) == 0,
		AskedAt:       time.Now(),
	}
	answer, err := r.handler(r.ctx, q)
	if err != nil {
		xlog.Debug("ask_user: no answer", "question_id", q.ID, "error", err)
		return fmt.Sprintf("No answer from the user: %v", err), nil, nil
	}
	return answer.String(), answer, nil
}

func newAskUserTool(o *Options) ToolDefinitionInterface {
	return NewToolDefinition(
		&askUserRunner{ctx: o.context, handler: o.userQuestionHandler, agentID: o.agentID},
		AskUserArgs{},
		UserQuestionToolName,
		"Ask the user a clarifying question and wait for the answer. Use it when the task is ambiguous and a wrong guess would waste work. Prefer a few short options; the answer comes back as the tool result.",
	)
}

// prepareUserQuestionTool returns the ask_user tool when WithUserQuestions is
// set, nil otherwise. ExecuteTools and Prefill both call it, so the tool set
// a prefill primes never drifts from the tool set a real run sends.
func prepareUserQuestionTool(o *Options) ToolDefinitionInterface {
	if o.userQuestionHandler == nil {
		return nil
	}
	return newAskUserTool(o)
}

// withoutPreparedUserQuestionTools returns a copy without ask_user tools that
// Cogito injected for an earlier ExecuteTools call. User-provided tools named
// ask_user are left alone.
func withoutPreparedUserQuestionTools(tools Tools) Tools {
	filtered := make(Tools, 0, len(tools))
	for _, tool := range tools {
		definition, ok := tool.(*ToolDefinition[AskUserArgs])
		if ok && definition.Name == UserQuestionToolName {
			if _, generated := definition.ToolRunner.(*askUserRunner); generated {
				continue
			}
		}
		filtered = append(filtered, tool)
	}
	return filtered
}

// withPreparedUserQuestionTool replaces any runner bound by an outer
// ExecuteTools call with the runner for the current context and agent.
func withPreparedUserQuestionTool(tool ToolDefinitionInterface) Option {
	return func(o *Options) {
		o.tools = withoutPreparedUserQuestionTools(o.tools)
		if tool != nil {
			o.tools = append(o.tools, tool)
		}
	}
}

// containsToolChoice reports whether any choice names the given tool.
func containsToolChoice(choices []*ToolChoice, name string) bool {
	for _, choice := range choices {
		if choice.Name == name {
			return true
		}
	}
	return false
}

// toolChoicesFirst returns choices reordered so those named name come first,
// keeping the relative order of everything else.
func toolChoicesFirst(choices []*ToolChoice, name string) []*ToolChoice {
	out := make([]*ToolChoice, 0, len(choices))
	for _, choice := range choices {
		if choice.Name == name {
			out = append(out, choice)
		}
	}
	for _, choice := range choices {
		if choice.Name != name {
			out = append(out, choice)
		}
	}
	return out
}

func isAskUserFailure(resultData any) bool {
	_, answered := resultData.(UserAnswer)
	return !answered
}
