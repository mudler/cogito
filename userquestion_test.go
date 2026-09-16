package cogito

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sashabaranov/go-openai"
)

func TestUserAnswerString(t *testing.T) {
	cases := []struct {
		name string
		in   UserAnswer
		want string
	}{
		{"selected only", UserAnswer{Selected: []string{"print", "validate"}}, "Selected: print, validate"},
		{"text only", UserAnswer{Text: "both, please"}, "Answer: both, please"},
		{"selected and text", UserAnswer{Selected: []string{"print"}, Text: "and log"}, "Selected: print\nAnswer: and log"},
		{"empty", UserAnswer{}, ""},
		{"blank text is empty", UserAnswer{Text: "   "}, ""},
	}
	for _, c := range cases {
		if got := c.in.String(); got != c.want {
			t.Errorf("%s: got %q, want %q", c.name, got, c.want)
		}
	}
}

func TestUserQuestionValidate(t *testing.T) {
	withOptions := UserQuestion{Question: "Print or validate?", Options: []string{"print", "validate"}}
	freeToo := UserQuestion{Question: "Print or validate?", Options: []string{"print", "validate"}, AllowFreeText: true}
	noOptions := UserQuestion{Question: "What is the flag name?", AllowFreeText: true}

	cases := []struct {
		name    string
		q       UserQuestion
		a       UserAnswer
		wantErr error // nil means valid
	}{
		{"one option", withOptions, UserAnswer{Selected: []string{"print"}}, nil},
		{"several options", withOptions, UserAnswer{Selected: []string{"print", "validate"}}, nil},
		{"empty answer", withOptions, UserAnswer{}, ErrInvalidAnswer},
		{"blank text only", withOptions, UserAnswer{Text: " "}, ErrInvalidAnswer},
		{"unknown label", withOptions, UserAnswer{Selected: []string{"delete"}}, ErrInvalidAnswer},
		{"free text not allowed", withOptions, UserAnswer{Text: "both"}, ErrInvalidAnswer},
		{"free text allowed", freeToo, UserAnswer{Text: "both"}, nil},
		{"option plus free text", freeToo, UserAnswer{Selected: []string{"print"}, Text: "and log"}, nil},
		{"text for a question without options", noOptions, UserAnswer{Text: "--dry-run"}, nil},
		{"selection for a question without options", noOptions, UserAnswer{Selected: []string{"x"}}, ErrInvalidAnswer},
	}
	for _, c := range cases {
		err := c.q.Validate(c.a)
		if c.wantErr == nil && err != nil {
			t.Errorf("%s: unexpected error %v", c.name, err)
		}
		if c.wantErr != nil && !errors.Is(err, c.wantErr) {
			t.Errorf("%s: got %v, want %v", c.name, err, c.wantErr)
		}
	}
}

type handleResult struct {
	a   UserAnswer
	err error
}

// handleAsync runs reg.Handle in a goroutine and returns the channel its
// result lands on.
func handleAsync(t *testing.T, reg *QuestionRegistry, ctx context.Context, q UserQuestion) <-chan handleResult {
	t.Helper()
	out := make(chan handleResult, 1)
	go func() {
		a, err := reg.Handle(ctx, q)
		out <- handleResult{a, err}
	}()
	return out
}

// waitNotified returns the next notification or fails the test after 2s.
func waitNotified(t *testing.T, notified <-chan UserQuestion) UserQuestion {
	t.Helper()
	select {
	case q := <-notified:
		return q
	case <-time.After(2 * time.Second):
		t.Fatal("registry never notified")
		return UserQuestion{}
	}
}

// waitPending polls until reg has n pending questions or 2s pass.
func waitPending(t *testing.T, reg *QuestionRegistry, n int) {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for len(reg.Pending()) < n && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if got := len(reg.Pending()); got < n {
		t.Fatalf("pending = %d, want at least %d", got, n)
	}
}

func TestQuestionRegistryAnswerUnknownID(t *testing.T) {
	reg := NewQuestionRegistry(nil)
	if err := reg.Answer("nope", UserAnswer{Text: "x"}); !errors.Is(err, ErrQuestionNotFound) {
		t.Fatalf("got %v, want ErrQuestionNotFound", err)
	}
}

func TestQuestionRegistryAnswerFlow(t *testing.T) {
	notified := make(chan UserQuestion, 1)
	reg := NewQuestionRegistry(func(q UserQuestion) { notified <- q })
	q := UserQuestion{ID: "q1", Question: "Print or validate?", Options: []string{"print", "validate"}, AskedAt: time.Now()}

	res := handleAsync(t, reg, context.Background(), q)
	if got := waitNotified(t, notified); got.ID != "q1" {
		t.Fatalf("notified %+v, want q1", got)
	}
	if p := reg.Pending(); len(p) != 1 || p[0].ID != "q1" {
		t.Fatalf("pending = %+v, want [q1]", p)
	}

	// Bad answers are rejected and keep the question pending.
	for _, bad := range []UserAnswer{{}, {Selected: []string{"delete"}}, {Text: "both"}} {
		if err := reg.Answer("q1", bad); !errors.Is(err, ErrInvalidAnswer) {
			t.Fatalf("Answer(%+v) = %v, want ErrInvalidAnswer", bad, err)
		}
	}
	if len(reg.Pending()) != 1 {
		t.Fatal("a rejected answer must keep the question pending")
	}

	if err := reg.Answer("q1", UserAnswer{Selected: []string{"print"}}); err != nil {
		t.Fatalf("valid answer rejected: %v", err)
	}
	select {
	case r := <-res:
		if r.err != nil || len(r.a.Selected) != 1 || r.a.Selected[0] != "print" {
			t.Fatalf("Handle returned %+v, %v", r.a, r.err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Handle did not return after Answer")
	}
	if len(reg.Pending()) != 0 {
		t.Fatal("answered question still pending")
	}
	if err := reg.Answer("q1", UserAnswer{Selected: []string{"print"}}); !errors.Is(err, ErrQuestionNotFound) {
		t.Fatalf("second answer = %v, want ErrQuestionNotFound", err)
	}
}

func TestQuestionRegistryPendingIsOldestFirst(t *testing.T) {
	reg := NewQuestionRegistry(nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	base := time.Now()
	handleAsync(t, reg, ctx, UserQuestion{ID: "later", Question: "b", AskedAt: base.Add(time.Second)})
	handleAsync(t, reg, ctx, UserQuestion{ID: "earlier", Question: "a", AskedAt: base})
	waitPending(t, reg, 2)
	p := reg.Pending()
	if p[0].ID != "earlier" || p[1].ID != "later" {
		t.Fatalf("pending order = %+v", p)
	}
}

func TestQuestionRegistryHandleCancelledClearsPending(t *testing.T) {
	var notifications atomic.Int32
	reg := NewQuestionRegistry(func(UserQuestion) { notifications.Add(1) })
	ctx, cancel := context.WithCancel(context.Background())
	res := handleAsync(t, reg, ctx, UserQuestion{ID: "q1", Question: "?", AskedAt: time.Now()})
	waitPending(t, reg, 1)
	cancel()
	select {
	case r := <-res:
		if !errors.Is(r.err, ErrQuestionCancelled) {
			t.Fatalf("Handle returned %v, want ErrQuestionCancelled", r.err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Handle did not return after cancel")
	}
	if len(reg.Pending()) != 0 {
		t.Fatal("cancelled question still pending")
	}
	if n := notifications.Load(); n != 1 {
		t.Fatalf("notified %d times, want 1", n)
	}
	if err := reg.Answer("q1", UserAnswer{Text: "late"}); !errors.Is(err, ErrQuestionNotFound) {
		t.Fatalf("late answer = %v, want ErrQuestionNotFound", err)
	}
}

func TestQuestionRegistryAnswerDeliveredBeforeCancelWins(t *testing.T) {
	reg := NewQuestionRegistry(nil)
	ctx, cancel := context.WithCancel(context.Background())
	res := handleAsync(t, reg, ctx, UserQuestion{ID: "q1", Question: "?", AskedAt: time.Now()})
	waitPending(t, reg, 1)
	if err := reg.Answer("q1", UserAnswer{Text: "yes"}); err != nil {
		t.Fatal(err)
	}
	cancel()
	select {
	case r := <-res:
		if r.err != nil || r.a.Text != "yes" {
			t.Fatalf("Handle returned %+v, %v; want the answer", r.a, r.err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Handle did not return")
	}
}

func TestQuestionRegistryOwnsQuestionOptions(t *testing.T) {
	notified := make(chan UserQuestion, 1)
	reg := NewQuestionRegistry(func(q UserQuestion) { notified <- q })
	options := []string{"print", "validate"}
	res := handleAsync(t, reg, context.Background(), UserQuestion{
		ID: "q1", Question: "Print or validate?", Options: options, AskedAt: time.Now(),
	})

	shown := waitNotified(t, notified)
	options[0] = "caller mutation"
	shown.Options[1] = "notifier mutation"
	pending := reg.Pending()
	if got := pending[0].Options; got[0] != "print" || got[1] != "validate" {
		t.Fatalf("pending options = %v, want an owned copy", got)
	}
	pending[0].Options[0] = "pending mutation"
	if got := reg.Pending()[0].Options[0]; got != "print" {
		t.Fatalf("pending options[0] = %q after caller mutation, want print", got)
	}

	if err := reg.Answer("q1", UserAnswer{Selected: []string{"print"}}); err != nil {
		t.Fatalf("answer using original option: %v", err)
	}
	if r := <-res; r.err != nil {
		t.Fatalf("Handle returned error: %v", r.err)
	}
}

func TestQuestionRegistryOwnsAnswerSelection(t *testing.T) {
	reg := NewQuestionRegistry(nil)
	res := handleAsync(t, reg, context.Background(), UserQuestion{
		ID: "q1", Question: "Print or validate?", Options: []string{"print"}, AskedAt: time.Now(),
	})
	waitPending(t, reg, 1)
	answer := UserAnswer{Selected: []string{"print"}}
	if err := reg.Answer("q1", answer); err != nil {
		t.Fatal(err)
	}
	answer.Selected[0] = "caller mutation"
	if r := <-res; r.err != nil || len(r.a.Selected) != 1 || r.a.Selected[0] != "print" {
		t.Fatalf("Handle returned %+v, %v; want owned answer selection", r.a, r.err)
	}
}

func TestQuestionRegistryRejectsDuplicatePendingID(t *testing.T) {
	reg := NewQuestionRegistry(nil)
	first := handleAsync(t, reg, context.Background(), UserQuestion{ID: "same", Question: "first", AskedAt: time.Now()})
	waitPending(t, reg, 1)

	_, err := reg.Handle(context.Background(), UserQuestion{ID: "same", Question: "second", AskedAt: time.Now()})
	if err == nil || !strings.Contains(err.Error(), "already pending") {
		t.Fatalf("duplicate Handle error = %v, want descriptive already-pending error", err)
	}
	if p := reg.Pending(); len(p) != 1 || p[0].Question != "first" {
		t.Fatalf("duplicate replaced pending question: %+v", p)
	}
	if err := reg.Answer("same", UserAnswer{Text: "answer"}); err != nil {
		t.Fatal(err)
	}
	if r := <-first; r.err != nil || r.a.Text != "answer" {
		t.Fatalf("first Handle returned %+v, %v", r.a, r.err)
	}
}

func TestQuestionRegistryAlreadyCancelledDoesNotPublish(t *testing.T) {
	var notifications atomic.Int32
	reg := NewQuestionRegistry(func(UserQuestion) { notifications.Add(1) })
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := reg.Handle(ctx, UserQuestion{ID: "q1", Question: "?", AskedAt: time.Now()})
	if !errors.Is(err, ErrQuestionCancelled) {
		t.Fatalf("Handle error = %v, want ErrQuestionCancelled", err)
	}
	if n := notifications.Load(); n != 0 {
		t.Fatalf("notifications = %d, want 0", n)
	}
	if p := reg.Pending(); len(p) != 0 {
		t.Fatalf("pending = %+v, want none", p)
	}
}

func TestQuestionRegistryNotifierCanInspectAndAnswer(t *testing.T) {
	var reg *QuestionRegistry
	reg = NewQuestionRegistry(func(q UserQuestion) {
		if p := reg.Pending(); len(p) != 1 || p[0].ID != q.ID {
			t.Errorf("pending during notification = %+v", p)
		}
		if err := reg.Answer(q.ID, UserAnswer{Text: "yes"}); err != nil {
			t.Errorf("Answer during notification: %v", err)
		}
	})

	res := handleAsync(t, reg, context.Background(), UserQuestion{ID: "q1", Question: "?", AskedAt: time.Now()})
	select {
	case r := <-res:
		if r.err != nil || r.a.Text != "yes" {
			t.Fatalf("Handle returned %+v, %v", r.a, r.err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("notifier could not inspect and answer without deadlock")
	}
}

// neverHandler is for tests where the model must not ask: it blocks until ctx
// ends so an unexpected question fails loudly by timeout.
func neverHandler(ctx context.Context, _ UserQuestion) (UserAnswer, error) {
	<-ctx.Done()
	return UserAnswer{}, ErrQuestionCancelled
}

func userFragment(text string) Fragment {
	return NewEmptyFragment().AddMessage(UserMessageRole, text)
}

func TestAskUserAnswerBecomesToolResultWithoutExtraModelCall(t *testing.T) {
	llm := newSequenceLLM(toolTurn(UserQuestionToolName, `{"question":"Print or validate?","options":["print","validate"]}`))
	asked := make(chan UserQuestion, 1)
	reg := NewQuestionRegistry(func(q UserQuestion) { asked <- q })
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	t.Cleanup(cancel)

	type out struct {
		f   Fragment
		err error
	}
	done := make(chan out, 1)
	go func() {
		f, err := ExecuteTools(llm, userFragment("add a dry-run flag"), WithContext(ctx), WithUserQuestions(reg.Handle))
		done <- out{f, err}
	}()

	var q UserQuestion
	select {
	case q = <-asked:
	case <-time.After(2 * time.Second):
		t.Fatal("handler was never invoked")
	}
	if q.ID == "" || q.AskedAt.IsZero() {
		t.Fatalf("question not stamped: %+v", q)
	}
	if q.Question != "Print or validate?" || len(q.Options) != 2 || q.AllowFreeText {
		t.Fatalf("unexpected question %+v", q)
	}
	if q.AgentID != "" {
		t.Fatalf("root agent must ask with an empty AgentID, got %q", q.AgentID)
	}
	if p := reg.Pending(); len(p) != 1 || p[0].ID != q.ID {
		t.Fatalf("pending = %+v", p)
	}
	select {
	case <-done:
		t.Fatal("ExecuteTools returned before the question was answered")
	case <-time.After(100 * time.Millisecond):
	}
	if got := llm.completions(); got != 1 {
		t.Fatalf("model calls while waiting = %d, want 1", got)
	}

	if err := reg.Answer(q.ID, UserAnswer{Selected: []string{"print"}}); err != nil {
		t.Fatal(err)
	}
	var r out
	select {
	case r = <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("ExecuteTools did not resume after the answer")
	}
	if r.err != nil {
		t.Fatal(r.err)
	}

	var toolMsgs []openai.ChatCompletionMessage
	for _, m := range r.f.Messages {
		if m.Role == ToolMessageRole.String() {
			toolMsgs = append(toolMsgs, m)
		}
	}
	if len(toolMsgs) != 1 || toolMsgs[0].Content != "Selected: print" {
		t.Fatalf("tool messages = %+v, want exactly one with the rendered answer", toolMsgs)
	}
	// One selection call before the question, one final Ask after it, nothing
	// in between: the answer entered the conversation as the tool result.
	if got := llm.completions(); got != 1 {
		t.Fatalf("selection calls = %d, want 1", got)
	}
	if got := llm.askCount(); got != 1 {
		t.Fatalf("final asks = %d, want 1", got)
	}
	if len(reg.Pending()) != 0 {
		t.Fatal("registry still holds the answered question")
	}
	if len(r.f.Status.ToolResults) != 1 {
		t.Fatalf("tool results = %d, want 1", len(r.f.Status.ToolResults))
	}
	if _, ok := r.f.Status.ToolResults[0].ResultData.(UserAnswer); !ok {
		t.Fatalf("ResultData should carry the UserAnswer, got %T", r.f.Status.ToolResults[0].ResultData)
	}
}

func TestAskUserCancelledContextAsksOnceAndReturnsContextError(t *testing.T) {
	llm := newSequenceLLM(toolTurn(UserQuestionToolName, `{"question":"?"}`))
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	t.Cleanup(cancel)
	var calls atomic.Int32
	handler := func(ctx context.Context, q UserQuestion) (UserAnswer, error) {
		calls.Add(1)
		<-ctx.Done()
		return UserAnswer{}, ErrQuestionCancelled
	}
	done := make(chan error, 1)
	go func() {
		// WithMaxAttempts(3): a tool that returned an error would be retried,
		// which for ask_user means asking again. It must not.
		_, err := ExecuteTools(llm, userFragment("x"), WithContext(ctx), WithUserQuestions(handler), WithMaxAttempts(3))
		done <- err
	}()
	deadline := time.Now().Add(2 * time.Second)
	for calls.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if calls.Load() == 0 {
		t.Fatal("handler was never invoked")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("ExecuteTools returned %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("ExecuteTools did not return after cancel")
	}
	if n := calls.Load(); n != 1 {
		t.Fatalf("handler invoked %d times, want exactly 1", n)
	}
}

func TestAskUserRunnerReportsProblemsAsResultText(t *testing.T) {
	var calls atomic.Int32
	failing := func(ctx context.Context, q UserQuestion) (UserAnswer, error) {
		calls.Add(1)
		return UserAnswer{}, errors.New("transport down")
	}
	r := &askUserRunner{ctx: context.Background(), handler: failing}

	out, data, err := r.Run(AskUserArgs{Question: "   "})
	if err != nil || data != nil || !strings.HasPrefix(out, "Error:") {
		t.Fatalf("empty question: got (%q, %v, %v), want an Error result and nil error", out, data, err)
	}
	if calls.Load() != 0 {
		t.Fatal("an empty question must not reach the handler")
	}

	out, _, err = r.Run(AskUserArgs{Question: "ok?"})
	if err != nil || !strings.Contains(out, "transport down") {
		t.Fatalf("handler failure: got (%q, %v), want the failure in the result text and nil error", out, err)
	}
	if calls.Load() != 1 {
		t.Fatalf("handler invoked %d times, want 1", calls.Load())
	}

	// A question without options always accepts free text.
	var seen UserQuestion
	recording := func(ctx context.Context, q UserQuestion) (UserAnswer, error) {
		seen = q
		return UserAnswer{Text: "x"}, nil
	}
	r = &askUserRunner{ctx: context.Background(), handler: recording, agentID: "child-1"}
	if _, _, err := r.Run(AskUserArgs{Question: "name?"}); err != nil {
		t.Fatal(err)
	}
	if !seen.AllowFreeText || seen.AgentID != "child-1" {
		t.Fatalf("question = %+v, want AllowFreeText and the runner's agentID", seen)
	}
}

func TestAskUserToolIsInjectedOnlyWithTheOption(t *testing.T) {
	echo := newNamedTool("echo")
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	t.Cleanup(cancel)

	without := newSequenceLLM()
	_, _ = ExecuteTools(without, userFragment("x"), WithTools(echo))
	if contains(without.toolNames(0), UserQuestionToolName) {
		t.Fatalf("ask_user offered without WithUserQuestions: %v", without.toolNames(0))
	}

	with := newSequenceLLM()
	_, _ = ExecuteTools(with, userFragment("x"), WithContext(ctx), WithTools(echo), WithUserQuestions(neverHandler))
	if countOf(with.toolNames(0), UserQuestionToolName) != 1 {
		t.Fatalf("ask_user should be offered exactly once: %v", with.toolNames(0))
	}

	// Prefill mirrors both cases, including the ask_user schema.
	pWithout := &captureLLM{}
	if err := Prefill(context.Background(), pWithout, userFragment("x"), WithTools(echo)); err != nil {
		t.Fatal(err)
	}
	pWith := &captureLLM{}
	if err := Prefill(context.Background(), pWith, userFragment("x"), WithTools(echo), WithUserQuestions(neverHandler)); err != nil {
		t.Fatal(err)
	}
	names := func(req openai.ChatCompletionRequest) []string {
		var out []string
		for _, tl := range req.Tools {
			if tl.Function != nil {
				out = append(out, tl.Function.Name)
			}
		}
		return out
	}
	if got, want := names(pWithout.last), without.toolNames(0); !reflect.DeepEqual(got, want) {
		t.Fatalf("prefill tools without option = %v, real turn = %v", got, want)
	}
	if got, want := names(pWith.last), with.toolNames(0); !reflect.DeepEqual(got, want) {
		t.Fatalf("prefill tools with option = %v, real turn = %v", got, want)
	}
}
