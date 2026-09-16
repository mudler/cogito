package cogito

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mudler/cogito/structures"
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

func TestAutoPlanSubtaskOffersOneFreshAskUserTool(t *testing.T) {
	llm := newSequenceLLM(
		toolTurn("json", `{"extract_boolean":true}`),
		toolTurn("json", `{"goal":"clarify the change"}`),
		toolTurn("json", `{"description":"ask before changing","subtasks":["clarify scope"]}`),
		toolTurn(UserQuestionToolName, `{"question":"Which scope?"}`),
		toolTurn("json", `{"extract_boolean":true}`),
	)
	var asked UserQuestion
	handler := func(_ context.Context, q UserQuestion) (UserAnswer, error) {
		asked = q
		return UserAnswer{Text: "the current package"}, nil
	}

	result, err := ExecuteTools(llm, userFragment("make the change"),
		EnableAutoPlan,
		WithUserQuestions(handler),
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(llm.requests) <= 3 {
		t.Fatalf("auto-plan did not reach the subtask tool request: %d requests", len(llm.requests))
	}
	if got := countOf(llm.toolNames(3), UserQuestionToolName); got != 1 {
		t.Fatalf("planned subtask offered ask_user %d times, want 1: %v", got, llm.toolNames(3))
	}
	if asked.Question != "Which scope?" || asked.AgentID != "" {
		t.Fatalf("question = %+v, want root question handled by the current runner", asked)
	}
	if got := countOf(result.Status.ToolsCalled.Names(), UserQuestionToolName); got != 1 {
		t.Fatalf("executed ask_user %d times, want 1", got)
	}
}

func TestTODOPlanWorkPhasePreservesQuestionHandlerAndAgentID(t *testing.T) {
	worker := newSequenceLLM(
		toolTurn(UserQuestionToolName, `{"question":"Proceed?"}`),
		toolTurn("json", `{"todos":[{"id":"1","description":"clarify scope","completed":true}]}`),
	)
	reviewer := newSequenceLLM(toolTurn("json", `{"extract_boolean":true}`))
	var asked UserQuestion
	handler := func(_ context.Context, q UserQuestion) (UserAnswer, error) {
		asked = q
		return UserAnswer{Text: "yes"}, nil
	}
	todos := &structures.TODOList{TODOs: []structures.TODO{{
		ID: "1", Description: "clarify scope",
	}}}

	_, err := ExecutePlan(
		worker,
		userFragment("make the change"),
		&structures.Plan{Description: "clarify first", Subtasks: []string{"clarify scope"}},
		&structures.Goal{Goal: "make the change"},
		WithReviewerLLM(reviewer),
		WithTODOs(todos),
		WithUserQuestions(handler),
		withAgentIDStamp("planned-child"),
	)
	if err != nil {
		t.Fatal(err)
	}
	if got := countOf(worker.toolNames(0), UserQuestionToolName); got != 1 {
		t.Fatalf("TODO work phase offered ask_user %d times, want 1: %v", got, worker.toolNames(0))
	}
	if asked.Question != "Proceed?" || asked.AgentID != "planned-child" {
		t.Fatalf("question = %+v, want handler and agent id preserved in TODO work phase", asked)
	}
}

func TestSubAgentQuestionCarriesAgentID(t *testing.T) {
	asked := make(chan UserQuestion, 1)
	handler := func(ctx context.Context, q UserQuestion) (UserAnswer, error) {
		asked <- q
		return UserAnswer{Text: "yes"}, nil
	}
	runner := &spawnAgentRunner{
		llm:         newSequenceLLM(toolTurn(UserQuestionToolName, `{"question":"ok?"}`)),
		parentTools: Tools{newNamedTool("echo")},
		parentOpts:  []Option{WithUserQuestions(handler)},
		manager:     NewAgentManager(),
		ctx:         context.Background(),
	}
	if _, _, err := runner.Run(SpawnAgentArgs{Task: "ask", Background: false}); err != nil {
		t.Fatal(err)
	}
	select {
	case q := <-asked:
		if q.AgentID == "" {
			t.Fatal("a sub-agent's question must carry its agent id")
		}
	default:
		t.Fatal("the sub-agent never asked")
	}
}

func TestSubAgentAllowListControlsAskUser(t *testing.T) {
	offered := func(tools []string) []string {
		var got []string
		runner := &spawnAgentRunner{
			llm:         newInspectingLLM(func(_ Fragment, names []string) { got = names }),
			parentTools: Tools{newNamedTool("echo")},
			parentOpts:  []Option{WithUserQuestions(neverHandler)},
			manager:     NewAgentManager(),
			ctx:         context.Background(),
		}
		if _, _, err := runner.Run(SpawnAgentArgs{Task: "t", Background: false, Tools: tools}); err != nil {
			t.Fatal(err)
		}
		return got
	}
	if got := offered(nil); countOf(got, UserQuestionToolName) != 1 {
		t.Fatalf("no allow-list: ask_user should be inherited once, got %v", got)
	}
	if got := offered([]string{"echo"}); contains(got, UserQuestionToolName) {
		t.Fatalf("allow-list without ask_user must drop it, got %v", got)
	}
	if got := offered([]string{"echo", UserQuestionToolName}); countOf(got, UserQuestionToolName) != 1 {
		t.Fatalf("allow-list naming ask_user should offer it once, got %v", got)
	}
}

func TestSubAgentDefinitionAllowListControlsAskUser(t *testing.T) {
	offered := func(tools []string) []string {
		var got []string
		runner := &spawnAgentRunner{
			llm:              newInspectingLLM(func(_ Fragment, names []string) { got = names }),
			parentTools:      Tools{newNamedTool("echo")},
			parentOpts:       []Option{WithUserQuestions(neverHandler)},
			manager:          NewAgentManager(),
			ctx:              context.Background(),
			agentDefinitions: []AgentDefinition{{Name: "worker", Tools: tools}},
		}
		if _, _, err := runner.Run(SpawnAgentArgs{Task: "t", AgentType: "worker"}); err != nil {
			t.Fatal(err)
		}
		return got
	}
	if got := offered([]string{"echo"}); contains(got, UserQuestionToolName) {
		t.Fatalf("definition without ask_user must drop it, got %v", got)
	}
	if got := offered([]string{"echo", UserQuestionToolName}); countOf(got, UserQuestionToolName) != 1 {
		t.Fatalf("definition naming ask_user should offer it once, got %v", got)
	}
}

func TestParentAndChildEachOfferAskUserOnce(t *testing.T) {
	// The parent spawns a foreground child on the same (scripted) model:
	// request 0 is the parent's selection, request 1 the child's.
	llm := newSequenceLLM(
		toolTurn("spawn_agent", `{"task":"look","background":false}`),
		replyTurn("child done"),
	)
	_, err := ExecuteTools(llm, userFragment("delegate"),
		WithTools(newNamedTool("echo")),
		EnableAgentSpawning,
		WithUserQuestions(neverHandler),
	)
	if err != nil {
		t.Fatal(err)
	}
	if got := llm.completions(); got < 2 {
		t.Fatalf("expected the parent and the child to each select once, got %d requests", got)
	}
	if n := countOf(llm.toolNames(0), UserQuestionToolName); n != 1 {
		t.Fatalf("parent offered ask_user %d times, want 1: %v", n, llm.toolNames(0))
	}
	if n := countOf(llm.toolNames(1), UserQuestionToolName); n != 1 {
		t.Fatalf("child offered ask_user %d times, want 1: %v", n, llm.toolNames(1))
	}
}

func TestSubAgentQuestionsWithDispatcherAndFallback(t *testing.T) {
	t.Run("dispatcher handles run without local ask_user", func(t *testing.T) {
		var spec AgentRunSpec
		var handlerCalls atomic.Int32
		runner := &spawnAgentRunner{
			llm:         newSequenceLLM(),
			parentTools: Tools{newNamedTool("echo")},
			parentOpts: []Option{WithUserQuestions(func(context.Context, UserQuestion) (UserAnswer, error) {
				handlerCalls.Add(1)
				return UserAnswer{Text: "yes"}, nil
			})},
			manager: NewAgentManager(),
			ctx:     context.Background(),
			dispatcher: func(_ context.Context, got AgentRunSpec) (Fragment, error) {
				spec = got
				return NewFragment(openai.ChatCompletionMessage{Role: "assistant", Content: "remote done"}), nil
			},
		}
		if _, _, err := runner.Run(SpawnAgentArgs{Task: "remote", Background: false}); err != nil {
			t.Fatal(err)
		}
		if contains(spec.Tools, UserQuestionToolName) {
			t.Fatalf("dispatcher received process-local ask_user: %v", spec.Tools)
		}
		if got := handlerCalls.Load(); got != 0 {
			t.Fatalf("local question handler called %d times for dispatched run", got)
		}
	})

	t.Run("fallback asks locally with the child id", func(t *testing.T) {
		asked := make(chan UserQuestion, 1)
		runner := &spawnAgentRunner{
			llm:         newSequenceLLM(toolTurn(UserQuestionToolName, `{"question":"fallback?"}`)),
			parentTools: Tools{newNamedTool("echo")},
			parentOpts: []Option{WithUserQuestions(func(_ context.Context, q UserQuestion) (UserAnswer, error) {
				asked <- q
				return UserAnswer{Text: "yes"}, nil
			})},
			manager: NewAgentManager(),
			ctx:     context.Background(),
			dispatcher: func(context.Context, AgentRunSpec) (Fragment, error) {
				return Fragment{}, ErrDispatchFallback
			},
		}
		if _, _, err := runner.Run(SpawnAgentArgs{Task: "fallback", Background: false}); err != nil {
			t.Fatal(err)
		}
		select {
		case q := <-asked:
			if q.AgentID == "" {
				t.Fatal("fallback question must carry the child id")
			}
		default:
			t.Fatal("fallback did not run the local question handler")
		}
	})
}

func TestCompletedSubAgentResumeRetainsQuestionHandlerAndID(t *testing.T) {
	manager := NewAgentManager()
	asked := make(chan UserQuestion, 1)
	childHandler := func(_ context.Context, q UserQuestion) (UserAnswer, error) {
		asked <- q
		return UserAnswer{Text: "yes"}, nil
	}
	spawner := &spawnAgentRunner{
		llm:         newSequenceLLM(replyTurn("initial done")),
		parentTools: Tools{newNamedTool("echo")},
		parentOpts:  []Option{WithUserQuestions(childHandler)},
		manager:     manager,
		ctx:         context.Background(),
	}
	if _, _, err := spawner.Run(SpawnAgentArgs{Task: "initial", Background: false}); err != nil {
		t.Fatal(err)
	}
	agents := manager.List()
	if len(agents) != 1 {
		t.Fatalf("spawned agents = %d, want 1", len(agents))
	}
	agent := agents[0]
	var parentHandlerCalls atomic.Int32
	resumer := &sendAgentMessageRunner{
		manager: manager,
		ctx:     context.Background(),
		llm:     newSequenceLLM(toolTurn(UserQuestionToolName, `{"question":"resume?"}`)),
		subOpts: []Option{WithUserQuestions(func(context.Context, UserQuestion) (UserAnswer, error) {
			parentHandlerCalls.Add(1)
			return UserAnswer{Text: "wrong handler"}, nil
		})},
	}
	if _, _, err := resumer.Run(SendAgentMessageArgs{AgentID: agent.ID, Message: "continue"}); err != nil {
		t.Fatal(err)
	}
	select {
	case q := <-asked:
		if q.AgentID != agent.ID {
			t.Fatalf("resumed question AgentID = %q, want %q", q.AgentID, agent.ID)
		}
	default:
		t.Fatal("resumed child did not use its stored question handler")
	}
	if got := parentHandlerCalls.Load(); got != 0 {
		t.Fatalf("resume used the current parent handler %d times", got)
	}
}

func TestCompletedSubAgentResumeRetainsExcludedQuestionPermission(t *testing.T) {
	manager := NewAgentManager()
	spawner := &spawnAgentRunner{
		llm:         newSequenceLLM(replyTurn("initial done")),
		parentTools: Tools{newNamedTool("echo")},
		parentOpts:  []Option{WithUserQuestions(neverHandler)},
		manager:     manager,
		ctx:         context.Background(),
	}
	if _, _, err := spawner.Run(SpawnAgentArgs{Task: "initial", Tools: []string{"echo"}}); err != nil {
		t.Fatal(err)
	}
	agents := manager.List()
	if len(agents) != 1 {
		t.Fatalf("spawned agents = %d, want 1", len(agents))
	}
	var offered []string
	resumer := &sendAgentMessageRunner{
		manager: manager,
		ctx:     context.Background(),
		llm:     newInspectingLLM(func(_ Fragment, names []string) { offered = names }),
		subOpts: []Option{WithUserQuestions(neverHandler)},
	}
	if _, _, err := resumer.Run(SendAgentMessageArgs{AgentID: agents[0].ID, Message: "continue"}); err != nil {
		t.Fatal(err)
	}
	if contains(offered, UserQuestionToolName) {
		t.Fatalf("resume restored ask_user excluded by the child's allow-list: %v", offered)
	}
}

// recordingRunner records when echo runs and exposes that event to tests.
type recordingRunner struct {
	record  func(string)
	started chan<- struct{}
}

func (r recordingRunner) Run(EchoArgs) (string, any, error) {
	if r.record != nil {
		r.record("echo")
	}
	if r.started != nil {
		close(r.started)
	}
	return "ok", nil, nil
}

type executeToolsResult struct {
	fragment Fragment
	err      error
}

func executeToolsAsync(llm LLM, f Fragment, opts ...Option) <-chan executeToolsResult {
	done := make(chan executeToolsResult, 1)
	go func() {
		result, err := ExecuteTools(llm, f, opts...)
		done <- executeToolsResult{fragment: result, err: err}
	}()
	return done
}

func waitForSignal(t *testing.T, ch <-chan struct{}, description string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for %s", description)
	}
}

func waitForExecuteTools(t *testing.T, done <-chan executeToolsResult) executeToolsResult {
	t.Helper()
	select {
	case result := <-done:
		return result
	case <-time.After(2 * time.Second):
		t.Fatal("ExecuteTools did not return")
		return executeToolsResult{}
	}
}

func TestAskUserRunsFirstAndAloneInAParallelBatch(t *testing.T) {
	var mu sync.Mutex
	var sequence []string
	record := func(s string) {
		mu.Lock()
		sequence = append(sequence, s)
		mu.Unlock()
	}
	asked := make(chan struct{})
	answer := make(chan struct{})
	echoStarted := make(chan struct{})
	handler := func(context.Context, UserQuestion) (UserAnswer, error) {
		record("asked")
		close(asked)
		<-answer
		record("answered")
		return UserAnswer{Text: "go"}, nil
	}
	echo := NewToolDefinition[EchoArgs](recordingRunner{record: record, started: echoStarted}, EchoArgs{}, "echo", "echo")
	llm := newSequenceLLM(toolsTurn(
		toolCall{"echo", `{"text":"x"}`},
		toolCall{UserQuestionToolName, `{"question":"go?"}`},
	))
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	t.Cleanup(cancel)
	done := executeToolsAsync(llm, userFragment("do both"),
		WithContext(ctx),
		WithTools(echo),
		EnableParallelToolExecution,
		WithUserQuestions(handler),
	)

	waitForSignal(t, asked, "question handler")
	select {
	case <-echoStarted:
		t.Fatal("echo ran before the question was answered")
	case <-time.After(100 * time.Millisecond):
	}
	close(answer)
	result := waitForExecuteTools(t, done)
	if result.err != nil {
		t.Fatal(result.err)
	}
	mu.Lock()
	defer mu.Unlock()
	want := []string{"asked", "answered", "echo"}
	if !reflect.DeepEqual(sequence, want) {
		t.Fatalf("execution order = %v, want %v", sequence, want)
	}
}

type rendezvousRunner struct {
	started    chan<- struct{}
	peer       <-chan struct{}
	overlapped *atomic.Bool
}

func (r rendezvousRunner) Run(EchoArgs) (string, any, error) {
	close(r.started)
	select {
	case <-r.peer:
		r.overlapped.Store(true)
		return "ok", nil, nil
	case <-time.After(500 * time.Millisecond):
		return "timed out waiting for parallel sibling", nil, errors.New("parallel sibling did not start")
	}
}

func TestAskUserCustomToolKeepsParallelExecutionWithoutQuestionHandler(t *testing.T) {
	askStarted := make(chan struct{})
	echoStarted := make(chan struct{})
	var askOverlapped atomic.Bool
	var echoOverlapped atomic.Bool
	customAsk := NewToolDefinition[EchoArgs](rendezvousRunner{
		started: askStarted, peer: echoStarted, overlapped: &askOverlapped,
	}, EchoArgs{}, UserQuestionToolName, "custom ask")
	echo := NewToolDefinition[EchoArgs](rendezvousRunner{
		started: echoStarted, peer: askStarted, overlapped: &echoOverlapped,
	}, EchoArgs{}, "echo", "echo")
	llm := newSequenceLLM(toolsTurn(
		toolCall{UserQuestionToolName, `{"text":"x"}`},
		toolCall{"echo", `{"text":"x"}`},
	))

	_, err := ExecuteTools(llm, userFragment("do both"),
		WithTools(customAsk, echo),
		EnableParallelToolExecution,
		WithMaxAttempts(1),
	)
	if err != nil {
		t.Fatal(err)
	}
	if !askOverlapped.Load() || !echoOverlapped.Load() {
		t.Fatalf("custom ask_user and echo did not overlap: ask=%v echo=%v", askOverlapped.Load(), echoOverlapped.Load())
	}
}

func TestAskUserFailureSkipsParallelSiblingsAndPreservesToolResults(t *testing.T) {
	var handlerCalls atomic.Int32
	var echoCalls atomic.Int32
	handler := func(context.Context, UserQuestion) (UserAnswer, error) {
		handlerCalls.Add(1)
		return UserAnswer{}, errors.New("question transport failed")
	}
	echo := NewToolDefinition[EchoArgs](recordingRunner{record: func(string) { echoCalls.Add(1) }}, EchoArgs{}, "echo", "echo")
	llm := newSequenceLLM(toolsTurn(
		toolCall{"echo", `{"text":"x"}`},
		toolCall{UserQuestionToolName, `{"question":"go?"}`},
	))

	f, err := ExecuteTools(llm, userFragment("do both"),
		WithTools(echo),
		EnableParallelToolExecution,
		WithUserQuestions(handler),
		WithMaxAttempts(3),
	)
	if err != nil {
		t.Fatal(err)
	}
	assertFailedQuestionBatch(t, f, handlerCalls.Load(), echoCalls.Load())
}

func TestAskUserCancellationSkipsParallelSiblingsAndPreservesToolResults(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	called := make(chan struct{})
	var handlerCalls atomic.Int32
	var echoCalls atomic.Int32
	handler := func(ctx context.Context, _ UserQuestion) (UserAnswer, error) {
		handlerCalls.Add(1)
		close(called)
		<-ctx.Done()
		return UserAnswer{}, ErrQuestionCancelled
	}
	echo := NewToolDefinition[EchoArgs](recordingRunner{record: func(string) { echoCalls.Add(1) }}, EchoArgs{}, "echo", "echo")
	llm := newSequenceLLM(toolsTurn(
		toolCall{"echo", `{"text":"x"}`},
		toolCall{UserQuestionToolName, `{"question":"go?"}`},
	))
	done := executeToolsAsync(llm, userFragment("do both"),
		WithContext(ctx),
		WithTools(echo),
		EnableParallelToolExecution,
		WithUserQuestions(handler),
		WithMaxAttempts(3),
	)

	waitForSignal(t, called, "question handler")
	cancel()
	result := waitForExecuteTools(t, done)
	if !errors.Is(result.err, context.Canceled) {
		t.Fatalf("ExecuteTools returned %v, want context.Canceled", result.err)
	}
	assertFailedQuestionBatch(t, result.fragment, handlerCalls.Load(), echoCalls.Load())
}

func TestAskUserAnswerAfterCancellationSkipsParallelSiblingsAndPreservesAnswer(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	var handlerCalls atomic.Int32
	var echoCalls atomic.Int32
	handler := func(context.Context, UserQuestion) (UserAnswer, error) {
		handlerCalls.Add(1)
		cancel()
		return UserAnswer{Text: "accepted"}, nil
	}
	echo := NewToolDefinition[EchoArgs](recordingRunner{record: func(string) { echoCalls.Add(1) }}, EchoArgs{}, "echo", "echo")
	llm := newSequenceLLM(toolsTurn(
		toolCall{"echo", `{"text":"x"}`},
		toolCall{UserQuestionToolName, `{"question":"go?"}`},
	))

	f, err := ExecuteTools(llm, userFragment("do both"),
		WithContext(ctx),
		WithTools(echo),
		EnableParallelToolExecution,
		WithUserQuestions(handler),
		WithMaxAttempts(3),
	)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("ExecuteTools returned %v, want context.Canceled", err)
	}
	if got := handlerCalls.Load(); got != 1 {
		t.Fatalf("question handler called %d times, want 1", got)
	}
	if got := echoCalls.Load(); got != 0 {
		t.Fatalf("echo called %d times after cancellation, want 0", got)
	}
	if len(f.Status.ToolResults) != 2 {
		t.Fatalf("tool results = %d, want 2", len(f.Status.ToolResults))
	}
	statuses := make(map[string]ToolStatus, len(f.Status.ToolResults))
	for _, status := range f.Status.ToolResults {
		statuses[status.Name] = status
	}
	questionStatus, ok := statuses[UserQuestionToolName]
	answer, answered := questionStatus.ResultData.(UserAnswer)
	if !ok || !questionStatus.Executed || !answered || answer.Text != "accepted" || questionStatus.Result != "Answer: accepted" {
		t.Fatalf("ask_user status = %+v, answer = %+v; want preserved accepted answer", questionStatus, answer)
	}
	echoStatus, ok := statuses["echo"]
	if !ok || echoStatus.Executed {
		t.Fatalf("echo status = %+v, want preserved but not executed", echoStatus)
	}
	if names := f.Status.ToolsCalled.Names(); countOf(names, UserQuestionToolName) != 1 || countOf(names, "echo") != 0 {
		t.Fatalf("tools called = %v, want only ask_user", names)
	}
	assertQuestionBatchResultIDs(t, f)
}

func assertFailedQuestionBatch(t *testing.T, f Fragment, handlerCalls, echoCalls int32) {
	t.Helper()
	if handlerCalls != 1 {
		t.Fatalf("question handler called %d times, want 1", handlerCalls)
	}
	if echoCalls != 0 {
		t.Fatalf("echo called %d times without an answer, want 0", echoCalls)
	}
	if len(f.Status.ToolResults) != 2 {
		t.Fatalf("tool results = %d, want 2", len(f.Status.ToolResults))
	}
	statuses := make(map[string]ToolStatus, len(f.Status.ToolResults))
	for _, status := range f.Status.ToolResults {
		statuses[status.Name] = status
	}
	questionStatus, ok := statuses[UserQuestionToolName]
	if !ok || !questionStatus.Executed || questionStatus.ResultData != nil {
		t.Fatalf("ask_user status = %+v, want executed failure without UserAnswer data", questionStatus)
	}
	echoStatus, ok := statuses["echo"]
	if !ok || echoStatus.Executed {
		t.Fatalf("echo status = %+v, want preserved but not executed", echoStatus)
	}
	if names := f.Status.ToolsCalled.Names(); countOf(names, UserQuestionToolName) != 1 || countOf(names, "echo") != 0 {
		t.Fatalf("tools called = %v, want only ask_user", names)
	}
	assertQuestionBatchResultIDs(t, f)
}

func assertQuestionBatchResultIDs(t *testing.T, f Fragment) {
	t.Helper()
	assistantToolCallIDs := map[string]bool{}
	toolMessageIDs := map[string]bool{}
	for _, message := range f.Messages {
		for _, call := range message.ToolCalls {
			assistantToolCallIDs[call.ID] = true
		}
		if message.Role == ToolMessageRole.String() {
			toolMessageIDs[message.ToolCallID] = true
		}
	}
	if len(assistantToolCallIDs) != 2 || len(toolMessageIDs) != 2 {
		t.Fatalf("assistant tool-call IDs = %v, tool-message IDs = %v; want two of each", assistantToolCallIDs, toolMessageIDs)
	}
	for id := range assistantToolCallIDs {
		if !toolMessageIDs[id] {
			t.Fatalf("tool messages missing result for %s: %v", id, toolMessageIDs)
		}
	}
	for _, status := range f.Status.ToolResults {
		if !assistantToolCallIDs[status.ToolArguments.ID] {
			t.Fatalf("tool status %q has unknown call ID %q", status.Name, status.ToolArguments.ID)
		}
	}
}
