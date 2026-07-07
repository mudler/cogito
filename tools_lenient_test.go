package cogito

import (
	"strings"
	"sync"
	"testing"
)

func TestNormalizeToolName(t *testing.T) {
	cases := []struct{ in, want string }{
		{"foo", "foo"},
		{"Foo", "foo"},
		{"functions.foo", "foo"},
		{"tools/foo", "foo"},
		{"namespace::foo", "foo"},
		{"foo-bar", "foo_bar"},
		{"Foo-Bar", "foo_bar"},
		{"functions.Coach_Regeln", "coach_regeln"},
		{"  foo  ", "foo"},
		{"", ""},
		{"functions.", ""}, // nothing after the separator → empty key
	}
	for _, c := range cases {
		if got := normalizeToolName(c.in); got != c.want {
			t.Errorf("normalizeToolName(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestFindLenient(t *testing.T) {
	tools := Tools{newNamedTool("coach_regeln")}

	// exact + lenient variants local models routinely emit
	for _, n := range []string{
		"coach_regeln",           // exact
		"functions.coach_regeln", // namespaced
		"tools/coach_regeln",
		"Coach_Regeln", // case
		"coach-regeln", // separator
	} {
		if tools.Find(n) == nil {
			t.Errorf("Find(%q) = nil, want match", n)
		}
	}

	for _, n := range []string{"other_tool", "", "functions."} {
		if tools.Find(n) != nil {
			t.Errorf("Find(%q) matched, want nil", n)
		}
	}
}

// An exact match must always win, even when a lenient candidate appears earlier
// in the slice — the lenient pass is a fallback, never an override.
func TestFindExactWinsOverLenient(t *testing.T) {
	tools := Tools{newNamedTool("functions.foo"), newNamedTool("foo")}
	got := tools.Find("foo")
	if got == nil || got.Tool().Function.Name != "foo" {
		name := "<nil>"
		if got != nil {
			name = got.Tool().Function.Name
		}
		t.Errorf(`Find("foo") = %q, want exact "foo"`, name)
	}
}

// A lenient match must hand the tool-call callback the CANONICAL tool name —
// the name that is executed — not the model's spelling. Otherwise an approval
// hook configured for "echo" would never see a call the model spelled "Echo".
func TestLenientToolCallCallbackSeesCanonicalName(t *testing.T) {
	for _, spelled := range []string{"Echo", "functions.echo", "tools/echo"} {
		t.Run(spelled, func(t *testing.T) {
			var mu sync.Mutex
			var seen []string
			cb := func(tc *ToolChoice, _ *SessionState) ToolCallDecision {
				mu.Lock()
				seen = append(seen, tc.Name)
				mu.Unlock()
				return ToolCallDecision{Approved: true}
			}
			var echoMu sync.Mutex
			echoCount := 0
			llm := newScriptedLLM(spelled, `{"text": "hi"}`, "done")
			f := NewEmptyFragment().AddMessage(UserMessageRole, "say hi")
			if _, err := ExecuteTools(llm, f, WithTools(newEchoTool(&echoMu, &echoCount)), WithToolCallBack(cb), WithIterations(1)); err != nil && err != ErrNoToolSelected {
				t.Fatalf("ExecuteTools: %v", err)
			}
			if len(seen) != 1 || seen[0] != "echo" {
				t.Fatalf("callback saw %v, want [echo]", seen)
			}
			if echoCount != 1 {
				t.Fatalf("echo ran %d times, want 1", echoCount)
			}
		})
	}
}

// When two tools normalize to the same name, a lenient lookup must not pick
// either one: it is not found, and the error names the candidates.
func TestFindAmbiguousLenientIsNotFound(t *testing.T) {
	tools := Tools{newNamedTool("github/list_issues"), newNamedTool("gitlab/list_issues")}

	if got := tools.Find("list_issues"); got != nil {
		t.Fatalf(`Find("list_issues") = %q, want nil (ambiguous)`, got.Tool().Function.Name)
	}
	if got := tools.Find("functions.list_issues"); got != nil {
		t.Fatalf(`Find("functions.list_issues") = %q, want nil (ambiguous)`, got.Tool().Function.Name)
	}
	// exact names still resolve
	for _, n := range []string{"github/list_issues", "gitlab/list_issues"} {
		if got := tools.Find(n); got == nil || got.Tool().Function.Name != n {
			t.Fatalf("Find(%q) did not return the exact tool", n)
		}
	}
	err := tools.notFoundError("list_issues")
	if err == nil || !strings.Contains(err.Error(), "github/list_issues") || !strings.Contains(err.Error(), "gitlab/list_issues") {
		t.Fatalf("notFoundError = %v, want both candidates named", err)
	}
	if e := (Tools{newNamedTool("foo")}).notFoundError("bar"); e == nil || strings.Contains(e.Error(), "ambiguous") {
		t.Fatalf("plain not-found error = %v, want no ambiguity note", e)
	}
}

// End to end: a model call to an ambiguous name must execute neither tool.
func TestAmbiguousLenientToolCallExecutesNothing(t *testing.T) {
	var mu sync.Mutex
	a, b := 0, 0
	toolA := NewToolDefinition[EchoArgs](echoRunner{mu: &mu, count: &a}, EchoArgs{}, "github/list_issues", "a")
	toolB := NewToolDefinition[EchoArgs](echoRunner{mu: &mu, count: &b}, EchoArgs{}, "gitlab/list_issues", "b")
	llm := newScriptedLLM("list_issues", `{"text": "x"}`, "done")
	f := NewEmptyFragment().AddMessage(UserMessageRole, "list issues")
	_, _ = ExecuteTools(llm, f, WithTools(toolA, toolB), WithIterations(1))
	if a != 0 || b != 0 {
		t.Fatalf("ambiguous name executed a tool (github=%d, gitlab=%d), want neither", a, b)
	}
}
