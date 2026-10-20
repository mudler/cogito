package cogito

import (
	"encoding/json"
	"reflect"
	"testing"
)

func executionFixture(t *testing.T, id, agent string) executionCandidate {
	t.Helper()
	c, err := freezeExecution(id, "protocol-"+id, agent, "counter", map[string]any{"nested": map[string]any{"value": "secret"}})
	if err != nil {
		t.Fatal("freeze failed")
	}
	return c
}
func executionStartFor(c executionCandidate) executionStart {
	return executionStart{candidate: c, bridge: true, contextBody: true, helper: c.agent != "", propagated: true, engineAgent: c.agent}
}
func executionReady(c executionCandidate) []executionEvent {
	return []executionEvent{executionStartFor(c), executionPolicyReply{c, 0, true}, executionApprovalReply{c, 0, true}}
}
func executionTrace(t *testing.T, c executionCandidate, events []executionEvent, rules []string) executionModel {
	t.Helper()
	m := executionModel{}
	for i, e := range events {
		before := executionKey(m)
		tr := reduceExecution(m, c.invocation, e)
		if tr.rule != rules[i] {
			t.Fatalf("step %d: got %s want %s", i, tr.rule, rules[i])
		}
		if executionKey(m) != before {
			t.Fatal("mutated prior snapshot")
		}
		executionCheck(t, m, c.invocation, e, tr)
		m = tr.model
	}
	return m
}

// Structural keys are test-only, contain no raw arguments/results in diagnostics,
// and include phase type (encoding an empty interface directly loses its tag).
func executionKey(m executionModel) string {
	// Candidate has private fields, so explicitly encode its comparable values.
	type row struct {
		Values      [10]string
		Revision    uint64
		Entries     uint8
		Phase, Text string
	}
	rows := []row{}
	for _, r := range m.calls {
		text := ""
		switch p := r.phase.(type) {
		case executionJournal:
			text = p.text
		case executionDone:
			text = p.text
		}
		a, b := r.candidate, r.evaluated
		rows = append(rows, row{[10]string{a.invocation, a.protocol, a.agent, a.name, a.arguments, b.invocation, b.protocol, b.agent, b.name, b.arguments}, r.revision, r.entries, reflect.TypeOf(r.phase).Name(), text})
	}
	b, _ := json.Marshal(rows)
	return string(b)
}

// Independent guards: these assertions deliberately do not consult EX rule IDs
// to authorize invocation/publication. Canonical equality alone is insufficient:
// phase, revision, frozen policy candidate and cancellation history also matter.
func executionCheck(t *testing.T, before executionModel, id string, e executionEvent, tr executionTransition) {
	t.Helper()
	for _, r := range tr.model.calls {
		if r.entries > 1 {
			t.Fatal("body entered more than once")
		}
		switch r.phase.(type) {
		case executionPolicy, executionClosed, executionUnknown:
		case executionApproval, executionBridge, executionEntered, executionProjection, executionJournal, executionDone:
			if r.evaluated != r.candidate || !r.candidate.valid() {
				t.Fatal("unevaluated candidate advanced")
			}
		default:
			t.Fatal("unrecognized phase")
		}
	}
	var old *executionRecord
	for i := range before.calls {
		if before.calls[i].candidate.invocation == id {
			old = &before.calls[i]
		}
	}
	for _, effect := range tr.effects {
		switch ef := effect.(type) {
		case executionInvoke:
			b, ok := e.(executionBody)
			if !ok || old == nil {
				t.Fatal("invoke without body request")
			}
			if _, ok := old.phase.(executionBridge); !ok {
				t.Fatal("invoke outside live bridge")
			}
			if old.entries != 0 || b.candidate != old.candidate || b.revision != old.revision || old.evaluated != old.candidate || ef.candidate != old.candidate {
				t.Fatal("invoke binding mismatch")
			}
		case executionPublish:
			if _, ok := e.(executionJournaled); !ok || old == nil {
				t.Fatal("publication without journal acceptance")
			}
			j, ok := old.phase.(executionJournal)
			if !ok || ef.protocol != old.candidate.protocol || ef.text != j.text {
				t.Fatal("publication is not accepted projection")
			}
		case executionEvaluate, executionAskApproval, executionIntercept, executionProject, executionWriteJournal, executionCancelBody:
		default:
			t.Fatal("unrecognized effect")
		}
	}
}

func TestExecutionModelHappyPath(t *testing.T) {
	c := executionFixture(t, "a", "")
	events := append(executionReady(c), executionBody{c, 0}, executionBodyFinished{true}, executionProjected{"safe"}, executionJournaled{}, executionReturn{})
	m := executionTrace(t, c, events, []string{"EX01", "EX03", "EX05", "EX07", "EX08", "EX09", "EX10", "EX00"})
	if p, ok := m.calls[0].phase.(executionDone); !ok || p.text != "safe" || m.calls[0].entries != 1 {
		t.Fatal("wrong completion")
	}
}
func TestExecutionModelSubstitution(t *testing.T) {
	c := executionFixture(t, "a", "")
	m := executionTrace(t, c, append(executionReady(c), executionSubstitute{"recorded"}, executionJournaled{}, executionBody{c, 0}), []string{"EX01", "EX03", "EX05", "EX11", "EX10", "EX00"})
	p, ok := m.calls[0].phase.(executionDone)
	if !ok || p.text != "recorded" || m.calls[0].entries != 0 {
		t.Fatal("substitution did not preserve result without execution")
	}
}
func TestExecutionModelFrozenArguments(t *testing.T) {
	args := map[string]any{"nested": map[string]any{"v": []any{"original"}}}
	c, err := freezeExecution("a", "p", "", "n", args)
	if err != nil {
		t.Fatal(err)
	}
	args["nested"].(map[string]any)["v"].([]any)[0] = "mutated"
	copy := c.copiedArguments()
	copy["nested"].(map[string]any)["v"].([]any)[0] = "changed"
	if c.arguments != "{\"nested\":{\"v\":[\"original\"]}}" {
		t.Fatal("snapshot aliased")
	}
}
func TestExecutionModelCanonicalMatch(t *testing.T) {
	a, _ := freezeExecution("a", "p", "", "n", map[string]any{"b": 2, "a": 1})
	b, _ := freezeExecution("a", "p", "", "n", map[string]any{"a": 1, "b": 2})
	if a != b || !a.valid() {
		t.Fatal("canonical match failed")
	}
}
func TestExecutionModelInvalidCandidate(t *testing.T) {
	if _, err := freezeExecution("a", "p", "", "n", map[string]any{"bad": make(chan int)}); err == nil {
		t.Fatal("unsupported arguments accepted")
	}
	c := executionFixture(t, "a", "")
	c.arguments = "{} {}"
	executionTrace(t, c, []executionEvent{executionStartFor(c)}, []string{"EX02"})
}

func executionAdmissionFailure(t *testing.T, change func(*executionStart)) {
	t.Helper()
	c := executionFixture(t, "a", "")
	e := executionStartFor(c)
	change(&e)
	m := executionTrace(t, c, []executionEvent{e, executionBody{c, 0}}, []string{"EX02", "EX00"})
	if _, ok := m.calls[0].phase.(executionClosed); !ok {
		t.Fatal("admission did not fail closed")
	}
}
func TestExecutionModelMissingBridge(t *testing.T) {
	executionAdmissionFailure(t, func(e *executionStart) { e.bridge = false })
}
func TestExecutionModelMissingContextBody(t *testing.T) {
	executionAdmissionFailure(t, func(e *executionStart) { e.contextBody = false })
}
func TestExecutionModelForgedHelper(t *testing.T) {
	executionAdmissionFailure(t, func(e *executionStart) { e.candidate.agent = "forged" })
}
func TestExecutionModelUnsupportedRemote(t *testing.T) {
	executionAdmissionFailure(t, func(e *executionStart) {
		e.helper = true
		e.propagated = false
		e.engineAgent = "helper"
		e.candidate.agent = "helper"
	})
}
func TestExecutionModelHelperPaths(t *testing.T) {
	for _, path := range []string{"foreground", "background", "nested", "resumed"} {
		t.Run(path, func(t *testing.T) {
			c := executionFixture(t, "a", path)
			executionTrace(t, c, append(executionReady(c), executionBody{c, 0}), []string{"EX01", "EX03", "EX05", "EX07"})
		})
	}
}
func TestExecutionModelPolicyVeto(t *testing.T) {
	c := executionFixture(t, "a", "")
	executionTrace(t, c, []executionEvent{executionStartFor(c), executionPolicyReply{c, 0, false}, executionApprovalReply{c, 0, true}, executionBody{c, 0}}, []string{"EX01", "EX04", "EX00", "EX00"})
}
func TestExecutionModelApprovalVeto(t *testing.T) {
	c := executionFixture(t, "a", "")
	executionTrace(t, c, []executionEvent{executionStartFor(c), executionPolicyReply{c, 0, true}, executionApprovalReply{c, 0, false}, executionBody{c, 0}}, []string{"EX01", "EX03", "EX04", "EX00"})
}
func TestExecutionModelApprovalFastPathCannotBypassPolicy(t *testing.T) {
	c := executionFixture(t, "a", "")
	executionTrace(t, c, []executionEvent{executionStartFor(c), executionApprovalReply{c, 0, true}, executionBody{c, 0}}, []string{"EX01", "EX00", "EX00"})
}
func TestExecutionModelModificationRechecks(t *testing.T) {
	c := executionFixture(t, "a", "")
	changed := c
	changed.name = "other"
	events := append(executionReady(c), executionModify{changed}, executionBody{changed, 1}, executionPolicyReply{c, 0, true}, executionPolicyReply{changed, 1, true}, executionApprovalReply{changed, 1, true}, executionBody{changed, 1})
	executionTrace(t, c, events, []string{"EX01", "EX03", "EX05", "EX06", "EX00", "EX00", "EX03", "EX05", "EX07"})
}
func TestExecutionModelInPlaceMutationRechecks(t *testing.T) {
	c := executionFixture(t, "a", "")
	args := c.copiedArguments()
	args["nested"].(map[string]any)["value"] = "changed"
	changed, err := freezeExecution(c.invocation, c.protocol, c.agent, c.name, args)
	if err != nil {
		t.Fatal(err)
	}
	executionTrace(t, c, append(executionReady(c), executionModify{changed}, executionBody{c, 0}, executionBody{changed, 1}), []string{"EX01", "EX03", "EX05", "EX06", "EX00", "EX00"})
}
func TestExecutionModelEqualModificationInvalidatesRevision(t *testing.T) {
	c := executionFixture(t, "a", "")
	executionTrace(t, c, append(executionReady(c), executionModify{c}, executionPolicyReply{c, 0, true}, executionBody{c, 0}), []string{"EX01", "EX03", "EX05", "EX06", "EX00", "EX00"})
}
func TestExecutionModelIdentityModificationRejected(t *testing.T) {
	c := executionFixture(t, "a", "")
	for _, field := range []string{"invocation", "protocol", "agent"} {
		t.Run(field, func(t *testing.T) {
			changed := c
			switch field {
			case "invocation":
				changed.invocation = "other"
			case "protocol":
				changed.protocol = "other"
			case "agent":
				changed.agent = "other"
			}
			executionTrace(t, c, append(executionReady(c), executionModify{changed}, executionBody{c, 0}), []string{"EX01", "EX03", "EX05", "EX02", "EX00"})
		})
	}
}
func TestExecutionModelDuplicateReentrantBody(t *testing.T) {
	c := executionFixture(t, "a", "")
	executionTrace(t, c, append(executionReady(c), executionBody{c, 0}, executionBody{c, 0}, executionBodyFinished{true}, executionBody{c, 0}), []string{"EX01", "EX03", "EX05", "EX07", "EX00", "EX08", "EX00"})
}
func TestExecutionModelExpiredBody(t *testing.T) {
	c := executionFixture(t, "a", "")
	executionTrace(t, c, append(executionReady(c), executionReturn{}, executionBody{c, 0}), []string{"EX01", "EX03", "EX05", "EX16", "EX00"})
}
func TestExecutionModelCancellationBeforeEntry(t *testing.T) {
	c := executionFixture(t, "a", "")
	m := executionTrace(t, c, append(executionReady(c), executionCancel{}, executionBody{c, 0}), []string{"EX01", "EX03", "EX05", "EX12", "EX00"})
	if m.calls[0].entries != 0 {
		t.Fatal("cancelled body entered")
	}
}
func TestExecutionModelCancellationAfterEntryUnknown(t *testing.T) {
	c := executionFixture(t, "a", "")
	m := executionTrace(t, c, append(executionReady(c), executionBody{c, 0}, executionCancel{}, executionBodyFinished{true}, executionJournaled{}), []string{"EX01", "EX03", "EX05", "EX07", "EX13", "EX00", "EX00"})
	if _, ok := m.calls[0].phase.(executionUnknown); !ok {
		t.Fatal("cancelled execution not unknown")
	}
}
func TestExecutionModelBodyFailureNoRetry(t *testing.T) {
	c := executionFixture(t, "a", "")
	executionTrace(t, c, append(executionReady(c), executionBody{c, 0}, executionBodyFinished{false}, executionStartFor(c), executionBody{c, 0}), []string{"EX01", "EX03", "EX05", "EX07", "EX14", "EX00", "EX00"})
}
func TestExecutionModelProjectionFailureNoPublication(t *testing.T) {
	c := executionFixture(t, "a", "")
	executionTrace(t, c, append(executionReady(c), executionBody{c, 0}, executionBodyFinished{true}, executionFailure{}, executionProjected{"unsafe"}, executionJournaled{}), []string{"EX01", "EX03", "EX05", "EX07", "EX08", "EX14", "EX00", "EX00"})
}
func TestExecutionModelJournalFailureUnknown(t *testing.T) {
	c := executionFixture(t, "a", "")
	executionTrace(t, c, append(executionReady(c), executionSubstitute{"safe"}, executionFailure{}, executionJournaled{}), []string{"EX01", "EX03", "EX05", "EX11", "EX14", "EX00"})
}
func TestExecutionModelReturnAfterEntryUnknown(t *testing.T) {
	c := executionFixture(t, "a", "")
	executionTrace(t, c, append(executionReady(c), executionBody{c, 0}, executionReturn{}, executionBodyFinished{true}), []string{"EX01", "EX03", "EX05", "EX07", "EX15", "EX00"})
}
func TestExecutionModelWrongBindingCannotEnter(t *testing.T) {
	c := executionFixture(t, "a", "")
	changed := c
	changed.arguments = "{}"
	executionTrace(t, c, append(executionReady(c), executionBody{changed, 0}, executionBody{c, 1}, executionBody{c, 0}), []string{"EX01", "EX03", "EX05", "EX00", "EX00", "EX07"})
}
func TestExecutionModelLateOutOfOrder(t *testing.T) {
	c := executionFixture(t, "a", "")
	executionTrace(t, c, []executionEvent{executionJournaled{}, executionBodyFinished{true}, executionStartFor(c), executionProjected{"unsafe"}, executionSubstitute{"unsafe"}}, []string{"EX00", "EX00", "EX01", "EX00", "EX00"})
}
func TestExecutionModelMaxAttemptsCannotReplay(t *testing.T) {
	c := executionFixture(t, "a", "")
	events := append(executionReady(c), executionBody{c, 0}, executionFailure{})
	rules := []string{"EX01", "EX03", "EX05", "EX07", "EX14"}
	for attempt := 0; attempt < 5; attempt++ {
		events = append(events, executionStartFor(c), executionBody{c, 0})
		rules = append(rules, "EX00", "EX00")
	}
	executionTrace(t, c, events, rules)
}

// The checker bounds each call to one observed modification and two argument
// values. It explores each root/helper call exhaustively within that alphabet,
// then all order-preserving interleavings of two fixed successful traces. This
// is not an exhaustive unbounded two-call product or a scheduler fairness proof.
func executionAlphabet(c executionCandidate, m executionModel) []executionEvent {
	changed := c
	changed.arguments = "{}"
	revision := uint64(0)
	if len(m.calls) > 0 {
		revision = m.calls[0].revision
	}
	events := []executionEvent{executionStartFor(c), executionPolicyReply{c, 0, true}, executionPolicyReply{c, 0, false}, executionApprovalReply{c, 0, true}, executionApprovalReply{c, 0, false}, executionBody{c, 0}, executionBodyFinished{true}, executionBodyFinished{false}, executionSubstitute{"safe"}, executionProjected{"safe"}, executionJournaled{}, executionCancel{}, executionReturn{}, executionFailure{}}
	if revision == 0 {
		events = append(events, executionModify{changed})
	}
	events = append(events, executionPolicyReply{changed, 1, true}, executionPolicyReply{changed, 1, false}, executionApprovalReply{changed, 1, true}, executionApprovalReply{changed, 1, false}, executionBody{changed, 1})
	return events
}
func executionTerminal(m executionModel) bool {
	if len(m.calls) == 0 {
		return false
	}
	for _, r := range m.calls {
		switch r.phase.(type) {
		case executionDone, executionUnknown, executionClosed:
		default:
			return false
		}
	}
	return true
}
func TestExecutionModelBoundedChecker(t *testing.T) {
	states, edges, witnesses := 0, 0, 0
	rules := map[string]int{}
	for _, agent := range []string{"", "engine-helper"} {
		c := executionFixture(t, "a", agent)
		type node struct {
			m    executionModel
			path []executionEvent
		}
		queue := []node{{}}
		seen := map[string]bool{executionKey(executionModel{}): true}
		for head := 0; head < len(queue); head++ {
			n := queue[head]
			states++
			// An explicit cancellation is a progress witness from every nonterminal
			// state. This is reachability of a safe exit, not eventual success/fairness.
			if !executionTerminal(n.m) && len(n.m.calls) > 0 {
				tr := reduceExecution(n.m, c.invocation, executionCancel{})
				executionCheck(t, n.m, c.invocation, executionCancel{}, tr)
				if !executionTerminal(tr.model) {
					t.Fatalf("no cancellation exit at depth %d", len(n.path))
				}
				witnesses++
			}
			for _, e := range executionAlphabet(c, n.m) {
				before := executionKey(n.m)
				tr := reduceExecution(n.m, c.invocation, e)
				edges++
				rules[tr.rule]++
				executionCheck(t, n.m, c.invocation, e, tr)
				if executionKey(n.m) != before {
					t.Fatal("checker found mutation")
				}
				k := executionKey(tr.model)
				if !seen[k] {
					seen[k] = true
					path := append(append([]executionEvent(nil), n.path...), e)
					queue = append(queue, node{tr.model, path})
				}
			}
		}
	}
	if states < 40 || edges < 400 || witnesses == 0 {
		t.Fatal("checker unexpectedly empty")
	}
	for _, rule := range []string{"EX00", "EX01", "EX03", "EX04", "EX05", "EX06", "EX07", "EX08", "EX09", "EX10", "EX11", "EX12", "EX13", "EX14", "EX15", "EX16", "EX17"} {
		if rules[rule] == 0 {
			t.Fatalf("checker missing %s", rule)
		}
	}
	t.Logf("bounded single-call root/helper checker: states=%d transitions=%d safe-exit witnesses=%d", states, edges, witnesses)
}

func TestExecutionModelTwoCallsEqualArgumentsInterleavings(t *testing.T) {
	a, b := executionFixture(t, "a", ""), executionFixture(t, "b", "engine-helper")
	if a.arguments != b.arguments || a.invocation == b.invocation || a.protocol == b.protocol {
		t.Fatal("invalid two-call bounds")
	}
	traces := [][]executionEvent{
		append(executionReady(a), executionBody{a, 0}, executionBody{a, 0}, executionBodyFinished{true}, executionProjected{"safe-a"}, executionJournaled{}),
		append(executionReady(b), executionBody{b, 0}, executionBody{b, 0}, executionBodyFinished{true}, executionProjected{"safe-b"}, executionJournaled{}),
	}
	candidates := []executionCandidate{a, b}
	witnesses, edges := 0, 0
	var walk func(executionModel, int, int)
	walk = func(m executionModel, i, j int) {
		if i == len(traces[0]) && j == len(traces[1]) {
			witnesses++
			if !executionTerminal(m) || len(m.calls) != 2 {
				t.Fatal("two calls did not complete")
			}
			for _, r := range m.calls {
				if r.entries != 1 {
					t.Fatal("equal arguments merged or duplicate body entered")
				}
			}
			return
		}
		for actor := 0; actor < 2; actor++ {
			positions := []int{i, j}
			pos := positions[actor]
			if pos == len(traces[actor]) {
				continue
			}
			id := candidates[actor].invocation
			e := traces[actor][pos]
			before := executionKey(m)
			tr := reduceExecution(m, id, e)
			edges++
			executionCheck(t, m, id, e, tr)
			if executionKey(m) != before {
				t.Fatal("interleaving mutated history")
			}
			for _, old := range m.calls {
				if old.candidate.invocation != id {
					found := false
					for _, next := range tr.model.calls {
						if next.candidate.invocation == old.candidate.invocation {
							found = true
							if !reflect.DeepEqual(old, next) {
								t.Fatal("cross-call contamination")
							}
						}
					}
					if !found {
						t.Fatal("other call disappeared")
					}
				}
			}
			if actor == 0 {
				walk(tr.model, i+1, j)
			} else {
				walk(tr.model, i, j+1)
			}
		}
	}
	walk(executionModel{}, 0, 0)
	if witnesses != 12870 {
		t.Fatalf("actual interleavings %d want 12870", witnesses)
	}
	t.Logf("two equal-argument distinct-ID root/helper traces: complete interleavings=%d transitions=%d (includes sequential extremes and parallel/reentrant duplicate orderings)", witnesses, edges)
}

func TestExecutionModelCancelPendingSubstitutionUnknown(t *testing.T) {
	c := executionFixture(t, "a", "")
	m := executionTrace(t, c, append(executionReady(c), executionSubstitute{"safe"}, executionCancel{}, executionJournaled{}, executionBody{c, 0}), []string{"EX01", "EX03", "EX05", "EX11", "EX13", "EX00", "EX00"})
	if _, ok := m.calls[0].phase.(executionUnknown); !ok {
		t.Fatal("pending durable write not unknown")
	}
}
func TestExecutionModelEffectsCarryExactBindings(t *testing.T) {
	c := executionFixture(t, "a", "")
	m := executionModel{}
	events := append(executionReady(c), executionBody{c, 0}, executionBodyFinished{true}, executionProjected{"safe"}, executionJournaled{})
	expected := []executionEffect{executionEvaluate{c, 0}, executionAskApproval{c, 0}, executionIntercept{c, 0}, executionInvoke{c, 0}, executionProject{c.protocol}, executionWriteJournal{c.protocol, "safe"}, executionPublish{c.protocol, "safe"}}
	for i, e := range events {
		tr := reduceExecution(m, c.invocation, e)
		if !reflect.DeepEqual(tr.effects, []executionEffect{expected[i]}) {
			t.Fatalf("effect mismatch at step %d", i)
		}
		m = tr.model
	}
}
