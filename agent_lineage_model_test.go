package cogito

import (
	"fmt"
	"reflect"
	"testing"
)

func lnScope(id string, generation ...uint8) lineageScope {
	s := lineageScope{ID: id}
	if len(generation) > 0 {
		s.Generation = generation[0]
	}
	return s
}
func lnAllocate(id, parent string) lineageAllocate {
	return lineageAllocate{ID: id, Task: "identical task", Capture: lineageCapture{Parent: lnScope(parent), Manager: "shared", Sink: "captured-" + id, Grants: "grant-" + id}}
}
func lnStep(t *testing.T, m lineageModel, e lineageEvent, rule string, kinds ...string) lineageModel {
	t.Helper()
	before := lnKey(m)
	tr := reduceLineage(m, e)
	if tr.Rule != rule {
		t.Fatalf("%T: rule %s want %s", e, tr.Rule, rule)
	}
	var got []string
	for _, ef := range tr.Effects {
		got = append(got, ef.Kind)
	}
	if !reflect.DeepEqual(got, kinds) {
		t.Fatalf("%s effects %v want %v", rule, got, kinds)
	}
	if err := checkLineageTransition(m, e, tr); err != nil {
		t.Fatal(err)
	}
	if before != lnKey(m) {
		t.Fatal("mutated prior snapshot")
	}
	return tr.Model
}
func lnRunning(t *testing.T, m lineageModel, id, parent string) lineageModel {
	m = lnStep(t, m, lnAllocate(id, parent), "LN01-allocate")
	m = lnStep(t, m, lineageRegister{lnScope(id), lnScope(parent)}, "LN02-register", "register", "spawn-callback")
	return lnStep(t, m, lineageStart{lnScope(id)}, "LN03-start", "policy", "lifecycle-running", "body")
}
func lnComplete(t *testing.T, m lineageModel, s lineageScope) lineageModel {
	m = lnStep(t, m, lineageFinish{s, "result"}, "LN04-result", "publish-result", "lifecycle-completed", "completion-callback")
	return lnStep(t, m, lineageSettle{s}, "LN05-settle", "inject-parent", "close-generation")
}
func lnFixture(t *testing.T) lineageModel {
	m := lnRunning(t, lineageModel{}, "p", "")
	return lnRunning(t, m, "q", "")
}
func lnKey(m lineageModel) string {
	ids := make([]string, 0, len(m.Agents))
	for id := range m.Agents {
		ids = append(ids, id)
	}
	sortStrings(ids)
	key := ""
	for _, id := range ids {
		a := m.Agents[id]
		key += fmt.Sprintf("%q:%q:%#v:%d:%T:%#v;", id, a.Task, a.Capture, a.Generation, a.Phase, a.Phase)
	}
	return key
}
func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

func TestLineageRootEmptyParent(t *testing.T) {
	m := lnRunning(t, lineageModel{}, "p", "")
	if m.Agents["p"].Capture.Parent != (lineageScope{}) {
		t.Fatal("root parent")
	}
}
func TestLineageLifecycleOrderedTrace(t *testing.T) {
	m := lnRunning(t, lineageModel{}, "p", "")
	m = lnComplete(t, m, lnScope("p"))
	if !lineageTerminal(m.Agents["p"].Phase) {
		t.Fatal("not completed")
	}
}
func TestLineageChildBeforePolicy(t *testing.T) { m := lnFixture(t); lnRunning(t, m, "a", "p") }
func TestLineageIdenticalTasksParallelBranchesReverseOrder(t *testing.T) {
	m := lnFixture(t)
	m = lnRunning(t, m, "a", "p")
	m = lnRunning(t, m, "b", "q")
	m = lnComplete(t, m, lnScope("b"))
	if !reflect.DeepEqual(m.pending(lnScope("p")), []string{"a"}) || len(m.pending(lnScope("q"))) != 0 {
		t.Fatal("branch leakage")
	}
	lnComplete(t, m, lnScope("a"))
}
func TestLineageNestedCallbacksCapture(t *testing.T) {
	m := lnRunning(t, lnFixture(t), "a", "p")
	m = lnRunning(t, m, "b", "a")
	tr := reduceLineage(m, lineageFinish{lnScope("b"), "ok"})
	for _, ef := range tr.Effects {
		if ef.Capture != m.Agents["b"].Capture {
			t.Fatal("closure capture changed")
		}
	}
}
func TestLineageForgedParentRegistration(t *testing.T) {
	m := lnStep(t, lnFixture(t), lnAllocate("a", "p"), "LN01-allocate")
	lnStep(t, m, lineageRegister{lnScope("a"), lnScope("q")}, "LN00-ignored")
}
func TestLineageStaleRegistration(t *testing.T) {
	m := lnRunning(t, lnFixture(t), "a", "p")
	lnStep(t, m, lineageRegister{lnScope("a"), lnScope("p")}, "LN00-ignored")
}
func TestLineageMissingParent(t *testing.T) {
	lnStep(t, lineageModel{}, lnAllocate("a", "missing"), "LN00-ignored")
}
func TestLineageParentMustRun(t *testing.T) {
	m := lnStep(t, lineageModel{}, lnAllocate("p", ""), "LN01-allocate")
	lnStep(t, m, lnAllocate("a", "p"), "LN00-ignored")
}
func TestLineageSharedManagerRequired(t *testing.T) {
	m := lnFixture(t)
	e := lnAllocate("a", "p")
	e.Capture.Manager = "private"
	lnStep(t, m, e, "LN00-ignored")
}
func TestLineageDuplicateAllocation(t *testing.T) {
	m := lnFixture(t)
	lnStep(t, m, lnAllocate("p", ""), "LN00-ignored")
}
func TestLineageOutOfOrderFinish(t *testing.T) {
	m := lnStep(t, lineageModel{}, lnAllocate("p", ""), "LN01-allocate")
	lnStep(t, m, lineageFinish{lnScope("p"), "early"}, "LN00-ignored")
}
func TestLineageDuplicateCompletion(t *testing.T) {
	m := lnComplete(t, lnFixture(t), lnScope("p"))
	lnStep(t, m, lineageSettle{lnScope("p")}, "LN00-ignored")
}
func TestLineageCancellationLateResult(t *testing.T) {
	m := lnFixture(t)
	m = lnStep(t, m, lineageCancel{lnScope("p")}, "LN06-cancel", "cancel-target", "publish-result", "lifecycle-cancelled", "completion-callback")
	m = lnStep(t, m, lineageFinish{lnScope("p"), "late"}, "LN00-ignored")
	lnStep(t, m, lineageSettle{lnScope("p")}, "LN05-settle", "inject-parent", "close-generation")
}
func TestLineageCompletionWinsCancellation(t *testing.T) {
	m := lnFixture(t)
	m = lnStep(t, m, lineageFinish{lnScope("p"), "ok"}, "LN04-result", "publish-result", "lifecycle-completed", "completion-callback")
	lnStep(t, m, lineageCancel{lnScope("p")}, "LN00-ignored")
}
func TestLineageResumePreservesIdentityGrantsSink(t *testing.T) {
	m := lnRunning(t, lnFixture(t), "a", "p")
	old := m.Agents["a"]
	m = lnComplete(t, m, lnScope("a"))
	m = lnStep(t, m, lineageResume{lnScope("a")}, "LN07-resume", "new-generation")
	a := m.Agents["a"]
	if old.Capture != a.Capture || old.ID != a.ID || a.Generation != 1 {
		t.Fatal("lost identity")
	}
	lnStep(t, m, lineageStart{lnScope("a", 1)}, "LN08-resume-start", "lifecycle-running", "body")
}
func TestLineageOverlappingResumeSerialized(t *testing.T) {
	m := lnComplete(t, lnFixture(t), lnScope("p"))
	m = lnStep(t, m, lineageResume{lnScope("p")}, "LN07-resume", "new-generation")
	lnStep(t, m, lineageResume{lnScope("p", 1)}, "LN00-ignored")
}
func TestLineageSeparateResumeConcurrent(t *testing.T) {
	m := lnFixture(t)
	m = lnComplete(t, m, lnScope("p"))
	m = lnComplete(t, m, lnScope("q"))
	m = lnStep(t, m, lineageResume{lnScope("p")}, "LN07-resume", "new-generation")
	m = lnStep(t, m, lineageResume{lnScope("q")}, "LN07-resume", "new-generation")
	if _, ok := m.Agents["p"].Phase.(lineageResuming); !ok {
		t.Fatal("lost resume")
	}
}
func TestLineageLateGenerationEvents(t *testing.T) {
	m := lnComplete(t, lnFixture(t), lnScope("p"))
	m = lnStep(t, m, lineageResume{lnScope("p")}, "LN07-resume", "new-generation")
	for _, e := range []lineageEvent{lineageFinish{lnScope("p"), "late"}, lineageCancel{lnScope("p")}, lineageSettle{lnScope("p")}, lineageRegister{lnScope("p"), lnScope("")}} {
		m = lnStep(t, m, e, "LN00-ignored")
	}
}
func TestLineageGenerationSeparateDone(t *testing.T) {
	m := lnFixture(t)
	for g := uint8(0); g < 3; g++ {
		if g > 0 {
			m = lnStep(t, m, lineageResume{lnScope("p", g-1)}, "LN07-resume", "new-generation")
			m = lnStep(t, m, lineageStart{lnScope("p", g)}, "LN08-resume-start", "lifecycle-running", "body")
		}
		m = lnComplete(t, m, lnScope("p", g))
		lnStep(t, m, lineageSettle{lnScope("p", g)}, "LN00-ignored")
	}
}
func TestLineageParkingOnlyDirectChildren(t *testing.T) {
	m := lnRunning(t, lnFixture(t), "a", "p")
	m = lnRunning(t, m, "b", "a")
	tr := reduceLineage(m, lineagePark{lnScope("a")})
	if !reflect.DeepEqual(tr.Effects[0].Pending, []string{"b"}) {
		t.Fatal("self,ancestor,sibling parked")
	}
	lnStep(t, m, lineagePark{lnScope("a")}, "LN09-park", "wait-direct-children")
}
func TestLineageParkingGenerationIsolation(t *testing.T) {
	m := lnRunning(t, lnFixture(t), "a", "p")
	m = lnComplete(t, m, lnScope("p"))
	m = lnStep(t, m, lineageResume{lnScope("p")}, "LN07-resume", "new-generation")
	if len(m.pending(lnScope("p", 1))) != 0 {
		t.Fatal("old child parked")
	}
}
func TestLineageSiblingSurvivesCancellation(t *testing.T) {
	m := lnRunning(t, lnFixture(t), "a", "p")
	m = lnRunning(t, m, "b", "p")
	old := m.Agents["b"]
	m = lnStep(t, m, lineageCancel{lnScope("a")}, "LN06-cancel", "cancel-target", "publish-result", "lifecycle-cancelled", "completion-callback")
	if !reflect.DeepEqual(old, m.Agents["b"]) {
		t.Fatal("sibling wiped")
	}
}
func TestLineageIndependentInvariantRejectsCycle(t *testing.T) {
	m := lnFixture(t)
	a := m.Agents["p"]
	a.Capture.Parent = lnScope("p")
	m.Agents["p"] = a
	if checkLineage(m) == nil {
		t.Fatal("cycle accepted")
	}
}
func TestLineageCheckTransitionRejectsTampering(t *testing.T) {
	m := lnFixture(t)
	e := lineageFinish{lnScope("p"), "ok"}
	tr := reduceLineage(m, e)
	tr.Effects[0].Capture.Sink = "forged"
	if checkLineageTransition(m, e, tr) == nil {
		t.Fatal("forged sink accepted")
	}
	tr = reduceLineage(m, e)
	tr.Rule = "LN07-resume"
	if checkLineageTransition(m, e, tr) == nil {
		t.Fatal("guard accepted")
	}
}

// Enumerate two specified branch programs, not a whole Cartesian event space.
// Each child has generations 0..2. All cross-branch order-preserving schedules
// are explored with canonical state deduplication. Invalid/late vocabulary is
// checked separately by named traces, and cancel alternatives below.
func TestLineageBoundedInterleavings(t *testing.T) {
	initial := lnFixture(t)
	program := func(id, parent string) []lineageEvent {
		es := []lineageEvent{lnAllocate(id, parent), lineageRegister{lnScope(id), lnScope(parent)}, lineageStart{lnScope(id)}}
		for g := uint8(0); g < 3; g++ {
			if g > 0 {
				es = append(es, lineageResume{lnScope(id, g-1)}, lineageStart{lnScope(id, g)})
			}
			es = append(es, lineageFinish{lnScope(id, g), "ok"}, lineageSettle{lnScope(id, g)})
		}
		return es
	}
	programs := [2][]lineageEvent{program("a", "p"), program("b", "q")}
	type node struct {
		m    lineageModel
		i, j int
		path string
	}
	queue := []node{{m: initial}}
	seen := map[string]bool{}
	edges, terminal, cancelEdges := 0, 0, 0
	for len(queue) > 0 {
		n := queue[0]
		queue = queue[1:]
		key := fmt.Sprintf("%d/%d/%s", n.i, n.j, lnKey(n.m))
		if seen[key] {
			continue
		}
		seen[key] = true
		if err := checkLineage(n.m); err != nil {
			t.Fatalf("%s: %v", n.path, err)
		}
		if n.i == len(programs[0]) && n.j == len(programs[1]) {
			terminal++
			continue
		}
		successors := 0
		for branch := 0; branch < 2; branch++ {
			index := n.i
			if branch == 1 {
				index = n.j
			}
			if index == len(programs[branch]) {
				continue
			}
			e := programs[branch][index]
			tr := reduceLineage(n.m, e)
			if err := checkLineageTransition(n.m, e, tr); err != nil {
				t.Fatalf("%s/%T: %v", n.path, e, err)
			}
			if tr.Rule == "LN00-ignored" {
				t.Fatalf("dead program %s", n.path)
			}
			edges++
			successors++
			child := node{m: tr.Model, i: n.i, j: n.j, path: n.path + fmt.Sprintf(" %d:%s", branch, tr.Rule)}
			if branch == 0 {
				child.i++
			} else {
				child.j++
			}
			queue = append(queue, child)
		}
		if successors == 0 {
			t.Fatal("nonterminal dead end")
		}
		// Independent bounded cancellation continuation must reach terminal in two
		// steps without touching siblings, at every cancellable discovered node.
		for id, a := range n.m.Agents {
			e := lineageCancel{lnScope(id, a.Generation)}
			tr := reduceLineage(n.m, e)
			if tr.Rule != "LN06-cancel" {
				continue
			}
			cancelEdges++
			if err := checkLineageTransition(n.m, e, tr); err != nil {
				t.Fatal(err)
			}
			settle := lineageSettle{lnScope(id, a.Generation)}
			done := reduceLineage(tr.Model, settle)
			if err := checkLineageTransition(tr.Model, settle, done); err != nil {
				t.Fatal(err)
			}
			if !lineageTerminal(done.Model.Agents[id].Phase) {
				t.Fatal("cancel deadlock")
			}
		}
	}
	t.Logf("bounded two-parent/two-child generations 0..2: states=%d program-edges=%d terminal=%d cancellation-continuations=%d", len(seen), edges, terminal, cancelEdges)
	if len(seen) != 196 || edges != 364 || terminal != 1 {
		t.Fatalf("unexpected bounded counts: %d/%d/%d", len(seen), edges, terminal)
	}
}

func TestLineageRegistrationAfterParentGenerationRetired(t *testing.T) {
	m := lnStep(t, lnFixture(t), lnAllocate("a", "p"), "LN01-allocate")
	m = lnComplete(t, m, lnScope("p"))
	m = lnStep(t, m, lineageResume{lnScope("p")}, "LN07-resume", "new-generation")
	lnStep(t, m, lineageRegister{lnScope("a"), lnScope("p")}, "LN00-ignored")
}
