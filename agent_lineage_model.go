package cogito

import (
	"fmt"
	"reflect"
	"sort"
)

// LN rules specify engine lineage, not host grant authority or actual MCP.
// This model does not change AgentState or runtime options. A future shell must
// use private withAgentIDStamp capture (never model arguments), share the manager,
// callbacks and helper definitions, and serialize transitions per target. It must
// not Close a child runner or wipe the shared cancellation registry on completion.
// Effects are ordered obligations: registration and spawn precede policy;
// running precedes body; result/terminal callback precede local injection/done.
// Callback sinks and grants are opaque captured identities, not authorization.
// The engine, not a caller-supplied ParentID, constructs lineageAllocate.

type lineagePhase interface{ lineagePhase() }
type lineageAllocated struct{}
type lineageRegistered struct{}
type lineageRunning struct{}
type lineageCompleting struct {
	Result  string
	Outcome lineageOutcome
}
type lineageCompleted struct{ Result string }
type lineageCancelled struct{ Result string }
type lineageResuming struct{}

func (lineageAllocated) lineagePhase()  {}
func (lineageRegistered) lineagePhase() {}
func (lineageRunning) lineagePhase()    {}
func (lineageCompleting) lineagePhase() {}
func (lineageCompleted) lineagePhase()  {}
func (lineageCancelled) lineagePhase()  {}
func (lineageResuming) lineagePhase()   {}

type lineageOutcome uint8

const (
	lineageSuccess lineageOutcome = iota
	lineageCancellation
)

type lineageScope struct {
	ID         string
	Generation uint8
}
type lineageCapture struct {
	Parent                lineageScope
	Grants, Sink, Manager string
}
type lineageAgent struct {
	ID, Task   string
	Capture    lineageCapture
	Generation uint8
	Phase      lineagePhase
}
type lineageModel struct{ Agents map[string]lineageAgent }

type lineageEvent interface{ lineageEvent() }
type lineageAllocate struct {
	ID, Task string
	Capture  lineageCapture
}
type lineageRegister struct {
	Target lineageScope
	Parent lineageScope
}
type lineageStart struct{ Target lineageScope }
type lineageFinish struct {
	Target lineageScope
	Result string
}
type lineageSettle struct{ Target lineageScope }
type lineageCancel struct{ Target lineageScope }
type lineageResume struct{ Target lineageScope }
type lineagePark struct{ Target lineageScope }

func (lineageAllocate) lineageEvent() {}
func (lineageRegister) lineageEvent() {}
func (lineageStart) lineageEvent()    {}
func (lineageFinish) lineageEvent()   {}
func (lineageSettle) lineageEvent()   {}
func (lineageCancel) lineageEvent()   {}
func (lineageResume) lineageEvent()   {}
func (lineagePark) lineageEvent()     {}

type lineageEffect struct {
	Kind    string
	Target  lineageScope
	Capture lineageCapture
	Value   string
	Pending []string
}
type lineageTransition struct {
	Model   lineageModel
	Rule    string
	Effects []lineageEffect
}

func lineageTerminal(p lineagePhase) bool {
	switch p.(type) {
	case lineageCompleted, lineageCancelled:
		return true
	}
	return false
}

// pending is execution-local: only registered, unsettled direct children of this
// exact generation. Neither shared-manager membership nor task text defines it.
func (m lineageModel) pending(parent lineageScope) []string {
	var ids []string
	for id, a := range m.Agents {
		if a.Capture.Parent != parent {
			continue
		}
		switch a.Phase.(type) {
		case lineageRegistered, lineageRunning, lineageCompleting, lineageResuming:
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids
}
func lineageTarget(e lineageEvent) lineageScope {
	switch e := e.(type) {
	case lineageRegister:
		return e.Target
	case lineageStart:
		return e.Target
	case lineageFinish:
		return e.Target
	case lineageSettle:
		return e.Target
	case lineageCancel:
		return e.Target
	case lineageResume:
		return e.Target
	case lineagePark:
		return e.Target
	case lineageAllocate:
		return lineageScope{ID: e.ID}
	}
	panic("unknown lineage event")
}

// reduceLineage clones the map and never mutates previous snapshots. LN00 is an
// explicit invalid/stale-event no-op; LN01..LN09 have stable guards below.
func reduceLineage(m lineageModel, e lineageEvent) lineageTransition {
	n := lineageModel{Agents: make(map[string]lineageAgent, len(m.Agents))}
	for id, a := range m.Agents {
		n.Agents[id] = a
	}
	tr := lineageTransition{Model: n, Rule: "LN00-ignored"}
	target := lineageTarget(e)
	a, exists := n.Agents[target.ID]
	effect := func(kind, value string) {
		tr.Effects = append(tr.Effects, lineageEffect{Kind: kind, Target: lineageScope{a.ID, a.Generation}, Capture: a.Capture, Value: value})
	}
	if e, ok := e.(lineageAllocate); ok {
		c := e.Capture
		if exists || e.ID == "" || c.Manager == "" || c.Sink == "" {
			return tr
		}
		if c.Parent.ID == "" {
			if c.Parent.Generation != 0 {
				return tr
			}
		} else {
			p, ok := n.Agents[c.Parent.ID]
			if !ok || p.Generation != c.Parent.Generation || p.Capture.Manager != c.Manager {
				return tr
			}
			if _, ok := p.Phase.(lineageRunning); !ok {
				return tr
			}
		}
		a = lineageAgent{ID: e.ID, Task: e.Task, Capture: c, Phase: lineageAllocated{}}
		n.Agents[a.ID] = a
		tr.Rule = "LN01-allocate"
		return tr
	}
	if !exists || a.Generation != target.Generation {
		return tr
	}
	switch e := e.(type) {
	case lineageRegister:
		if _, ok := a.Phase.(lineageAllocated); !ok || e.Parent != a.Capture.Parent {
			return tr
		}
		if a.Capture.Parent.ID != "" {
			p, ok := n.Agents[a.Capture.Parent.ID]
			if !ok || p.Generation != a.Capture.Parent.Generation {
				return tr
			}
			if _, ok := p.Phase.(lineageRunning); !ok {
				return tr
			}
		}
		a.Phase = lineageRegistered{}
		tr.Rule = "LN02-register"
		effect("register", "")
		effect("spawn-callback", "")
	case lineageStart:
		switch a.Phase.(type) {
		case lineageRegistered:
			tr.Rule = "LN03-start"
			effect("policy", "")
		case lineageResuming:
			tr.Rule = "LN08-resume-start"
		case lineageAllocated, lineageRunning, lineageCompleting, lineageCompleted, lineageCancelled:
			return tr
		}
		a.Phase = lineageRunning{}
		effect("lifecycle-running", "")
		effect("body", "")
	case lineageFinish:
		if _, ok := a.Phase.(lineageRunning); !ok {
			return tr
		}
		a.Phase = lineageCompleting{e.Result, lineageSuccess}
		tr.Rule = "LN04-result"
		effect("publish-result", e.Result)
		effect("lifecycle-completed", e.Result)
		effect("completion-callback", e.Result)
	case lineageCancel:
		switch a.Phase.(type) {
		case lineageAllocated, lineageRegistered, lineageRunning, lineageResuming:
		case lineageCompleting, lineageCompleted, lineageCancelled:
			return tr
		}
		a.Phase = lineageCompleting{"cancelled", lineageCancellation}
		tr.Rule = "LN06-cancel"
		effect("cancel-target", "")
		effect("publish-result", "cancelled")
		effect("lifecycle-cancelled", "cancelled")
		effect("completion-callback", "cancelled")
	case lineageSettle:
		p, ok := a.Phase.(lineageCompleting)
		if !ok {
			return tr
		}
		if p.Outcome == lineageCancellation {
			a.Phase = lineageCancelled{p.Result}
		} else {
			a.Phase = lineageCompleted{p.Result}
		}
		tr.Rule = "LN05-settle"
		effect("inject-parent", p.Result)
		effect("close-generation", "")
	case lineageResume:
		if !lineageTerminal(a.Phase) || a.Generation == 255 {
			return tr
		}
		a.Generation++
		a.Phase = lineageResuming{}
		tr.Rule = "LN07-resume"
		effect("new-generation", "")
	case lineagePark:
		if _, ok := a.Phase.(lineageRunning); !ok {
			return tr
		}
		tr.Rule = "LN09-park"
		effect("wait-direct-children", "")
		tr.Effects[0].Pending = n.pending(target)
	case lineageAllocate:
		panic("allocation handled above")
	}
	n.Agents[a.ID] = a
	return tr
}

// checkLineage is independent of reduceLineage. These are structural safety
// invariants; completion reachability and event/effect guards are checked below.
func checkLineage(m lineageModel) error {
	for id, a := range m.Agents {
		if id == "" || a.ID != id || a.Capture.Manager == "" || a.Capture.Sink == "" {
			return fmt.Errorf("invalid capture %q", id)
		}
		switch p := a.Phase.(type) {
		case lineageAllocated, lineageRegistered:
			if a.Generation != 0 {
				return fmt.Errorf("reallocated generation")
			}
		case lineageResuming:
			if a.Generation == 0 {
				return fmt.Errorf("zero resume")
			}
		case lineageRunning, lineageCompleted, lineageCancelled:
		case lineageCompleting:
			if p.Outcome != lineageSuccess && p.Outcome != lineageCancellation {
				return fmt.Errorf("invalid outcome")
			}
		default:
			return fmt.Errorf("unknown phase %T", p)
		}
		if a.Capture.Parent.ID == "" {
			if a.Capture.Parent.Generation != 0 {
				return fmt.Errorf("root generation")
			}
			continue
		}
		p, ok := m.Agents[a.Capture.Parent.ID]
		if !ok || p.Capture.Manager != a.Capture.Manager || p.Generation < a.Capture.Parent.Generation {
			return fmt.Errorf("invalid parent")
		}
		seen := map[string]bool{id: true}
		for p.ID != "" {
			if seen[p.ID] {
				return fmt.Errorf("cycle")
			}
			seen[p.ID] = true
			p = m.Agents[p.Capture.Parent.ID]
		}
	}
	return nil
}

// checkLineageTransition checks guards separately from canonical reducer replay.
// Exact replay is useful for tampered effects, but is NOT the independent oracle.
func checkLineageTransition(before lineageModel, e lineageEvent, tr lineageTransition) error {
	if err := checkLineage(tr.Model); err != nil {
		return err
	}
	target := lineageTarget(e)
	old, exists := before.Agents[target.ID]
	next := tr.Model.Agents[target.ID]
	if tr.Rule != "LN01-allocate" && tr.Rule != "LN00-ignored" {
		if !exists || old.Generation != target.Generation || old.ID != next.ID || old.Task != next.Task || old.Capture != next.Capture {
			return fmt.Errorf("identity changed")
		}
	}
	for id, a := range before.Agents {
		if id != target.ID && !reflect.DeepEqual(a, tr.Model.Agents[id]) {
			return fmt.Errorf("sibling changed")
		}
	}
	valid := false
	switch tr.Rule {
	case "LN00-ignored":
		valid = reflect.DeepEqual(before.Agents, tr.Model.Agents) || (len(before.Agents) == 0 && len(tr.Model.Agents) == 0)
	case "LN01-allocate":
		ev, ok := e.(lineageAllocate)
		valid = ok && !exists && next.Capture == ev.Capture && next.Generation == 0
	case "LN02-register":
		ev, ok := e.(lineageRegister)
		_, phase := old.Phase.(lineageAllocated)
		valid = ok && phase && ev.Parent == old.Capture.Parent
		if old.Capture.Parent.ID != "" {
			p, exists := before.Agents[old.Capture.Parent.ID]
			_, running := p.Phase.(lineageRunning)
			valid = valid && exists && running && p.Generation == old.Capture.Parent.Generation
		}
	case "LN03-start":
		_, ok := e.(lineageStart)
		_, phase := old.Phase.(lineageRegistered)
		valid = ok && phase
	case "LN04-result":
		_, ok := e.(lineageFinish)
		_, phase := old.Phase.(lineageRunning)
		valid = ok && phase
	case "LN05-settle":
		_, ok := e.(lineageSettle)
		_, phase := old.Phase.(lineageCompleting)
		valid = ok && phase
	case "LN06-cancel":
		_, ok := e.(lineageCancel)
		switch old.Phase.(type) {
		case lineageAllocated, lineageRegistered, lineageRunning, lineageResuming:
			valid = ok
		}
	case "LN07-resume":
		_, ok := e.(lineageResume)
		valid = ok && lineageTerminal(old.Phase) && old.Generation < 255 && next.Generation == old.Generation+1
	case "LN08-resume-start":
		_, ok := e.(lineageStart)
		_, phase := old.Phase.(lineageResuming)
		valid = ok && phase
	case "LN09-park":
		_, ok := e.(lineagePark)
		_, phase := old.Phase.(lineageRunning)
		valid = ok && phase
	}
	if !valid {
		return fmt.Errorf("rule guard %s", tr.Rule)
	}
	expected := reduceLineage(before, e)
	if !reflect.DeepEqual(expected, tr) {
		return fmt.Errorf("noncanonical transition %s", tr.Rule)
	}
	return nil
}
