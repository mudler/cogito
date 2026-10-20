package cogito

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
)

// This file is a pure decision model, not an interceptor API or a dispatch
// implementation. EX rules are stable review anchors for future wiring.
// A shell must serialize events (including concurrent/reentrant Body requests),
// execute effects once, and supply trusted context/identity facts. Correlation
// identifiers are never authority. Legacy execution is deliberately not modeled.

type executionCandidate struct {
	invocation, protocol, agent, name, arguments string
}

// freezeExecution owns a canonical JSON snapshot, including nested values. No
// caller-owned map or slice survives. Numbers retain their JSON representation.
func freezeExecution(invocation, protocol, agent, name string, arguments map[string]any) (executionCandidate, error) {
	b, err := json.Marshal(arguments)
	if err != nil || arguments == nil || invocation == "" || protocol == "" || name == "" {
		return executionCandidate{}, errors.New("invalid execution candidate")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	var owned map[string]any
	if d.Decode(&owned) != nil {
		return executionCandidate{}, errors.New("invalid execution arguments")
	}
	b, err = json.Marshal(owned)
	if err != nil {
		return executionCandidate{}, errors.New("invalid execution arguments")
	}
	return executionCandidate{invocation, protocol, agent, name, string(b)}, nil
}

func (c executionCandidate) copiedArguments() map[string]any {
	d := json.NewDecoder(bytes.NewBufferString(c.arguments))
	d.UseNumber()
	var result map[string]any
	if d.Decode(&result) != nil {
		return nil
	}
	return result
}

func (c executionCandidate) valid() bool {
	d := json.NewDecoder(bytes.NewBufferString(c.arguments))
	d.UseNumber()
	var a map[string]any
	if d.Decode(&a) != nil || a == nil {
		return false
	}
	var trailing any
	if d.Decode(&trailing) != io.EOF {
		return false
	}
	frozen, err := freezeExecution(c.invocation, c.protocol, c.agent, c.name, a)
	return err == nil && frozen == c
}

type executionPhase interface{ executionPhase() }
type executionPolicy struct{}
type executionApproval struct{}
type executionBridge struct{}
type executionEntered struct{}
type executionProjection struct{}
type executionJournal struct{ text string }
type executionDone struct{ text string }
type executionClosed struct{}
type executionUnknown struct{}

func (executionPolicy) executionPhase()     {}
func (executionApproval) executionPhase()   {}
func (executionBridge) executionPhase()     {}
func (executionEntered) executionPhase()    {}
func (executionProjection) executionPhase() {}
func (executionJournal) executionPhase()    {}
func (executionDone) executionPhase()       {}
func (executionClosed) executionPhase()     {}
func (executionUnknown) executionPhase()    {}

type executionRecord struct {
	candidate executionCandidate
	phase     executionPhase
	// evaluated is the exact candidate approved by mandatory policy. Revision
	// distinguishes modification-away-and-back and stale policy/approval replies.
	evaluated executionCandidate
	revision  uint64
	entries   uint8
}
type executionModel struct{ calls []executionRecord }

type executionEvent interface{ executionEvent() }

// Admission facts come from the engine, never tool arguments. helper covers
// foreground/background/nested/resumed paths; remote/custom without propagation
// is unsupported. MaxAttempts is intentionally absent from this secure model.
type executionStart struct {
	candidate           executionCandidate
	bridge, contextBody bool
	helper, propagated  bool
	engineAgent         string
}
type executionPolicyReply struct {
	candidate executionCandidate
	revision  uint64
	allow     bool
}
type executionApprovalReply struct {
	candidate executionCandidate
	revision  uint64
	allow     bool
}
type executionModify struct{ candidate executionCandidate }
type executionBody struct {
	candidate executionCandidate
	revision  uint64
}
type executionBodyFinished struct{ ok bool }

// Substitute carries only an already authorized text projection, not raw Data.
// Unsupported replay formats are represented by executionFailure.
type executionSubstitute struct{ text string }
type executionProjected struct{ text string }
type executionJournaled struct{}
type executionCancel struct{}
type executionReturn struct{}
type executionFailure struct{}

func (executionStart) executionEvent()         {}
func (executionPolicyReply) executionEvent()   {}
func (executionApprovalReply) executionEvent() {}
func (executionModify) executionEvent()        {}
func (executionBody) executionEvent()          {}
func (executionBodyFinished) executionEvent()  {}
func (executionSubstitute) executionEvent()    {}
func (executionProjected) executionEvent()     {}
func (executionJournaled) executionEvent()     {}
func (executionCancel) executionEvent()        {}
func (executionReturn) executionEvent()        {}
func (executionFailure) executionEvent()       {}

type executionEffect interface{ executionEffect() }
type executionEvaluate struct {
	candidate executionCandidate
	revision  uint64
}
type executionAskApproval struct {
	candidate executionCandidate
	revision  uint64
}
type executionIntercept struct {
	candidate executionCandidate
	revision  uint64
}
type executionInvoke struct {
	candidate executionCandidate
	revision  uint64
}
type executionProject struct{ protocol string }
type executionWriteJournal struct{ protocol, text string }
type executionPublish struct{ protocol, text string }
type executionCancelBody struct{ invocation string }

func (executionEvaluate) executionEffect()     {}
func (executionAskApproval) executionEffect()  {}
func (executionIntercept) executionEffect()    {}
func (executionInvoke) executionEffect()       {}
func (executionProject) executionEffect()      {}
func (executionWriteJournal) executionEffect() {}
func (executionPublish) executionEffect()      {}
func (executionCancelBody) executionEffect()   {}

type executionTransition struct {
	model   executionModel
	effects []executionEffect
	rule    string
}

// reduceExecution has no I/O, callbacks, clocks, mutable global state, or raw
// result/error payloads. Unknown and Closed never retry, whatever MaxAttempts.
// EX00 ignores absent, stale, duplicate, late and out-of-order events.
func reduceExecution(m executionModel, invocation string, event executionEvent) executionTransition {
	out := executionTransition{model: m, rule: "EX00"}
	index := -1
	for i, r := range m.calls {
		if r.candidate.invocation == invocation {
			index = i
			break
		}
	}
	if e, ok := event.(executionStart); ok {
		if index >= 0 {
			return out
		}
		c := e.candidate
		// Reject conflicting protocol IDs too; neither ID grants permission.
		for _, r := range m.calls {
			if r.candidate.protocol == c.protocol {
				return out
			}
		}
		r := executionRecord{candidate: c, phase: executionClosed{}}
		out.rule = "EX02"
		if c.invocation == invocation && c.valid() && e.bridge && e.contextBody &&
			c.agent == e.engineAgent && ((e.helper && e.propagated && c.agent != "") || (!e.helper && c.agent == "")) {
			r.phase = executionPolicy{}
			out.rule = "EX01"
			out.effects = []executionEffect{executionEvaluate{c, 0}}
		}
		// Failed admission is retained under the requested invocation, even when
		// the supplied correlation was forged. It cannot later acquire a body.
		r.candidate.invocation = invocation
		out.model.calls = append(append([]executionRecord(nil), m.calls...), r)
		return out
	}
	if index < 0 {
		return out
	}
	r := m.calls[index]
	switch r.phase.(type) {
	case executionClosed, executionUnknown, executionDone:
		return out
	}
	set := func(phase executionPhase, rule string, effects ...executionEffect) {
		r.phase = phase
		out.rule = rule
		out.effects = effects
	}
	switch e := event.(type) {
	case executionModify:
		switch r.phase.(type) {
		case executionPolicy, executionApproval, executionBridge:
			// Invocation/protocol/engine identity remain frozen. Only name and
			// arguments can be modified, including an observed in-place mutation.
			c := e.candidate
			if !c.valid() || c.invocation != r.candidate.invocation || c.protocol != r.candidate.protocol || c.agent != r.candidate.agent || r.revision == ^uint64(0) {
				set(executionClosed{}, "EX02")
			} else {
				r.candidate = c
				r.evaluated = executionCandidate{}
				r.revision++
				set(executionPolicy{}, "EX06", executionEvaluate{c, r.revision})
			}
		}
	case executionPolicyReply:
		if _, ok := r.phase.(executionPolicy); ok && e.candidate == r.candidate && e.revision == r.revision {
			if e.allow {
				r.evaluated = r.candidate
				set(executionApproval{}, "EX03", executionAskApproval{r.candidate, r.revision})
			} else {
				set(executionClosed{}, "EX04")
			}
		}
	case executionApprovalReply:
		if _, ok := r.phase.(executionApproval); ok && e.candidate == r.candidate && e.revision == r.revision && r.evaluated == r.candidate {
			if e.allow {
				set(executionBridge{}, "EX05", executionIntercept{r.candidate, r.revision})
			} else {
				set(executionClosed{}, "EX04")
			}
		}
	case executionBody:
		if _, ok := r.phase.(executionBridge); ok && r.entries == 0 && e.candidate == r.candidate && e.revision == r.revision && r.evaluated == r.candidate && r.candidate.valid() {
			r.entries = 1
			set(executionEntered{}, "EX07", executionInvoke{r.candidate, r.revision})
		}
	case executionBodyFinished:
		if _, ok := r.phase.(executionEntered); ok {
			if e.ok {
				set(executionProjection{}, "EX08", executionProject{r.candidate.protocol})
			} else {
				set(executionUnknown{}, "EX14")
			}
		}
	case executionProjected:
		if _, ok := r.phase.(executionProjection); ok {
			set(executionJournal{e.text}, "EX09", executionWriteJournal{r.candidate.protocol, e.text})
		}
	case executionSubstitute:
		if _, ok := r.phase.(executionBridge); ok {
			set(executionJournal{e.text}, "EX11", executionWriteJournal{r.candidate.protocol, e.text})
		}
	case executionJournaled:
		if p, ok := r.phase.(executionJournal); ok {
			set(executionDone{p.text}, "EX10", executionPublish{r.candidate.protocol, p.text})
		}
	case executionCancel:
		_, pendingJournal := r.phase.(executionJournal)
		if r.entries == 0 && !pendingJournal {
			set(executionClosed{}, "EX12")
		} else {
			set(executionUnknown{}, "EX13", executionCancelBody{invocation})
		}
	case executionReturn:
		// Return before accepted completion expires the body. A consumed body or
		// pending journal is ambiguous, even for zero-body recorded substitution.
		switch r.phase.(type) {
		case executionEntered, executionProjection, executionJournal:
			set(executionUnknown{}, "EX15")
		case executionPolicy, executionApproval, executionBridge:
			set(executionClosed{}, "EX16")
		}
	case executionFailure:
		switch r.phase.(type) {
		case executionEntered, executionProjection, executionJournal:
			set(executionUnknown{}, "EX14")
		case executionPolicy, executionApproval, executionBridge:
			set(executionClosed{}, "EX17")
		}
	case executionStart: // Already handled above.
	}
	if out.rule != "EX00" {
		out.model.calls = append([]executionRecord(nil), m.calls...)
		out.model.calls[index] = r
	}
	return out
}
