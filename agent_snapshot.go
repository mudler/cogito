package cogito

import "slices"

// cloneAgentFragment owns both histories independently: cloning must not heal a
// stale projection or discard an envelope. Parent links may form a graph.
func cloneAgentFragment(f Fragment) Fragment {
	return *cloneAgentFragmentGraph(&f, make(map[*Fragment]*Fragment))
}

func cloneAgentFragmentGraph(f *Fragment, seen map[*Fragment]*Fragment) *Fragment {
	if f == nil {
		return nil
	}
	if copy, ok := seen[f]; ok {
		return copy
	}
	copy := *f
	seen[f] = &copy
	copy.ConversationMessages = CloneMessages(f.ConversationMessages)
	copy.Messages = PortableMessages(WrapMessages(f.Messages))
	copy.ParentFragment = cloneAgentFragmentGraph(f.ParentFragment, seen)
	copy.PendingNativeParts = slices.Clone(f.PendingNativeParts)
	copy.Multimedia = slices.Clone(f.Multimedia)
	if f.Status != nil {
		status := *f.Status
		status.ToolsCalled = slices.Clone(status.ToolsCalled)
		status.ToolResults = slices.Clone(status.ToolResults)
		status.Plans = slices.Clone(status.Plans)
		status.PastActions = slices.Clone(status.PastActions)
		status.ReasoningLog = slices.Clone(status.ReasoningLog)
		status.InjectedMessages = slices.Clone(status.InjectedMessages)
		copy.Status = &status
	}
	return &copy
}

// Caller holds the manager lock when snapshotting a registered, mutable state.
// Execution capabilities and application-defined status values remain opaque;
// conversation payloads never share mutable storage with the registry.
func snapshotAgent(agent *AgentState) *AgentState {
	if agent == nil {
		return nil
	}
	copy := *agent
	if agent.Fragment != nil {
		f := cloneAgentFragment(*agent.Fragment)
		copy.Fragment = &f
	}
	return &copy
}
