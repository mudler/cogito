# Execution interception

Secure embedders pass both `WithExecutionInterceptor(bridge)` and
`WithRequiredExecutionInterceptor()`. Missing bridges, legacy-only native
bodies and custom/remote agent dispatchers fail closed. Legacy execution
without these options retains its retry and approval behavior.

`ExecutionCall` exposes InvocationID, ToolCallID, AgentID, Name and Arguments
through accessors. Each Arguments call decodes an independent nested copy.
Formatting the call redacts all fields. IDs provide correlation, not authority.
The engine assigns invocation correlation before policy and preserves protocol
IDs from the model. `tool_args.go` therefore preserves parsed IDs; selection
only generates an ID when the provider supplies none.

Install mandatory vetoes with `WithExecutionPolicy`. The old approval callback
cannot distinguish a mandatory veto from an approval fast path, so the separate
hook is necessary. It runs before every approval evaluation, including skips.
SessionState.ExecutionCall is the immutable candidate evaluated by policy;
ToolChoice.InvocationID supplies matching correlation to existing callbacks.
Approval callbacks may modify arguments/name in place or return Modified.
Every observed change triggers another policy and approval evaluation, with
an eight-evaluation bound. Identity changes and textual Adjustment requests
fail closed in bridged mode. Callbacks must not concurrently mutate objects
while returning them. Retained callback maps/pointers do not own final dispatch.
Transient mutations undone before callback return cannot be observed; only the
final returned candidate is executed and must equal the evaluated candidate.

The bridge may return a replacement ExecutionResult without calling ToolBody.
That successful result follows the normal protocol tool-message/status path,
including the next model request and exported fragment. Data is live transport
only, not a durable replay encoding. Errors discard text/data and return a
fixed rejection without native error logging or result callbacks. Native body
errors and interceptor errors never trigger MaxAttempts retries in bridged mode.

ToolBody accepts a context, is single-use under concurrency, checks invocation
and body cancellation before entry, and expires when the interceptor returns.
Return cancels an already-entered native context; it cannot undo an external
effect. MCP ExecuteContext passes that context to actual ClientSession.CallTool.
Custom context-aware definitions implement ContextToolDefinitionInterface.
Typed ToolDefinition runners opt in with ContextTool.RunContext; merely wrapping
a legacy Run implementation does not qualify for required mode.

Foreground, background, nested and resumed local helpers inherit required mode,
bridge and policy; the engine stamps their registry ID independently of approval.
Required mode enables helper-tool preparation in descendants so nested helpers
use fresh runners rather than losing the spawn seam. Agent-tool selection still
propagates. Background/detached helpers retain their existing independent
lifetime and must be cancelled through their registered agent cancellation.

## Model mapping and limits

The accepted pure model remains unchanged. The shell reuses its canonical
executionCandidate/freezing/copy implementation. Admission corresponds to EX02;
policy, approval and reevaluation to EX03–EX06; one native entry to EX07;
cancellation, expiry and failure to EX12–EX17. Live zero-body substitution is
the transport portion of EX11. Host projection (EX08), journal acceptance
(EX09/EX11) and durable publication (EX10) are deliberately not implemented
inside this engine. The full reducer models host facts unavailable here, so
forcing its durable transitions into this shell would invent journal evidence.

Tests use actual in-process MCP transports, fake model completions and counter
bodies. They establish engine interception, not durable recovery, broker
settlement, authorization, or provider exactly-once effects. The host must bind
trusted identities, authorize/project results and settle its durable journal
before returning success from the interceptor. A consumed failed call remains
ambiguous to that host; Cogito never automatically retries it.
