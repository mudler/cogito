# Provider messages and conversation ownership

Cogito supports an optional, non-streaming `MessageLLM` capability alongside the unchanged `LLM` interface. It carries provider data without flattening that data into text.

This API does not add a native Claude adapter or guarantee support for every backend. An adapter must implement provider parsing, replay, and validation.

## Message carrier

A `Message` contains two representations:

- `Portable`: an `openai.ChatCompletionMessage` for portable content, tool calls, and correlation IDs.
- `Provider`: an optional `*ProviderMessage` for provider-specific replay.

`ProviderMessage` contains `Version`, `Scope`, and ordered `Blocks` of `json.RawMessage`. Version `1` is the initial schema. Blocks retain unknown fields, string values, and number lexemes. JSON re-encoding need not retain insignificant whitespace or equivalent string escaping.

Treat received envelopes as immutable history. Do not rebuild provider blocks from portable content or execution arguments. Do not merge, reorder, or partially retain provider-backed assistant messages.

`ProviderScope` binds an envelope to `Provider`, `Protocol`, `Endpoint`, `Model`, and optional `Account`. Use a normalized, credential-free origin and base path for `Endpoint`. Use a stable, non-secret account or profile identifier for `Account`. Never put credentials, tokens, or authorization headers in scope fields.

The adapter must validate the version, scope, block integrity, and portable projection before sending a request. Cogito carries this provenance; it does not authenticate it or enforce provider-specific replay rules. A scope match alone does not prove that restored data is trustworthy.

## Completion dispatch

`MessageLLM` embeds `LLM` and adds this method:

```go
CreateMessageCompletion(context.Context, cogito.CompletionRequest) (cogito.CompletionReply, cogito.LLMUsage, error)
```

Use `CompleteMessages` to select the optional capability:

```go
package example

import (
    "context"

    "github.com/mudler/cogito"
    "github.com/sashabaranov/go-openai"
)

func Complete(ctx context.Context, llm cogito.LLM, f cogito.Fragment) (cogito.CompletionReply, cogito.LLMUsage, error) {
    messages, err := f.Conversation()
    if err != nil {
        return cogito.CompletionReply{}, cogito.LLMUsage{}, err
    }
    return cogito.CompleteMessages(ctx, llm, cogito.CompletionRequest{
        Options: openai.ChatCompletionRequest{Model: "your-model"},
        Messages: messages,
    })
}
```

`CompletionRequest.Messages` is authoritative. `Options` carries the other OpenAI request settings. `Options.Messages` must be `nil`; even a nonnil empty slice is an error.

`CompletionReply.Choices` carries authoritative `CompletionChoice` values. Each choice contains `Index`, `Message`, and `FinishReason`. `Response` contains the compatibility `LLMReply`, including its `ChatCompletionResponse`.

Adapters must return matching choice order, indexes, finish reasons, and portable messages in both representations. Choice indexes must be nonnegative and unique, but need not be contiguous. Choice slices must also match in length and nilness. `CompleteMessages` rejects inconsistent replies rather than selecting one representation.

`CompleteMessages` copies request options, messages, and successful replies into independently owned storage. Nested SDK fields and raw byte buffers do not alias the caller's data. Unsupported option values, such as functions or cyclic data, produce an error instead of shared storage.

For an existing `LLM`, `CompleteMessages` converts portable-only messages into `CreateChatCompletion` requests. Any nonnil `Provider` causes `ErrProviderMessagesUnsupported` before the legacy backend call. It never silently discards opaque history.

The legacy transport projection replaces nonblank, syntactically invalid historical tool arguments with `{}` for template compatibility. This does not rewrite durable history or rich replay. Blank arguments retain the legacy no-parameters convention; that convention does not establish native wire validity.

Custom legacy `Ask` implementations should call `Fragment.LegacyMessages()` before contacting their backend. That method rejects opaque history and stale projections. Direct calls to arbitrary custom clients bypass the protections of `CompleteMessages`.

## Migrate fragment access

`Fragment.ConversationMessages` is authoritative when it is nonnil. `Fragment.Messages` remains a separately owned portable compatibility projection.

| Value | Meaning |
| --- | --- |
| `ConversationMessages == nil` | Legacy mode: `Conversation()` wraps `Messages`. |
| Nonnil, empty `ConversationMessages` | Intentional authoritative empty history, with a matching nonnil empty `Messages`. |
| Nonnil, nonempty `ConversationMessages` | Authoritative envelopes with an exactly matching portable projection. |

`CloneMessages`, `WrapMessages`, and `PortableMessages` preserve nil versus empty slices. `CloneMessages` deep-copies envelopes. `WrapMessages` copies portable messages without inventing provider data. `PortableMessages` copies only the portable projection; it is not a lossless export.

Replace direct writes to `Messages` with these operations:

- `WithConversation(messages)` explicitly replaces history and synchronizes the projection.
- `AppendMessages(messages...)` appends owned envelopes.
- `AddMessage`, `AddToolMessage`, and the other fragment mutators preserve the conversation mode and existing envelopes.

For example, initialize an authoritative conversation, then append portable input:

```go
package example

import (
    "github.com/mudler/cogito"
    "github.com/sashabaranov/go-openai"
)

func NewConversation() cogito.Fragment {
    f := cogito.NewEmptyFragment().WithConversation([]cogito.Message{})
    return f.AppendMessages(cogito.Message{
        Portable: openai.ChatCompletionMessage{
            Role: "user", Content: "Describe this project.",
        },
    })
}
```

`WithConversation(nil)` clears both histories and selects legacy mode. It is not equivalent to an authoritative empty slice.

`Conversation()` returns an owned snapshot without request normalization. It reports `*ErrConversationProjectionMismatch` when direct edits make `Messages` inconsistent, including differences in nested values or nilness. `AppendMessages` does not repair an existing mismatch. Resolve the conflict explicitly with `WithConversation` using the intended complete history; do not flatten provider data to hide the error.

Use `GetConversation()` for the lossless counterpart of `GetMessages()`. It adds the force-text instruction when tool calls exist and normalizes portable system instructions. It retains non-system envelopes intact and ordered. It rejects opaque system messages when normalization would require rewriting them.

`GetMessages()` is a display or legacy projection, not a replay API or a projection validator. Use `Conversation()` for persistence and ordinary history access. Use `LegacyMessages()` for guarded legacy `Ask` requests.

### Manipulators and wrappers

Use `WithConversationManipulator(func([]Message) ([]Message, error))` for envelope-aware request changes. The callback receives owned messages; Cogito copies its result before use. An error aborts the request. Retain each provider-backed envelope intact when keeping it.

`WithMessagesManipulator` remains available for portable-only history. Envelope-aware request builders reject it on provider-backed history before invoking the callback. They also reject configuring both manipulator options.

Wrappers must preserve capabilities deliberately. Embedding an `LLM` field does not expose an underlying `MessageLLM` method through Go's method set. Implement and forward `CreateMessageCompletion` only when the underlying client supports it. Preserve `StreamingLLM` and `NativePartsAware` independently where applicable. Cogito's usage-counting wrappers preserve these optional capabilities without advertising unsupported ones.

### Agent snapshots

`AgentManager.Get`, `List`, and `Wait` return snapshots, not live registry entries. Their conversation payloads use independent storage, including parent histories. Fetch another snapshot to observe progress. Editing a returned fragment does not update the running agent; use manager operations such as `Inject`.

Execution capabilities and application-defined status data remain opaque. The snapshot contract does not promise deep cloning of arbitrary application objects.

## Tool execution and immutable history

`ToolChoice.OriginalCall` contains an owned copy of the received tool call before execution changes. It is diagnostic provenance, not a replay key. Synthetic choices may leave it nil. The original assistant envelope remains authoritative.

`ToolChoice.Arguments` is an execution-only map. Parsing accepts exactly one JSON object and rejects trailing values, arrays, and `null`. Blank arguments retain the no-parameters convention.

Parsed numbers use `json.Number` at every nesting level, rather than `float64`. Dynamic tools and callbacks must handle this type. Use `Int64()` or `Float64()` with error handling, or retain `String()` when exact numeric text matters. Do not convert a large integer through `float64` when precision matters. Typed tool runners continue to decode into their declared argument types through JSON.

Policy callbacks can change execution arguments without rewriting original calls or provider blocks. Cogito independently owns rich callback inputs and `Modified` values. Editing `OriginalCall` does not replace the retained provenance.

### Whole-batch correction

Cogito validates all tool arguments before executing any sibling in the selected batch. If any arguments are invalid, no tool in that batch executes. The correction exchange retains the complete original assistant and closes every call ID with a tool result. Invalid calls receive their argument error; valid siblings receive a not-executed result that requests resubmission of the whole batch.

Closed correction exchanges remain in returned history on retry exhaustion, transport failure, cancellation, or a later unavailable tool. This includes forced-reasoning paths. Provider-backed calls with missing or duplicate IDs fail closed; Cogito does not invent native correlation IDs. Malformed or ambiguous envelopes do not receive a fabricated valid history.

### Rich policy preflight and adjustment

The following policy applies to non-streaming provider-backed batches and their replacement chains. Cogito preflights callbacks before any tool execution. After integrity and context checks, callback precedence is:

1. `Approved: false` rejects the batch, even when `Skip` is true.
2. `Approved: true, Skip: true` skips that call while allowing siblings.
3. A nonnil `Modified` selects execution changes instead of adjustment feedback.
4. A nonempty `Adjustment` requests replacement of the whole batch.

Rejection stops later callbacks and prevents all execution in that batch. Callbacks cannot change original correlation IDs. Cogito retains the original assistant and one result per call, in original order, including skips and unexecuted calls.

Ordinary tool execution errors become result text. A tool made unavailable by a policy modification receives a failed terminal event without dispatch; rich parallel preflight prevents all dispatch, while sequential execution retains earlier results and cancels the remaining calls. The legacy path continues valid siblings. Context failures retain actual results once and close the unexecuted remainder. Cancellation prevents further sequential tool execution; already-started parallel work is joined. These rules do not roll back side effects that already occurred.

The first adjustment supersedes the whole original batch with zero tool executions. Cogito closes every original ID and appends portable feedback to history. It then selects a complete replacement using the same counted LLM, request options, and prepared tools. Every replacement requires fresh approval, including portable replacements within a rich-origin chain.

`WithMaxAdjustmentAttempts(N)` permits `N` additional selections in that local chain. The default is `5`; nonpositive values permit none. Exhaustion returns an error with the closed batch and feedback retained. It never executes an unapproved candidate.

Ordinary legacy adjustment keeps its existing per-tool semantics, including legacy streaming. The legacy path does not enforce `WithMaxAdjustmentAttempts`. Do not rely on this option as a universal loop bound.

### Cancellation and streaming boundary

The rich carrier is non-streaming. Cogito prefers `MessageLLM` over the legacy streaming capability for rich completion dispatch. This is not a native provider streaming API.

Once a successful selection reaches rich policy preflight, Cogito retains the original assistant and closes each valid call on interruption. A configured `WithStreamCallback` does not change this boundary for a selected provider envelope: `MessageLLM` selection still uses the non-streaming path.

Before that initial policy boundary, cancellation after receiving a response or selection of an unavailable tool can discard the pending assistant. Already-closed correction exchanges remain in history. This is not a guarantee that every received pending batch persists.

An adjustment chain has a stronger boundary: a valid replacement batch with unique nonempty IDs is closed without execution even if cancellation or unavailable-tool selection prevents it from reaching policy preflight. Invalid IDs fail closed without fabricated correlation IDs.

Legacy selections cancelled after lifecycle admission receive paired protocol replies, even for calls never dispatched. Cancellation before admission can still discard a pending selection; already-closed correction exchanges remain durable.

## Compaction and persistence

Compaction uses portable role/content and tool-result text for summaries. It does not serialize provider blocks, signatures, redacted data, or private reasoning into the summary request. The summary becomes new portable content, not reconstructed provider history.

Retained envelopes remain complete and independently owned. Compaction moves the retention boundary to keep assistant/tool-result groups together. It rejects incomplete, orphaned, or ambiguous exchanges instead of splitting them. On validation, summary, or cancellation failure, it returns the original fragment rather than a partial replacement.

For durable sessions, save the envelopes from `Conversation()`, not `GetMessages()`. Restore them with `WithConversation` after validating your storage schema and provenance. Preserve the distinction between absent history and intentional empty history. Protect stored provider data as sensitive session data.

Your application owns persistence, migration, access control, and corruption handling. Your adapter owns route binding and replay validation before network access. Neither portable exports nor old sessions without provider blocks can recover lost signatures or opaque state.

Render portable content for display. Never dump entire envelopes into templates, logs, or user interfaces. Portable content can itself contain sensitive user or tool data; this API is not a general secret scrubber. Adapters must exclude private provider data from portable display fields.

See the [README](../README.md) for the existing tool and agent APIs.

### Tool lifecycle observers

`WithToolLifecycleCallback` observes execution independently of approval. Each
admitted call emits `queued` once, `running` once at actual dispatch (not per
retry), and `terminal` once. Terminal outcomes are `completed`, `failed`,
`denied`, `skipped`, `cancelled`, or `superseded`. Invalid selections (unavailable
tools or duplicate nonempty IDs) fail before admission, approval, or execution.
Supplied IDs remain stable. `Index` is zero-based within the admitted batch;
`AgentID` is empty at the root and identifies inherited child executions.

Terminal events arrive when individual calls finish, including parallel calls.
`WithToolCallResultCallback` remains ordered by selection and runs after the
batch joins. Avoid processing the same result through both callbacks. Pending
cancelled calls receive protocol replies but no execution history or legacy
result callback. Denial terminates the denied call and cancels pending siblings.
Adjustment preserves explicit skips, supersedes pending calls, and admits a
replacement batch requiring fresh approval. Rich sink calls are skipped without
execution; existing sink parking and resumption behavior remains unchanged.

Invocations sharing a lifecycle option are serialized, including inherited child
agents. Callbacks must return promptly: do not panic, re-enter execution, or wait
synchronously for another tool or agent. Treat payloads as read-only, not as
independently deep-owned snapshots. Synchronize observer state read outside the
callback. Execution arguments remain detached from provider history; policy
edits do not rewrite `OriginalCall`, native envelopes, or numeric lexemes.

Tools may implement `ContextToolDefinitionInterface.ExecuteContext`; typed
runners may additionally implement `ContextTool[T].RunContext`. Contextual
execution takes precedence. `ToolDefinition.Execute` delegates with a background
context. Cancellation and deadline errors are not retried; other failures use
at least one attempt, even when the configured maximum is zero. Cogito waits for
all dispatched tools to return. A legacy tool that succeeds after cancellation
still emits `completed`, then the enclosing operation returns the context error.
