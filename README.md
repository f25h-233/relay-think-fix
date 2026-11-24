# relay-think-fix

A local proxy that fixes **silently dropped reasoning** on Anthropic-compatible
relay providers, so Claude Code (and other Anthropic clients) actually get
thinking blocks back when the upstream model did think.

If you pay for a relay that resells DeepSeek / GLM / Qwen behind an
Anthropic-shaped API, and your client keeps showing "thinking" with nothing in
it, or no thinking at all on tool-calling turns, this is probably why.

## The problem

Reasoning works when you test the relay by hand, then disappears in real
agentic use. Three distinct failures were measured on real relays; each has its
own cause and its own fix.

### 1. Reasoning vanishes on tool-result turns

Reproduced on three independent relays with two model families
(deepseek-v4.1-flash, glm-5.3-flash).

A request whose last user turn is a bare `tool_result` comes back with
`reasoning_tokens: 0` from upstream. The model really does skip reasoning -
this is not the client discarding it. The relay's own billing receipt shows it:

```
billing_usage.openai_usage.completion_tokens_details.reasoning_tokens = 0
```

Replace the `tool_result` block with a plain text block and reasoning returns
immediately (370 / 236 / 460 characters across runs).

Because every agentic loop is mostly tool-result turns, this is where a session
loses its reasoning: measured 288 consecutive turns with zero reasoning in one
real session.

**Fix:** append a short instruction when the last user turn is a pure
`tool_result`. It must go into the top-level `system` field, not `messages` -
a `role:"system"` entry inside `messages` is invalid Anthropic shape, and relays
that translate to OpenAI flatten it into a user turn, so the model reads it as
the human talking (observed: the nudge text kept arriving as a user message).

### 2. Empty reasoning even after the nudge

About 2 in 10 nudged turns still come back as a shell: `thinking_chars <= 1`
and `reasoning_tokens <= 1`.

**Fix:** buffer the response until the thinking block is decided; if it came
back empty, retry with a stronger instruction.

Repeating that instruction **verbatim** works better than paraphrasing it.
Measured on an OpenAI-protocol relay, 10 samples per arm:

| approach | non-empty | median reasoning_tokens |
|---|---|---|
| no nudge | 7/10 | 114 |
| nudge, once | 10/10 | 156 |
| **nudge, repeated twice** | **10/10** | **245** |
| "must think" x3 (different wording) | 10/10 | 154 |

Verbatim repetition of the full instruction beats escalating the language.

### 3. redacted_thinking with readable text

Some relays stream readable reasoning inside a `redacted_thinking` block.
Claude Code's schema for that block is `{type, data}` only, so the
`thinking_delta` text is discarded.

**Fix:** rewrite `content_block_start` redacted_thinking into thinking,
borrowing the relay's opaque id as the signature.

## Where the money goes

The naive implementation of fix 2 - "abort as soon as it looks empty, then
retry" - is expensive in a way that is easy to miss.

Measured on a ~500k-token conversation:

```
aborted attempt   input=529,763  cache_read=0        cost Y0.053
the retry         input=514,126  cache_read=509,568  cost Y0.002
```

The aborted attempt was billed **at full input price with no cache hit** -
25x the cost of the retry that actually succeeded. Over one afternoon this
pattern accounted for 87% of the bill on that account.

Two things follow, and both are implemented here:

- **Decide late.** Only conclude from a state that is actually conclusive:
  thinking text has arrived, or the response is complete. Do not cut a stream
  halfway to "save time" - you pay for the whole prompt either way.
- **Gate retries by prompt size.** A retry re-sends everything. Above
  `OD_PROXY_RETRY_MAX_INPUT` (default 60k estimated tokens) retrying is not
  worth it. The exception is a turn that already committed to an answer or a
  tool call with no reasoning at all: nothing useful is discarded there, so it
  retries regardless of size.

Caveat worth stating plainly: relays differ in how they bill an aborted
stream. One relay was measured charging full price for a cut connection; that
result may not generalise. Measure your own relay before trusting the numbers.

## What this does not fix

- **Reasoning that is hidden rather than absent.** Some relays return a
  thinking block whose content is a single space character. That is reasoning
  that happened and was blanked out; the proxy cannot recover text the relay
  never sent. Tell the two apart by whether a thinking block exists at all.
- **Relays without an Anthropic endpoint at all.** A relay that only speaks
  OpenAI `/chat/completions` needs a protocol translator in front of it (for
  example CLIProxyAPI). This proxy must sit on the Anthropic side of that
  translator, because it inspects and rewrites Anthropic content blocks.
- **The model choosing not to think on simple turns.** If the upstream decides
  a turn is trivial, the retry will not invent reasoning.

## Install

Python 3.9+, no third-party dependencies.

```bash
git clone https://github.com/<you>/relay-think-fix
cd relay-think-fix
```

Point it at your relay and start it:

```bash
OD_PROXY_UPSTREAM=https://your-relay.example.com python3 relay_think_fix.py
```

Then set your client's base URL to the local listener:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:18997
```

For several relays, one listener each, so switching providers in your client
does not need a restart:

```bash
OD_PROXY_ROUTES=18997=relay-a.example.com,18996=relay-b.example.com \
  python3 relay_think_fix.py
```

On Windows, edit `start-proxy.bat` and double-click it.

### Runtime upstream switching

```bash
curl "http://127.0.0.1:18997/__admin/upstream"
curl "http://127.0.0.1:18997/__admin/upstream?set=relay-b.example.com"
```

No restart, no effect on other listeners, takes effect on the next request.

## Verifying it works on your relay

Watch the log. Every request gets one JSONL line with the numbers that matter:

```json
{"t":"...","port":18997,"kind":"sse","injected":true,"retried":true,
 "stats":{"thinking_chars":1432,"reasoning_tokens":530,
          "decided_by":"thinking_ok","blocks":["thinking","tool_use"]}}
```

`decided_by` tells you why the proxy stopped waiting:

| value | meaning |
|---|---|
| `thinking_ok` | reasoning arrived - normal |
| `no_thinking_tool` | tool call with no reasoning - aborted and retried |
| `no_thinking_answering` | text started with no reasoning - aborted and retried |
| `complete` | response finished with no reasoning (retried if the prompt was small) |
| `timed_out` | upstream stalled; passed through unjudged |

To inspect what the relay actually sent, dump raw frames:

```bash
OD_PROXY_RAW_DUMP=18997 OD_PROXY_RAW_DIR=./raw python3 relay_think_fix.py
```

Then look for these three shapes in `raw-*.jsonl`:

- `content_block_start` with type `thinking`, then `thinking_delta` frames →
  real reasoning.
- `content_block_start` with type `thinking`, then a delta containing a single
  space → reasoning happened and was blanked by the relay.
- no thinking block at all → the model really did not reason.

## Configuration

See `example.env` for every option. The ones that matter:

| variable | default | what it does |
|---|---|---|
| `OD_PROXY_REFLECT` | `1` | inject the nudge on tool-result turns |
| `OD_PROXY_INJECT_TARGET` | `system` | where the nudge goes (`system` is correct) |
| `OD_PROXY_EMPTY_RETRIES` | `3` | retries on empty reasoning |
| `OD_PROXY_RETRY_MAX_INPUT` | `60000` | skip retries above this prompt size |
| `OD_PROXY_ABORT_TEXT_MIN` | `20` | text chars that prove "answering blind" |
| `OD_PROXY_EARLY_RELEASE` | `3` | thinking chars that prove "not empty" |
| `OD_PROXY_DECIDE_TIMEOUT` | `180` | give up waiting on a stalled upstream |

## License

AGPL-3.0. See `LICENSE`.
