#!/usr/bin/env python
"""Anthropic-protocol "thinking fixup" proxy for relay providers.

Sits on 127.0.0.1, forwards /v1/messages to a chosen upstream, and fixes three
relay-side problems that make deepseek/glm models look like they never think:

  1. redacted_thinking with readable text.
     Some relays stream readable reasoning inside a `redacted_thinking` SSE
     block.  Claude Code's schema for that block is `{type, data}` only, so
     the `thinking_delta` text is discarded.
     -> rewrite `content_block_start` redacted_thinking into `thinking`
        (signature borrowed from the relay's opaque id).

  2. reasoning dropped on tool-result turns.
     Measured on three independent Anthropic-compatible relays (deepseek-v4.1-flash
     and glm-5.3-flash): when the last user turn is a
     bare `tool_result`, upstream `reasoning_tokens` is 0 - the model really
     skips reasoning.  Appending a short instruction restores it.
     -> opt-in: append that nudge when the tail is a pure tool_result turn.

     The nudge goes into the TOP-LEVEL `system` field, never into `messages`.
     Anthropic's messages array allows only user/assistant; a `role:"system"`
     entry there is invalid, and relays that translate to OpenAI flatten it
     into a user turn - the model then reads it as the human talking, which
     showed up as the nudge text arriving as a user message turn after turn.
     A top-level system block becomes an OpenAI system message, which is what
     an operator instruction is supposed to be.
     Set OD_PROXY_INJECT_TARGET=messages for the old (broken) shape.

  3. empty reasoning even after the nudge (shell turns: thinking_chars<=1,
     reasoning_tokens<=1).  Measured ~2/10 injected turns on one relay.
     -> when a nudge was injected for this request, buffer the response up to
        the point where the thinking block is decided; if it came back empty,
        retry the upstream once with a stronger instruction.  Only injected
        (tool-result) turns are buffered, so ordinary turns stream untouched.

Upstreams: each instance serves ONE upstream, set by OD_PROXY_UPSTREAM.
To cover several relays run one instance per port (see od_think_proxy.bat), or
use OD_PROXY_ROUTES for a small port->upstream table (see below).

Env:
  OD_PROXY_PORT            default 18997
  OD_PROXY_UPSTREAM        upstream host, scheme optional (required in practice)
  OD_PROXY_LOG             default <this dir>/.cache/od_proxy.jsonl ("0" disables)
  OD_PROXY_REFLECT         default 1: append the nudge on tool-result turns.
                           Set 0 to disable, or set custom text.
  OD_PROXY_REFLECT_MODE    "tools" (default) / "always"
  OD_PROXY_INJECT_ROLE     "system" (default) / "user"
  OD_PROXY_EMPTY_RETRY     default 1: retry once when an injected turn came back
                           with empty reasoning
  OD_PROXY_EMPTY_MIN       default 10: thinking chars at or below this counts
                           as empty
  OD_PROXY_EMPTY_RETRIES   default 3: retries per request
  OD_PROXY_RETRY_MAX_INPUT default 60000: retry only when the prompt is this
                           small (chars/3 estimate). A retry re-sends the whole
                           prompt; on a 500k-token turn that is real money.
  OD_PROXY_EARLY_RELEASE   default 3: thinking chars that prove the block is
                           non-empty, so the response can be released early
  OD_PROXY_ABORT_TEXT_MIN  default 20: text chars that, together with no
                           thinking, prove the model is answering blind. The
                           response is then dropped and retried. This path is
                           NOT limited by OD_PROXY_RETRY_MAX_INPUT: the prompt
                           is a cache hit and only a few dozen output tokens
                           are thrown away.
  OD_PROXY_DECIDE_TIMEOUT  default 180: seconds to wait for a response to
                           finish before passing it through unjudged
  OD_PROXY_ROUTES          optional "port=host,port=host" table
"""
import http.server
import http.client
import json
import os
import ssl
import sys
import threading
import time

PORT = int(os.environ.get('OD_PROXY_PORT', '18997'))
UPSTREAM = os.environ.get('OD_PROXY_UPSTREAM', '')
LOG_PATH = os.environ.get(
    'OD_PROXY_LOG',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache', 'od_proxy.jsonl'))

DEFAULT_NUDGE = '先分析工具输出，再决定下一步要做什么。'
_refl_env = os.environ.get('OD_PROXY_REFLECT', '1').strip()
if _refl_env in ('0', '', 'off', 'false', 'no'):
    REFLECT_TEXT = None
elif _refl_env in ('1', 'on', 'true', 'yes'):
    REFLECT_TEXT = DEFAULT_NUDGE
else:
    REFLECT_TEXT = _refl_env
REFLECT_MODE = os.environ.get('OD_PROXY_REFLECT_MODE', 'tools')
INJECT_ROLE = os.environ.get('OD_PROXY_INJECT_ROLE', 'system').strip().lower()
if INJECT_ROLE not in ('system', 'user'):
    INJECT_ROLE = 'system'
INJECT_TARGET = os.environ.get('OD_PROXY_INJECT_TARGET', 'system').strip().lower()
if INJECT_TARGET not in ('system', 'messages'):
    INJECT_TARGET = 'system'
EMPTY_RETRY = os.environ.get('OD_PROXY_EMPTY_RETRY', '1').strip().lower() not in (
    '0', '', 'off', 'false', 'no')
try:
    EMPTY_MIN = int(os.environ.get('OD_PROXY_EMPTY_MIN', '10'))
except ValueError:
    EMPTY_MIN = 10
try:
    EMPTY_RETRIES = int(os.environ.get('OD_PROXY_EMPTY_RETRIES', '3'))
except ValueError:
    EMPTY_RETRIES = 3
# Cost gate: a retry re-sends the whole prompt. On a ~500k-token conversation
# that is real money for one shot at a maybe. Bill a retry only when the input
# is small enough that the extra call is cheap.
try:
    EMPTY_RETRY_MAX_INPUT = int(os.environ.get('OD_PROXY_RETRY_MAX_INPUT', '60000'))
except ValueError:
    EMPTY_RETRY_MAX_INPUT = 60000
try:
    EARLY_RELEASE = int(os.environ.get('OD_PROXY_EARLY_RELEASE', '3'))
except ValueError:
    EARLY_RELEASE = 3
# Abort-and-retry trigger. The model is answering (text is streaming) while no
# thinking has arrived. Both conditions together prove it is answering WITHOUT
# thinking - a lone small thinking count could also just mean the response is
# slow and has not reached the reasoning yet.
try:
    ABORT_TEXT_MIN = int(os.environ.get('OD_PROXY_ABORT_TEXT_MIN', '20'))
except ValueError:
    ABORT_TEXT_MIN = 20
try:
    DECIDE_TIMEOUT = float(os.environ.get('OD_PROXY_DECIDE_TIMEOUT', '180'))
except ValueError:
    DECIDE_TIMEOUT = 180.0

# Repeating the instruction verbatim measurably raises reasoning depth
# (2026-11-24, one relay, n=10 per arm): one pass ~156 median reasoning
# tokens, two passes ~245, no nudge 7/10 with a third of calls returning a
# 1-token shell. Verbatim repetition beats paraphrasing.
_STRONG_ONCE = ('先完整推理工具返回了什么、下一步该做什么，把推理写进 thinking，'
                '再输出回答或调用工具。不要跳过推理。')
STRONG_NUDGE = _STRONG_ONCE * 2
# Raw SSE capture. Set OD_PROXY_RAW_DUMP to a port ("18994") or "all".
# Every upstream SSE frame is written verbatim to <dir>/raw-<port>-<time>.jsonl
# so a turn that looks empty in the stats can be inspected byte for byte.
RAW_DUMP = os.environ.get('OD_PROXY_RAW_DUMP', '').strip()
RAW_DUMP_DIR = os.environ.get('OD_PROXY_RAW_DIR') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '.cache', 'raw')
NUDGE_MARK = '先分析工具输出'

STRONG_MARK = '先完整推理工具返回了什么'

_lock = threading.Lock()


def norm_upstream(u):
    u = (u or '').strip()
    if u.startswith('http://'):
        return 'http:' + u[len('http://'):]
    if u.startswith('https://'):
        return 'https:' + u[len('https://'):]
    return 'https:' + u


def parse_routes(raw):
    routes = []
    for part in (raw or '').split(','):
        part = part.strip()
        if not part or '=' not in part:
            continue
        port, host = part.split('=', 1)
        try:
            routes.append((int(port.strip()), norm_upstream(host)))
        except ValueError:
            continue
    return routes


def log(rec):
    if LOG_PATH in ('0', '', 'off'):
        return
    try:
        d = os.path.dirname(LOG_PATH)
        if d:
            os.makedirs(d, exist_ok=True)
        with _lock:
            if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 5 * 1024 * 1024:
                os.replace(LOG_PATH, LOG_PATH + '.1')
            with open(LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    except Exception:
        pass


def rewrite(j):
    """redacted_thinking -> thinking, preserving the streamed text."""
    if j.get('type') == 'content_block_start':
        cb = j.get('content_block') or {}
        if cb.get('type') == 'redacted_thinking':
            sig = cb.get('data') or cb.get('signature') or 'od-proxy-sig'
            cb['type'] = 'thinking'
            cb['thinking'] = cb.get('thinking') or ''
            cb['signature'] = sig
            cb.pop('data', None)
            return True
    return False


def _usage_reasoning_tokens(u):
    if not isinstance(u, dict):
        return None
    bu = u.get('billing_usage') or {}
    ou = (bu.get('openai_usage') or {}) if isinstance(bu, dict) else {}
    rt = ((ou.get('completion_tokens_details') or {}).get('reasoning_tokens')
          if isinstance(ou, dict) else None)
    if rt is None:
        rt = (u.get('output_tokens_details') or {}).get('reasoning_tokens')
    return rt


def sse_stats(j, stats):
    t = j.get('type')
    if t == 'content_block_start':
        bt = (j.get('content_block') or {}).get('type')
        stats['blocks'].append(bt)
        if bt == 'tool_use':
            stats['tool_uses'] = stats.get('tool_uses', 0) + 1
    elif t == 'content_block_delta':
        d = j.get('delta') or {}
        if d.get('type') == 'thinking_delta':
            stats['thinking_chars'] += len(d.get('thinking') or '')
        elif d.get('type') == 'text_delta':
            stats['text_chars'] += len(d.get('text') or '')
    elif t == 'message_delta':
        rt = _usage_reasoning_tokens(j.get('usage'))
        if rt is not None:
            stats['reasoning_tokens'] = rt


def _content_types(msg):
    c = msg.get('content')
    if isinstance(c, str):
        return ['text']
    if isinstance(c, list):
        return [b.get('type') for b in c if isinstance(b, dict)]
    return []


def raw_dump(port, text):
    """Append one upstream frame to the raw capture file, if enabled."""
    if not RAW_DUMP:
        return
    if RAW_DUMP not in ('all', str(port)):
        return
    try:
        os.makedirs(RAW_DUMP_DIR, exist_ok=True)
        fn = os.path.join(RAW_DUMP_DIR, 'raw-%s-%s.jsonl' % (
            port, time.strftime('%Y%m%d-%H%M')))
        with _lock:
            with open(fn, 'a', encoding='utf-8') as f:
                f.write(text + chr(10))
    except Exception:
        pass


def _rough_input_tokens(req):
    """Rough input size in tokens (chars/3), enough for a cost gate."""
    try:
        return len(json.dumps(req, ensure_ascii=False)) // 3
    except Exception:
        return 0


def _retry_allowed(req):
    """Retry only when re-sending this prompt is cheap.

    A retry re-sends the entire prompt. On a 500k-token conversation one extra
    attempt costs real money for a single shot at a maybe - and the measured
    case is that it usually comes back empty anyway.
    """
    if not EMPTY_RETRY:
        return False
    return _rough_input_tokens(req) <= EMPTY_RETRY_MAX_INPUT


def _is_pure_tool_result_tail(req):
    msgs = req.get('messages') or []
    for m in reversed(msgs):
        if not isinstance(m, dict):
            continue
        if m.get('role') != 'user':
            continue
        types = _content_types(m)
        if not types:
            continue
        if 'tool_result' not in types:
            return False
        return all(t == 'tool_result' for t in types)
    return False


def _texts_of(msg):
    c = msg.get('content')
    if isinstance(c, str):
        return [c]
    out = []
    if isinstance(c, list):
        for b in c:
            if isinstance(b, dict) and b.get('type') == 'text':
                out.append(b.get('text') or '')
    return out


def _system_texts(req):
    s = req.get('system')
    if isinstance(s, str):
        return [s]
    out = []
    if isinstance(s, list):
        for b in s:
            if isinstance(b, dict) and b.get('type') == 'text':
                out.append(b.get('text') or '')
    return out


def _already_nudged(req):
    if any(NUDGE_MARK in t or STRONG_MARK in t for t in _system_texts(req)):
        return True
    msgs = req.get('messages') or []
    for m in reversed(msgs[-3:]):
        if not isinstance(m, dict):
            continue
        if m.get('role') not in ('user', 'system'):
            continue
        if any(NUDGE_MARK in t or STRONG_MARK in t for t in _texts_of(m)):
            return True
    return False


def _append_system_block(req, text):
    """Append a text block to the top-level `system` field.

    Appended AFTER any existing block, so Claude Code's cache_control
    breakpoints stay untouched: the cached prefix is unchanged.
    """
    s = req.get('system')
    if s is None:
        req['system'] = [{"type": "text", "text": text}]
        return True
    if isinstance(s, str):
        req['system'] = [{"type": "text", "text": s},
                         {"type": "text", "text": text}]
        return True
    if isinstance(s, list):
        req['system'] = list(s) + [{"type": "text", "text": text}]
        return True
    return False


def maybe_inject_reflection(req):
    if not REFLECT_TEXT or not isinstance(req, dict):
        return False
    if REFLECT_MODE == 'tools' and not req.get('tools'):
        return False
    if not _is_pure_tool_result_tail(req):
        return False
    if _already_nudged(req):
        return False
    if INJECT_TARGET == 'system':
        return _append_system_block(req, REFLECT_TEXT)
    req.setdefault('messages', []).append(
        {"role": INJECT_ROLE, "content": [{"type": "text", "text": REFLECT_TEXT}]})
    return True


def strengthen(body_bytes):
    """Replace the injected nudge with the stronger one. Returns new body bytes or None."""
    try:
        req = json.loads(body_bytes)
    except Exception:
        return None
    s = req.get('system')
    if isinstance(s, str):
        if NUDGE_MARK in s or STRONG_MARK in s:
            req['system'] = STRONG_NUDGE
            return json.dumps(req, ensure_ascii=False).encode()
    elif isinstance(s, list):
        for b in reversed(s):
            if isinstance(b, dict) and b.get('type') == 'text' and (
                    NUDGE_MARK in (b.get('text') or '') or STRONG_MARK in (b.get('text') or '')):
                b['text'] = STRONG_NUDGE
                return json.dumps(req, ensure_ascii=False).encode()
    msgs = req.get('messages') or []
    for m in reversed(msgs):
        if not isinstance(m, dict) or m.get('role') not in ('user', 'system'):
            continue
        c = m.get('content')
        if isinstance(c, str) and (NUDGE_MARK in c or STRONG_MARK in c):
            m['content'] = STRONG_NUDGE
            return json.dumps(req, ensure_ascii=False).encode()
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get('type') == 'text' and (
                        NUDGE_MARK in (b.get('text') or '') or STRONG_MARK in (b.get('text') or '')):
                    b['text'] = STRONG_NUDGE
                    return json.dumps(req, ensure_ascii=False).encode()
    return None


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'od-think-proxy/1.5'
    upstream = UPSTREAM

    def log_message(self, *args):
        pass

    def _write_chunk(self, data):
        self.wfile.write(f'{len(data):X}\r\n'.encode() + data + b'\r\n')

    # ---------- admin ----------
    def do_GET(self):
        if self.path.startswith('/__admin/upstream'):
            from urllib.parse import urlparse, parse_qs
            q = parse_qs(urlparse(self.path).query)
            if 'set' in q and q['set'] and q['set'][0].strip():
                new = norm_upstream(q['set'][0].strip())
                type(self).upstream = new
                log({'t': time.strftime('%Y-%m-%dT%H:%M:%S'), 'port': self.server.server_port,
                     'kind': 'admin_switch', 'upstream': new})
            else:
                new = self.upstream
            payload = json.dumps({'port': self.server.server_port,
                                  'upstream': new,
                                  'inject_role': INJECT_ROLE,
                                  'inject_target': INJECT_TARGET,
                                  'empty_retry': EMPTY_RETRY,
                                  'empty_min': EMPTY_MIN,
                                  'retry_max_input': EMPTY_RETRY_MAX_INPUT,
                                  'early_release': EARLY_RELEASE,
                                  'abort_text_min': ABORT_TEXT_MIN,
                                  'decide_timeout': DECIDE_TIMEOUT,
                                  'version': self.server_version}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = True
            return
        self.send_response(404)
        self.send_header('Content-Length', '0')
        self.end_headers()
        self.close_connection = True

    # ---------- forwarding ----------
    def _forward(self, body, fwd, up):
        """Open a request upstream. Returns (conn, resp). Raises on connect errors."""
        if up.startswith('http:'):
            host = up[len('http:'):].lstrip('/')
            conn = http.client.HTTPConnection(host, timeout=600)
        else:
            conn = http.client.HTTPSConnection(
                up[len('https:'):].lstrip('/'), timeout=600,
                context=ssl.create_default_context())
        h = dict(fwd)
        h['content-length'] = str(len(body))
        conn.request('POST', self.path, body=body, headers=h)
        return conn, conn.getresponse()

    def _send_error(self, e, up):
        payload = json.dumps({'type': 'error',
                              'error': {'type': 'proxy_error', 'message': repr(e)}}).encode()
        self.send_response(599)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True
        log({'t': time.strftime('%Y-%m-%dT%H:%M:%S'), 'port': self.server.server_port,
             'upstream': up, 'kind': 'connect_error', 'err': repr(e)[:200]})

    def _send_buffered_json(self, status, headers, data, injected, retried, up, t0, stats):
        self.send_response(status)
        for k, v in headers:
            if k.lower() in ('transfer-encoding', 'content-length', 'connection'):
                continue
            self.send_header(k, v)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
        self.close_connection = True
        log({'t': time.strftime('%Y-%m-%dT%H:%M:%S'), 'port': self.server.server_port,
             'upstream': up, 'path': self.path, 'status': status,
             'kind': 'nonstream', 'injected': injected, 'retried': retried,
             'ms': int((time.time() - t0) * 1000), **stats})

    def _json_thinking_chars(self, data):
        try:
            j = json.loads(data)
        except Exception:
            return None, None
        chars = 0
        rt = _usage_reasoning_tokens(j.get('usage'))
        for b in j.get('content') or []:
            if isinstance(b, dict) and b.get('type') == 'thinking':
                chars += len(b.get('thinking') or '')
        return chars, rt

    def do_POST(self):
        try:
            n = int(self.headers.get('content-length', 0) or 0)
        except ValueError:
            n = 0
        body = self.rfile.read(n)
        try:
            req = json.loads(body)
        except Exception:
            req = None

        injected = False
        if isinstance(req, dict):
            try:
                injected = maybe_inject_reflection(req)
            except Exception:
                injected = False
            if injected:
                body = json.dumps(req, ensure_ascii=False).encode()

        fwd = {k: v for k, v in self.headers.items()
               if k.lower() not in ('host', 'content-length', 'connection',
                                    'accept-encoding', 'transfer-encoding')}
        if not any(k.lower() == 'user-agent' for k in fwd):
            fwd['user-agent'] = 'claude-cli/2.1.303 (external, cli)'

        up = self.upstream
        t0 = time.time()
        attempts = 0
        while True:
            try:
                conn, resp = self._forward(body, fwd, up)
            except Exception as e:
                self._send_error(e, up)
                return

            ctype = resp.getheader('content-type', '')
            if 'text/event-stream' not in ctype:
                data = resp.read()
                try:
                    conn.close()
                except Exception:
                    pass
                stats = {}
                chars, rt = self._json_thinking_chars(data)
                stats['thinking_chars'] = chars
                stats['reasoning_tokens'] = rt
                empty = (chars is not None and chars <= EMPTY_MIN
                         and (rt is None or rt <= 2))
                if injected and EMPTY_RETRY and empty and resp.status == 200 \
                        and attempts < EMPTY_RETRIES and _retry_allowed(req):
                    new_body = strengthen(body)
                    if new_body:
                        attempts += 1
                        body = new_body
                        continue
                self._send_buffered_json(resp.status, resp.getheaders(), data,
                                         injected, attempts > 0, up, t0, stats)
                return

            # ---- SSE ----
            if not (injected and EMPTY_RETRY):
                self._stream_plain(resp, conn, req, injected, up, t0)
                return

            buffered, stats, decided, reason = self._read_until_decided(resp)
            stats['decided_by'] = reason

            # Reasoning is missing and the model already committed to an
            # answer or a tool call. Either way it acted blind, and retrying
            # costs only the prompt - which is a cache hit - plus whatever few
            # output tokens were produced. These two bypass the size gate on
            # purpose: an unthinking action in a 500k-token session is exactly
            # the expensive mistake worth redoing.
            acting_without_thinking = reason in ('no_thinking_answering',
                                                 'no_thinking_tool')
            # A response that ran to completion with no reasoning anywhere:
            # nothing was discarded, so this one honours the size gate.
            completed_empty = (reason == 'complete'
                               and stats['thinking_chars'] <= EMPTY_MIN
                               and (stats.get('reasoning_tokens') is None
                                    or stats['reasoning_tokens'] <= 2))

            allowed = (EMPTY_RETRY and attempts < EMPTY_RETRIES and
                       (acting_without_thinking or _retry_allowed(req)))
            if (acting_without_thinking or completed_empty) and allowed:
                new_body = strengthen(body)
                if new_body:
                    try:
                        conn.close()
                    except Exception:
                        pass
                    attempts += 1
                    body = new_body
                    continue

            self._stream_buffered(buffered, resp, up, t0, stats, injected,
                                  attempts > 0, decided)
            return

    def _read_until_decided(self, resp):
        # returns (buffered, stats, decided, reason)
        """Buffer SSE lines until thinking emptiness is *safely* known.

        Only two states justify a conclusion:
          - enough thinking text arrived (non-empty; release immediately), or
          - message_stop arrived (response complete, stats trustworthy).

        Anything earlier is unsafe: interleaved thinking allows several
        thinking blocks, so the first empty one proves nothing, and aborting
        mid-generation bills the whole prompt at full price (measured: Y0.053
        with cache_read=0, against Y0.002 on the warm retry).

        If the upstream stalls without finishing, stop waiting after
        DECIDE_TIMEOUT seconds and pass the buffered content through
        unjudged. A slow upstream must not become a lost response.
        """
        stats = {'blocks': [], 'thinking_chars': 0, 'text_chars': 0,
                 'rewritten': 0, 'reasoning_tokens': None, 'tool_uses': 0}
        buffered = []
        decided = False
        reason = "eof"
        deadline = time.time() + DECIDE_TIMEOUT
        while True:
            if DECIDE_TIMEOUT > 0 and time.time() > deadline:
                stats['timed_out'] = True
                break
            line = resp.readline()
            raw_dump(self.server.server_port, line.decode('utf-8', 'replace').rstrip())
            if not line:
                break
            out = line
            if line.startswith(b'data:'):
                raw = line[5:].strip()
                if raw and raw != b'[DONE]':
                    try:
                        j = json.loads(raw)
                    except Exception:
                        j = None
                    if isinstance(j, dict):
                        sse_stats(j, stats)
                        if rewrite(j):
                            stats['rewritten'] += 1
                            out = b'data: ' + json.dumps(
                                j, ensure_ascii=False).encode() + b'\n'
                        t = j.get('type')
                        # Only two conclusions are safe.
                        # 1) Enough thinking text arrived -> non-empty, release now
                        #    so a healthy turn keeps streaming at full speed.
                        # 2) message_stop -> the response is complete, the stats
                        #    are trustworthy. Nothing earlier: interleaved
                        #    thinking allows several thinking blocks, so the
                        #    first empty one proves nothing, and aborting mid
                        #    generation bills the whole prompt at full price.
                        # Release as soon as the block is provably non-empty.
                        # A low bar (default 3 chars) keeps first-byte latency
                        # down on slow upstreams while still disqualifying the
                        # 0-1 char shells we actually see.
                        if stats['thinking_chars'] > EARLY_RELEASE:
                            decided = True
                            reason = "thinking_ok"
                        elif (ABORT_TEXT_MIN > 0
                              and stats['text_chars'] >= ABORT_TEXT_MIN):
                            # No thinking yet AND the model is already emitting
                            # its answer. That pair locks it down: it is
                            # answering without reasoning, not merely slow.
                            decided = True
                            reason = "no_thinking_answering"
                        elif stats.get('tool_uses', 0) > 0:
                            # A tool call with no reasoning is the same failure
                            # as an answer with no reasoning - the model acted
                            # blind. Nothing was written to discard, so aborting
                            # costs only the prompt (a cache hit) - skip the
                            # text threshold and the size gate entirely.
                            decided = True
                            reason = "no_thinking_tool"
                        elif t == 'message_stop':
                            decided = True
                            reason = "complete"
            buffered.append(out)
            if decided:
                break
        return buffered, stats, decided, reason

    def _stream_plain(self, resp, conn, req, injected, up, t0):
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() in ('transfer-encoding', 'content-length', 'connection'):
                continue
            self.send_header(k, v)
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()

        stats = {'blocks': [], 'thinking_chars': 0, 'text_chars': 0, 'rewritten': 0,
                 'reasoning_tokens': None, 'tool_uses': 0}
        try:
            while True:
                line = resp.readline()
                raw_dump(self.server.server_port, line.decode('utf-8', 'replace').rstrip())
                if not line:
                    break
                out = line
                if line.startswith(b'data:'):
                    raw = line[5:].strip()
                    if raw and raw != b'[DONE]':
                        try:
                            j = json.loads(raw)
                        except Exception:
                            j = None
                        if isinstance(j, dict):
                            sse_stats(j, stats)
                            if rewrite(j):
                                stats['rewritten'] += 1
                                out = b'data: ' + json.dumps(
                                    j, ensure_ascii=False).encode() + b'\n'
                self._write_chunk(out)
            self.wfile.write(b'0\r\n\r\n')
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True
            try:
                conn.close()
            except Exception:
                pass
        log({'t': time.strftime('%Y-%m-%dT%H:%M:%S'), 'port': self.server.server_port,
             'upstream': up, 'path': self.path, 'status': resp.status, 'kind': 'sse',
             'injected': injected, 'retried': False,
             'ms': int((time.time() - t0) * 1000),
             'messages': len((req or {}).get('messages') or []),
             'stats': stats})

    def _stream_buffered(self, buffered, resp, up, t0, stats, injected, retried, decided):
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() in ('transfer-encoding', 'content-length', 'connection'):
                continue
            self.send_header(k, v)
        self.send_header('Transfer-Encoding', 'chunked')
        self.end_headers()
        try:
            for out in buffered:
                self._write_chunk(out)
            if decided:
                while True:
                    line = resp.readline()
                    raw_dump(self.server.server_port, line.decode('utf-8', 'replace').rstrip())
                    if not line:
                        break
                    out = line
                    if line.startswith(b'data:'):
                        raw = line[5:].strip()
                        if raw and raw != b'[DONE]':
                            try:
                                j = json.loads(raw)
                            except Exception:
                                j = None
                            if isinstance(j, dict):
                                sse_stats(j, stats)
                                if rewrite(j):
                                    stats['rewritten'] += 1
                                    out = b'data: ' + json.dumps(
                                        j, ensure_ascii=False).encode() + b'\n'
                    self._write_chunk(out)
            self.wfile.write(b'0\r\n\r\n')
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.close_connection = True
        log({'t': time.strftime('%Y-%m-%dT%H:%M:%S'), 'port': self.server.server_port,
             'upstream': up, 'path': self.path, 'status': resp.status, 'kind': 'sse',
             'injected': injected, 'retried': retried,
             'ms': int((time.time() - t0) * 1000), 'stats': stats})


def make_server(port, upstream):
    h = type('BoundHandler', (Handler,), {'upstream': upstream})
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', port), h)
    srv.daemon_threads = True
    return srv


def main():
    routes = parse_routes(os.environ.get('OD_PROXY_ROUTES', ''))
    if not routes:
        routes = [(PORT, norm_upstream(UPSTREAM))]
    servers = []
    for port, upstream in routes:
        srv = make_server(port, upstream)
        servers.append(srv)
        print(f'od-think-proxy listening on 127.0.0.1:{port} -> {upstream} '
              f'(reflect={"on" if REFLECT_TEXT else "off"}, mode={REFLECT_MODE}, '
              f'role={INJECT_ROLE}, empty_retry={EMPTY_RETRY}/{EMPTY_RETRIES})', flush=True)
    for srv in servers[1:]:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for srv in servers:
            try:
                srv.server_close()
            except Exception:
                pass


if __name__ == '__main__':
    main()
