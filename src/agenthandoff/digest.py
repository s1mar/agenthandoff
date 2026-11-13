"""handoff_digest: answer a question about files with a local model, checked by code.

The steps:
1. prepare(): read the files, label every line (L00412: ...) and pack the lines
   into chunks. This happens before any model work, so a bad path or an input that
   is too big is reported at once.
2. run_prepared(): plant one "recall check" line in a copy of the largest chunk (_plant_probe).
3. Map: send each chunk to the model, which returns claims, each with an exact
   quote. Code finds each quote in its chunk (chunk.resolve) and works out
   file:line. Claims whose quote is not found are dropped and counted.
4. Merge (_merge): write the answer from the kept claims, citing them by ID only
   ([C3]). A big claim set is first narrowed: the model picks the relevant claims,
   a batch at a time. Claims from several files get a short note per file, then
   one call combines the notes. IDs that do not exist are replaced with [?].
5. Return a DigestResult. If the model never reported the planted line, the
   result says "RECALL CHECK: FAILED", telling the agent to check for itself.
api.run_digest() calls these two steps and logs every run, including failed ones,
to <home>/logs/digest.jsonl (log_record below).
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import chunk as ck
from .backend import Backend
from .config import CONTEXT_TOKENS, Settings
from .errors import InputTooLarge, NoInput, ThinkingOnly

MAX_CLAIMS_PER_CHUNK = 12    # 8 cut off a 9-item list in one chunk on the eval suite
MERGE_INPUT_TOKENS = 8000    # claims sent to the merge call; with its reply this fits the context
REPLY_ROOM = 2048            # a merge call's reply and prompt text; a small window gets fewer claims
MERGE_BATCH_TOKENS = 2500    # claims per "pick the relevant ones" call; more claims than this get picked first
MERGE_QUOTE_CHARS = 200      # each claim's quote, cut to this, goes to the merge with it
MAX_SHOWN_CLAIMS = 12        # claims printed in the result; the cited ones come first
MAX_SHOWN_PLACES = 3         # places listed for a quote found in several places
MAX_SPLITS = 2               # a cut-off chunk is halved and retried, at most this deep (4 parts)
MAX_INPUT_TOKENS = 200_000   # per digest; above this, narrow the paths
PROBE_WORDS = "recall check"

MAP_SYSTEM = f"""You read one chunk of files for a coding agent and pull out the facts that help answer its question.
Each line of the text starts with a label like L00042: that is not part of the line.
Rules:
- Every claim must have a "quote": text copied character for character from ONE line of the text, without the label. Pick the most specific part of the line, at least a few words.
- Never write line numbers or file names in place of a quote.
- If nothing in this chunk helps answer the question, return no claims.
- At most {MAX_CLAIMS_PER_CHUNK} claims, the ones that matter most for the question. Each claim is one sentence of at most 15 words."""

# The recall check is its own short call on one chunk with a planted line. It used to ride
# in the answer call ("always add the recall line as a claim"), but a small model drops an
# off-topic line when it has good claims to give: on 25 real digests the check
# failed 17 times while the right line was in the claims 24 times. Asked alone, it found the
# code 25 of 25 times, and 9 of 9 in full 6,000-token log chunks. It asks for the code, not
# the words, because a document can mention "recall check" itself (README, STATUS did).
# The example code RC-0000 is never planted (codes are 1000-9999), so a guess fails.
RECALL_SYSTEM = ('You check that a text was read to the end. Somewhere in it, one line contains a code '
                 'made of "RC-" and four digits, like RC-0000. Reply with that code only.')
RECALL_REMINDER = "\n\nNow reply with the RC- code from the text above."

MAP_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {
        "type": "object",
        "properties": {"claim": {"type": "string"}, "quote": {"type": "string"}},
        "required": ["claim", "quote"]}}},
    "required": ["claims"],
}

MERGE_SYSTEM = """You write the answer to a coding agent's question from claims about some files.
Each claim has a sentence written by a helper, and a quote: text code found in the files. The quote is true; the sentence may be wrong.
Rules:
- Use only the claims given. Do not add facts. Where a sentence and its quote disagree, believe the quote, and leave out a claim whose quote does not support it.
- Give exact values (numbers, names, paths) as the quotes show them.
- Cite claims by ID in square brackets, like [C3]. Every sentence that states a fact cites at least one claim.
- Answer as much of the question as the claims allow, even if they cover only part of it. Then, in one sentence, say what the claims leave out.
- Only if no claim is relevant at all, say so in one sentence.
- At most 200 words. No preamble."""

MERGE_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}

# Small models pick relevant items far better than they write a synthesis from
# hundreds of them, so a big claim set is first narrowed by picking, batch by batch.
SELECT_SYSTEM = """You help answer a coding agent's question. You get some claims found in files, each with an ID like C3, a sentence, and a quote from the files.
Pick every claim that helps answer the question: facts, values, names, causes, steps, or a description of what a file does. When unsure, pick it.
Reply with JSON: {"ids": ["C3", "C7"]}. If none helps, {"ids": []}."""

SELECT_SCHEMA = {"type": "object", "properties": {"ids": {"type": "array", "items": {"type": "string"}}},
                 "required": ["ids"]}

# Claims from several files are written up one file at a time, then combined. Asked
# for "a summary of each module" from claims about functions, the model refused;
# asked what one file contributes, it answers.
FILE_NOTE_SYSTEM = """You help answer a coding agent's question. You get the claims found in ONE file; other helpers handle the other files.
Each claim has a sentence written by a helper, and a quote: text code found in the file. The quote is true; the sentence may be wrong.
Write one or two sentences on what this file contributes to the answer: what it does or defines, with exact names and values.
Rules:
- Use only the claims given. Where a sentence and its quote disagree, believe the quote.
- Cite claims by ID in square brackets, like [C3], after each fact.
- Never say the claims do not answer the question; give what they show about this file.
- At most 60 words."""

COMBINE_SYSTEM = """You write the answer to a coding agent's question from notes, one note per file, with claims cited like [C3].
Rules:
- Combine the notes into one answer that covers every file that matters for the question. Keep each fact's [C..] citations exactly as written.
- If the question asks about each file or module, give one line per file.
- If it asks about a sequence of calls or events, put the steps in order.
- Do not add facts that are not in the notes.
- At most 300 words. No preamble."""

BRACKET_RE = re.compile(r"\[([^\[\]]*\bC\d+\b[^\[\]]*)\]", re.I)   # [C1], [C1, C2], [C1-C3], [c4]
CLAIM_ID_RE = re.compile(r"\bC(\d+)\b", re.I)


@dataclass
class Claim:
    id: str              # "C1", "C2", ... in the order found
    claim: str           # the model's sentence; may be wrong, the quote is what code checked
    quote: str
    hits: list[ck.Hit]

    def pointer(self) -> str:
        """'path:12' or 'path:12-14'; for a quote in several places the first
        MAX_SHOWN_PLACES and how many more (a log line repeated 300 times)."""
        shown = ", ".join(f"{h.path}:{h.start}" if h.start == h.end else f"{h.path}:{h.start}-{h.end}"
                          for h in self.hits[:MAX_SHOWN_PLACES])
        more = len(self.hits) - MAX_SHOWN_PLACES
        return f"{shown} (+{more} more)" if more > 0 else shown


@dataclass
class DigestResult:
    question: str
    answer: str
    claims: list[Claim]
    recall_ok: bool | None          # None when the recall check was turned off
    stats: dict = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    picked: set[str] = field(default_factory=set)    # claim IDs the merge judged relevant

    def shown_claims(self) -> tuple[list[Claim], int]:
        """Claims to print (cited first, then picked, then in order) and how many were left out."""
        cited = set(CLAIM_ID_RE.findall(self.answer))
        rank = lambda c: 0 if c.id[1:] in cited else 1 if c.id in self.picked else 2  # noqa: E731
        ordered = sorted(self.claims, key=rank)          # stable: file order within each rank
        shown = ordered[:max(MAX_SHOWN_CLAIMS, len(cited & {c.id[1:] for c in self.claims}))]
        return shown, len(self.claims) - len(shown)

    def render(self) -> str:
        """The text an agent reads. Kept short: that is the point of the tool."""
        out = [f"ANSWER: {self.answer}", ""]
        shown, omitted = self.shown_claims()
        if shown:
            out.append("CLAIMS (quotes found in the files by code; pointers are file:line):")
            for c in shown:
                out.append(f"[{c.id}] {c.pointer()}  {c.claim}")
                quote = c.quote if len(c.quote) <= 200 else c.quote[:200] + "..."
                out.append(f"      > {quote}")
            if omitted:
                out.append(f"({omitted} more claims not shown; use --json to see all)")
            out.append("")
        s = self.stats
        if s.get("unreadable_chunks"):
            out.append(f"WARNING: the model's reply for {s['unreadable_chunks']} of {s['chunks']} chunks "
                       "could not be used, so facts in those parts are missing.")
        if self.recall_ok is False:
            out.append("RECALL CHECK: FAILED. The model missed a planted line, so this digest may be "
                       "missing things. Check the files yourself for anything that matters.")
        elif self.recall_ok:
            out.append("RECALL CHECK: passed")
        out.append(f"STATS: {s.get('files', 0)} files, {s.get('chunks', 0)} chunks, about "
                   f"{s.get('est_input_tokens', 0):,} tokens read locally, {len(self.claims)} claims kept, "
                   f"{s.get('dropped', 0)} dropped, {s.get('seconds', 0):.1f} s, model {s.get('model')}")
        if s.get("dropped_reasons"):
            out.append("DROPPED: " + ", ".join(f"{k} x{v}" for k, v in sorted(s["dropped_reasons"].items())))
        if self.skipped:
            more = " ..." if len(self.skipped) > 10 else ""
            out.append(f"SKIPPED {len(self.skipped)}: " + "; ".join(self.skipped[:10]) + more)
        return "\n".join(out)

    def to_dict(self) -> dict:
        return {"answer": self.answer, "recall_ok": self.recall_ok, "stats": self.stats,
                "skipped": self.skipped,
                "claims": [{"id": c.id, "claim": c.claim, "quote": c.quote, "pointer": c.pointer()}
                           for c in self.claims]}


def parse_json_reply(text: str) -> dict | None:
    """The JSON object in a model reply, tolerating code fences and text around it."""
    text = text.strip()
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            data = json.loads(text[start:end + 1])
        except ValueError:
            return None
    return data if isinstance(data, dict) else None


def _plant_probe(chunks: list[ck.Chunk], rng: random.Random) -> tuple[ck.Chunk, str]:
    """A copy of the largest chunk (the hardest to read to the end) with a recall line
    inserted at a random place away from its first and last fifth (models remember the
    start and end of a text best, so the middle is the real test). The chunks the answer
    call reads are left as they are. Returns the copy and its unique marker."""
    marker = f"RC-{rng.randint(1000, 9999)}"
    lines = list(max(chunks, key=lambda c: ck.est_tokens(c.render())).lines)
    pos = rng.randint(len(lines) // 5, max(len(lines) // 5, 4 * len(lines) // 5))
    num = lines[min(pos, len(lines) - 1)].num
    lines.insert(pos, ck.Line(None, num, f"ERROR {PROBE_WORDS}: unique marker {marker} was planted here"))
    return ck.Chunk(lines), marker


def _recall_call(chunk: ck.Chunk, marker: str, backend: Backend, stats: "_Stats") -> bool:
    """Ask for the planted code alone. Passes only if the exact random code comes back."""
    try:
        reply = backend.chat(RECALL_SYSTEM, chunk.render() + RECALL_REMINDER, None, max_output=60)
    except ThinkingOnly:
        stats["thinking_parts"] += 1
        return False
    stats.count_call(reply)
    return marker in (reply.text or "")


class _Stats(dict):
    """The stats dict, with helpers to count calls and drops."""

    def count_call(self, reply):
        self["calls"] += 1
        if reply.input_tokens is None or reply.output_tokens is None:
            self["server_counts_missing"] += 1
        self["input_tokens"] += reply.input_tokens or 0
        self["output_tokens"] += reply.output_tokens or 0

    def drop(self, reason: str):
        self["dropped"] += 1
        self["dropped_reasons"][reason] = self["dropped_reasons"].get(reason, 0) + 1


def _map_chunk(chunk: ck.Chunk, question: str, backend: Backend, stats: _Stats) -> list[tuple]:
    """Ask the model about one chunk. Returns [(claim_text, quote, hits)].

    A reply that was cut off is retried as two half chunks (up to MAX_SPLITS deep).
    The usual cause is a chunk that is more tokens than estimated (logs full of
    numbers and IDs), which fills the context before the reply is written."""
    unreadable_before = stats["unreadable_parts"]
    found = _map_part(chunk, question, backend, stats, MAX_SPLITS)
    if stats["unreadable_parts"] > unreadable_before:
        stats["unreadable_chunks"] += 1
    return found


def _map_part(chunk: ck.Chunk, question: str, backend: Backend, stats: _Stats, splits_left: int) -> list[tuple]:
    try:
        reply = backend.chat(MAP_SYSTEM, f"Question: {question}\n\n{chunk.render()}",
                             MAP_SCHEMA, max_output=900)
    except ThinkingOnly as e:
        # Seen once: one reply of many came back as thinking only. That part is
        # unreadable; the rest of the digest still counts (run_prepared re-raises if
        # no map reply was usable, so a model that always thinks still gets the fix).
        stats["thinking_parts"] += 1
        stats["unreadable_parts"] += 1
        stats.thinking_error = e
        return []
    stats["map_replies"] += 1
    stats.count_call(reply)
    parsed = parse_json_reply(reply.text)
    if reply.truncated and splits_left > 0 and len(chunk.lines) > 1:
        stats["split_chunks"] += 1
        half = len(chunk.lines) // 2
        found = []
        for part in (ck.Chunk(chunk.lines[:half]), ck.Chunk(chunk.lines[half:])):
            found += _map_part(part, question, backend, stats, splits_left - 1)
        return found
    if reply.truncated or parsed is None or not isinstance(parsed.get("claims"), list):
        stats["unreadable_parts"] += 1
        return []

    found = []
    for raw in parsed["claims"]:
        if not isinstance(raw, dict):
            stats.drop("malformed claim")
            continue
        quote = str(raw.get("quote") or "")
        text = str(raw.get("claim") or "").strip()
        res = ck.resolve(quote, chunk)
        if not res.ok:
            stats.drop(res.reason)
            continue
        if len(found) >= MAX_CLAIMS_PER_CHUNK:
            stats.drop("over the per-chunk limit")
            continue
        found.append((text, quote.strip(), res.hits))
    return found


def _merge_entry(c: Claim) -> str:
    quote = " ".join(c.quote.split())
    quote = quote if len(quote) <= MERGE_QUOTE_CHARS else quote[:MERGE_QUOTE_CHARS] + "..."
    return f"{c.id} ({c.pointer()}): {c.claim}\n    quote: {quote}"


def chunk_limit(backend, chunk_tokens: int) -> int:
    """File text per map call: chunk_tokens, or half the backend's window if that is less."""
    return min(chunk_tokens, getattr(backend, "context_tokens", CONTEXT_TOKENS) // 2)


def merge_budget(backend) -> int:
    """Claim text per merge call: MERGE_INPUT_TOKENS, or less so it fits the backend's window."""
    window = getattr(backend, "context_tokens", CONTEXT_TOKENS)
    return max(1024, min(MERGE_INPUT_TOKENS, window - REPLY_ROOM))


def _pick_for_merge(claims: list[Claim], budget: int = MERGE_INPUT_TOKENS) -> list[Claim]:
    """The claims that fit `budget` tokens, taken in turns from each file (the
    first claim of every file, then the second of every file, ...) and listed in
    file order. Taken in plain file order, one big file filled the budget: on a
    26-file package the merge saw a few files and said it had no summary."""
    by_file: dict[str, list[Claim]] = {}
    for c in claims:
        by_file.setdefault(c.hits[0].path if c.hits else "", []).append(c)
    queues = list(by_file.values())
    picked, used = set(), 0
    for rank in range(max(len(q) for q in queues)):
        for q in queues:
            if rank >= len(q):
                continue
            used += ck.est_tokens(_merge_entry(q[rank])) + 1   # + the newline that joins entries
            if used > budget:
                return [c for c in claims if c.id in picked]
            picked.add(q[rank].id)
    return list(claims)


def _batches(claims: list[Claim], budget: int) -> list[list[Claim]]:
    """Claims in file order, cut into runs of at most `budget` tokens (at least one each)."""
    out, cur, used = [], [], 0
    for c in claims:
        cost = ck.est_tokens(_merge_entry(c)) + 1
        if cur and used + cost > budget:
            out.append(cur)
            cur, used = [], 0
        cur.append(c)
        used += cost
    return out + [cur] if cur else out


def _ask(system: str, user: str, backend: Backend, stats: _Stats, max_output: int) -> str:
    reply = backend.chat(system, user, MERGE_SCHEMA, max_output=max_output)
    stats.count_call(reply)
    return str((parse_json_reply(reply.text) or {}).get("answer") or "").strip()


def _select(question: str, claims: list[Claim], backend: Backend, stats: _Stats) -> list[Claim]:
    """The claims the model picks as relevant, batch by batch, in file order. Only
    run when the claims are more than one batch; on a few claims picking can only lose."""
    batches = _batches(claims, min(MERGE_BATCH_TOKENS, merge_budget(backend)))
    if len(batches) == 1:
        return claims
    picked = set()
    for batch in batches:
        body = "\n".join(_merge_entry(c) for c in batch)
        reply = backend.chat(SELECT_SYSTEM, f"Question: {question}\n\nClaims:\n{body}", SELECT_SCHEMA, max_output=300)
        stats.count_call(reply)
        ids = (parse_json_reply(reply.text) or {}).get("ids")
        allowed = {c.id for c in batch}
        picked |= {str(i).strip().upper() for i in ids if str(i).strip().upper() in allowed} \
            if isinstance(ids, list) else set()
    chosen = [c for c in claims if c.id in picked]
    stats["merge_picked"] = len(chosen)
    return chosen or claims          # nothing picked (or unreadable replies): keep them all


def _merge(question: str, claims: list[Claim], backend: Backend, stats: _Stats) -> tuple[str, set[str]]:
    """Write the answer from the claims. Returns (answer, IDs of the claims judged relevant).

    1. A big claim set is narrowed first: the model picks the relevant claims, a batch at a time.
    2. Claims from one file: one call writes the answer.
       Claims from several files: one short note per file, then one call combines the notes.
       If the combined answer cites no claim at all, it was a refusal ("the claims do not
       say..."), so the per-file notes are given instead.
    3. Citations are checked: unknown IDs become [?]."""
    chosen = _select(question, claims, backend, stats)
    budget = merge_budget(backend)
    by_file: dict[str, list[Claim]] = {}
    for c in chosen:
        by_file.setdefault(c.hits[0].path if c.hits else "", []).append(c)
    if len(by_file) == 1:
        listed = _pick_for_merge(chosen, budget)
        stats["merge_claims_cut"] = len(chosen) - len(listed)
        body = "\n".join(_merge_entry(c) for c in listed)
        answer = _ask(MERGE_SYSTEM, f"Question: {question}\n\nClaims:\n{body}", backend, stats, 700)
    else:
        notes = []
        for path, group in by_file.items():
            listed = _pick_for_merge(group, budget)
            body = "\n".join(_merge_entry(c) for c in listed)
            note = _ask(FILE_NOTE_SYSTEM, f"Question: {question}\n\nFile: {path}\nClaims:\n{body}", backend, stats, 200)
            if note:
                notes.append(f"{path}: {note}")
        fits = ck.est_tokens("\n".join(notes)) <= budget     # too many notes: list them, do not overflow
        answer = _ask(COMBINE_SYSTEM, f"Question: {question}\n\nNotes:\n" + "\n".join(notes), backend, stats, 900) \
            if notes and fits else ""
        if notes and not fits:
            stats["combine_skipped"] = 1
            answer = "By file:\n" + "\n".join(f"- {n}" for n in notes)
        elif notes and not CLAIM_ID_RE.search(answer):
            stats["combine_refused"] = 1
            answer = "By file:\n" + "\n".join(f"- {n}" for n in notes)
    picked = {c.id for c in chosen} if len(chosen) < len(claims) else set()
    if not answer:
        stats["bad_merge_reply"] = 1
        return "(the merge step returned no answer; read the claims below)", picked
    return _check_cites(answer, chosen, stats), picked


def _check_cites(answer: str, listed: list[Claim], stats: _Stats) -> str:
    valid = {c.id[1:] for c in listed}

    def check_bracket(match):        # keep known IDs, replace unknown ones with "?"
        def check_id(m):
            if m.group(1) in valid:
                return f"C{m.group(1)}"
            stats["unknown_cites"] += 1
            return "?"
        return "[" + CLAIM_ID_RE.sub(check_id, match.group(1)) + "]"
    return BRACKET_RE.sub(check_bracket, answer)


@dataclass
class Prepared:
    """The input of one digest, read and chunked, plus the stats it will fill in."""
    paths: list[str]
    chunks: list[ck.Chunk]
    skipped: list[str]
    stats: _Stats


def prepare(paths: list[str], include: list[str] | None, display_root: Path | None,
            chunk_tokens: int, max_input_tokens: int = MAX_INPUT_TOKENS) -> Prepared:
    """Read and chunk the input. Raises NoInput if nothing is readable and
    InputTooLarge above max_input_tokens (estimated)."""
    files, skipped = ck.collect_files(paths, include)
    lines, unreadable = ck.read_lines(files, display_root)
    skipped += unreadable
    if not lines:
        why = "; ".join(skipped[:5]) or "the files are empty"
        raise NoInput(f"no readable text in the given paths ({why}).")
    est_total = ck.total_cost(lines)
    if est_total > max_input_tokens:
        raise InputTooLarge(f"the input is about {est_total:,} tokens, above the {max_input_tokens:,} limit. "
                            "Narrow the paths or use --include.")
    chunks = ck.pack(lines, chunk_tokens)
    stats = _Stats(files=len(files), chunks=len(chunks), chunk_tokens=chunk_tokens, est_input_tokens=est_total,
                   model="?", calls=0,
                   input_tokens=0, output_tokens=0, dropped=0, dropped_reasons={}, unknown_cites=0,
                   unreadable_chunks=0, unreadable_parts=0, split_chunks=0, thinking_parts=0, map_replies=0, server_counts_missing=0,
                   seconds=0.0)
    return Prepared(paths, chunks, skipped, stats)


def run_prepared(prepared: Prepared, question: str, backend: Backend,
                 recall_check: bool = True, seed: int | None = None) -> DigestResult:
    """Run the model steps on prepared input. `seed` makes the probe placement
    repeatable (tests, evals). Errors from the backend are raised as they come;
    prepared.stats then shows how far it got."""
    started = time.monotonic()
    stats = prepared.stats
    stats["model"] = getattr(backend, "model", "") or getattr(backend, "name", "") or "?"   # a CLI may have no model
    rng = random.Random(seed)
    probe_chunk, stats["probe_marker"] = _plant_probe(prepared.chunks, rng) if recall_check else (None, None)
    claims: list[Claim] = []
    seen_places = set()
    try:
        for chunk in prepared.chunks:
            for text, quote, hits in _map_chunk(chunk, question, backend, stats):
                if tuple(hits) in seen_places:
                    stats.drop("duplicate")
                    continue
                seen_places.add(tuple(hits))
                claims.append(Claim(f"C{len(claims) + 1}", text, quote, hits))
        if stats["map_replies"] == 0 and stats["thinking_parts"]:
            raise stats.thinking_error
        probe_seen = _recall_call(probe_chunk, stats["probe_marker"], backend, stats) if recall_check else None
        picked: set[str] = set()
        if claims:
            answer, picked = _merge(question, claims, backend, stats)
        else:
            answer = "Nothing in the given files answers this question (no claim survived the quote check)."
    finally:
        stats["seconds"] = time.monotonic() - started

    result = DigestResult(question, answer, claims, probe_seen, stats, prepared.skipped,
                          picked)
    stats["output_est_tokens"] = ck.est_tokens(result.render())
    return result


def run(paths: list[str], question: str, backend: Backend, settings: Settings,
        include: list[str] | None = None, display_root: Path | None = None,
        max_input_tokens: int = MAX_INPUT_TOKENS, recall_check: bool = True,
        seed: int | None = None) -> DigestResult:
    """prepare() then run_prepared(), with no lock, GPU check or log. For tests and
    scripts that bring their own backend; agents go through api.run_digest()."""
    prepared = prepare(paths, include, display_root, chunk_limit(backend, settings.chunk_tokens), max_input_tokens)
    return run_prepared(prepared, question, backend, recall_check, seed)


def log_record(settings: Settings, record: dict):
    """Append one record to <home>/logs/digest.jsonl. Never raises: a full disk or a
    read-only home must not stop a digest, so a failure is reported on stderr."""
    try:
        folder = Path(settings.home) / "logs"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "digest.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "kind": "digest", **record}) + "\n")
    except OSError as e:
        print(f"agenthandoff: could not write the log: {e}", file=sys.stderr)
