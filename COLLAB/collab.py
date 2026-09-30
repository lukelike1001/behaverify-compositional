#!/usr/bin/env python3
"""
collab.py -- shared scratchpad for two agents working the same branch.

Everything here is APPEND-ONLY. Two agents editing one file concurrently is the
whole problem, so nothing is ever rewritten in place: the discussion is appended
to LOG.md, and claim status lives in claims.jsonl as a stream of events that
`status` folds into a current view. Git then merges cleanly instead of
conflicting, and no agent can silently overwrite the other's work.

The claim ledger is the point. A claim is not established because the agent who
made it is confident -- it is established when the OTHER agent independently
checks it. `status` shows what is still unverified, which is the work queue.

Usage:
    ./collab.py post   <agent> "<message>"
    ./collab.py claim  <agent> "<statement>" [--evidence "<how to check>"]
    ./collab.py verify <agent> <id> "<what you did to confirm>"
    ./collab.py refute <agent> <id> "<what you found instead>"
    ./collab.py amend  <agent> <id> "<correction to your own claim>"
    ./collab.py status [--all]
    ./collab.py show   <id>
    ./collab.py log [-n N]

Start at STATE.md for where the project is, and DEAD_ENDS.md for what has
already been tried and closed. This file is the machinery; those two are the
memory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "LOG.md"
CLAIMS = ROOT / "claims.jsonl"
AGENTS = ("claude", "gemini", "luke")


def _summary(text: str, width: int = 140) -> str:
    """First sentence or so, enough to scan the log without opening a claim."""
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[:width].rsplit(" ", 1)[0] + "..."


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _append(path: Path, text: str) -> None:
    """Single append; atomic enough for two writers on one machine."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


@dataclass
class Ledger:
    """The append-only claim event stream, folded into current state."""

    events: list[dict]

    @classmethod
    def load(cls) -> "Ledger":
        """Read the whole event stream."""
        if not CLAIMS.exists():
            return cls(events=[])
        lines = CLAIMS.read_text(encoding="utf-8").splitlines()
        return cls(events=[json.loads(line) for line in lines if line.strip()])

    def next_id(self) -> int:
        """One past the highest claim id issued so far."""
        made = [e["id"] for e in self.events if e["kind"] == "claim"]
        return max(made, default=0) + 1

    def fold(self) -> dict[int, dict]:
        """Current state per claim id: statement, author, status, checks."""
        state: dict[int, dict] = {}
        for event in self.events:
            claim_id = event["id"]
            if event["kind"] == "claim":
                state[claim_id] = {
                    "statement": event["statement"],
                    "author": event["agent"],
                    "evidence": event.get("evidence", ""),
                    "status": "unverified",
                    "checks": [],
                }
            elif claim_id in state:
                # an amendment corrects the claim; only the OTHER agent's
                # verdict moves its status
                if event["kind"] in ("verify", "refute"):
                    state[claim_id]["status"] = (
                        "verified" if event["kind"] == "verify" else "refuted"
                    )
                state[claim_id]["checks"].append(
                    f"{event['kind']} by {event['agent']}: {event['note']}"
                )
        return state

    def record(self, event: dict) -> None:
        """Append one event; never rewrites earlier ones."""
        _append(CLAIMS, json.dumps(event) + "\n")


def post(agent: str, message: str) -> None:
    """Append a signed, timestamped entry to the discussion log."""
    _append(LOG, f"\n## [{agent}] {_now()}\n\n{message}\n")
    print(f"posted to {LOG.name}")


def _note(agent: str, line: str) -> None:
    """One-line pointer in the log; the full text stays in the ledger.

    Writing claims into LOG.md in full duplicated the ledger word for word and
    grew the file to 204 KB, of which 88 percent was a second copy of
    claims.jsonl. The log is now the discussion only, and `status` or `show`
    is how you read a claim.
    """
    _append(LOG, f"\n## [{agent}] {_now()}\n\n{line}\n")


def claim(agent: str, statement: str, evidence: str) -> None:
    """Register a new claim as unverified and announce it in the log."""
    ledger = Ledger.load()
    claim_id = ledger.next_id()
    ledger.record({"kind": "claim", "id": claim_id, "agent": agent, "when": _now(),
                   "statement": statement, "evidence": evidence})
    _note(agent, f"**CLAIM #{claim_id}** (unverified) — "
                 f"{_summary(statement)}  `./collab.py show {claim_id}`")
    print(f"claim #{claim_id} recorded as unverified")


def amend(agent: str, claim_id: int, note: str) -> None:
    """Correct your OWN claim without erasing what it originally said.

    A claim is append-only, so an author who finds a number wrong cannot fix
    it in place, and a refutation by the other agent is the wrong instrument
    for self-correction. Without this, a corrected figure lives only in a
    discussion post and `show` keeps printing the stale one -- which is
    exactly how claim #62's overstated contract count survived long enough to
    be repeated in a later summary.
    """
    ledger = Ledger.load()
    state = ledger.fold()
    if claim_id not in state:
        raise SystemExit(f"no claim #{claim_id}")
    if state[claim_id]["author"] != agent:
        raise SystemExit(
            f"claim #{claim_id} is {state[claim_id]['author']}'s; "
            f"use refute, not amend"
        )
    ledger.record({"kind": "amend", "id": claim_id, "agent": agent,
                   "when": _now(), "note": note})
    _note(agent, f"**AMEND #{claim_id}** — {_summary(note)}  "
                 f"`./collab.py show {claim_id}`")
    print(f"claim #{claim_id} amended by {agent}")


def judge(agent: str, claim_id: int, note: str, kind: str) -> None:
    """Verify or refute another agent's claim. Self-checking is refused."""
    ledger = Ledger.load()
    state = ledger.fold()
    if claim_id not in state:
        raise SystemExit(f"no claim #{claim_id}")
    if state[claim_id]["author"] == agent:
        raise SystemExit(
            f"claim #{claim_id} was made by {agent}; the OTHER agent must check it"
        )
    ledger.record({"kind": kind, "id": claim_id, "agent": agent,
                   "when": _now(), "note": note})
    _note(agent, f"**{kind.upper()} #{claim_id}** — {_summary(note)}  "
                 f"`./collab.py show {claim_id}`")
    print(f"claim #{claim_id} marked {kind}d by {agent}")


def status(show_all: bool) -> None:
    """Print current claim states, unverified and refuted first."""
    state = Ledger.load().fold()
    if not state:
        print("no claims yet")
        return
    order = {"refuted": 0, "unverified": 1, "verified": 2}
    for claim_id, entry in sorted(state.items(), key=lambda kv: (order[kv[1]["status"]], kv[0])):
        if not show_all and entry["status"] == "verified":
            continue
        mark = {"unverified": "?", "verified": "+", "refuted": "x"}[entry["status"]]
        print(f"[{mark}] #{claim_id} ({entry['author']}, {entry['status']}) {entry['statement']}")
        if entry["evidence"] and entry["status"] == "unverified":
            print(f"      check: {entry['evidence']}")
        for check in entry["checks"]:
            print(f"      {check}")
    pending = sum(1 for e in state.values() if e["status"] == "unverified")
    print(f"\n{pending} unverified of {len(state)}")


def show(claim_id: int) -> None:
    """Print one claim in full: statement, evidence, and every check on it."""
    state = Ledger.load().fold()
    entry = state.get(claim_id)
    if entry is None:
        raise SystemExit(f"no claim #{claim_id}")
    print(f"#{claim_id} ({entry['author']}, {entry['status']})\n")
    print(entry["statement"])
    if entry["evidence"]:
        print(f"\nhow to check: {entry['evidence']}")
    for check in entry["checks"]:
        print(f"\n{check}")


def main() -> None:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    poster = sub.add_parser("post")
    poster.add_argument("agent", choices=AGENTS)
    poster.add_argument("message")
    claimer = sub.add_parser("claim")
    claimer.add_argument("agent", choices=AGENTS)
    claimer.add_argument("statement")
    claimer.add_argument("--evidence", default="")
    for kind in ("verify", "refute", "amend"):
        judger = sub.add_parser(kind)
        judger.add_argument("agent", choices=AGENTS)
        judger.add_argument("id", type=int)
        judger.add_argument("note")
    reporter = sub.add_parser("status")
    reporter.add_argument("--all", action="store_true")
    shower = sub.add_parser("show")
    shower.add_argument("id", type=int)
    tailer = sub.add_parser("log")
    tailer.add_argument("-n", type=int, default=40)
    args = parser.parse_args()

    if args.command == "post":
        post(args.agent, args.message)
    elif args.command == "claim":
        claim(args.agent, args.statement, args.evidence)
    elif args.command == "amend":
        amend(args.agent, args.id, args.note)
    elif args.command in ("verify", "refute"):
        judge(args.agent, args.id, args.note, args.command)
    elif args.command == "status":
        status(args.all)
    elif args.command == "show":
        show(args.id)
    elif args.command == "log":
        print("\n".join(LOG.read_text(encoding="utf-8").splitlines()[-args.n:])
              if LOG.exists() else "(empty)")


if __name__ == "__main__":
    main()
