#!/usr/bin/env python3
"""Offline audit labels for Tau Airline validation jsonl files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


WRITE_TOOLS = {
    "book_reservation",
    "cancel_reservation",
    "update_reservation_baggages",
    "update_reservation_passengers",
    "update_reservation_flights",
    "send_certificate",
}
READ_TOOLS = {
    "list_all_airports",
    "search_direct_flight",
    "search_onestop_flight",
    "get_user_details",
    "get_reservation_details",
    "calculate",
}
ESCALATION_TOOLS = {"transfer_to_human_agents"}

COMPLETION_CLAIMS = [
    ("cancel_reservation", re.compile(r"\b(?:reservation|flight|booking)\b.{0,80}\b(?:has been|was|is)\s+cancel(?:led|ed)\b", re.I)),
    ("cancel_reservation", re.compile(r"\b(?:refund|refunded)\b.{0,80}\b(?:processed|issued|credited|initiated)\b", re.I)),
    ("book_reservation", re.compile(r"\b(?:booking|reservation)\b.{0,80}\b(?:confirmed|booked|created|complete|completed)\b", re.I)),
    ("update_reservation_flights", re.compile(r"\b(?:reservation|flight|flights)\b.{0,80}\b(?:updated|changed|modified|moved|rescheduled)\b", re.I)),
    ("update_reservation_baggages", re.compile(r"\b(?:baggage|bags|checked bags)\b.{0,80}\b(?:updated|added|changed|included)\b", re.I)),
    ("update_reservation_passengers", re.compile(r"\b(?:passenger|passengers)\b.{0,80}\b(?:removed|updated|changed|added)\b", re.I)),
    ("send_certificate", re.compile(r"\b(?:certificate|voucher|travel voucher|gift card)\b.{0,80}\b(?:issued|sent|added|created|provided)\b", re.I)),
    ("send_certificate", re.compile(r"\b(?:compensation)\b.{0,80}\b(?:sent|issued|provided|processed)\b", re.I)),
]

ASK_OR_WAIT_PATTERNS = [
    re.compile(r"\bi will\b.{0,80}\b(?:check|look|fetch|search|verify|proceed|start)\b", re.I),
    re.compile(r"\blet me\b.{0,80}\b(?:check|look|fetch|search|verify)\b", re.I),
    re.compile(r"\bcould you\b.{0,80}\b(?:provide|confirm|tell|share|specify)\b", re.I),
    re.compile(r"\bplease\b.{0,80}\b(?:provide|confirm|tell|share|specify|wait)\b", re.I),
    re.compile(r"\bwould you like\b.{0,80}\b(?:to proceed|me to proceed|to confirm)\b", re.I),
    re.compile(r"\bcan you\b.{0,80}\b(?:provide|confirm|tell|share|specify)\b", re.I),
]

REFUSAL_OR_EXPLANATION_PATTERNS = [
    re.compile(r"\b(?:cannot|can't|unable to|not allowed|not eligible|not possible|policy|according to policy)\b", re.I),
    re.compile(r"\b(?:transfer(?:red)? to (?:a )?human|human agent|escalat(?:e|ed|ing))\b", re.I),
    re.compile(r"\b(?:policy states|policy requires|must be transferred|requires human)\b", re.I),
]

READ_ANSWER_PATTERNS = [
    re.compile(r"\b(?:available|not available|flight number|reservation details|policy states)\b", re.I),
]

MUTATING_INTENT_PATTERNS = [
    re.compile(r"\b(?:book|reserve|booking|reservation)\b", re.I),
    re.compile(r"\b(?:cancel|refund|voucher|certificate|compensation)\b", re.I),
    re.compile(r"\b(?:change|modify|update|move|reschedule|upgrade|add|remove)\b", re.I),
    re.compile(r"\b(?:baggage|bags|passenger|insurance)\b", re.I),
]


def _clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_tool_calls(text: str) -> tuple[list[str], int]:
    names: list[str] = []
    malformed = 0
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text or "", re.I | re.S):
        block = match.group(1).strip()
        name = None
        try:
            value = json.loads(block)
            if isinstance(value, dict) and isinstance(value.get("name"), str):
                name = value["name"]
        except json.JSONDecodeError:
            pass
        if name is None:
            m = re.search(r'"name"\s*:\s*"([^"]+)"', block)
            if m:
                name = m.group(1)
        if name is None:
            m = re.search(r"\bname\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", block)
            if m:
                name = m.group(1)
        if name is None:
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\b", block)
            if m:
                name = m.group(1)
                malformed += 1
        if name is None:
            malformed += 1
        else:
            names.append(name)
    return names, malformed


def first_user_text(output: str) -> str:
    for part in re.split(r"\buser\b", output, flags=re.I)[1:]:
        text = part.split("assistant", 1)[0]
        if "<tool_response>" not in text:
            return _clean(text)
    return ""


def assistant_text(output: str) -> str:
    chunks = re.split(r"\bassistant\b", output, flags=re.I)[1:]
    return " ".join(_clean(chunk.split("user", 1)[0]) for chunk in chunks)


def has_valid_refusal_or_explanation_or_transfer(output: str, tool_names: list[str]) -> bool:
    text = assistant_text(output)
    if any(name in ESCALATION_TOOLS for name in tool_names):
        return True
    if any(pattern.search(text) for pattern in REFUSAL_OR_EXPLANATION_PATTERNS):
        return True
    has_read_tool = any(name in READ_TOOLS for name in tool_names)
    return has_read_tool and any(pattern.search(text) for pattern in READ_ANSWER_PATTERNS)


def mutating_intent(row: dict[str, Any], output: str) -> bool:
    text = " ".join([_clean(row.get("input", "")), first_user_text(output), assistant_text(output)[:1000]])
    return any(pattern.search(text) for pattern in MUTATING_INTENT_PATTERNS)


def fake_completion_claim(output: str, tool_names: list[str]) -> tuple[int, list[str]]:
    text = assistant_text(output)
    reasons = []
    tool_set = set(tool_names)
    for expected_tool, pattern in COMPLETION_CLAIMS:
        if not pattern.search(text):
            continue
        if expected_tool.startswith("update_reservation_"):
            has_tool = any(name.startswith("update_reservation_") for name in tool_set)
        else:
            has_tool = expected_tool in tool_set
        if not has_tool:
            reasons.append(expected_tool)
    return int(bool(reasons)), sorted(set(reasons))


def no_progress_flag(output: str, tool_names: list[str], fake_claim: bool, has_valid: bool) -> int:
    text = assistant_text(output)
    write_count = sum(1 for name in tool_names if name in WRITE_TOOLS)
    effective_count = sum(1 for name in tool_names if name in WRITE_TOOLS or name in READ_TOOLS or name in ESCALATION_TOOLS)
    ask_hits = sum(1 for pattern in ASK_OR_WAIT_PATTERNS if pattern.search(text))
    has_completion_signal = fake_claim or any(pattern.search(text) for _, pattern in COMPLETION_CLAIMS)
    if has_completion_signal or any(name in ESCALATION_TOOLS for name in tool_names):
        return 0
    if effective_count == 0 and ask_hits > 0:
        return 1
    if write_count == 0 and ask_hits >= 2 and not has_valid:
        return 1
    if effective_count <= 1 and ask_hits > 0 and len(text) > 120 and not has_valid:
        return 1
    return 0


def audit_row(row: dict[str, Any], db_match_field: str = "db_state_match") -> dict[str, Any]:
    output = str(row.get("output", ""))
    tool_names, parsed_malformed = parse_tool_calls(output)
    row_malformed = float(row.get("malformed_tool_call_count", 0) or 0)
    malformed_critical_tool_call = int(parsed_malformed > 0 or row_malformed > 0)
    executed_mutating = float(row.get("executed_mutating_tool_call_count", 0) or 0)
    if executed_mutating == 0:
        executed_mutating = float(sum(1 for name in tool_names if name in WRITE_TOOLS))

    fake_claim, fake_reasons = fake_completion_claim(output, tool_names)
    has_valid = has_valid_refusal_or_explanation_or_transfer(output, tool_names)
    no_progress = no_progress_flag(output, tool_names, bool(fake_claim), has_valid)
    db_match = float(row.get(db_match_field, 0) or 0)

    mut_success = int(
        db_match == 1.0
        and executed_mutating > 0
        and fake_claim == 0
        and malformed_critical_tool_call == 0
    )
    valid_noop = int(
        db_match == 1.0
        and executed_mutating == 0
        and fake_claim == 0
        and no_progress == 0
        and has_valid
    )
    suspicious_noop = int(
        db_match == 1.0
        and executed_mutating == 0
        and (fake_claim == 1 or no_progress == 1 or not has_valid)
    )

    return {
        **row,
        "audit_tool_names": tool_names,
        "audit_db_match_field": db_match_field,
        "audit_db_match_value": db_match,
        "audit_mutating_intent": int(mutating_intent(row, output)),
        "audit_fake_completion_claim": fake_claim,
        "audit_fake_completion_reasons": fake_reasons,
        "audit_no_progress_flag": no_progress,
        "audit_malformed_critical_tool_call": malformed_critical_tool_call,
        "audit_has_valid_refusal_or_explanation_or_transfer": int(has_valid),
        "audit_mutating_success": mut_success,
        "audit_valid_noop_success": valid_noop,
        "audit_suspicious_noop_success": suspicious_noop,
        "audit_valid_success": int(mut_success or valid_noop),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("-o", "--output-jsonl", type=Path)
    parser.add_argument(
        "--db-match-field",
        choices=["db_state_match", "outcome_reward"],
        default="db_state_match",
        help="Field used for audit success/no-op labels. Use outcome_reward for old API validation files without DB match.",
    )
    args = parser.parse_args()

    output_path = args.output_jsonl
    if output_path is None:
        output_path = args.input_jsonl.with_name(args.input_jsonl.stem + "_audited.jsonl")

    rows = []
    with args.input_jsonl.open() as f:
        for line in f:
            if line.strip():
                rows.append(audit_row(json.loads(line), db_match_field=args.db_match_field))

    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    keys = [
        "outcome_reward",
        "db_state_match",
        "forced_done",
        "audit_fake_completion_claim",
        "audit_no_progress_flag",
        "audit_mutating_success",
        "audit_valid_noop_success",
        "audit_suspicious_noop_success",
        "audit_valid_success",
    ]
    print(f"wrote {output_path}")
    print(f"n={len(rows)}")
    for key in keys:
        values = [float(row.get(key, 0) or 0) for row in rows]
        print(f"{key}/mean={sum(values) / len(values):.4f} count={sum(1 for value in values if value != 0)}")


if __name__ == "__main__":
    main()
