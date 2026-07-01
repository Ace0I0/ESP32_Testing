from __future__ import annotations

from pathlib import Path
from typing import Any


MAX_CONTEXT_TOKENS = 8192
MAX_PROMPT_TOKENS = 6500
MAX_KNOWLEDGE_TOKENS = 3500

MIN_RESPONSE_TOKENS = 128
DEFAULT_RESPONSE_TOKENS = 256
MAX_RESPONSE_TOKENS = 512
FULL_REPORT_RESPONSE_TOKENS = 768

SAFETY_BUFFER_TOKENS = 700
MAX_SELECTED_FILES = 3

DEBUG_BUDGET = True

# Token budget manager for the prompt-building pipeline.
# intent_router.py provides selected_file_details, then this module decides:
# - how much context each selected file may contribute,
# - how many response tokens the model may use,
# - whether the final knowledge context must be trimmed before the LLM call.

# These priorities only decide which file gets more or less context when the
# same file appears in different routing roles.
REASON_PRIORITY = {
    "primary_intent": 100,
    "secondary_intent": 70,
    "topic_support": 40,
    "fallback": 10,
}

# Response multipliers stay separate from context budgets so the router can keep
# one file small on input while still allowing a larger answer if needed.
REASON_RESPONSE_MULTIPLIER = {
    "primary_intent": 1.0,
    "secondary_intent": 0.5,
    "topic_support": 0.25,
    "fallback": 0.2,
}

# Per-file budgets are tuned for Raspberry Pi use, not for filling the full
# context window. Bigger files still need to be trimmed aggressively.
FILE_BUDGETS = {
    "eas_contacts.txt": {
        "context_tokens": 300,
        "response_tokens": 160,
        "priority": 8,
    },
    "eas_diseases.txt": {
        "context_tokens": 900,
        "response_tokens": 384,
        "priority": 7,
    },
    "eas_creatures.txt": {
        "context_tokens": 1000,
        "response_tokens": 384,
        "priority": 7,
    },
    "eas_agencies.txt": {
        "context_tokens": 500,
        "response_tokens": 256,
        "priority": 6,
    },
    "eas_locations.txt": {
        "context_tokens": 500,
        "response_tokens": 256,
        "priority": 6,
    },
    "eas_current_activity.txt": {
        "context_tokens": 700,
        "response_tokens": 256,
        "priority": 8,
    },
    "eas_general.txt": {
        "context_tokens": 600,
        "response_tokens": 256,
        "priority": 5,
    },
}

LONG_RESPONSE_PHRASES = [
    "full report",
    "detailed report",
    "complete briefing",
    "long explanation",
    "give me everything",
    "full explanation",
    "detailed breakdown",
]


def estimate_tokens(text: str) -> int:
    """Use a rough character-to-token estimate for lightweight budgeting."""
    return max(1, len(text) // 4) if text else 0


def _file_budget(file_name: str) -> dict[str, int]:
    """Return per-file budget settings, falling back to general EAS defaults."""
    return FILE_BUDGETS.get(file_name, FILE_BUDGETS["eas_general.txt"])


def _selection_strength(item: dict[str, Any]) -> tuple[int, int, int]:
    """Build a sortable strength tuple for a routed file detail item."""
    file_name = str(item.get("file", ""))
    reason = str(item.get("reason", "fallback"))
    intent_score = int(item.get("intent_score", 0))
    file_priority = int(_file_budget(file_name).get("priority", 0))
    return (
        REASON_PRIORITY.get(reason, 0),
        intent_score,
        file_priority,
    )


def sort_selected_files(selected_file_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort files by reason priority, then intent score, then file priority."""
    return sorted(selected_file_details, key=_selection_strength, reverse=True)


def dedupe_selected_files(selected_file_details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the strongest version of each file while preserving the best reason."""
    strongest_by_file: dict[str, dict[str, Any]] = {}

    for item in selected_file_details:
        file_name = str(item.get("file", ""))
        if not file_name:
            continue

        candidate = dict(item)
        current = strongest_by_file.get(file_name)

        if current is None or _selection_strength(candidate) > _selection_strength(current):
            strongest_by_file[file_name] = candidate

    return list(strongest_by_file.values())


def calculate_context_budget_for_file(file_name: str, reason: str) -> int:
    """Scale a file's input-context budget based on why it was selected."""
    base = int(_file_budget(file_name)["context_tokens"])

    if reason == "primary_intent":
        return base
    if reason == "secondary_intent":
        return int(base * 0.6)
    if reason == "topic_support":
        return int(base * 0.35)
    return int(base * 0.25)


def _long_answer_requested(user_text: str) -> bool:
    """Detect explicit requests that should allow a longer response cap."""
    lower = user_text.lower()
    return any(phrase in lower for phrase in LONG_RESPONSE_PHRASES)


def calculate_response_tokens(selected_file_details: list[dict[str, Any]], user_text: str = "") -> int:
    """Calculate the max_tokens value passed into the local LLM request.

    This is called from load_llm_sam.build_system_prompt(); if answers are too
    short or too long, start debugging here and in REASON_RESPONSE_MULTIPLIER.
    """
    total = 0

    for item in selected_file_details:
        file_name = str(item.get("file", ""))
        reason = str(item.get("reason", "fallback"))
        base_tokens = int(_file_budget(file_name)["response_tokens"])
        multiplier = REASON_RESPONSE_MULTIPLIER.get(reason, 0.25)
        total += int(base_tokens * multiplier)

    total = total or DEFAULT_RESPONSE_TOKENS
    final = max(MIN_RESPONSE_TOKENS, min(total, MAX_RESPONSE_TOKENS))

    if _long_answer_requested(user_text):
        final = min(FULL_REPORT_RESPONSE_TOKENS, max(final, MAX_RESPONSE_TOKENS))

    return final


def trim_to_token_budget(text: str, max_tokens: int) -> str:
    """Trim text to the rough token cap while trying to end on a clean boundary."""
    max_chars = max_tokens * 4

    if len(text) <= max_chars:
        return text

    trimmed = text[:max_chars]
    last_period = trimmed.rfind(".")
    last_newline = trimmed.rfind("\n")
    cut_point = max(last_period, last_newline)

    if cut_point > max_chars * 0.7:
        trimmed = trimmed[: cut_point + 1]

    return trimmed.rstrip() + "\n[Context trimmed to fit token budget.]"


def build_budgeted_knowledge_context(
    selected_file_details: list[dict[str, Any]],
    knowledge_dir: str,
    max_knowledge_tokens: int = MAX_KNOWLEDGE_TOKENS,
) -> tuple[str, list[dict[str, Any]]]:
    """Load, trim, and assemble the selected knowledge files under budget.

    This is the main bridge between the router output and the final prompt.
    The returned debug metadata is printed by load_llm_sam.print_prompt_debug()
    so missing files, trims, and allocation mistakes can be traced later.
    """
    details = sort_selected_files(dedupe_selected_files(selected_file_details))[:MAX_SELECTED_FILES]
    knowledge_root = Path(knowledge_dir)

    context_blocks: list[str] = []
    debug_metadata: list[dict[str, Any]] = []
    total_used_tokens = 0

    for item in details:
        file_name = str(item.get("file", ""))
        reason = str(item.get("reason", "fallback"))
        path = knowledge_root / file_name
        allocated_context_tokens = calculate_context_budget_for_file(file_name, reason)

        if total_used_tokens >= max_knowledge_tokens:
            break

        header = f"[Knowledge: {file_name} | reason: {reason}]"
        header_tokens = estimate_tokens(header + "\n")
        allowed_context_tokens = min(allocated_context_tokens, max(0, max_knowledge_tokens - total_used_tokens - header_tokens))

        if allowed_context_tokens <= 0:
            debug_metadata.append(
                {
                    "file": file_name,
                    "reason": reason,
                    "allocated_context_tokens": 0,
                    "estimated_used_tokens": 0,
                    "trimmed": True,
                    "exists": path.exists(),
                }
            )
            continue

        if not path.exists():
            debug_metadata.append(
                {
                    "file": file_name,
                    "reason": reason,
                    "allocated_context_tokens": allowed_context_tokens,
                    "estimated_used_tokens": 0,
                    "trimmed": False,
                    "exists": False,
                }
            )
            continue

        raw_text = path.read_text(encoding="utf-8").strip()
        trimmed_text = trim_to_token_budget(raw_text, allowed_context_tokens)
        trimmed = trimmed_text != raw_text
        block = f"{header}\n{trimmed_text}".strip()
        used_tokens = estimate_tokens(block)

        if total_used_tokens + used_tokens > max_knowledge_tokens:
            remaining_tokens = max_knowledge_tokens - total_used_tokens
            if remaining_tokens <= 0:
                break

            allowed_context_tokens = max(0, remaining_tokens - header_tokens)
            trimmed_text = trim_to_token_budget(raw_text, allowed_context_tokens)
            trimmed = True
            block = f"{header}\n{trimmed_text}".strip()
            used_tokens = estimate_tokens(block)

        context_blocks.append(block)
        debug_metadata.append(
            {
                "file": file_name,
                "reason": reason,
                "allocated_context_tokens": allowed_context_tokens,
                "estimated_used_tokens": estimate_tokens(trimmed_text),
                "trimmed": trimmed,
                "exists": True,
            }
        )
        total_used_tokens += used_tokens

    return "\n\n".join(context_blocks), debug_metadata


def enforce_prompt_budget(
    prompt_without_knowledge: str,
    knowledge_context: str,
    user_text: str,
    max_prompt_tokens: int = MAX_PROMPT_TOKENS,
) -> str:
    """Trim knowledge further if the total prompt would exceed the prompt cap."""
    available_for_knowledge = max_prompt_tokens - estimate_tokens(prompt_without_knowledge) - estimate_tokens(user_text)

    if available_for_knowledge <= 0:
        fallback = "[Knowledge omitted due to prompt budget.]"
        fallback_tokens = estimate_tokens(prompt_without_knowledge) + estimate_tokens(fallback) + estimate_tokens(user_text)
        if fallback_tokens <= max_prompt_tokens:
            return fallback
        return ""

    if estimate_tokens(knowledge_context) <= available_for_knowledge:
        return knowledge_context

    return trim_to_token_budget(knowledge_context, available_for_knowledge)


def print_budget_debug(budget_debug: dict[str, Any]) -> None:
    """Print the budget portion of the prompt debug report."""
    if not DEBUG_BUDGET:
        return

    print(f"[Budget] Max context tokens: {budget_debug.get('max_context_tokens', MAX_CONTEXT_TOKENS)}")
    print(f"[Budget] Max prompt tokens: {budget_debug.get('max_prompt_tokens', MAX_PROMPT_TOKENS)}")
    print(f"[Budget] Max knowledge tokens: {budget_debug.get('max_knowledge_tokens', MAX_KNOWLEDGE_TOKENS)}")
    print("[Budget] Selected files:")

    for item in budget_debug.get("selected_file_details", []):
        print(
            f"  - {item.get('file')} | reason={item.get('reason')} | "
            f"allocated={item.get('allocated_context_tokens')} | "
            f"used={item.get('estimated_used_tokens')} | trimmed={item.get('trimmed')}"
        )

    print(f"[Budget] Response token limit: {budget_debug.get('response_tokens', DEFAULT_RESPONSE_TOKENS)}")
    print(f"[Budget] Estimated final prompt tokens: {budget_debug.get('estimated_final_prompt_tokens', 0)}")
