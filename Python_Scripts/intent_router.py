from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz


# Rule-based intent router for Artemis knowledge selection.
#
# Intent describes what the user wants to do.
# Topics describe what the user is talking about.
# The router keeps those two ideas separate so a topic word alone does not
# automatically win file selection.

DEBUG_ROUTER = True
MAX_SELECTED_FILES = 3

# Raspberry Pi terminal input usually already uses plain ASCII quotes, so this is
# mostly a conservative fallback for pasted text or odd input sources.
SMART_QUOTES_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
    }
)

# Each intent entry defines a primary knowledge file, optional secondary files,
# phrases, proximity rules, and weak keywords. Strong phrases and nearby-word
# combinations should win over loose keyword matches.
INTENTS: dict[str, dict[str, Any]] = {
    "contact_report": {
        "primary_file": "eas_contacts.txt",
        "secondary_files": ["eas_agencies.txt"],
        "secondary_triggers": ["agency", "authority", "department", "bureau", "official"],
        "strong_phrases": [
            "who do i contact",
            "who should i contact",
            "who do i call",
            "who should i call",
            "where do i report",
            "how do i report",
            "who do i notify",
            "who should i notify",
            "contact an agency",
            "contact the agency",
            "report this",
            "report an incident",
            "report the incident",
            "emergency contact",
            "nearest contact",
            "who handles this",
            "who is responsible for this",
            "which agency do i contact",
            "what number do i call",
        ],
        "combo_rules": [
            ("who", "contact", 12),
            ("who", "call", 12),
            ("who", "notify", 12),
            ("where", "report", 11),
            ("how", "report", 11),
            ("contact", "agency", 10),
            ("report", "incident", 10),
            ("call", "hotline", 10),
            ("number", "call", 8),
        ],
        "keywords": [
            "contact",
            "call",
            "notify",
            "hotline",
            "phone",
            "number",
            "agency",
            "authority",
            "responsible",
            "dispatcher",
            "dispatch",
        ],
    },
    "disease_info": {
        "primary_file": "eas_diseases.txt",
        "secondary_files": [],
        "secondary_triggers": [],
        "strong_phrases": [
            "what is this disease",
            "what is the disease",
            "what are the symptoms",
            "what symptoms does it cause",
            "how does it spread",
            "how is it spread",
            "is it contagious",
            "is it infectious",
            "how is it treated",
            "what causes this disease",
            "what is the infection",
            "how dangerous is the disease",
            "what should i do if infected",
            "how do i know if i am infected",
        ],
        "combo_rules": [
            ("disease", "symptoms", 12),
            ("virus", "symptoms", 12),
            ("infection", "symptoms", 12),
            ("disease", "spread", 12),
            ("virus", "spread", 12),
            ("infection", "spread", 12),
            ("how", "spread", 10),
            ("is", "contagious", 10),
            ("what", "symptoms", 10),
            ("infected", "do", 9),
            ("quarantine", "disease", 10),
        ],
        "keywords": [
            "disease",
            "virus",
            "infection",
            "infected",
            "symptom",
            "symptoms",
            "spread",
            "contagious",
            "infectious",
            "treatment",
            "treated",
            "pathogen",
            "quarantine",
            "contamination",
        ],
    },
    "creature_info": {
        "primary_file": "eas_creatures.txt",
        "secondary_files": [],
        "secondary_triggers": [],
        "strong_phrases": [
            "what is this creature",
            "what is this organism",
            "what creature is this",
            "how do i identify it",
            "how do i identify this creature",
            "what does it look like",
            "is it dangerous",
            "how dangerous is it",
            "how do i survive it",
            "what is a mimic",
            "what is the mimic",
            "what is vita carnis",
            "what is nature's mockery",
            "how do i avoid it",
            "what should i do if i see it",
        ],
        "combo_rules": [
            ("what", "creature", 12),
            ("what", "organism", 12),
            ("identify", "creature", 11),
            ("identify", "organism", 11),
            ("dangerous", "creature", 10),
            ("survive", "creature", 10),
            ("avoid", "creature", 10),
            ("what", "mimic", 12),
            ("what", "vita", 10),
            ("nature", "mockery", 10),
        ],
        "keywords": [
            "creature",
            "organism",
            "entity",
            "mimic",
            "mimik",
            "vita carnis",
            "carnis",
            "nature's mockery",
            "natures mockery",
            "dangerous",
            "identify",
            "identification",
            "survive",
            "avoid",
            "specimen",
        ],
    },
    "agency_info": {
        "primary_file": "eas_agencies.txt",
        "secondary_files": ["eas_contacts.txt"],
        "secondary_triggers": ["contact", "report", "call", "notify"],
        "strong_phrases": [
            "what agency is this",
            "what does the agency do",
            "which agency handles this",
            "which department handles this",
            "who has authority",
            "who is in charge",
            "what organization is responsible",
            "what is this agency",
            "who issued the warning",
            "who sent the broadcast",
        ],
        "combo_rules": [
            ("which", "agency", 12),
            ("what", "agency", 12),
            ("agency", "handles", 11),
            ("department", "handles", 11),
            ("who", "authority", 10),
            ("who", "issued", 10),
            ("who", "broadcast", 10),
        ],
        "keywords": [
            "agency",
            "department",
            "organization",
            "authority",
            "official",
            "government",
            "division",
            "bureau",
            "issued",
            "warning",
            "broadcast",
        ],
    },
    "location_info": {
        "primary_file": "eas_locations.txt",
        "secondary_files": [],
        "secondary_triggers": [],
        "strong_phrases": [
            "where is this happening",
            "where did this happen",
            "what area is affected",
            "which area is affected",
            "what zone is affected",
            "where is the location",
            "where was it seen",
            "where was it last seen",
            "what site is this",
            "where should i evacuate",
            "which region is affected",
        ],
        "combo_rules": [
            ("where", "happening", 12),
            ("where", "seen", 12),
            ("where", "location", 12),
            ("area", "affected", 11),
            ("zone", "affected", 11),
            ("region", "affected", 11),
            ("where", "evacuate", 10),
            ("evacuation", "area", 10),
        ],
        "keywords": [
            "where",
            "location",
            "area",
            "zone",
            "site",
            "facility",
            "region",
            "evacuation",
            "evacuate",
            "sector",
            "district",
            "coordinates",
            "seen",
        ],
    },
    "current_activity_status": {
        "primary_file": "eas_current_activity.txt",
        "secondary_files": ["eas_general.txt"],
        "secondary_triggers": ["general", "broadcast", "alert", "warning", "status"],
        "strong_phrases": [
            "what is happening",
            "what is happening now",
            "what is going on",
            "what is the current alert",
            "what is the current warning",
            "what is the current status",
            "what is the situation",
            "what is the latest update",
            "any current activity",
            "what changed",
            "status report",
            "give me a status report",
            "current emergency",
            "active emergency",
        ],
        "combo_rules": [
            ("current", "alert", 12),
            ("current", "warning", 12),
            ("current", "status", 12),
            ("latest", "update", 10),
            ("what", "happening", 12),
            ("status", "report", 11),
            ("active", "emergency", 10),
        ],
        "keywords": [
            "current",
            "status",
            "activity",
            "active",
            "latest",
            "update",
            "warning",
            "alert",
            "situation",
            "emergency",
            "happening",
            "now",
        ],
    },
    "safety_protocol": {
        "primary_file": "eas_general.txt",
        "secondary_files": ["eas_contacts.txt", "eas_locations.txt"],
        "secondary_triggers": ["contact", "call", "notify", "where", "location", "evacuate", "shelter"],
        "strong_phrases": [
            "what should i do",
            "what do i do",
            "how do i stay safe",
            "how can i stay safe",
            "should i evacuate",
            "should i shelter",
            "where should i go",
            "is it safe",
            "is this safe",
            "what is the protocol",
            "what are the instructions",
            "emergency instructions",
            "safety instructions",
        ],
        "combo_rules": [
            ("what", "do", 12),
            ("stay", "safe", 11),
            ("should", "evacuate", 11),
            ("should", "shelter", 11),
            ("is", "safe", 10),
            ("safety", "instructions", 10),
            ("emergency", "instructions", 10),
            ("what", "protocol", 10),
        ],
        "keywords": [
            "safe",
            "safety",
            "protocol",
            "instructions",
            "evacuate",
            "evacuation",
            "shelter",
            "avoid",
            "danger",
            "warning",
            "emergency",
            "secure",
            "lockdown",
        ],
    },
    "general_eas_info": {
        "primary_file": "eas_general.txt",
        "secondary_files": [],
        "secondary_triggers": [],
        "strong_phrases": [
            "what does this mean",
            "explain the alert",
            "explain this warning",
            "what is eas mode",
            "what is this broadcast",
            "what does the warning mean",
            "what does the alert mean",
            "give me general information",
        ],
        "combo_rules": [
            ("explain", "alert", 12),
            ("explain", "warning", 12),
            ("what", "mean", 11),
            ("what", "broadcast", 11),
            ("general", "information", 10),
        ],
        "keywords": [
            "explain",
            "meaning",
            "general",
            "broadcast",
            "alert",
            "warning",
            "eas",
            "information",
            "info",
        ],
    },
}

# Topic detection is separate from intent routing.
# Topics only describe the subject being discussed, and they should not
# override a different intent unless the scoring rules also support it.
TOPICS: dict[str, list[str]] = {
    "disease": [
        "disease",
        "virus",
        "infection",
        "infected",
        "pathogen",
        "deep root",
        "deep root virus",
        "deep root disease",
        "desease",
    ],
    "creature": [
        "creature",
        "organism",
        "entity",
        "mimic",
        "mimik",
        "vita carnis",
        "carnis",
        "nature's mockery",
        "natures mockery",
    ],
    "agency": [
        "agency",
        "department",
        "organization",
        "authority",
        "bureau",
        "government",
    ],
    "location": [
        "location",
        "where",
        "area",
        "zone",
        "site",
        "facility",
        "region",
        "sector",
        "district",
        "evacuation",
        "evacuate",
        "coordinates",
    ],
    "contact": [
        "contact",
        "contakt",
        "hotline",
        "phone",
        "number",
        "call",
        "notify",
    ],
    "current_activity": [
        "current",
        "status",
        "latest",
        "active",
        "happening now",
        "what is happening",
        "now",
        "alert",
        "warning",
    ],
}

TOPIC_MATCH_THRESHOLD = 90
TOPIC_BACKUP_THRESHOLD = 92

# File metadata stays separate from scoring so the router can keep returning the
# older filename list while also exposing richer details for the budget manager.
FILE_TO_INTENT = {
    config["primary_file"]: intent_name for intent_name, config in INTENTS.items()
}

TOPIC_TO_FILE = {
    "disease": "eas_diseases.txt",
    "creature": "eas_creatures.txt",
    "agency": "eas_agencies.txt",
    "location": "eas_locations.txt",
    "contact": "eas_contacts.txt",
    "current_activity": "eas_current_activity.txt",
}


def remove_consecutive_duplicate_words(text: str) -> str:
    """Collapse only consecutive duplicate words while preserving word order."""
    words = text.split()
    cleaned: list[str] = []

    for word in words:
        if not cleaned or cleaned[-1] != word:
            cleaned.append(word)

    return " ".join(cleaned)


def normalize_user_text(user_text: str) -> str:
    """Normalize user text safely without changing word order.

    The smart-quote translation is kept as a fallback, but terminal input on the
    Raspberry Pi will usually already arrive as plain quotes.
    """
    text = user_text.translate(SMART_QUOTES_TRANSLATION)
    text = text.lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"[\u0000-\u001f\u007f-\u009f]", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = remove_consecutive_duplicate_words(text)
    return text.strip()


def _comparison_text(normalized_text: str) -> str:
    """Build a punctuation-light comparison string for fuzzy topic checks."""
    text = normalized_text.replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    """Split text into ordered word tokens."""
    return [token for token in text.split() if token]


def _phrase_positions(tokens: list[str], phrase_tokens: list[str]) -> list[int]:
    """Return every exact starting position for a phrase token sequence."""
    if not tokens or not phrase_tokens or len(phrase_tokens) > len(tokens):
        return []

    positions: list[int] = []
    phrase_length = len(phrase_tokens)

    for index in range(len(tokens) - phrase_length + 1):
        if tokens[index : index + phrase_length] == phrase_tokens:
            positions.append(index)

    return positions


def words_near(text: str, word_a: str, word_b: str, max_words_between: int = 4) -> bool:
    """Check whether two words or phrases appear within a small window."""
    tokens = _tokenize(text)
    left_tokens = _tokenize(word_a)
    right_tokens = _tokenize(word_b)

    left_positions = _phrase_positions(tokens, left_tokens)
    right_positions = _phrase_positions(tokens, right_tokens)

    if not left_positions or not right_positions:
        return False

    left_length = len(left_tokens)
    right_length = len(right_tokens)

    for left_start in left_positions:
        left_end = left_start + left_length - 1
        for right_start in right_positions:
            right_end = right_start + right_length - 1
            gap_forward = right_start - left_end - 1
            gap_reverse = left_start - right_end - 1
            if 0 <= gap_forward <= max_words_between or 0 <= gap_reverse <= max_words_between:
                return True

    return False


def _contains_exact_term(text: str, term: str) -> bool:
    """Match a whole word or exact phrase without relying on fuzzy scoring."""
    if " " in term or "'" in term:
        return term in text

    pattern = rf"\b{re.escape(term)}\b"
    return re.search(pattern, text) is not None


def _score_phrase(phrase: str, normalized_text: str) -> tuple[int, str | None]:
    """Score a strong intent phrase using exact and fuzzy phrase matching."""
    if _contains_exact_term(normalized_text, phrase):
        return 14, f"exact phrase matched: {phrase}"

    # Strong phrases are the highest-value signals. Exact hits should dominate,
    # while fuzzy phrase matches provide a smaller but still meaningful boost.
    partial_score = fuzz.partial_ratio(phrase, normalized_text)
    if partial_score >= 90:
        return 10, f"strong fuzzy phrase matched: {phrase} ({partial_score})"
    if partial_score >= 84:
        return 6, f"fuzzy phrase matched: {phrase} ({partial_score})"

    return 0, None


def _score_keyword(keyword: str, normalized_text: str) -> tuple[int, str | None]:
    """Score a weak keyword signal when no stronger phrase match is present."""
    if _contains_exact_term(normalized_text, keyword):
        return 2, f"keyword matched: {keyword}"

    # Keyword fuzzing is intentionally weaker than phrase scoring so that
    # generic words like disease, alert, or contact do not dominate intent.
    if len(keyword) < 4:
        return 0, None

    partial_score = fuzz.partial_ratio(keyword, normalized_text)
    if partial_score >= 92:
        return 1, f"fuzzy keyword matched: {keyword} ({partial_score})"

    return 0, None


def _score_combo(normalized_text: str, word_a: str, word_b: str, score: int) -> tuple[int, str | None]:
    """Score a proximity rule when two terms appear near each other."""
    if words_near(normalized_text, word_a, word_b):
        return score, f"combo matched: {word_a} near {word_b}"
    return 0, None


def score_intents(normalized_text: str) -> tuple[dict[str, int], list[str]]:
    """Score every intent using phrases, proximity, and keyword fallbacks."""
    scores: dict[str, int] = {intent: 0 for intent in INTENTS}
    reasons: list[str] = []

    for intent_name, config in INTENTS.items():
        intent_score = 0

        for phrase in config["strong_phrases"]:
            score, reason = _score_phrase(phrase, normalized_text)
            if score:
                intent_score += score
                reasons.append(f"{reason} for {intent_name}")

        for word_a, word_b, combo_score in config["combo_rules"]:
            score, reason = _score_combo(normalized_text, word_a, word_b, combo_score)
            if score:
                intent_score += score
                reasons.append(f"{reason} for {intent_name}")

        for keyword in config["keywords"]:
            score, reason = _score_keyword(keyword, normalized_text)
            if score:
                intent_score += score
                reasons.append(f"{reason} for {intent_name}")

        scores[intent_name] = intent_score

    return scores, reasons


def classify_confidence(scores: dict[str, int]) -> str:
    """Convert raw intent scores into a simple confidence tier."""
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ranked:
        return "low"

    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    if top_score >= 12 and top_score - second_score >= 6:
        return "high"
    if top_score >= 8 and top_score - second_score >= 3:
        return "medium"
    return "low"


def _sorted_intents(scores: dict[str, int]) -> list[tuple[str, int]]:
    """Sort intents from highest score to lowest score."""
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _should_load_secondary_files(
    intent_name: str,
    normalized_text: str,
    topics: list[str],
    intent_scores: dict[str, int],
    second_intent: str | None,
    second_score: int,
) -> bool:
    """Decide whether a winning intent needs any extra supporting files."""
    config = INTENTS[intent_name]
    if not config["secondary_files"]:
        return False

    secondary_triggers = config.get("secondary_triggers", [])

    def has_trigger() -> bool:
        for trigger in secondary_triggers:
            if _contains_exact_term(normalized_text, trigger):
                return True

            if len(trigger) >= 4 and fuzz.partial_ratio(trigger, normalized_text) >= 92:
                return True

        return False

    if intent_name == "contact_report":
        return "agency" in topics or has_trigger()

    if intent_name == "agency_info":
        return "contact" in topics or has_trigger()

    if intent_name == "safety_protocol":
        return "contact" in topics or "location" in topics or has_trigger()

    if intent_name == "current_activity_status":
        return has_trigger()

    return False


def select_files(
    intent_scores: dict[str, int],
    topics: list[str],
    normalized_text: str | None = None,
) -> list[str]:
    """Pick the smallest safe set of knowledge files for the routed intent."""
    ranked = _sorted_intents(intent_scores)
    if not ranked:
        return ["eas_general.txt"]

    top_intent, top_score = ranked[0]
    second_intent, second_score = ranked[1] if len(ranked) > 1 else (None, 0)
    confidence = classify_confidence(intent_scores)
    selected_files: list[str] = []

    def add_file(filename: str) -> None:
        if filename not in selected_files:
            selected_files.append(filename)

    if confidence == "low":
        add_file("eas_general.txt")
        primary_file = INTENTS[top_intent]["primary_file"]
        if top_score > 0 and primary_file != "eas_general.txt":
            add_file(primary_file)
        return selected_files[:MAX_SELECTED_FILES]

    primary_file = INTENTS[top_intent]["primary_file"]
    add_file(primary_file)

    if confidence == "high":
        if normalized_text is None:
            normalized_text = ""

        if _should_load_secondary_files(
            top_intent,
            normalized_text,
            topics,
            intent_scores,
            second_intent,
            second_score,
        ):
            for filename in INTENTS[top_intent]["secondary_files"]:
                add_file(filename)

    elif confidence == "medium" and second_intent and second_score >= 4:
        add_file(INTENTS[second_intent]["primary_file"])

    return selected_files[:MAX_SELECTED_FILES]


def _detect_topic(normalized_text: str, topic_name: str, aliases: list[str]) -> tuple[bool, str | None]:
    """Check whether a topic alias appears exactly or fuzzily in the input."""
    comparison_text = _comparison_text(normalized_text)

    for alias in aliases:
        alias_normalized = normalize_user_text(alias)
        alias_comparison = _comparison_text(alias_normalized)

        if _contains_exact_term(normalized_text, alias_normalized):
            return True, f"topic detected: {topic_name} ({alias_normalized})"

        if alias_comparison and alias_comparison in comparison_text:
            return True, f"topic detected: {topic_name} ({alias_comparison})"

        partial_score = fuzz.partial_ratio(alias_comparison or alias_normalized, comparison_text)
        if partial_score >= TOPIC_MATCH_THRESHOLD:
            return True, f"topic detected: {topic_name} ({alias_normalized}, {partial_score})"

        backup_score = fuzz.token_set_ratio(alias_comparison or alias_normalized, comparison_text)
        if backup_score >= TOPIC_BACKUP_THRESHOLD:
            return True, f"topic detected: {topic_name} ({alias_normalized}, backup {backup_score})"

    return False, None


def detect_topics(normalized_text: str) -> list[str]:
    """Detect subject matter labels without deciding the final intent."""
    topics: list[str] = []

    for topic_name, aliases in TOPICS.items():
        matched, _ = _detect_topic(normalized_text, topic_name, aliases)
        if matched:
            topics.append(topic_name)

    return topics


def _filter_topics_for_intent(topics: list[str], intent: str) -> list[str]:
    """Drop redundant topic labels that are already implied by the intent."""
    if intent in {"contact_report", "agency_info"} and "contact" in topics:
        filtered = [topic for topic in topics if topic != "contact"]
        if filtered:
            return filtered

    return topics


def _add_file_detail(
    details: list[dict[str, Any]],
    seen_files: set[str],
    file_name: str,
    reason: str,
    intent_name: str,
    intent_scores: dict[str, int],
) -> None:
    """Append one routed file detail unless that file was already selected."""
    if not file_name or file_name in seen_files:
        return

    details.append(
        {
            "file": file_name,
            "reason": reason,
            "intent": intent_name,
            "intent_score": intent_scores.get(intent_name, 0),
        }
    )
    seen_files.add(file_name)


def build_selected_file_details(
    selected_files: list[str],
    topics: list[str],
    intent_scores: dict[str, int],
    confidence: str,
) -> list[dict[str, Any]]:
    """Convert the selected file list into structured metadata for budgeting.

    This is the handoff point between intent routing and token budgeting.
    token_budget.py relies on file, reason, intent, and intent_score to decide
    how much context each selected knowledge file receives.
    """
    ranked = _sorted_intents(intent_scores)
    if not ranked:
        return []

    top_intent, top_score = ranked[0]
    second_intent, second_score = ranked[1] if len(ranked) > 1 else (None, 0)
    primary_file = INTENTS[top_intent]["primary_file"]
    secondary_file = INTENTS[second_intent]["primary_file"] if second_intent else None
    details: list[dict[str, Any]] = []
    seen_files: set[str] = set()

    _add_file_detail(details, seen_files, primary_file, "primary_intent", top_intent, intent_scores)

    if (
        second_intent
        and secondary_file
        and secondary_file in selected_files
        and secondary_file != primary_file
        and confidence in {"medium", "low"}
        and second_score > 0
    ):
        _add_file_detail(details, seen_files, secondary_file, "secondary_intent", second_intent, intent_scores)

    for file_name in selected_files:
        if file_name in seen_files:
            continue

        if file_name == "eas_general.txt":
            if top_intent == "general_eas_info":
                _add_file_detail(details, seen_files, file_name, "primary_intent", top_intent, intent_scores)
                continue

            if file_name in INTENTS[top_intent].get("secondary_files", []):
                _add_file_detail(details, seen_files, file_name, "topic_support", FILE_TO_INTENT.get(file_name, top_intent), intent_scores)
                continue

            if confidence == "low":
                _add_file_detail(details, seen_files, file_name, "fallback", FILE_TO_INTENT.get(file_name, top_intent), intent_scores)
                continue

        if file_name == primary_file or file_name == secondary_file:
            continue

        mapped_intent = FILE_TO_INTENT.get(file_name, top_intent)
        reason = "topic_support" if mapped_intent != top_intent else "secondary_intent"
        _add_file_detail(details, seen_files, file_name, reason, mapped_intent, intent_scores)

    for topic in topics:
        file_name = TOPIC_TO_FILE.get(topic)
        if not file_name or file_name in seen_files or file_name == primary_file:
            continue

        mapped_intent = FILE_TO_INTENT.get(file_name, top_intent)
        _add_file_detail(details, seen_files, file_name, "topic_support", mapped_intent, intent_scores)

    return details


def route_intent(user_text: str) -> dict[str, Any]:
    """Main public entry point for intent, topic, and file routing.

    The controller calls this once per user input before any knowledge files are
    loaded.
    """
    normalized_text = normalize_user_text(user_text)
    intent_scores, scoring_reasons = score_intents(normalized_text)
    ranked = _sorted_intents(intent_scores)
    top_intent = ranked[0][0] if ranked else "general_eas_info"
    confidence = classify_confidence(intent_scores)

    topics = detect_topics(normalized_text)
    topics = _filter_topics_for_intent(topics, top_intent)
    all_intent_scores_zero = all(score == 0 for score in intent_scores.values())
    unrecognized = all_intent_scores_zero and not topics

    selected_files = select_files(intent_scores, topics, normalized_text)
    selected_file_details = build_selected_file_details(selected_files, topics, intent_scores, confidence)

    debug_reasons = list(scoring_reasons)
    for topic in topics:
        debug_reasons.append(f"topic detected: {topic}")

    if unrecognized:
        debug_reasons.append("input not recognized: all intent scores were zero and no topics were detected")

    return {
        "normalized_text": normalized_text,
        "intent": top_intent,
        "intent_scores": intent_scores,
        "confidence": confidence,
        "topics": topics,
        "all_intent_scores_zero": all_intent_scores_zero,
        "unrecognized": unrecognized,
        "selected_files": selected_files,
        "selected_file_details": selected_file_details,
        "debug_reasons": debug_reasons,
    }


def print_router_debug(route: dict[str, Any]) -> None:
    """Print routing decisions and the reason trail when debug mode is on."""
    if not DEBUG_ROUTER:
        return

    normalized_text = route.get("normalized_text", "")
    intent = route.get("intent", "unknown")
    confidence = route.get("confidence", "low")
    topics = route.get("topics", [])
    unrecognized = route.get("unrecognized", False)
    selected_files = route.get("selected_files", [])
    scores = route.get("intent_scores", {})
    reasons = route.get("debug_reasons", [])

    print(f"[Router] User text: {normalized_text}")
    print(f"[Router] Intent: {intent}")
    print(f"[Router] Confidence: {confidence}")
    print(f"[Router] Unrecognized: {unrecognized}")
    print(
        "[Router] Scores: "
        + ", ".join(f"{name}={score}" for name, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])))
    )
    print(f"[Router] Topics: {', '.join(topics) if topics else 'none'}")
    print(f"[Router] Files: {', '.join(selected_files) if selected_files else 'none'}")

    if reasons:
        print("[Router] Reasons:")
        for reason in reasons:
            print(f"  - {reason}")
