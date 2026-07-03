from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz


# Rule-based intent/topic router for Artemis.
#
# Intent answers: "What is the user trying to do?"
# Topic answers: "What subject should Artemis look up?"
# Intent is scored first and controls depth/behavior. Topic is scored second and
# controls which memory/knowledge files support the answer.

MAX_SELECTED_FILES = 4
FUZZY_PHRASE_THRESHOLD = 88
FUZZY_KEYWORD_THRESHOLD = 88
FUZZY_TOPIC_THRESHOLD = 88

# SET TO FALSE TO DISABLE DEBUG MODE
DEBUG_ROUTER = True

SEVERITY_PHRASES = {
    "urgent": [
        "i need help",
        "help me",
        "something is in my house",
        "someone is in my house",
        "someone is inside",
        "there is someone inside",
        "i think someone broke in",
        "i hear something downstairs",
        "something followed me",
        "i am scared",
        "i am in danger",
    ],
    "elevated": [
        "i heard something",
        "something is outside",
        "i saw something",
        "someone is watching",
        "there is a noise",
        "door is open",
        "window is open",
    ],
}

# INTENT_RULES is the main weighted rule table for deciding what the user is
# trying to do.
#
# Format:
# - The outer key is the intent name returned in route["primary_intent"].
# - "phrases" maps exact phrase strings to score weights.
# - "keywords" maps single-word triggers to score weights.
# - "combos" lists pairs of nearby words plus a score weight:
#   ("word_a", "word_b", score).
# - "depth" tells the prompt builder how much answer/detail the intent usually
#   needs.
# - "needs_knowledge" tells the router whether this intent should load local
#   memory/knowledge files.
#
# Score meaning:
# - Higher scores mean stronger evidence for that intent.
# - Strong phrases should usually have the highest weights because they show
#   clear user intent, such as "who should i contact".
# - Combos are middle-strength evidence because they catch flexible wording
#   where two important words appear near each other.
# - Keywords are weaker evidence because a single word can be ambiguous.
#
# Decision process:
# - score_intents() adds every matching phrase, keyword, and combo score.
# - The intent with the highest total becomes the primary intent.
# - Nearby lower-scoring intents can become secondary intents if they are close
#   enough to the winner.
# - If every intent scores 0, the router treats the input as out_of_scope unless
#   topic/severity logic adds a stronger route.
#
# RapidFuzz is used as a backup for phrase/keyword typos. Fuzzy matches add less
# score than exact matches so a misspelling can help routing without overpowering
# clearer exact rules.
INTENT_RULES: dict[str, dict[str, Any]] = {
    "identity_query": {
        "phrases": {
            "who are you": 12,
            "what are you": 12,
            "your name": 8,
            "who is artemis": 10,
        },
        "keywords": {
            "identity": 3,
            "name": 2,
        },
        "combos": [
            ("who", "you", 8),
        ],
        "depth": "none",
        "needs_knowledge": False,
    },
    "mode_query": {
        "phrases": {
            "what mode are you in": 14,
            "current mode": 8,
            "which mode": 8,
            "what is eas mode": 6,
        },
        "keywords": {
            "mode": 4,
        },
        "combos": [
            ("mode", "you", 6),
        ],
        "depth": "none",
        "needs_knowledge": False,
    },
    "general_chat": {
        "phrases": {
            "hello": 6,
            "hello artemis": 10,
            "hi": 6,
            "hey": 6,
            "thank you": 6,
            "thanks": 6,
        },
        "keywords": {
            "hello": 4,
            "hi": 4,
            "hey": 4,
            "thanks": 3,
        },
        "combos": [],
        "depth": "none",
        "needs_knowledge": False,
    },
    "current_status": {
        "phrases": {
            "anything to report": 14,
            "anything reported": 14,
            "any reports": 14,
            "any updates": 14,
            "current status": 12,
            "latest report": 12,
            "recent report": 12,
            "recent activity": 12,
            "anything happening": 12,
            "what is happening": 12,
            "status report": 10,
        },
        "keywords": {
            "current": 3,
            "currently": 3,
            "active": 3,
            "recent": 3,
            "recently": 3,
            "latest": 3,
            "new": 2,
            "today": 3,
            "now": 3,
            "lately": 3,
            "status": 3,
            "update": 3,
            "updates": 3,
            "alert": 2,
            "alerts": 2,
            "happening": 3,
            "reported": 3,
            "reports": 3,
            "anything": 2,
        },
        "combos": [
            ("anything", "report", 9),
            ("any", "reports", 9),
            ("report", "recently", 8),
            ("updates", "today", 8),
            ("anything", "happening", 8),
        ],
        "depth": "normal",
        "needs_knowledge": True,
    },
    "contact_reporting": {
        "phrases": {
            "who should i contact": 16,
            "who do i contact": 16,
            "how do i report": 16,
            "where do i report": 16,
            "report this": 12,
            "what number do i call": 14,
            "what office": 10,
            "who should i call": 12,
            "who do i call": 12,
        },
        "keywords": {
            "contact": 4,
            "phone": 3,
            "email": 3,
            "address": 3,
            "office": 4,
            "agency": 3,
            "call": 3,
            "notify": 3,
        },
        "combos": [
            ("who", "contact", 12),
            ("how", "report", 12),
            ("where", "report", 12),
            ("number", "call", 10),
            ("report", "this", 8),
        ],
        "depth": "normal",
        "needs_knowledge": True,
    },
    "explanation": {
        "phrases": {
            "what is": 8,
            "what are": 8,
            "tell me about": 12,
            "explain": 10,
            "what does this mean": 12,
            "what does": 7,
        },
        "keywords": {
            "explain": 5,
            "meaning": 4,
            "describe": 4,
            "about": 2,
        },
        "combos": [
            ("what", "disease", 8),
            ("what", "creature", 8),
            ("tell", "about", 8),
        ],
        "depth": "normal",
        "needs_knowledge": True,
    },
    "safety_instruction": {
        "phrases": {
            "what should i do": 16,
            "what do i do": 14,
            "how do i stay safe": 14,
            "should i evacuate": 12,
            "should i shelter": 12,
            "is it safe": 10,
            "safety instructions": 10,
            "i need help": 18,
            "help me": 18,
            "i am in danger": 20,
            "i am scared": 16,
            "something is in my house": 22,
            "someone is in my house": 22,
            "someone is inside": 20,
            "i think someone broke in": 22,
        },
        "keywords": {
            "safe": 3,
            "safety": 4,
            "protocol": 3,
            "instructions": 4,
            "evacuate": 4,
            "shelter": 4,
            "symptoms": 2,
            "symptom": 2,
            "danger": 5,
            "scared": 4,
            "inside": 3,
            "downstairs": 4,
            "followed": 4,
        },
        "combos": [
            ("should", "do", 12),
            ("what", "do", 10),
            ("has", "symptoms", 9),
            ("someone", "symptoms", 9),
            ("stay", "safe", 10),
            ("someone", "inside", 14),
            ("something", "house", 14),
            ("someone", "house", 14),
            ("door", "open", 8),
            ("window", "open", 8),
        ],
        "depth": "deep",
        "needs_knowledge": True,
    },
    "location_query": {
        "phrases": {
            "where is": 14,
            "where are": 12,
            "what area": 10,
            "which area": 10,
            "what zone": 10,
            "which zone": 10,
        },
        "keywords": {
            "where": 5,
            "location": 4,
            "located": 4,
            "area": 3,
            "zone": 3,
            "station": 4,
            "sector": 3,
            "coordinates": 4,
        },
        "combos": [
            ("where", "station", 10),
            ("where", "location", 10),
        ],
        "depth": "normal",
        "needs_knowledge": True,
    },
    "out_of_scope": {
        "phrases": {},
        "keywords": {},
        "combos": [],
        "depth": "none",
        "needs_knowledge": False,
    },
}


TOPIC_RULES: dict[str, dict[str, int]] = {
    "safety": {
        "safety": 5,
        "safe": 4,
        "protocol": 4,
        "urgent": 5,
        "help": 4,
        "danger": 5,
        "scared": 4,
        "house": 3,
        "inside": 4,
        "downstairs": 4,
        "outside": 3,
        "watching": 4,
        "noise": 3,
        "door is open": 5,
        "window is open": 5,
        "someone is inside": 7,
        "something is in my house": 8,
        "someone is in my house": 8,
    },
    "disease": {
        "disease": 5,
        "diseases": 5,
        "virus": 5,
        "infection": 5,
        "infected": 4,
        "symptom": 5,
        "symptoms": 5,
        "outbreak": 4,
        "medical": 4,
        "contagion": 4,
        "deep root": 8,
        "nature's mockery": 5,
    },
    "creature": {
        "creature": 5,
        "creatures": 5,
        "entity": 5,
        "entities": 5,
        "monster": 4,
        "mimic": 6,
        "mimics": 6,
        "woodcrawler": 6,
        "vita carnis": 8,
        "gemini": 5,
        "fake people": 6,
        "wretch": 5,
    },
    "agency": {
        "agency": 5,
        "agencies": 5,
        "department": 4,
        "office": 4,
        "authority": 4,
        "bureau": 4,
        "official": 3,
    },
    "location": {
        "where": 4,
        "location": 5,
        "located": 5,
        "area": 4,
        "zone": 4,
        "station": 5,
        "sector": 4,
        "coordinates": 5,
        "facility": 4,
    },
    "contact": {
        "contact": 5,
        "phone": 5,
        "email": 5,
        "address": 5,
        "call": 4,
        "number": 4,
        "notify": 4,
        "report this": 4,
    },
    "current_activity": {
        "current": 4,
        "currently": 4,
        "recent": 4,
        "recently": 4,
        "latest": 4,
        "today": 4,
        "now": 4,
        "lately": 4,
        "status": 4,
        "update": 4,
        "updates": 4,
        "reported": 4,
        "reports": 4,
        "anything to report": 6,
        "anything reported": 6,
        "any reports": 6,
        "any updates": 6,
    },
    "general": {
        "hello": 3,
        "hi": 3,
        "hey": 3,
        "artemis": 2,
        "general": 3,
        "eas": 3,
    },
}


TOPIC_TO_FILE = {
    "safety": "eas_safety_protocols.txt",
    "disease": "eas_diseases.txt",
    "creature": "eas_creatures.txt",
    "agency": "eas_agencies.txt",
    "location": "eas_locations.txt",
    "contact": "eas_contacts.txt",
    "current_activity": "eas_current_activity.txt",
    "general": "eas_general.txt",
}


INTENT_PRIMARY_FILES = {
    "current_status": ["eas_current_activity.txt"],
    "contact_reporting": ["eas_contacts.txt", "eas_agencies.txt"],
    "safety_instruction": ["eas_safety_protocols.txt"],
    "location_query": ["eas_locations.txt"],
}


def normalize_text(text: str) -> str:
    """Normalize user input for repeatable rule matching."""
    normalized = text.lower().strip()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[\u0000-\u001f\u007f-\u009f]", " ", normalized)
    normalized = re.sub(r"[^a-z0-9\s']", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()

def tokenize_text(text: str) -> set[str]:
    """Return normalized word tokens for keyword scoring."""
    return set(normalize_text(text).split())


def _contains_term(normalized_text: str, term: str) -> bool:
    normalized_term = normalize_text(term)
    if not normalized_term:
        return False

    if " " in normalized_term or "'" in normalized_term:
        return normalized_term in normalized_text

    return re.search(rf"\b{re.escape(normalized_term)}\b", normalized_text) is not None


def _words_near(normalized_text: str, word_a: str, word_b: str, max_gap: int = 5) -> bool:
    tokens = normalized_text.split()
    positions_a = [index for index, token in enumerate(tokens) if token == word_a]
    positions_b = [index for index, token in enumerate(tokens) if token == word_b]

    return any(abs(left - right) <= max_gap for left in positions_a for right in positions_b)


def _similarity(left: str, right: str) -> int:
    """Return a 0-100 similarity score using rapidfuzz."""
    return int(fuzz.ratio(left, right))


def _partial_similarity(needle: str, haystack: str) -> int:
    """Return a partial phrase score using rapidfuzz."""
    if not needle or not haystack:
        return 0

    return int(fuzz.partial_ratio(needle, haystack))


def _best_token_similarity(term: str, tokens: set[str]) -> int:
    """Compare one keyword against all input tokens and return the best score."""
    if not term or not tokens:
        return 0

    return max(_similarity(term, token) for token in tokens)


def _score_rule_set(user_text: str, rules: dict[str, Any]) -> int:
    normalized = normalize_text(user_text)
    tokens = tokenize_text(normalized)
    score = 0

    for phrase, weight in rules.get("phrases", {}).items():
        if _contains_term(normalized, phrase):
            score += int(weight)
            continue

        fuzzy_score = _partial_similarity(normalize_text(phrase), normalized)
        if fuzzy_score >= FUZZY_PHRASE_THRESHOLD:
            # Fuzzy phrase matches are intentionally weaker than exact phrase
            # matches. They catch typos without letting near misses dominate.
            score += max(2, min(6, int(weight) // 2))

    for keyword, weight in rules.get("keywords", {}).items():
        if keyword in tokens:
            score += int(weight)
            continue

        if _best_token_similarity(normalize_text(keyword), tokens) >= FUZZY_KEYWORD_THRESHOLD:
            score += max(1, min(2, int(weight) // 2))

    for word_a, word_b, weight in rules.get("combos", []):
        if _words_near(normalized, word_a, word_b):
            score += int(weight)

    return score


def score_intents(user_text: str) -> dict[str, int]:
    """Score each intent with exact phrases, keywords, and proximity combos."""
    return {
        intent_name: _score_rule_set(user_text, rules)
        for intent_name, rules in INTENT_RULES.items()
    }


def score_topics(user_text: str) -> dict[str, int]:
    """Score each topic independently from intent."""
    normalized = normalize_text(user_text)
    tokens = tokenize_text(normalized)
    scores: dict[str, int] = {}

    for topic_name, weighted_terms in TOPIC_RULES.items():
        topic_score = 0
        for term, weight in weighted_terms.items():
            normalized_term = normalize_text(term)
            if " " in normalized_term or "'" in normalized_term:
                if normalized_term in normalized:
                    topic_score += weight
                elif _partial_similarity(normalized_term, normalized) >= FUZZY_TOPIC_THRESHOLD:
                    topic_score += max(1, min(3, weight // 2))
            elif normalized_term in tokens:
                topic_score += weight
            elif (
                not (topic_name == "current_activity" and normalized_term in {"reports", "reported"})
                and _best_token_similarity(normalized_term, tokens) >= FUZZY_TOPIC_THRESHOLD
            ):
                topic_score += max(1, min(3, weight // 2))

        scores[topic_name] = topic_score

    # "report" by itself is ambiguous. Only treat it as current activity when it
    # appears in a status-style shape such as "anything ... report".
    if _words_near(normalized, "anything", "report") or _contains_term(normalized, "any reports"):
        scores["current_activity"] += 5

    return scores


def score_severity(user_text: str) -> dict[str, int]:
    """Score urgent/elevated safety language separately from normal topics."""
    normalized = normalize_text(user_text)
    scores = {severity: 0 for severity in SEVERITY_PHRASES}

    for severity, phrases in SEVERITY_PHRASES.items():
        for phrase in phrases:
            normalized_phrase = normalize_text(phrase)
            if _contains_term(normalized, normalized_phrase):
                scores[severity] += 10 if severity == "urgent" else 6
                continue

            if _partial_similarity(normalized_phrase, normalized) >= FUZZY_PHRASE_THRESHOLD:
                scores[severity] += 5 if severity == "urgent" else 3

    return scores


def _severity_level(severity_scores: dict[str, int]) -> str:
    if severity_scores.get("urgent", 0) > 0:
        return "urgent"
    if severity_scores.get("elevated", 0) > 0:
        return "elevated"
    return "none"


def _rank_scores(scores: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _choose_primary_intent(intent_scores: dict[str, int]) -> str:
    ranked = _rank_scores(intent_scores)
    if not ranked or ranked[0][1] <= 0:
        return "out_of_scope"

    return ranked[0][0]


def _choose_secondary_intents(intent_scores: dict[str, int], primary_intent: str) -> list[str]:
    primary_score = intent_scores.get(primary_intent, 0)
    secondary: list[str] = []

    for intent_name, score in _rank_scores(intent_scores):
        if intent_name in {primary_intent, "out_of_scope"}:
            continue
        if score >= 6 and primary_score - score <= 8:
            secondary.append(intent_name)

    return secondary[:2]


def _choose_topics(topic_scores: dict[str, int], primary_intent: str) -> list[str]:
    topics = [topic for topic, score in _rank_scores(topic_scores) if score > 0]

    if not topics and primary_intent in {"identity_query", "mode_query", "general_chat"}:
        topics = ["general"]

    return topics


def _confidence(intent_scores: dict[str, int], topic_scores: dict[str, int]) -> int:
    top_intent = max(intent_scores.values(), default=0)
    top_topic = max(topic_scores.values(), default=0)
    return min(100, (top_intent * 5) + (top_topic * 3))


def _depth_for_route(primary_intent: str, topics: list[str]) -> str:
    if primary_intent == "out_of_scope":
        return "none"
    if primary_intent in {"identity_query", "mode_query", "general_chat"}:
        return "none"
    if primary_intent == "safety_instruction":
        return "deep"
    if primary_intent == "contact_reporting" and topics:
        return "normal"
    return str(INTENT_RULES.get(primary_intent, {}).get("depth", "shallow"))


def _needs_knowledge(primary_intent: str, topics: list[str]) -> bool:
    if primary_intent in {"identity_query", "mode_query", "general_chat", "out_of_scope"}:
        return False
    return bool(INTENT_RULES.get(primary_intent, {}).get("needs_knowledge", False) or topics)


def _add_unique(items: list[str], item: str) -> None:
    if item and item not in items:
        items.append(item)


def _select_files_from_route(route: dict[str, Any]) -> list[str]:
    primary_intent = route["primary_intent"]
    topics = route["topics"]
    severity = route.get("severity", "none")

    if not route["needs_knowledge"]:
        return []

    selected: list[str] = []

    for file_name in INTENT_PRIMARY_FILES.get(primary_intent, []):
        _add_unique(selected, file_name)

    if primary_intent == "explanation":
        for topic in topics:
            if topic in {"safety", "disease", "creature", "agency", "location", "general"}:
                _add_unique(selected, TOPIC_TO_FILE[topic])

    if primary_intent == "safety_instruction":
        if "disease" in topics:
            _add_unique(selected, "eas_diseases.txt")
            _add_unique(selected, "eas_contacts.txt")
        elif "creature" in topics:
            _add_unique(selected, "eas_creatures.txt")
            _add_unique(selected, "eas_contacts.txt")
        elif severity == "urgent":
            _add_unique(selected, "eas_contacts.txt")
        else:
            _add_unique(selected, "eas_general.txt")

    if primary_intent == "contact_reporting":
        if "disease" in topics:
            _add_unique(selected, "eas_diseases.txt")
        if "creature" in topics:
            _add_unique(selected, "eas_creatures.txt")

    if primary_intent == "current_status":
        for topic in topics:
            if topic in {"disease", "creature", "location", "agency"}:
                _add_unique(selected, TOPIC_TO_FILE[topic])

    if primary_intent == "location_query":
        _add_unique(selected, "eas_locations.txt")

    return selected[:MAX_SELECTED_FILES]


def _file_reason(file_name: str, primary_intent: str) -> str:
    if file_name in INTENT_PRIMARY_FILES.get(primary_intent, []):
        return "primary_intent"
    return "topic_support"


def _file_intent(file_name: str, primary_intent: str) -> str:
    if file_name in INTENT_PRIMARY_FILES.get(primary_intent, []):
        return primary_intent

    for topic, topic_file in TOPIC_TO_FILE.items():
        if file_name == topic_file:
            return topic

    return primary_intent


def build_selected_file_details(route: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert selected files into token_budget.py metadata."""
    primary_intent = route["primary_intent"]
    intent_scores = route["intent_scores"]
    details: list[dict[str, Any]] = []

    for file_name in route["selected_files"]:
        mapped_intent = _file_intent(file_name, primary_intent)
        details.append(
            {
                "file": file_name,
                "reason": _file_reason(file_name, primary_intent),
                "intent": mapped_intent,
                "intent_score": intent_scores.get(primary_intent, 0),
            }
        )

    return details


def analyze_query(user_text: str) -> dict[str, Any]:
    """Build a complete route object for one user prompt."""
    normalized_text = normalize_text(user_text)
    intent_scores = score_intents(normalized_text)
    topic_scores = score_topics(normalized_text)
    severity_scores = score_severity(normalized_text)
    severity = _severity_level(severity_scores)

    if severity == "urgent":
        intent_scores["safety_instruction"] += 24
        topic_scores["safety"] += 12
    elif severity == "elevated":
        intent_scores["safety_instruction"] += 14
        topic_scores["safety"] += 8

    primary_intent = _choose_primary_intent(intent_scores)
    secondary_intents = _choose_secondary_intents(intent_scores, primary_intent)
    topics = _choose_topics(topic_scores, primary_intent)
    confidence = _confidence(intent_scores, topic_scores)
    depth = _depth_for_route(primary_intent, topics)
    needs_knowledge = _needs_knowledge(primary_intent, topics)

    route: dict[str, Any] = {
        "normalized_text": normalized_text,
        "primary_intent": primary_intent,
        "secondary_intents": secondary_intents,
        "intent_scores": intent_scores,
        "topics": topics,
        "topic_scores": topic_scores,
        "severity": severity,
        "severity_scores": severity_scores,
        "confidence": confidence,
        "needs_knowledge": needs_knowledge,
        "depth": depth,
    }
    route["selected_files"] = _select_files_from_route(route)
    route["selected_file_details"] = build_selected_file_details(route)
    route["unrecognized"] = primary_intent == "out_of_scope" and not topics

    # Compatibility keys used by the current controller/debug code.
    route["intent"] = primary_intent
    route["route_intent"] = primary_intent
    route["route_topics"] = topics
    route["route_confidence"] = confidence
    route["route_depth"] = depth
    route["route_severity"] = severity
    route["debug_reasons"] = _build_debug_reasons(route)
    return route


def get_relevant_knowledge_files(user_text: str) -> list[str]:
    """Compatibility helper returning only the selected file names."""
    return analyze_query(user_text)["selected_files"]


def route_intent(user_text: str) -> dict[str, Any]:
    """Main controller entry point."""
    return analyze_query(user_text)


def _build_debug_reasons(route: dict[str, Any]) -> list[str]:
    reasons = [
        f"primary intent selected: {route['primary_intent']}",
        f"depth selected: {route['depth']}",
        "fuzzy scorer: rapidfuzz",
    ]

    if route["secondary_intents"]:
        reasons.append(f"secondary intents: {', '.join(route['secondary_intents'])}")

    if route["severity"] != "none":
        reasons.append(f"severity detected: {route['severity']}")

    if route["topics"]:
        reasons.append(f"topics detected: {', '.join(route['topics'])}")

    if route["unrecognized"]:
        reasons.append("input not recognized: all intent and topic scores were zero")

    return reasons


def _format_scores(scores: dict[str, int]) -> str:
    return ", ".join(f"{name}={score}" for name, score in _rank_scores(scores))


def print_router_debug(route: dict[str, Any]) -> None:
    """Print routing decisions and score details when debug mode is on."""
    if not DEBUG_ROUTER:
        return

    print(f"[Router] User text: {route.get('normalized_text', '')}")
    print(f"[Router] Intent: {route.get('primary_intent', 'unknown')}")
    print(f"[Router] Secondary intents: {', '.join(route.get('secondary_intents', [])) or 'none'}")
    print(f"[Router] Confidence: {route.get('confidence', 0)}")
    print(f"[Router] Depth: {route.get('depth', 'none')}")
    print(f"[Router] Severity: {route.get('severity', 'none')}")
    print(f"[Router] Needs knowledge: {route.get('needs_knowledge', False)}")
    print(f"[Router] Unrecognized: {route.get('unrecognized', False)}")
    print("[Router] Fuzzy scorer: rapidfuzz")
    print(f"[Router] Intent scores: {_format_scores(route.get('intent_scores', {}))}")
    print(f"[Router] Topic scores: {_format_scores(route.get('topic_scores', {}))}")
    print(f"[Router] Severity scores: {_format_scores(route.get('severity_scores', {}))}")
    print(f"[Router] Topics: {', '.join(route.get('topics', [])) or 'none'}")
    print(f"[Router] Files: {', '.join(route.get('selected_files', [])) or 'none'}")

    reasons = route.get("debug_reasons", [])
    if reasons:
        print("[Router] Reasons:")
        for reason in reasons:
            print(f"  - {reason}")
