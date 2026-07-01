from pathlib import Path

from intent_router import route_intent
from token_budget import (
    FULL_REPORT_RESPONSE_TOKENS,
    MAX_KNOWLEDGE_TOKENS,
    build_budgeted_knowledge_context,
    calculate_response_tokens,
    estimate_tokens,
)


def run_tests() -> None:
    knowledge_dir = Path(__file__).resolve().parents[1] / "memory" / "knowledge"

    contact_route = route_intent("who do i contact to report this disease?")
    contact_details = contact_route["selected_file_details"]
    assert contact_details[0]["file"] == "eas_contacts.txt"
    assert any(item["file"] == "eas_diseases.txt" and item["reason"] == "topic_support" for item in contact_details)
    assert calculate_response_tokens(contact_details, "who do i contact to report this disease?") <= 256

    contact_context, contact_metadata = build_budgeted_knowledge_context(contact_details, str(knowledge_dir))
    assert estimate_tokens(contact_context) <= MAX_KNOWLEDGE_TOKENS
    assert len(contact_metadata) <= 3

    disease_route = route_intent("what are the symptoms of the deep root virus?")
    disease_details = disease_route["selected_file_details"]
    assert disease_details[0]["file"] == "eas_diseases.txt"
    assert calculate_response_tokens(disease_details, "what are the symptoms of the deep root virus?") >= 256

    report_route = route_intent("give me a full report on the mimic")
    report_details = report_route["selected_file_details"]
    assert calculate_response_tokens(report_details, "give me a full report on the mimic") <= FULL_REPORT_RESPONSE_TOKENS


if __name__ == "__main__":
    run_tests()
    print("token budget tests passed")
