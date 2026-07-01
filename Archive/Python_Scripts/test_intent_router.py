from intent_router import route_intent


def run_tests() -> None:
    contact_route = route_intent("who do i contact to report this disease?")
    assert contact_route["intent"] == "contact_report"
    assert contact_route["topics"] == ["disease"]
    assert contact_route["selected_files"] == ["eas_contacts.txt"]

    disease_route = route_intent("what are the symptoms of the deep root virus?")
    assert disease_route["intent"] == "disease_info"
    assert disease_route["selected_files"] == ["eas_diseases.txt"]

    creature_route = route_intent("what is a mimic?")
    assert creature_route["intent"] == "creature_info"
    assert creature_route["selected_files"] == ["eas_creatures.txt"]

    location_route = route_intent("where was the mimic last seen?")
    assert location_route["intent"] == "location_info"
    assert location_route["selected_files"] == ["eas_locations.txt"]

    safety_route = route_intent("what should i do if i see nature's mockery?")
    assert safety_route["intent"] == "safety_protocol"
    assert safety_route["selected_files"] == ["eas_general.txt"]

    agency_route = route_intent("which agency issued the warning?")
    assert agency_route["intent"] == "agency_info"
    assert agency_route["selected_files"] == ["eas_agencies.txt"]

    current_route = route_intent("what is happening now?")
    assert current_route["intent"] == "current_activity_status"
    assert current_route["selected_files"] == ["eas_current_activity.txt"]

    general_route = route_intent("explain this alert")
    assert general_route["intent"] == "general_eas_info"
    assert general_route["selected_files"] == ["eas_general.txt"]

    misspelled_contact_route = route_intent("who do i contakt to report this desease?")
    assert misspelled_contact_route["intent"] == "contact_report"

    misspelled_creature_route = route_intent("what is a mimik?")
    assert misspelled_creature_route["intent"] == "creature_info"


if __name__ == "__main__":
    run_tests()
    print("intent router tests passed")
