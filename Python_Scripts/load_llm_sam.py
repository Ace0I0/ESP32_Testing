import sys
import time
import threading
import json
import subprocess
import urllib.request
from pathlib import Path


# Constants to be used for color coding certain output (such as errors and responses)
RESET = "\033[0m"

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
PURPLE = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

BOLD = "\033[1m"
DIM = "\033[2m"

# Load LM with this URL on PI
LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

SAM_BINARY = Path("/home/pi/SAM/sam")
OUTPUT_WAV = Path("/home/pi/SAM/robot_response.wav")

SCRIPT_DIR = Path(__file__).resolve().parent

def find_project_dir(start_dir: Path) -> Path:
    for directory in [start_dir, *start_dir.parents]:
        if (directory / "memory").is_dir():
            return directory

    return start_dir

PROJECT_DIR = find_project_dir(SCRIPT_DIR)
MEMORY_DIR = PROJECT_DIR / "memory"
MODES_DIR = MEMORY_DIR / "modes"

KNOWLEDGE_DIR = MEMORY_DIR / "knowledge"
LLM_DEBUG = True
KNOWLEDGE_CHAR_BUDGET = 12000
MAX_SECTIONS_PER_FILE = 3
GENERIC_RETRIEVAL_WORDS = {
    "active",
    "activity",
    "advisory",
    "alert",
    "current",
    "happening",
    "report",
    "reports",
    "status",
    "update",
    "watch",
    "what",
    "when",
    "where",
    "which",
    "should",
    "this",
    "that",
}

# Defines the basic identity of Artemis
IDENTITY_PATH = MEMORY_DIR / "identity.json"

# Singlular mode for now, 3 total in the future
MODE_PATH = MODES_DIR / "EAS.json"


# General json loader
def load_json_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{RED}Missing JSON file: {path}{RESET}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)

# loads identity json file
def load_identity() -> dict:
    return load_json_file(IDENTITY_PATH)

# loads mode file
def load_mode() -> dict:
    return load_json_file(MODE_PATH)

# general list formater (we have stuff saved in a "list" style in stuff like identity.json and EAS.json which is one of our modes)
def format_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)

def debug_log(message: str) -> None:
    if LLM_DEBUG:
        print(f"{DIM}{CYAN}[debug] {message}{RESET}")

def estimate_token_count(text: str) -> int:
    # Rough English estimate. llama.cpp's server timings below are the real source when available.
    return max(1, len(text) // 4) if text else 0

def keyword_matches(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return sum(lower.count(keyword) for keyword in keywords)

def has_any_keyword(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(keyword in lower for keyword in keywords)

def split_knowledge_sections(text: str) -> list[dict]:
    sections = []
    current_title = "Overview"
    current_lines = []

    for line in text.splitlines():
        if line.startswith("##"):
            if current_lines:
                sections.append(
                    {
                        "title": current_title,
                        "text": "\n".join(current_lines).strip(),
                    }
                )

            current_title = line.lstrip("#").strip() or "Untitled"
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "title": current_title,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return [section for section in sections if section["text"]]

def get_knowledge_query_keywords(user_text: str) -> list[str]:
    words = [
        word.strip(".,!?;:()[]{}\"'")
        for word in user_text.lower().split()
    ]
    words = [word for word in words if len(word) >= 4]

    synonym_groups = {
        "deep root": ["deep root", "root", "roots", "bulb", "bulbs", "sprout", "sprouts", "disease", "virus"],
        "nature's mockery": ["nature's mockery", "mockery", "plant", "fungus", "growth", "vine", "vines"],
        "woodcrawler": ["woodcrawler", "crawlspace", "floor", "floorboard", "under", "knocking"],
        "fake people": ["fake people", "duplicate", "person", "family", "imitate", "voice", "help"],
        "mimic": ["mimic", "person", "humanoid", "figure", "window", "outside"],
        "harvester": ["harvester", "bulb", "tendril", "woods", "trail"],
        "host of influence": ["host", "influence", "spore", "spores", "dust", "coordination"],
        "crawl": ["crawl", "tendril", "tendrils", "meat", "antenna", "signal"],
        "report": ["report", "contact", "phone", "email", "address", "agency", "office"],
        "current": ["current", "active", "status", "happening", "advisory", "watch"],
    }

    base_keywords = set(words)
    keywords = set(words)
    lower = user_text.lower()

    for trigger, additions in synonym_groups.items():
        if trigger in lower or any(word in base_keywords for word in additions):
            keywords.update(additions)

    specific_keywords = keywords - GENERIC_RETRIEVAL_WORDS
    return sorted(specific_keywords or keywords)

def score_section(section: dict, keywords: list[str]) -> int:
    title_score = keyword_matches(section["title"], keywords) * 4
    body_score = keyword_matches(section["text"], keywords)
    return title_score + body_score

# Loads the knowledge file which has things like lore (this has different categories as there is too much info to load it all at once)
def load_selected_knowledge(files: list[str], user_text: str) -> tuple[str, list[dict]]:
    sections = []
    debug_files = []
    query_keywords = get_knowledge_query_keywords(user_text)
    remaining_chars = KNOWLEDGE_CHAR_BUDGET

    for filename in files:
        path = KNOWLEDGE_DIR / filename
        if path.exists():
            raw_text = path.read_text(encoding="utf-8").strip()
            parsed_sections = split_knowledge_sections(raw_text)
            scored_sections = []

            for section in parsed_sections:
                score = score_section(section, query_keywords)
                if section["title"] == "Overview":
                    score += 1

                scored_sections.append((score, section))

            selected_sections = [
                section
                for score, section in sorted(scored_sections, key=lambda item: item[0], reverse=True)
                if score > 0
            ][:MAX_SECTIONS_PER_FILE]

            if not selected_sections and parsed_sections:
                selected_sections = parsed_sections[:1]

            selected_text_blocks = []

            for section in selected_sections:
                block = f"## {section['title']}\n{section['text']}"
                if len(block) > remaining_chars:
                    block = block[:remaining_chars].rstrip()

                if block:
                    selected_text_blocks.append(block)
                    remaining_chars -= len(block)

                if remaining_chars <= 0:
                    break

            text = "\n\n".join(selected_text_blocks)
            debug_files.append(
                {
                    "name": filename,
                    "path": str(path),
                    "exists": True,
                    "bytes": path.stat().st_size,
                    "raw_chars": len(raw_text),
                    "chars": len(text),
                    "estimated_tokens": estimate_token_count(text),
                    "sections_found": len(parsed_sections),
                    "sections_loaded": [section["title"] for section in selected_sections],
                }
            )
            if text:
                sections.append(f"Knowledge file: {filename}\n{text}")
        else:
            debug_files.append(
                {
                    "name": filename,
                    "path": str(path),
                    "exists": False,
                    "bytes": 0,
                    "raw_chars": 0,
                    "chars": 0,
                    "estimated_tokens": 0,
                    "sections_found": 0,
                    "sections_loaded": [],
                }
            )

        if remaining_chars <= 0:
            break

    return "\n\n".join(sections), debug_files

# formats examples listed in modes (may or may not remove this, but with that i have in mind rn, it should be useful)
def format_examples(examples: list[dict]) -> str:
    lines = []

    for example in examples:
        user = example.get("user", "").strip()
        assistant = example.get("assistant", "").strip()

        if user and assistant:
            lines.append(f"User: {user}\nArtemis: {assistant}")

    return "\n\n".join(lines)

# Sets up a check to see when we need to extend the max tokens and temperature, honestly later i wanna switch this to a physical switch rather than 
# just a detection for key words
def get_generation_settings(user_text: str) -> dict:
    lower = user_text.lower()

    # Current list of words to look out for
    long_request_words = ["broadcast", "psa", "announcement", "full alert", "longer", "detailed"]

    if any(word in lower for word in long_request_words):
        return {
            "max_tokens": 260,
            "temperature": 0.75,
        }

    return {
        "max_tokens": 160,
        "temperature": 0.65,
    }

# Decides which file is loaded depending on user input, detecting keywords like the previous function
# Update: Lot of keywords added and "what-if" scenarios used
def get_relevant_knowledge_files(user_text: str) -> list[str]:
    text = user_text.lower()

    files = ["eas_general.txt"] # We load the eas general txt file as this contains a basis for Artemis.

    contact_words = ["contact", "phone", "email", "address", "report this", "report it", "who should", "agency", "office", "call", "send this"]
    current_activity_words = ["activity", "alert", "update", "happening", "event", "events", "status", "current", "active", "right now", "today", "advisory"]
    disease_words = ["disease", "virus", "outbreak", "infection", "symptom", "symptoms", "contagion", "medical", "deep root", "root disease", "root virus", "bulb", "sprout"]
    creature_words = ["creature", "entity", "entities", "monster", "mimic", "woodcrawler", "vita carnis", "gemini", "nature's mockery", "fake people", "wretch", "harvester", "crawl", "singularity", "monolith", "host of influence", "person", "figure", "humanoid", "window", "outside", "wrong walking"]
    location_words = ["where", "location", "located", "place", "area", "zone", "sector", "station", "shelter", "lake", "road", "tree line"]

    wants_contact = has_any_keyword(text, contact_words)

    if wants_contact:
        files.append("eas_contacts.txt")
        files.append("eas_agencies.txt")

    if has_any_keyword(text, current_activity_words) and not wants_contact:
        files.append("eas_current_activity.txt")

    if has_any_keyword(text, disease_words):
        files.append("eas_diseases.txt")

    if has_any_keyword(text, creature_words):
        files.append("eas_creatures.txt")

    if has_any_keyword(text, location_words):
        files.append("eas_locations.txt")

    return list(dict.fromkeys(files))

# Using for time debugging, so i can see how long each response takes
def format_elapsed(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"

    return f"{seconds:.2f} sec"

# System prompt builder -> Accumulates all needed info so Artemis can function correctly based on the mode chosen
def build_system_prompt(identity: dict, mode: dict, user_text: str) -> tuple[str, dict]:
    name = identity.get("name", "Artemis")
    robot_type = identity.get("type", "retro animatronic robot")
    voice_engine = identity.get("voice_engine", "SAM")

    core_rules = format_list(identity.get("core_rules", []))
    tone = ", ".join(mode.get("tone", []))
    response_rules = format_list(mode.get("response_rules", []))
    examples = format_examples(mode.get("example_responses", []))

    selected_files = get_relevant_knowledge_files(user_text)
    knowledge_text, knowledge_debug = load_selected_knowledge(selected_files, user_text)

    system_prompt = (
        f"You are {name}, a {robot_type}.\n"
        f"Your voice engine is {voice_engine}.\n\n"
        "Core identity rules:\n"
        f"{core_rules}\n\n"
        f"Current mode: {mode.get('display_name', 'EAS Mode')}\n"
        f"Mode name: {mode.get('mode_name', 'eas')}\n"
        f"Mode description: {mode.get('description', '')}\n"
        f"Tone: {tone}\n\n"
        "Mode response rules:\n"
        f"{response_rules}\n\n"
        "Example responses:\n"
        f"{examples}\n\n"
        "Relevant local knowledge:\n"
        f"{knowledge_text}\n\n"
        "Final output rules:\n"
        f"- Respond as {name}.\n"
        "- Write in normal English.\n"
        "- Do not use markdown.\n"
        "- Keep the response suitable for old robotic text to speech.\n"
    )

    debug_info = {
        "knowledge_dir": str(KNOWLEDGE_DIR),
        "selected_files": selected_files,
        "knowledge_files": knowledge_debug,
        "knowledge_chars": len(knowledge_text),
        "knowledge_estimated_tokens": estimate_token_count(knowledge_text),
        "knowledge_char_budget": KNOWLEDGE_CHAR_BUDGET,
        "query_keywords": get_knowledge_query_keywords(user_text),
        "system_prompt_chars": len(system_prompt),
        "system_prompt_estimated_tokens": estimate_token_count(system_prompt),
        "user_chars": len(user_text),
        "user_estimated_tokens": estimate_token_count(user_text),
    }

    return system_prompt, debug_info

def ask_local_model(user_text: str, system_prompt: str) -> tuple[str, dict]:
    settings = get_generation_settings(user_text)
    payload = {
        "model": "local-qwen",
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_text,
            },
        ],
        "max_tokens": settings["max_tokens"],
        "temperature": settings["temperature"],
    }
    request_started = time.perf_counter()

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        LLAMA_SERVER_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))
    request_elapsed = time.perf_counter() - request_started

    message = result["choices"][0]["message"]["content"].strip()
    server_debug = {
        "request_elapsed": request_elapsed,
        "settings": settings,
        "usage": result.get("usage", {}),
        "timings": result.get("timings", {}),
        "response_chars": len(message),
        "response_estimated_tokens": estimate_token_count(message),
    }

    return message, server_debug

def print_prompt_debug(debug_info: dict) -> None:
    if not LLM_DEBUG:
        return

    debug_log(f"knowledge folder: {debug_info['knowledge_dir']}")
    debug_log(
        "knowledge retrieval: "
        f"budget={debug_info['knowledge_char_budget']} chars, "
        f"query_keywords={', '.join(debug_info['query_keywords'])}"
    )
    debug_log("selected knowledge files:")

    for file_info in debug_info["knowledge_files"]:
        status = "loaded" if file_info["exists"] else "missing"
        debug_log(
            f"  {file_info['name']} - {status}, "
            f"{file_info['bytes']} bytes, "
            f"{file_info['chars']}/{file_info['raw_chars']} chars loaded, "
            f"~{file_info['estimated_tokens']} tokens, "
            f"sections={file_info['sections_loaded']}"
        )

    debug_log(
        "prompt size: "
        f"system={debug_info['system_prompt_chars']} chars "
        f"(~{debug_info['system_prompt_estimated_tokens']} tokens), "
        f"knowledge={debug_info['knowledge_chars']} chars "
        f"(~{debug_info['knowledge_estimated_tokens']} tokens), "
        f"user={debug_info['user_chars']} chars "
        f"(~{debug_info['user_estimated_tokens']} tokens)"
    )

def print_server_debug(server_debug: dict) -> None:
    if not LLM_DEBUG:
        return

    usage = server_debug.get("usage", {})
    timings = server_debug.get("timings", {})
    settings = server_debug.get("settings", {})

    debug_log(
        "request settings: "
        f"max_tokens={settings.get('max_tokens')}, "
        f"temperature={settings.get('temperature')}"
    )
    debug_log(
        "server response: "
        f"wall={format_elapsed(server_debug.get('request_elapsed', 0.0))}, "
        f"response={server_debug.get('response_chars', 0)} chars "
        f"(~{server_debug.get('response_estimated_tokens', 0)} tokens)"
    )

    if usage:
        debug_log(
            "server usage: "
            f"prompt_tokens={usage.get('prompt_tokens')}, "
            f"completion_tokens={usage.get('completion_tokens')}, "
            f"total_tokens={usage.get('total_tokens')}"
        )

    if timings:
        prompt_ms = timings.get("prompt_ms")
        predicted_ms = timings.get("predicted_ms")
        prompt_n = timings.get("prompt_n")
        predicted_n = timings.get("predicted_n")
        debug_log(
            "llama.cpp timings: "
            f"prompt_eval={format_elapsed(prompt_ms / 1000) if prompt_ms is not None else 'n/a'} "
            f"({prompt_n} tokens), "
            f"generation={format_elapsed(predicted_ms / 1000) if predicted_ms is not None else 'n/a'} "
            f"({predicted_n} tokens)"
        )
    else:
        debug_log("llama.cpp timings: not included in server response")

# Makes the .wav file for the output to be heard
def make_sam_wav(text: str, identity: dict) -> None:
    voice_settings = identity.get("voice_settings", {})

    pitch = str(voice_settings.get("pitch", 95))
    speed = str(voice_settings.get("speed", 65))
    throat = str(voice_settings.get("throat", 130))
    mouth = str(voice_settings.get("mouth", 110))

    cmd = [
        str(SAM_BINARY),
        "-wav",
        str(OUTPUT_WAV),
        "-pitch",
        pitch,
        "-speed",
        speed,
        "-throat",
        throat,
        "-mouth",
        mouth,
        text,
    ]

    subprocess.run(cmd, check=True)

# Just a lil "spinning" animation so i can see if Artemis hung or not
def thinking_spinner(stop_event: threading.Event) -> None:
    frames = ["—", "/", "|", "\\"]

    index = 0

    while not stop_event.is_set():
        frame = frames[index % len(frames)]
        sys.stdout.write(f"\r{YELLOW}Thinking {frame}{RESET}")
        sys.stdout.flush()

        index += 1
        time.sleep(0.15)

    sys.stdout.write("\r" + " " * 30 + "\r")
    sys.stdout.flush()

    

def main() -> None:
    if not SAM_BINARY.exists():
        print(f"{RED}SAM binary not found at: {SAM_BINARY}{RESET}")
        return

    try:
        identity = load_identity()
        mode = load_mode()
    except Exception as exc:
        print(f"{RED}Failed to load Artemis memory files: {exc}{RESET}")
        return

    voice_settings = identity.get("voice_settings", {})

    print(f"{BOLD}{RED}Local Qwen + SAM test{RESET}")
    print(f"Loaded identity: {BOLD}{RED}{identity.get('name', 'Artemis')}{RESET}")
    print(f"Loaded mode: {BOLD}{RED}{mode.get('display_name', 'EAS Mode')}{RESET}")
    debug_log(f"LLM server URL: {LLAMA_SERVER_URL}")
    debug_log(f"memory folder: {MEMORY_DIR}")
    print(
        "Voice settings: "
        f"pitch = {voice_settings.get('pitch', 95)}, "
        f"speed = {voice_settings.get('speed', 65)}, "
        f"throat = {voice_settings.get('throat', 130)}, "
        f"mouth = {voice_settings.get('mouth', 110)}"
    )
    print(f"{BOLD}Type {RED}q{RESET}{BOLD} to quit.\n{RESET}")

    while True:
        user_text = input(GREEN + "User: " + RESET).strip()

        if user_text.lower() in {"q", "quit", "exit"}:
            break

        if not user_text:
            continue

        total_start = time.perf_counter()

        prompt_start = time.perf_counter()
        system_prompt, prompt_debug = build_system_prompt(identity, mode, user_text)
        prompt_elapsed = time.perf_counter() - prompt_start
        
        print()
        print_prompt_debug(prompt_debug)

        stop_event = threading.Event()

        spinner_thread = threading.Thread(
            target=thinking_spinner,
            args=(stop_event,),
            daemon=True,
        )

        spinner_thread.start()
        try:
            llm_start = time.perf_counter()
            robot_text, server_debug = ask_local_model(user_text, system_prompt)
            llm_elapsed = time.perf_counter() - llm_start
        finally:
            stop_event.set()
            spinner_thread.join()

        print_server_debug(server_debug)
        print(f"{PURPLE}{identity.get('name', 'Artemis')}{RESET}: {robot_text}")

        sam_start = time.perf_counter()
        #print("Generating SAM voice...") # Debugging statement
        make_sam_wav(robot_text, identity)
        sam_elapsed = time.perf_counter() - sam_start

        total_elapsed = time.perf_counter() - total_start

        #print(f"Created WAV: {OUTPUT_WAV}") # Another debugging statement

        # Debugging timing info so i can see what is taking the most amount of time to process
        print(f"{CYAN}{BOLD}Timing:{RESET} prompt={format_elapsed(prompt_elapsed)}, llm={format_elapsed(llm_elapsed)}, sam={format_elapsed(sam_elapsed)}, total={format_elapsed(total_elapsed)}")
        print()

if __name__ == "__main__":
    main()
