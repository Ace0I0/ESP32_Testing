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

BASE_DIR = Path(__file__).resolve().parent
MEMORY_DIR = BASE_DIR / "memory"
MODES_DIR = MEMORY_DIR / "modes"

# Commented this part out for now, we can deal with adding knowledge to Artemis later
#KNOWLEDGE_DIR = MEMORY_DIR / "knowledge"

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

# formats examples listed in modes (may or may not remove this, but with that i have in mind rn, it should be useful)
def format_examples(examples: list[dict]) -> str:
    lines = []

    for example in examples:
        user = example.get("user", "").strip()
        assistant = example.get("assistant", "").strip()

        if user and assistant:
            lines.append(f"User: {user}\nArtemis: {assistant}")

    return "\n\n".join(lines)

# System prompt builder - Accumulates all needed info so Artemis can function correctly based on the mode chosen
def build_system_prompt(identity: dict, mode: dict) -> str:
    name = identity.get("name", "Artemis")
    robot_type = identity.get("type", "retro animatronic robot")
    voice_engine = identity.get("voice_engine", "SAM")

    core_rules = format_list(identity.get("core_rules", []))
    tone = ", ".join(mode.get("tone", []))
    response_rules = format_list(mode.get("response_rules", []))
    examples = format_examples(mode.get("example_responses", []))

    # returns all info so it works as a system prompt for Artemis
    return (
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

        "Final output rules:\n"
        f"- Respond as {name}.\n"
        "- Write in normal English.\n"
        "- Do not use IPA or dictionary phonetic notation.\n"
        "- Do not use markdown.\n"
        "- Keep the response suitable for old robotic text to speech.\n" # lets see if that even works considering its a 1.5B model lol
    )

def ask_local_model(user_text: str, system_prompt: str) -> str:
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
        "max_tokens": 250,
        "temperature": 0.8,
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        LLAMA_SERVER_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["choices"][0]["message"]["content"].strip()

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
        system_prompt = build_system_prompt(identity, mode)
    except Exception as exc:
        print(f"{RED}Failed to load Artemis memory files: {exc}{RESET}")
        return

    voice_settings = identity.get("voice_settings", {})

    print(f"{BOLD}{RED}Local Qwen + SAM test{RESET}")
    print(f"Loaded identity: {BOLD}{RED}{identity.get('name', 'Artemis')}{RESET}")
    print(f"Loaded mode: {BOLD}{RED}{mode.get('display_name', 'EAS Mode')}{RESET}")
    print(
        "Voice settings: "
        f"pitch = {voice_settings.get('pitch', 95)}, "
        f"speed = {voice_settings.get('speed', 65)}, "
        f"throat = {voice_settings.get('throat', 130)}, "
        f"mouth = {voice_settings.get('mouth', 110)}"
    )
    print(f"{BOLD}Type {RED}q{RESET}{BOLD} to quit.\n{RESET}")

    while True:
        user_text = input(GREEN+"User: "+RESET).strip()

        if user_text.lower() in {"q", "quit", "exit"}:
            break

        if not user_text:
            continue

        print()

        stop_event = threading.Event()

        spinner_thread = threading.Thread(
            target=thinking_spinner,
            args=(stop_event,),
            daemon=True,
        )

        spinner_thread.start()

        try:
            robot_text = ask_local_model(user_text, system_prompt)
        finally:
            stop_event.set()
            spinner_thread.join()

        print(f"{PURPLE}{identity.get('name', 'Artemis')}{RESET}: {robot_text}")

        #print("Generating SAM voice...") # Debugging statement
        make_sam_wav(robot_text, identity)

        #print(f"Created WAV: {OUTPUT_WAV}") # Another debugging statement
        print()

if __name__ == "__main__":
    main()