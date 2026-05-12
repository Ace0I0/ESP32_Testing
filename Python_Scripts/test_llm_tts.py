import sys
import time
import threading
import json
import subprocess
import urllib.request
from pathlib import Path


LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

SAM_BINARY = Path("/home/pi/SAM/sam")
OUTPUT_WAV = Path("/home/pi/SAM/robot_response.wav")

SAM_PITCH = 95
SAM_SPEED = 65
SAM_THROAT = 130
SAM_MOUTH = 110

SYSTEM_PROMPT = (
    "You are Artemis. "
    "Artemis is a small retro animatronic robot with an old synthetic voice. "
    "You are not Qwen. "
    "You are not a chatbot. "
    "You are not an AI assistant. "
    "If asked your name, say your name is Artemis. "
    "If asked what you are, say you are an animatronic robot. "
    "Never mention Qwen, llama.cpp, language models, models, prompts, or training data. "
    "Stay in character. "
    #"Reply in 1 to 3 short sentences. "
    #"Use simple words that are easy for old text to speech to pronounce. "
    "Avoid long lists. "
    "Avoid markdown. "
    "Your tone is curious, mechanical, slightly eerie, and helpful."
)

def ask_local_model(user_text: str) -> str:
    payload = {
        "model": "local-qwen",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_text,
            },
        ],
        "max_tokens": 100,
        "temperature": 0.7,
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


def make_sam_wav(text: str) -> None:
    cmd = [
        str(SAM_BINARY),
        "-wav",
        str(OUTPUT_WAV),
        "-pitch",
        str(SAM_PITCH),
        "-speed",
        str(SAM_SPEED),
        "-throat",
        str(SAM_THROAT),
        "-mouth",
        str(SAM_MOUTH),
        text,
    ]

    subprocess.run(cmd, check=True)


def play_wav() -> None:
    subprocess.run(["aplay", str(OUTPUT_WAV)], check=False)

def thinking_animation(stop_event: threading.Event) -> None:
    frames = [
        "Thinking   ",
        "Thinking.  ",
        "Thinking.. ",
        "Thinking...",
    ]

    index = 0

    while not stop_event.is_set():
        sys.stdout.write("\r" + frames[index % len(frames)])
        sys.stdout.flush()
        index += 1
        time.sleep(0.35)

    sys.stdout.write("\r" + " " * 20 + "\r")
    sys.stdout.flush()

def main() -> None:
    if not SAM_BINARY.exists():
        print(f"SAM binary not found at: {SAM_BINARY}")
        return

    print("Local Qwen + SAM test")
    print("Type q to quit.\n")

    while True:
        user_text = input("You: ").strip()

        if user_text.lower() in {"q", "quit", "exit"}:
            break

        if not user_text:
            continue

        print()

        stop_event = threading.Event()

        animation_thread = threading.Thread(
            target=thinking_animation,
            args=(stop_event,),
            daemon=True,
        )

        animation_thread.start()

        try:
            robot_text = ask_local_model(user_text)
        finally:
            stop_event.set()
            animation_thread.join()

        print(f"Robot text: {robot_text}")

        print("Generating SAM voice...")
        make_sam_wav(robot_text)

        print(f"Created WAV: {OUTPUT_WAV}")

        print()


if __name__ == "__main__":
    main()