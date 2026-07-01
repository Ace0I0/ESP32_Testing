import subprocess
from pathlib import Path
from datetime import datetime

SAM_BINARY = Path("/home/pi/SAM/sam")
OUTPUT_DIR = Path("/home/pi/SAM/output")

# Default values found to be "acceptable"
DEFAULT_PITCH = 150
DEFAULT_SPEED = 80
DEFAULT_THROAT = 150
DEFAULT_MOUTH = 150

# Ask user for values
def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            print("Please enter a valid integer.")

# Ask user for TTS string
def ask_text(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Text cannot be empty.")

# Assemble file output name
def build_filename(text: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_text = "".join(c for c in text[:30] if c.isalnum() or c in (" ", "_", "-")).strip()
    safe_text = safe_text.replace(" ", "_")
    if not safe_text:
        safe_text = "speech"
    return f"{timestamp}_{safe_text}.wav"

# Automate the generation
def generate_sam_audio(
    text: str,
    pitch: int,
    speed: int,
    throat: int,
    mouth: int,
    output_path: Path,
) -> None:
    cmd = [
        str(SAM_BINARY),
        "-wav",
        str(output_path),
        "-pitch",
        str(pitch),
        "-speed",
        str(speed),
        "-throat",
        str(throat),
        "-mouth",
        str(mouth),
        text,
    ]

    print("\nRunning:")
    print(" ".join(cmd))
    print()

    subprocess.run(cmd, check=True)
    print(f"Created: {output_path}")


def main() -> None:
    if not SAM_BINARY.exists():
        print(f"SAM binary not found at: {SAM_BINARY}")
        print("Update the SAM_BINARY path in this script.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("SAM TTS Test Tool")
    print("Press Enter to use the default value.\n")

    while True:
        text = ask_text("Enter text to speak: ")
        pitch = ask_int("Pitch", DEFAULT_PITCH)
        speed = ask_int("Speed", DEFAULT_SPEED)
        throat = ask_int("Throat", DEFAULT_THROAT)
        mouth = ask_int("Mouth", DEFAULT_MOUTH)

        filename = build_filename(text)
        output_path = OUTPUT_DIR / filename

        try:
            generate_sam_audio(
                text=text,
                pitch=pitch,
                speed=speed,
                throat=throat,
                mouth=mouth,
                output_path=output_path,
            )
        except subprocess.CalledProcessError as exc:
            print(f"SAM failed with exit code {exc.returncode}")
        except Exception as exc:
            print(f"Unexpected error: {exc}")

        again = input("\nGenerate another clip? [y/N]: ").strip().lower()
        if again != "y":
            break
        print()

    print("Done.")


if __name__ == "__main__":
    main()