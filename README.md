# Testing hosting a local language model

![Status](https://img.shields.io/badge/status-prototype-orange)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205-red)
![LLM](https://img.shields.io/badge/model-Qwen2.5%201.5B-green)
![TTS](https://img.shields.io/badge/TTS-SAM-lightgrey)

Artemis AI Voice Robot is an offline (future work will include possible online capabilities) Raspberry Pi voice assistant prototype that combines a local large language model with form of retro-style speech style. The project is designed around a Raspberry Pi 5 with 16 GB of RAM, Qwen2.5 1.5B Instruct GGUF, and SAM text-to-speech.

The initial goal is to build a small, self-contained AI robot personality that can respond locally without relying on any major cloud APIs. Artemis is currently being worked on in order to focus on an Emergency Alert System inspired mode, mixing things like practical instruction and fictional lore-driven behavior.

## Project Goals
- Run a local LLM on a Raspberry Pi 5.
- Generate voice output using SAM TTS.
- Support configurable AI personalities through JSON files.
- Build toward multiple interaction modes, starting with Artemis EAS Mode.

## Current System Overview

```mermaid
graph TD;
    User Input --> Python Controller;

```