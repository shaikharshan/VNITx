# HackIITK 2026 — Deepfake & Prompt Injection Defense

**Project:** Securing Voice and Multimodal AI Agents Against Deepfakes and Prompt Injection  
**Event:** HackIITK 2026

This repository contains a multimodal security framework that detects and mitigates malicious **audio**, **visual**, and **cross‑modal** inputs targeting AI agents.

## Problem Statement (HackIITK 2026)
Voice‑enabled and multimodal AI agents are increasingly deployed in real‑world systems such as virtual assistants, customer support bots, authentication systems, and automated decision‑making platforms. These agents rely on audio, visual, and textual inputs to understand user intent and trigger actions.

Recent advances in generative AI have enabled highly realistic audio deepfakes, voice cloning, and adversarial multimodal inputs. Attackers can manipulate AI agents using synthetic voices, hidden audio commands, or visual prompt injections embedded in images and videos. Traditional security mechanisms are insufficient to defend against these emerging threats.

This challenge focuses on building a robust security framework that detects and mitigates malicious audio and visual inputs targeting multimodal AI agents.

## Objectives
**Audio Deepfake and Voice Cloning Detection**
- Detect AI‑generated or spoofed audio inputs.
- Identify voice cloning attacks and replay attacks.
- Analyze acoustic and spectral features for authenticity verification.

**Multimodal Prompt Injection Detection**
- Detect hidden or adversarial instructions embedded in images or videos.
- Identify mismatches between audio, visual, and textual modalities.
- Prevent unauthorized actions triggered by manipulated inputs.

**Risk Scoring and Mitigation**
- Assign a confidence/risk score to each input.
- Block, flag, or downgrade suspicious inputs before agent execution.
- Provide explanations for detected threats.

## Modules
- `Image_prompt_detection/` — image OCR + prompt injection + cross‑modal checks
- `Video_prompt_detection/` — video analysis (frames, deepfake checks, AV sync, risk scoring)

Each module has its own README with setup and run instructions.
