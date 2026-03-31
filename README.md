---
title: MINT Brain Content Analyser
emoji: 🧠
colorFrom: green
colorTo: gray
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# MINT Brain Content Analyser

Upload a short-form video and get second-by-second neural engagement predictions powered by Meta's TRIBE v2 foundation model.

## Setup

Set your Hugging Face token as a Space secret named `HF_TOKEN` — required to download the LLaMA 3.2 weights that TRIBE v2 uses internally.

**Settings → Variables and secrets → New secret → Name: `HF_TOKEN` → Value: your token**
