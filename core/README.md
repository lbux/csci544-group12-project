# Core Architecture (`/core`)

This directory contains the foundational classes and utility functions that power the rest of the pipeline. It acts as the shared library for our agents, data structures, and LLM communication.

## Code Flow & Components

1. **`schemas.py`**: The schemas of the pipeline. Defines strict Pydantic models (e.g., `DebateEvaluation`, `RedditThread`) to guarantee mathematically sound data passing and JSON output from the LLMs.
2. **`llm_client.py`**: A centralized wrapper for the OpenAI/Ollama API. It dynamically injects schema templates into system prompts to ensure smaller models (like LLaMA-3.2:3b) output correctly formatted JSON.
3. **`agents.py`**: Defines the `RedditDebateAgent` class. It manages conversation history and injects user alignment profiles into the prompt.
4. **`moderation.py`**: Contains the dual-signal architecture described in our paper. 
   - `ToxicityClassifier`: The Tier-1 fine-tuned RoBERTa model.
   - `ActiveModerator` & `ThreadStateTracker`: The Tier-2 system that calculates cumulative penalties and triggers neutral mediation.
5. **`reddit_utils.py`**: Helper functions for loading and parsing the raw `.jsonl` Reddit dumps.