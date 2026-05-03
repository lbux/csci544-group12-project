# Data Pipeline (`/data_pipeline`)

This directory is responsible for acquiring, scoring, and preparing the raw Reddit data into usable seed threads for the LLM simulations.

## Execution Flow

1. **`scraper.py`**: Connects to the Reddit API to download recent posts and comments from highly-debated subreddits (e.g., r/Abortiondebate). Outputs raw JSONL.
2. **`score_reddit.py`**: Ingests the raw data and processes every comment through our Tier-1 fine-tuned DeBERTa toxicity classifier, appending a `"toxicity"` score to each turn.
3. **`filtering.py`**: Scans the scored dataset to find "toxic chains"—continuous multi-turn escalations of hostility. It isolates these specific comment chains to serve as the highly-contentious seeds for our simulated agents.