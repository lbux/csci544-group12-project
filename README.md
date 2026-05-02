# Prerequisites

* Install [uv](https://docs.astral.sh/uv/getting-started/installation/)

* Install [ollama](https://ollama.com/download)

* A Jupyter Notebook Environment

# How to Run
1) Install dependencies

    Run this in the root directory to install packages via `uv`

    ```
    uv sync
    ```
2) Download Local Models

    Ensure Ollama is running and run the following commands (WARNING: This is almost 20GBs)
    ```
    ollama pull llama3.1:8b
    ollama pull llama3.2:3b
    ollama pull gemma4:e4b
    ollama pull fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:e4b
    ```
    (Note: The DeBERTa toxicity classifier does not need an Ollama pull; the code will automatically download it from HuggingFace on the first run.)
3) Run the Pipeline

    Open `run_pipeline.ipynb` and select Run All.

    The raw Reddit data is already provided in the `data/` folder, so the pipeline will automatically skip the web-scraping step and proceed directly to scoring, filtering, simulation, and evaluation.