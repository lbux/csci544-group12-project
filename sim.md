# Simulation Scripts

The Reddit simulation script should be run from the project root because its default input path is `out.jsonl`

Install dependencies if needed:

```bash
uv sync
```

Make sure Ollama is running and the model is available:

```bash
ollama list
ollama pull llama3.1:8b
```

## Naive Debate

This runs a simple two-agent debate using fixed personas defined inside the script. The current abortion topic is only one example case.


```bash
uv run python sim/naive_abortion_debate.py
```

Common options:

```bash
uv run python sim/naive_abortion_debate.py --rounds 3 --model llama3.1:8b --no-stream
```

Useful arguments:

- `--rounds`: number of debate rounds.
- `--model`: Ollama/OpenAI-compatible model name.
- `--stream` / `--no-stream`: enable or disable streaming output.
- `--first-agent`: choose which fixed agent speaks first, either `1` or `2`.
- `--out-dir`: output directory for debate history.

> To change the debate topic or agent prompts,: Change `TOPIC` and `AGENTS` in [sim/naive_abortion_debate.py](sim/naive_abortion_debate.py).

## Reddit Thread Debate

This loads a real Reddit thread from `out.jsonl`, selects a comment chain, aligns two agents with two observed Reddit users from that chain, and continues the debate. The current abortion topic is only one example case.


```bash
uv run python sim/reddit_abortion_debate.py
```

Preview the selected thread and aligned users without generating LLM replies:

```bash
uv run python sim/reddit_abortion_debate.py --no-generate
```

Common options:

```bash
uv run python sim/reddit_abortion_debate.py --rounds 3 --model llama3.1:8b --no-stream
```

Useful arguments:


- `--submission-index`: select a submission by line/index in the input file.
- `--submission-id`: select a submission by stable Reddit submission ID. This overrides `--submission-index`.
- `--comment-id`: use the comment path ending at this comment as the seed chain.
- `--min-seed-words`: minimum comment length when auto-selecting a seed chain.
- `--rounds`: number of generated debate rounds.
- `--first-agent`: choose which aligned agent speaks first, either `1` or `2`.
- `--max-context-turns`: number of recent history turns included in the LLM prompt.
- `--no-generate`: load, select, print, and save the seed thread without generating new replies.

### Notes

To change the topic: Change `TOPIC` in [sim/reddit_abortion_debate.py](sim/reddit_abortion_debate.py).

Agent prompts are automatically generated from the selected Reddit chain:

- The two agent identities are not fixed in advance; they are built from the selected Reddit chain by `build_alignment_profiles(...)`.
- The selected chain determines which two Reddit users the agents align with.
- `RedditDebateAgent._build_chat_messages(...)` controls the full prompt sent to the model.
- `RedditDebateAgent._format_alignment_evidence(...)` controls how each user's observed Reddit comments are shown in the prompt.

How to select the chain:

- Default behavior: read `out.jsonl`, use `--submission-index 0`, and automatically select the most toxic usable comment chain from that first submission.
- Use `--submission-index` to choose a submission by position in `out.jsonl`.
- Use `--submission-id` to choose a submission by Reddit submission ID.
- Use `--comment-id` to reproduce a specific chain by choosing the comment path ending at that comment.
- For reproducible runs, prefer using `--submission-id` and `--comment-id` together so both the Reddit post and the chain endpoint are fixed.
- If `--comment-id` is not provided, the script automatically selects the most toxic usable comment chain from the chosen submission.

Select a submission by index (e.g. first submission in `out.json`):

## Moderated Reddit Thread Debate

This extends the Reddit thread debate with an active moderation agent. It loads a real Reddit thread from `out.jsonl`, aligns two debate agents with observed Reddit users, generates new replies, uses the local CGA toxicity classifier as a first-pass filter, sends flagged generated replies to an LLM moderation judge, and asks an intervention model to step in when the thread crosses the intervention threshold.


```bash
uv run python sim/moderation_reddit_abortion_debate.py
```

Preview the selected thread and aligned users without generating LLM replies:

```bash
uv run python sim/moderation_reddit_abortion_debate.py --no-generate
```

Common options:

```bash
uv run python sim/moderation_reddit_abortion_debate.py --rounds 3 --model llama3.1:8b --judge-model llama3.2:3b --no-stream
```

Useful arguments:

- `--model`: model used by the two aligned debate agents.
- `--judge-model`: model used by the active moderator to classify each flagged generated turn.
- `--intervention-model`: model used to generate moderator interventions. If omitted, the script uses `--judge-model`.
- `--classifier`: local CGA classifier path. Use `skip` to send every generated turn directly to the moderation judge.
- `--toxicity-threshold`: minimum classifier score required before calling the moderation judge.
- `--intervention-threshold`: cumulative moderation penalty required before the active moderator intervenes.
- `--cooldown-turns`: number of judged toxic turns to wait after an intervention before another non-severe intervention can fire.
- `--submission-index`: select a submission by line/index in the input file.
- `--submission-id`: select a submission by stable Reddit submission ID. This overrides `--submission-index`.
- `--comment-id`: use the comment path ending at this comment as the seed chain.
- `--min-seed-words`: minimum comment length when auto-selecting a seed chain.
- `--rounds`: number of generated debate rounds.
- `--first-agent`: choose which aligned agent speaks first, either `1` or `2`.
- `--max-context-turns`: number of recent history turns included in the LLM prompt.
- `--no-generate`: load, select, print, and save the seed thread without generating new replies.

### Notes

The moderated script intentionally does not use the NetworkX graph pipeline. The active moderation logic is adapted from `ModerationOrchestrator._ingest_comment(...)`, but it only tracks toxicity, judge results, cumulative penalties, cooldown state, and optional moderator turns in the output JSONL.

By default, the script uses one OpenAI-compatible local Ollama endpoint:

```bash
http://localhost:11434/v1/
```

The debate agents, moderation judge, and intervention generator can use different model names through `--model`, `--judge-model`, and `--intervention-model` while still sharing the same endpoint.

## Experiment Matrix

To run the current experiment matrix, use:

```bash
bash sim/run_abortion_experiment_matrix_ollama.sh
```

This runs two debate models across three simulation settings:

- naive debate
- Reddit aligned debate
- Reddit aligned debate with active moderation

The debate models are:

- `gemma4:e4b`
- `fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:e4b`

The original experiment plan also included `qwen3.5:9b` and `fredrezones55/Qwen3.5-Uncensored-HauhauCS-Aggressive:9b`. However, those Qwen-based runs were excluded from the final matrix because they showed unstable thinking-mode behavior under Ollama's OpenAI-compatible interface. The current experiment matrix therefore focuses on the Gemma family pair only. Future work can revisit the Qwen family and extend the comparison to more model sizes once the local serving setup is more stable.

For the moderated simulation, the judge and intervention models are fixed to:

```bash
llama3.1:8b
```

This keeps the moderation standard separate from the Gemma debate models under comparison.

Current Ollama experiment settings:

- Input file: `out.jsonl`
- Debate rounds: `8`
- Base URL: `http://localhost:11434/v1/`
- API key: `ollama`
- Streaming: enabled by default
- Judge model: `llama3.1:8b`
- Intervention model: `llama3.1:8b`
- Toxicity classifier: `models/cga_deberta_onnx_int8`
- Toxicity threshold: `0.6`
- Intervention threshold: `10`
- Cooldown turns after an intervention: `2`

This matrix therefore contains:

```bash
2 debate models x 3 simulation settings = 6 runs
```

The script assumes `out.jsonl` exists and that the needed Ollama models have already been pulled. All terminal output is printed to the console and also saved to one timestamped log file:

```bash
sim_debate_records/logs/abortion_experiment_matrix_ollama_YYYYMMDD_HHMMSS.log
```


## Outputs

All simulation scripts save JSONL debate histories to:

```bash
sim_debate_records/
```
