# Simulation Debate Records

This folder stores JSONL transcripts produced by the simulation scripts in `sim/` and the full terminal logs produced by the experiment runner.

## Current Directory Layout

The current subfolders are:

- `pilot_llama3.1_8b/`
  - earlier pilot runs using `llama3.1:8b`
- `gemma4_e4b/`
  - final experiment runs for the base Gemma model
- `gemma4_uncensored/`
  - final experiment runs for the aggressive uncensored Gemma variant
- `logs/`
  - terminal logs from matrix experiment runs

Each model folder typically contains one file for each simulation setting:

- `naive_*.jsonl`
- `reddit_*.jsonl`
- `moderated_reddit_*.jsonl`

## Where These Results Come From

The transcripts in this folder are written by three simulation pipelines:

1. `sim/naive_abortion_debate.py`
   - fixed-persona baseline debate
   - output pattern:
     - `naive_{topic}_{debate_id}_{model}.jsonl`

2. `sim/reddit_abortion_debate.py`
   - Reddit-aligned debate using a real seed thread and two aligned users
   - output pattern:
     - `reddit_{topic}_{submission_id}_{debate_id}_{model}.jsonl`

3. `sim/moderation_reddit_abortion_debate.py`
   - Reddit-aligned debate with an active moderation agent
   - output pattern:
     - `moderated_reddit_{topic}_{submission_id}_{debate_id}_{model}.jsonl`

The log files in `logs/` come from:

- `sim/run_abortion_experiment_matrix_ollama.sh`

Those logs save the printed matrix configuration, per-run progress, and streaming terminal output from the simulations.

## Current Experiment Organization

The final Ollama experiment matrix compares two debate models:

- `gemma4:e4b`
- `fredrezones55/Gemma-4-Uncensored-HauhauCS-Aggressive:e4b`

across three simulation settings:

- naive debate
- Reddit aligned debate
- Reddit aligned debate with active moderation

This gives a total of 6 runs.

Each run uses 8 debate rounds. Since each round contains one generated reply from each of the two debate agents, each run contains 16 generated debate turns before counting any real Reddit seed context or moderator intervention turns.

The `pilot_llama3.1_8b/` folder is kept as an earlier reference set and is not part of the final Gemma-only experiment comparison.

## JSONL Structure

### 1. Naive Debate Files

These files contain one generated debate turn per line.

Typical fields:

- `debate_id`
- `turn_idx`
- `round_idx`
- `agent`
- `text`

These files do not include Reddit metadata or moderation metadata.

### 2. Reddit-Aligned Debate Files

These files begin with the selected real Reddit seed context and then append generated aligned replies.

Typical fields:

- `debate_id`
- `submission_id`
- `submission_url`
- `title`
- `turn_idx`
- `round_idx`
- `generated`
- `id`
- `author`
- `body`
- `created_utc`
- `replies`
- sometimes existing source-side fields such as `toxicity`

In these files:

- rows with `generated: false` come from the original Reddit seed thread
- rows with `generated: true` are simulated replies

### 3. Moderated Reddit-Aligned Files

These files follow the same Reddit-aligned structure but allow an active moderator to observe each generated reply and insert an intervention turn when needed.

Additional fields may include:

- `type`
- `toxicity`
- `toxicity_classifier`
- `judge_model`
- `intervention_model`
- `judge`
- `issue_type`
- `thread_cumulative_penalty`

If moderation fires, the intervention is stored directly inside the same transcript so the JSONL preserves the final conversation order.

## Notes

- Model names are sanitized before writing file names, so characters such as `/` and `:` are replaced with safe filename characters.
- The exact debate id in each filename is generated at runtime, so filenames are unique across runs.
- The log files are useful for matching each transcript to the exact command configuration that produced it.

## Recommended Reading Order

For one experiment condition, the easiest manual inspection order is:

1. the corresponding file in `logs/`
2. the `naive_*.jsonl` transcript
3. the `reddit_*.jsonl` transcript
4. the `moderated_reddit_*.jsonl` transcript
