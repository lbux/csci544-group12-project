# Debate Simulations (`/simulations`)

This directory contains the execution loops for the three experimental conditions detailed in our paper. Each script sets up the agents, runs the turn-by-turn debate, and saves the output to the `/sim_debate_records` directory.

## Experimental Conditions

1. **`naive_debate.py`**: The Baseline. Runs a blank-slate debate between two generic personas (e.g., Pro-Choice vs. Pro-Life) with no external user data or moderation.
2. **`reddit_aligned.py`**: The Grounded Setting. Loads a highly-toxic seed thread from the data pipeline, builds alignment profiles for two simulated agents based on the real users' comment history, and allows them to continue the argument.
3. **`moderated_reddit.py`**: The Intervened Setting. Runs the Reddit-aligned debate, but passes every generated turn through the `ActiveModerator`. If the cumulative toxicity penalty exceeds the threshold, the orchestrator inserts a neutral mediation turn to attempt de-escalation.