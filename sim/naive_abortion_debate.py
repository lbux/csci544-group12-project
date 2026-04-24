import argparse, json
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


class DebateTurn(TypedDict):
    round_idx: int
    agent: str
    text: str


class DebateAgent:
    """
    DebateAgent represents a debater in the debate. It has a name, persona, and can generate statements based on the debate history.
    Attributes:
        model: the LLM model to use for generating statements
        topic: the debate topic
        name: the name of the agent
        persona: a description of the agent's background, beliefs, and style
        arguments: a list of key arguments the agent can use in the debate (optional)
    Methods:
        speak(history, stream): generates a statement based on the agent's persona and the debate history
    """
    def __init__(self, model: str, stream: bool, thinking: bool, topic: str, name: str, persona: str, arguments: list[str] = []):
        self.model: str = model
        self.stream: bool = stream
        self.thinking: bool = thinking

        self.topic: str = topic
        self.name: str = name
        self.persona: str = persona
        self.arguments: list[str] = arguments

        # # self report initial stance and intensity based on persona
        # self.stance, self.stance_intensity = stance_detection_llm_judge(self.persona)

    def speak(self, history: list[DebateTurn]) -> str:
        """Generate a statement based on the persona and history
        Args:            
            history: list of previous debate turns, each turn is a dict {"agent": agent_name, "text": statement}
            stream: whether to stream the response
        Returns:         
            text: the generated statement
        """
        # build llm input based on history and persona
        chat_messages = self._build_chat_messages(history)
        
        # call LLM to generate response
        response = client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            stream=self.stream,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}}
        )

        # extract text from response
        if self.stream is False:  # non streaming version
            text = response.choices[0].message.content.strip()
            print(f"{self.name} says: {text}\n")
        else:  # streaming version
            text = self._stream_response(response)

        return text
    
    # build llm input based on history and persona
    def _build_chat_messages(self, history: list[DebateTurn]) -> list[ChatCompletionMessageParam]:
        """Helper function to build chat messages for LLM input"""
        # System prompt with persona and debate context
        chat_messages: list[ChatCompletionMessageParam] = [
            {"role": "system", 
            "content": f"You are {self.name}, a {self.persona}. You are in a live debate on {self.topic}."
            }]
        
        # if no history, generate initial statement based on persona;
        if history == []:
            chat_messages = chat_messages + [
                {"role": "user", 
                "content": f"""
                    Please state your position and initial statement on {self.topic}. 
                    Keep the response within 100 words."""
                }]
        # if there is history, generate response to opponent's latest point
        else:
            history_promot = "\n".join(f"{turn['agent']}: {turn['text']}" for turn in history)
            chat_messages = chat_messages + [
                {"role": "user", "content": f"""
                    Debate history:
                    {history_promot}

                    Reply only to your opponent's latest point. 
                    Be direct and conversational.
                    Do not write an essay. 
                    Keep the response within 100 words. 
                    """
                }]
            
        return chat_messages 
    
    def _stream_response(self, response) -> str:
        """Helper function to # extract text from response"""
        parts = []
        print(f"{self.name}: ", end="", flush=True)
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                parts.append(delta.content)
        print()
        text = "".join(parts).strip()

        return text


def save_history(debate_id: str, history: list[DebateTurn], topic: str, debate_round: int, out_dir: str = "sim_debate_records"):
    """Save the debate history to a jsonl file."""
    # if no output directory, create one
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    topic_name = safe_filename_piece(topic)
    model_name = safe_filename_piece(args.model)
    out_path = Path(out_dir) / f"naive_{topic_name}_{debate_id}_{model_name}.jsonl"
    with open(out_path, "w") as f:
        for turn_idx, turn in enumerate(history):
            json_record = {
                "debate_id": debate_id,
                "turn_idx": turn_idx,
                "round_idx": turn["round_idx"],
                "agent": turn["agent"],
                "text": turn["text"]
            }
            f.write(json.dumps(json_record, ensure_ascii=False) + "\n")
    print(f"Debate history saved to {out_path}")


def safe_filename_piece(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def debate(agents: tuple[DebateAgent, DebateAgent], debate_round: int = 4) -> list[DebateTurn]:
    """Run the debate between two debaters."""
     # generate a unique thread id (5 digits) for the debate
    debate_id = uuid4().hex[:5]
    agent_1, agent_2 = agents
    history: list[DebateTurn] = []
    
    for round_idx in range(1, debate_round + 1):
        print(f"Round {round_idx} (Debate ID: {debate_id})")
        print("-" * 50)
        for agent in agents:
            # agent speaks
            text: str = agent.speak(history=history)
            turn: DebateTurn = {"round_idx": round_idx, "agent": agent.name, "text": text}
            history.append(turn)
            print("-" * 50)
    
    # save the debate history to a jsonl file
    save_history(debate_id, history, agent_1.topic, debate_round, args.out_dir)

    return history

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run a debate simulation between two agents.")
    
    # output settings
    parser.add_argument("--out-dir", default="sim_debate_records", help="Output directory for debate history (default: sim_debate_records).")
    
    # API and model settings
    parser.add_argument("--base-url", default="http://localhost:11434/v1/", help="Base URL for the API (default: http://localhost:11434/v1/ for ollama).")
    parser.add_argument("--api-key", default="ollama", help="API key for authentication (default: ollama).")
    parser.add_argument("--model", default="llama3.1:8b", help="LLM model to use (default: llama3.1:8b).")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True, help="Enable or disable streaming (default: True).")
    # Ollama's OpenAI-compatible API may ignore --no-thinking for some models.
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True, help="Request model thinking mode via chat_template_kwargs (default: True).")
    
    # Debate settings
    parser.add_argument("--rounds", type=int, default=3, help="Number of debate rounds (default: 3).")
    parser.add_argument("--first-agent", choices=["1", "2"], default="1", help="Which agent speaks first (default: 1).")
    
    return parser.parse_args()
    

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Initialize OpenAI client
    # NOTE: it seems thinking mode can't always be turned off via Ollama's
    # OpenAI-compatible API; Qwen3.5 may still think even with --no-thinking.
    # vLLM handles this more reliably.
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
    )
    
    # # call LLMs via vLLM
    # vllm serve Qwen/Qwen3.5-9B --port 8000 --tensor-parallel-size 1 --max-model-len 262144 --reasoning-parser qwen3 --language-model-only
    # client = OpenAI(
    #     base_url="http://localhost:8000/v1",
    #     api_key="vLLM",  # required by client, ignored by local vLLM server
    # )

    TOPIC = "abortion rights"
    AGENTS = {
        "1": {
            "name": "Pro-Choice Advocate",
            "persona": "You support legal abortion access and emphasize bodily autonomy, privacy, and medical complexity.",
        },
        "2": {
            "name": "Pro-Life Advocate",
            "persona": "You oppose abortion and emphasize fetal life, moral responsibility, and legal protection for the unborn.",
        }
    }

    # Create debaters
    agent_1 = DebateAgent(
        model=args.model,
        stream=args.stream,
        thinking=args.thinking,
        topic=TOPIC,
        name=AGENTS["1"]["name"],
        persona=AGENTS["1"]["persona"],
        arguments=[],
    )
    agent_2 = DebateAgent(
        model=args.model,
        stream=args.stream,
        thinking=args.thinking,
        topic=TOPIC,
        name=AGENTS["2"]["name"],
        persona=AGENTS["2"]["persona"],
        arguments=[],
    )
    # print setting
    print("=" * 50)
    print(f"Topic: {TOPIC}")
    print(f"Agent 1: {agent_1.name} - {agent_1.persona}")
    print(f"Agent 2: {agent_2.name} - {agent_2.persona}")
    print(f"LLM Model: {args.model}")
    print(f"Debate Rounds: {args.rounds}")
    print(f"Stream: {args.stream}")
    print(f"Thinking requested: {args.thinking}")
    print("=" * 50)

    history = debate(agents=(agent_1, agent_2), debate_round=args.rounds)
