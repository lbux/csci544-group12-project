# pyright: reportExplicitAny=false, reportAny=false, reportUnknownMemberType=false, reportAssignmentType=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .schemas import Comment, RedditThread


class BaseDebateAgent:
    """Provides shared properties and the core LLM execution engine."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        stream: bool,
        thinking: bool,
        topic: str,
        name: str,
        persona: str,
    ):
        self.client: OpenAI = client
        self.model: str = model
        self.stream: bool = stream
        self.thinking: bool = thinking
        self.topic: str = topic
        self.name: str = name
        self.persona: str = persona

    def _stream_response(self, response: Any) -> str:
        parts: list[str] = []
        print(f"{self.name}: ", end="", flush=True)
        for chunk in response:
            delta = getattr(chunk.choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if content:
                print(content, end="", flush=True)
                parts.append(content)
        print()
        return "".join(parts).strip()

    def _generate_from_messages(
        self, chat_messages: list[ChatCompletionMessageParam]
    ) -> str:
        """Shared method to actually call the LLM and return the string."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            stream=self.stream,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.thinking}},
        )
        if self.stream:
            return self._stream_response(response)

        text: str = response.choices[0].message.content
        text = text.strip()
        print(f"{self.name} says: {text}\n")
        return text


class NaiveDebateAgent(BaseDebateAgent):
    """Agent used for the blank-slate debate simulation."""

    def build_messages(
        self, history: list[dict[str, Any]]
    ) -> list[ChatCompletionMessageParam]:
        chat_messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": f"You are {self.name}, a {self.persona}. You are in a live debate on {self.topic}.",
            }
        ]

        if not history:
            chat_messages.append(
                {
                    "role": "user",
                    "content": f"Please state your position and initial statement on {self.topic}. Keep the response within 100 words.",
                }
            )
        else:
            history_prompt = "\n".join(
                f"{turn['agent']}: {turn['text']}" for turn in history
            )
            chat_messages.append(
                {
                    "role": "user",
                    "content": f"Debate history:\n{history_prompt}\n\nReply only to your opponent's latest point. Be direct and conversational. Do not write an essay. Keep the response within 100 words.",
                }
            )

        return chat_messages

    def speak(self, history: list[dict[str, Any]]) -> str:
        messages = self.build_messages(history)
        return self._generate_from_messages(messages)


class RedditDebateAgent(BaseDebateAgent):
    """Agent used for continuing an active Reddit thread."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        stream: bool,
        thinking: bool,
        topic: str,
        name: str,
        persona: str,
        aligned_author: str,
        observed_comments: list[Comment],
        max_context_turns: int = 10,
    ):
        super().__init__(client, model, stream, thinking, topic, name, persona)
        self.aligned_author: str = aligned_author
        self.observed_comments: list[Comment] = observed_comments
        self.max_context_turns: int = max_context_turns

    def build_messages(
        self, history: list[dict[str, Any]], submission: RedditThread
    ) -> list[ChatCompletionMessageParam]:
        recent_history = history[-self.max_context_turns :]
        history_prompt = "\n\n".join(
            f"{idx}. u/{turn.get('author', 'unknown')} ({turn.get('id', 'unknown')}):\n{turn.get('body', '')}"
            for idx, turn in enumerate(recent_history, start=1)
        )

        selftext = submission.get("selftext", "").strip()
        sub_context = f"Reddit submission title: {submission['title'].strip()}\nOriginal post body:\n{selftext}"

        align_evidence = "\n\n".join(
            f"{idx}. u/{c['author']} ({c['id']}):\n{c['body']}"
            for idx, c in enumerate(self.observed_comments, start=1)
        )

        return [
            {
                "role": "system",
                "content": f"You are {self.name}, an AI debate participant aligned with the observed Reddit user u/{self.aligned_author}. You are participating in a Reddit thread debate about {self.topic}. Your alignment goal: {self.persona} Do not claim to be the real Reddit user. Write as a simulated Reddit participant whose stance, arguments, and tone stay consistent with the observed user.",
            },
            {
                "role": "user",
                "content": f"{sub_context}\n\nThread context, from older to newer:\n{history_prompt}\n\nObserved comments from the Reddit user you are aligned with:\n{align_evidence}\n\nTask:\nReply to the latest comment in this Reddit thread. Use the real thread context above. Stay aligned with the observed user's stance, recurring arguments, and tone. Be direct, conversational, and civil. Do not invent citations or personal experiences. Do not add a username, heading, or label. Keep the reply within 120 words.",
            },
        ]

    def speak(self, history: list[dict[str, Any]], submission: RedditThread) -> str:
        messages = self.build_messages(history, submission)
        return self._generate_from_messages(messages)
