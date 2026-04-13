from pathlib import Path

import ollama
import torch
from huggingface_hub import (
    snapshot_download,  # pyright: ignore[reportUnknownVariableType]
)
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from interfaces import ReasoningResult


class LLMReasoner:
    def __init__(self, model: str = "gemma4:e4b") -> None:
        self.model: str = model
        _ = ollama.pull(self.model)

    def analyze_intent(
        self, text: str, parent_text: str, root_context: str
    ) -> ReasoningResult:

        SYSTEM_MESSAGE: str = """
        You are an impartial Moderation Judge for a highly contentious discussion forum. 
        An automated toxicity classifier has flagged a User Message. Your objective is to read the Original Post (the topic), the Previous Comment (the immediate context), and the flagged User Message, and determine the true intent and severity of the message.

        EVALUATION RULES:
        1. STANCE NEUTRALITY: You must evaluate the *behavior and language* of the user, not their beliefs. Both sides of an argument must be held to the exact same behavioral standard.
        2. CONTEXT MATTERS: Sarcasm, idioms, and hyperbolic jokes might trigger the automated classifier but are not actual violations. 

        RUBRIC & CATEGORIES:
        You must strictly assign one of the following categories and an associated point value:

        - "fp" (False Positive) | 0 points
        The classifier made a mistake. The text contains flagged words used in a benign, joking, or purely academic context.

        - "flare" (Heated Debate) | 1 to 2 points
        The comment is harsh, highly sarcastic, or deeply critical of the *argument*. It has rough edges typical of contentious debates, but it attacks the *ideas*, not the *person*. It does not cross the line into abuse.

        - "toxic" (Ad Hominem/Abuse) | 3 to 5 points
        The comment attacks the *person* rather than the argument. It includes insults, severe condescension, harassment, or aggressive attempts to shut down the conversation. It is destructive to healthy debate.

        - "zero-tolerance" (Severe Violation) | 100 points
        The comment contains hate speech, extreme slurs, threats of real-world violence, doxing, or encouragement of self-harm. Warrants an immediate ban.

        FEW-SHOT EXAMPLES:

        Example 1:
        Context: Post about dietary preferences. Previous: "I love pineapple on pizza."
        User Message: "You are literally killing me with that opinion lol, I'm dying."
        Output:
        {
        "explanation": "The user used the words 'killing' and 'dying', which likely triggered the automated classifier. However, within the context of a pizza debate, this is clearly benign hyperbole and a joke.",
        "category": "fp",
        "points": 0
        }

        Example 2:
        Context: Post about economic policy. Previous: "We need to print more money to solve this."
        User Message: "That is completely absurd. Your data is 10 years out of date and frankly, it's a ridiculous and financially illiterate argument."
        Output:
        {
        "explanation": "The user is being highly combative and harsh. However, words like 'absurd', 'ridiculous', and 'financially illiterate' are directed entirely at the argument and the data, not attacking the user personally. This is acceptable heated debate.",
        "category": "flare",
        "points": 1
        }

        Example 3:
        Context: Post about local elections. Previous: "I think candidate A has better infrastructure plans."
        User Message: "You are a braindead moron. He will bankrupt us"
        Output:
        {
        "explanation": "The user is deploying direct ad hominem insults ('braindead moron'). This crosses the line from debating ideas to personal abuse.",
        "category": "toxic",
        "points": 5
        }

        TASK:
        Analyze the following inputs and output a valid JSON object matching the requested schema. Provide your step-by-step explanation first, then the category, then the points.
        """

        USER_MESSAGE: str = f"""TASK:
        Analyze the following inputs and output a valid JSON object matching the requested schema. Provide your step-by-step explanation first, then the category, then the points.

        Original Post:
        {root_context}

        Previous Comment:
        {parent_text}

        User Message:
        {text}
        """

        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": USER_MESSAGE},
        ]

        response: ollama.ChatResponse = ollama.chat(  # pyright: ignore[reportUnknownMemberType]
            self.model, messages, format=ReasoningResult.model_json_schema()
        )
        output = ReasoningResult.model_validate_json(response.message.content)  # pyright: ignore[reportArgumentType]
        return output


class ToxicityClassifier:
    def __init__(self) -> None:
        model_path = Path("models/cga_deberta_onnx_int8")

        if not model_path.exists():
            _ = snapshot_download(
                repo_id="lbux/cga-deberta-onnx-int8",
                local_dir=model_path,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)  # pyright: ignore[reportUnknownMemberType, reportUnannotatedClassAttribute]

        self.ort_model = ORTModelForSequenceClassification.from_pretrained(  # pyright: ignore[reportUnknownMemberType, reportUnannotatedClassAttribute]
            model_path,
            file_name="model_quantized.onnx",
        )

    def predict(self, text: str) -> float:

        # print(self.ort_model.config)

        inputs = self.tokenizer(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
            text, return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            outputs = self.ort_model(**inputs)  # pyright: ignore[reportUnknownVariableType]

        logits = outputs.logits  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        probs = torch.softmax(input=logits, dim=-1)  # pyright: ignore[reportUnknownArgumentType]
        toxicity_score = probs[0][1].item()

        return toxicity_score


# if __name__ == "__main__":
#     classifier = ToxicityClassifier()
#     reasoner = LLMReasoner()
