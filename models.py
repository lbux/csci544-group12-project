from pathlib import Path

import ollama
import torch
from huggingface_hub import (
    snapshot_download,  # pyright: ignore[reportUnknownVariableType]
)
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

from interfaces import ReasoningResult
from prompts import SYSTEM_MESSAGE


class LLMReasoner:
    def __init__(self, model: str = "gemma4:e4b", prompt: str = SYSTEM_MESSAGE) -> None:
        self.model: str = model
        self.prompt: str = prompt
        # self.client: Client = ollama.Client(
        #     host="http://localhost:14434",
        # )
        _ = ollama.pull(self.model)

    def analyze_intent(
        self, comment_body: str, parent_body: str, thread_context: str
    ) -> ReasoningResult:
        """Takes the comment, parent comment, and thread body as context to produce a reasoning result from the LLM Reasoner"""

        USER_MESSAGE: str = f"""
        Original Post:
        {parent_body}

        Previous Comment:
        {thread_context}

        User Message:
        {comment_body}
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
        """"Assigns a Toxicity score to the input using a pre-trained classifier"""
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
