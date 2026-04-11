import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer


class ToxicityClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("models/cga_deberta_onnx_int8")  # pyright: ignore[reportUnknownMemberType, reportUnannotatedClassAttribute]

        self.ort_model = ORTModelForSequenceClassification.from_pretrained(  # pyright: ignore[reportUnknownMemberType, reportUnannotatedClassAttribute]
            "models/cga_deberta_onnx_int8", file_name="model_quantized.onnx"
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
#     text = "Hello!!"
#     result = classifier.predict(text)
#     print(result)
