from optimum.onnxruntime import (  # pyright: ignore[reportMissingTypeStubs]
    ORTModelForSequenceClassification,
    ORTQuantizer,
)
from optimum.onnxruntime.configuration import (  # pyright: ignore[reportMissingTypeStubs]
    AutoQuantizationConfig,
)
from transformers import AutoTokenizer

model_path = "models/cga_deberta"
onnx_path = "models/cga_deberta_onnx"

model = ORTModelForSequenceClassification.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
    model_id=model_path, export=True
)

tokenizer = AutoTokenizer.from_pretrained(model_path)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

_ = model.save_pretrained(onnx_path)  # pyright: ignore[reportUnknownMemberType]
tokenizer.save_pretrained(onnx_path)  # pyright: ignore[reportUnknownMemberType]

qconfig = AutoQuantizationConfig.avx2(is_static=False, per_channel=True)

quantizer = ORTQuantizer.from_pretrained(onnx_path)

_ = quantizer.quantize(
    quantization_config=qconfig,
    save_dir="models/cga_deberta_onnx_int8",
)
