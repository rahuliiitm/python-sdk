"""
Export the fine-tuned attack classifier to ONNX format.

Exports both FP32 and INT8 quantized versions, and verifies the
output config.json contains the id2label mapping required by OnnxSession.

Usage:
    python scripts/export_onnx.py [--model-dir ./output/attack-classifier-v1/best]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent / "output" / "attack-classifier-v1" / "best"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent / "output" / "attack-classifier-v1" / "onnx"
        ),
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Exporting model from {model_dir} to ONNX...")

    # Export using optimum
    from optimum.onnxruntime import ORTModelForSequenceClassification

    model = ORTModelForSequenceClassification.from_pretrained(
        str(model_dir), export=True
    )

    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    print(f"  FP32 model saved to {output_dir}")

    # Copy tokenizer files
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    # Verify config.json has id2label
    config_path = output_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        if "id2label" not in config:
            raise ValueError("config.json missing id2label mapping")
        print(f"  Labels: {config['id2label']}")
    else:
        raise FileNotFoundError("config.json not found in export output")

    # Quantize to INT8
    print("\nQuantizing to INT8...")
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(str(output_dir))
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)

        quantized_dir = output_dir / "quantized"
        quantized_dir.mkdir(parents=True, exist_ok=True)
        quantizer.quantize(save_dir=str(quantized_dir), quantization_config=qconfig)
        print(f"  INT8 model saved to {quantized_dir}")

        # Copy quantized model to onnx/model_quantized.onnx
        quantized_model = quantized_dir / "model_quantized.onnx"
        if quantized_model.exists():
            target = output_dir / "onnx" / "model_quantized.onnx"
            shutil.copy2(quantized_model, target)
            print(f"  Copied to {target}")
    except Exception as e:
        print(f"  Quantization failed (non-critical): {e}")
        print("  FP32 model is still available")

    # Reorganize to expected layout:
    # output_dir/onnx/model.onnx
    # output_dir/onnx/model_quantized.onnx (if quantized)
    # output_dir/tokenizer.json, config.json, etc.
    onnx_model = output_dir / "model.onnx"
    if onnx_model.exists():
        target = output_dir / "onnx" / "model.onnx"
        if not target.exists():
            shutil.move(str(onnx_model), str(target))
            print(f"\n  Moved model.onnx to onnx/model.onnx")

    # Verify ONNX model loads
    print("\nVerifying ONNX model...")
    import onnxruntime as ort

    onnx_path = output_dir / "onnx" / "model.onnx"
    if onnx_path.exists():
        session = ort.InferenceSession(str(onnx_path))
        print(f"  Inputs:  {[i.name for i in session.get_inputs()]}")
        print(f"  Outputs: {[o.name for o in session.get_outputs()]}")
    else:
        print(f"  WARNING: {onnx_path} not found")

    # Print file sizes
    print("\nFile sizes:")
    for p in sorted(output_dir.rglob("*.onnx")):
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.relative_to(output_dir)}: {size_mb:.1f} MB")

    print(f"\nExport complete. Model ready at: {output_dir}")
    print("Next step: python scripts/evaluate_model.py")


if __name__ == "__main__":
    main()
