from __future__ import annotations
import json
import logging
import time
from typing import TYPE_CHECKING

import torch
from PIL import Image

from lab_extractor.schemas.config import PipelineConfig

if TYPE_CHECKING:
    from transformers import Pipeline

logger = logging.getLogger(__name__)


class MedGemmaEngine:
    """Wrapper around MedGemma 1.5 4B for lab report inference.

    In dry_run mode: no model is loaded, mock responses are returned.
    In normal mode: loads google/medgemma-1.5-4b-it via HuggingFace pipeline.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.pipe: Pipeline | None = None

        if not config.dry_run:
            self._load_model()

    def _load_model(self) -> None:
        """Load MedGemma via HuggingFace pipeline."""
        from transformers import pipeline

        logger.info("Loading MedGemma model: %s", self.config.model_name)

        if self.config.quantize_4bit:
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
                BitsAndBytesConfig,
            )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForImageTextToText.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(self.config.model_name)
            # Store as tuple for 4-bit path
            self.pipe = (model, processor)
        else:
            dtype = (
                torch.bfloat16
                if self.config.torch_dtype == "bfloat16"
                else torch.float16
            )
            self.pipe = pipeline(
                "image-text-to-text",
                model=self.config.model_name,
                torch_dtype=dtype,
                device=self.config.device,
            )
        logger.info("Model loaded successfully")

    def query(self, image: Image.Image, prompt: str) -> str:
        """Send image + prompt to MedGemma, return text response.

        In dry_run mode: returns deterministic mock JSON based on prompt keywords.
        In normal mode: calls MedGemma via HuggingFace pipeline.
        """
        if self.config.dry_run:
            return self._mock_response(prompt)

        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Handle 4-bit quantized path
        if isinstance(self.pipe, tuple):
            model, processor = self.pipe
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                model.device
            )
            output_ids = model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens
            )
            return processor.decode(output_ids[0], skip_special_tokens=True)

        # Standard pipeline path (EXACT MedGemma API)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        output = self.pipe(text=messages, max_new_tokens=self.config.max_new_tokens)
        return output[0]["generated_text"][-1]["content"]

    def _mock_response(self, prompt: str) -> str:
        """Return deterministic mock JSON for dry-run mode.

        Detects prompt type from keywords and returns realistic mock data.
        """
        prompt_lower = prompt.lower()

        if (
            "classify" in prompt_lower
            or "type of lab report" in prompt_lower
            or "report type" in prompt_lower
        ):
            return json.dumps(
                {
                    "report_type": "cbc",
                    "confidence": 0.92,
                    "reasoning": "Header contains 'Complete Blood Count' and CBC-specific markers (Hemoglobin, WBC, Platelets)",
                }
            )

        elif (
            "extract" in prompt_lower
            or "lab test results" in prompt_lower
            or "test results" in prompt_lower
        ):
            return json.dumps(
                {
                    "patient_info": {
                        "name": "Ahmed Hassan",
                        "age": "45",
                        "gender": "M",
                        "patient_id": "PT-2024-001",
                    },
                    "lab_name": "Cairo Medical Center",
                    "report_date": "2024-01-15",
                    "report_type": "cbc",
                    "tests": [
                        {
                            "test_name": "Hemoglobin",
                            "value": "14.2",
                            "unit": "g/dL",
                            "reference_range": {"low": 13.5, "high": 17.5},
                        },
                        {
                            "test_name": "WBC",
                            "value": "7.8",
                            "unit": "x10^3/uL",
                            "reference_range": {"low": 4.5, "high": 11.0},
                        },
                        {
                            "test_name": "Platelets",
                            "value": "220",
                            "unit": "x10^3/uL",
                            "reference_range": {"low": 150.0, "high": 400.0},
                        },
                        {
                            "test_name": "RBC",
                            "value": "5.1",
                            "unit": "x10^6/uL",
                            "reference_range": {"low": 4.7, "high": 6.1},
                        },
                        {
                            "test_name": "Hematocrit",
                            "value": "42.5",
                            "unit": "%",
                            "reference_range": {"low": 38.3, "high": 48.6},
                        },
                        {
                            "test_name": "MCV",
                            "value": "83.3",
                            "unit": "fL",
                            "reference_range": {"low": 80.0, "high": 100.0},
                        },
                        {
                            "test_name": "MCH",
                            "value": "27.8",
                            "unit": "pg",
                            "reference_range": {"low": 27.0, "high": 33.0},
                        },
                    ],
                    "metadata": {"lab_id": "LAB-001", "physician": "Dr. Mohamed Ali"},
                }
            )

        elif (
            "validate" in prompt_lower
            or "accuracy" in prompt_lower
            or "review" in prompt_lower
        ):
            return json.dumps(
                {"valid": True, "issues": [], "corrections": None, "confidence": 0.95}
            )

        else:
            # Generic fallback
            return json.dumps(
                {
                    "response": "Mock response for unrecognized prompt type",
                    "prompt_received": prompt[:100],
                }
            )
