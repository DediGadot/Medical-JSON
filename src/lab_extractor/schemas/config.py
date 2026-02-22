from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    model_name: str = "google/medgemma-1.5-4b-it"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 2048
    dry_run: bool = False
    output_dir: str = "output"
    max_retries: int = 3
    image_size: int = 896
    quantize_4bit: bool = False
    custom_extract_prompt: str | None = None  # Override extraction prompt (e.g. for CORD)
