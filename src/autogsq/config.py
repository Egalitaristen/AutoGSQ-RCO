"""Configuration schema and strict validation for AutoGSQ-RCO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union
import uuid
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictBaseModel(BaseModel):
    """Base model that forbids unknown attributes."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelConfig(StrictBaseModel):
    name: str = Field(..., description="HuggingFace model ID or local directory path")
    device: str = Field("cuda", description="Device to run on (e.g. cuda, cpu)")
    dtype: str = Field("bfloat16", description="Torch dtype (e.g. bfloat16, float16, float32)")


class DataConfig(StrictBaseModel):
    dataset_name: Literal["wikitext2", "c4", "fineweb_edu"] = Field(
        "fineweb_edu", description="Calibration dataset name"
    )
    num_samples: int = Field(4096, ge=1, description="Number of calibration samples")
    max_length: int = Field(4096, ge=16, description="Sequence length for calibration")


class GSQConfig(StrictBaseModel):
    bitwidth_options: List[Union[int, Literal["ternary"]]] = Field(
        default_factory=lambda: [2, 3, 4, "ternary"],
        description="Candidate bit-widths to generate per layer",
    )
    init_method: Literal["gptq", "rtn"] = Field(
        "gptq", description="Initial quantization method: gptq or rtn"
    )
    group_size: int = Field(128, ge=1, description="Quantization group size along input dim")
    temperature: Tuple[float, float] = Field(
        (2.0, 0.05), description="Temperature annealing [high, low]"
    )
    logit_scale: Tuple[float, float] = Field(
        (100.0, 500.0), description="Logit scale annealing [low, high]"
    )
    num_epochs: int = Field(10, ge=1, description="Number of GSQ optimization epochs per layer")
    optimizer: Literal["lion", "adam"] = Field("lion", description="Optimizer to use for GSQ")
    learning_rate: float = Field(1e-3, gt=0.0, description="Learning rate for quantizer logits")
    # Paper-quality training (GSQ reference defaults).
    steps_per_epoch: int = Field(64, ge=1, description="Gradient steps per epoch (annealing is per step)")
    lr_logits: float = Field(2e-4, gt=0.0, description="Lion LR for mask/sign/offset logits")
    lr_scales: float = Field(1e-4, gt=0.0, description="Lion LR for per-group scales")
    gs_weight_decay: float = Field(1.0, ge=0.0, description="Weight decay on logits (0 on scales)")
    warmup_ratio: float = Field(0.1, ge=0.0, le=0.5, description="Fraction of steps for LR warmup")
    gptq_nsamples: int = Field(128, ge=1, description="Calibration sequences for Hessian capture")
    gptq_damping: float = Field(0.01, gt=0.0, description="Relative damping for Hessian inversion")
    calib_max_length: int = Field(2048, ge=16, description="Sequence length for Hessian capture")
    calib_device: str = Field("cpu", description="Device for Hessian capture (cpu safe, cuda fast)")
    logits_dtype: Literal["bfloat16", "float32"] = Field(
        "bfloat16", description="Dtype for GSQ logits (reference: bf16)"
    )

    @field_validator("bitwidth_options")
    @classmethod
    def validate_bitwidths(cls, v: List[Union[int, str]]) -> List[Union[int, str]]:
        if not v:
            raise ValueError("bitwidth_options cannot be empty")
        for item in v:
            if item == "ternary":
                continue
            if not isinstance(item, int) or item < 1 or item > 16:
                raise ValueError(f"Invalid bitwidth option: {item}. Must be integer (1-16) or 'ternary'.")
        return v


class RCOConfig(StrictBaseModel):
    target_bpw: float = Field(..., gt=0.0, le=16.0, description="Target bits per weight budget")
    eval_dataset: Literal["wikitext2", "c4", "fineweb_edu"] = Field(
        "wikitext2", description="Dataset for RCO allocation evaluation"
    )
    num_steps: int = Field(2000, ge=10, description="Number of RCO optimization steps")
    lr: float = Field(0.01, gt=0.0, description="Learning rate for RCO optimizer")
    temperature: float = Field(1.0, gt=0.0, description="Initial Gumbel temperature for RCO")
    temperature_min: float = Field(0.1, gt=0.0, description="Minimum Gumbel temperature for RCO")
    temperature_schedule: Literal["exponential", "linear"] = Field(
        "exponential", description="Gumbel temperature schedule (reference: exponential)"
    )
    # Task-loss refinement (RCO reference: forward passes on soft-mixed model).
    refine_steps: int = Field(50, ge=0, description="Task-loss RCO refinement steps after MSE init (0 = skip)")
    refine_batches: int = Field(2, ge=1, description="Minibatches per refinement step")
    refine_max_length: int = Field(512, ge=16, description="Sequence length for refinement batches")


class GGUFConfig(StrictBaseModel):
    gguf_type_tolerance_bpw: float = Field(
        0.15, ge=0.0, description="Max allowed drift between RCO bits and nearest GGUF type"
    )
    fallback_on_no_match: Literal["round_down", "round_up", "error"] = Field(
        "round_down", description="Behavior when no GGUF type is within tolerance"
    )


class OutputConfig(StrictBaseModel):
    checkpoint_dir: str = Field("./runtime", description="Directory for intermediate checkpoints")
    run_id: Optional[str] = Field(None, description="Unique run ID, auto-generated if omitted")

    def get_run_id(self) -> str:
        if self.run_id is not None and self.run_id.strip():
            return self.run_id.strip()
        return f"run_{uuid.uuid4().hex[:8]}"


class PipelineConfig(StrictBaseModel):
    model: ModelConfig
    data: DataConfig
    gsq: GSQConfig
    rco: RCOConfig
    gguf: GGUFConfig
    output: OutputConfig


def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """Load and strictly validate configuration from a YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        raise ValueError(f"Config YAML must define a mapping, got {type(raw_data)}")

    return PipelineConfig.model_validate(raw_data)


def save_config(config: PipelineConfig, out_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
