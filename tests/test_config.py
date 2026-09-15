"""Unit tests for configuration loading and strict Pydantic validation."""

from pathlib import Path
import pytest
from pydantic import ValidationError

from autogsq.config import load_config, PipelineConfig


def test_load_valid_example_config():
    """Verify example YAML configurations load and strictly validate without error."""
    config_path = Path("configs/examples/qwen3_8b_2p75bpw.yaml")
    cfg = load_config(config_path)

    assert isinstance(cfg, PipelineConfig)
    assert cfg.model.name == "Qwen/Qwen3-8B"
    assert cfg.rco.target_bpw == 2.75
    assert cfg.gsq.bitwidth_options == [2, 3, 4, "ternary"]
    assert cfg.gsq.group_size == 128


def test_strict_validation_rejects_unknown_keys(tmp_path: Path):
    """Verify that unknown keys in the YAML file cause validation errors (strict mode)."""
    invalid_yaml = tmp_path / "invalid_config.yaml"
    invalid_yaml.write_text(
        """
model:
  name: "test-model"
  device: "cpu"
  dtype: "float32"
  unknown_model_param: 12345

data:
  dataset_name: "wikitext2"
  num_samples: 128
  max_length: 512

gsq:
  bitwidth_options: [2, 4]
  init_method: "rtn"
  group_size: 128
  temperature: [2.0, 0.05]
  logit_scale: [100.0, 500.0]
  num_epochs: 2
  optimizer: "lion"
  learning_rate: 0.001

rco:
  target_bpw: 3.0
  eval_dataset: "wikitext2"
  num_steps: 100
  lr: 0.01
  temperature: 1.0
  temperature_min: 0.1

gguf:
  gguf_type_tolerance_bpw: 0.15
  fallback_on_no_match: "round_down"

output:
  checkpoint_dir: "./runtime"
  run_id: null
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError) as exc_info:
        load_config(invalid_yaml)

    assert "extra_forbidden" in str(exc_info.value) or "unknown_model_param" in str(exc_info.value)
