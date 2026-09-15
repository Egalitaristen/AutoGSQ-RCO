"""HuggingFace -> GGUF canonical mapping: tensor names, arch KV, tokenizer KV.

llama.cpp only loads GGUF files whose tensors use canonical names
(``blk.N.attn_q.weight`` rather than ``model.layers.N.self_attn.q_proj.weight``)
and whose KV carries architecture scalars and the tokenizer. This module
derives all of that from a model's ``config.json`` + ``tokenizer.json`` so
``assemble`` works for any HF-style model, not just ones we hard-code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# HF model_type -> GGUF architecture tag.
ARCH_FROM_HF: Dict[str, str] = {
    "llama": "llama",
    "qwen3": "qwen3",
    "qwen3_moe": "qwen3moe",
    "qwen2": "qwen2",
    "qwen2_moe": "qwen2moe",
    "mistral": "llama",
    "mixtral": "llama",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "phi3": "phi3",
    "smollm3": "llama",
}

# (HF regex, GGUF template) for decoder-block tensors. `{i}` = layer index.
# Ordered most-specific first; anything unmatched passes through unchanged
# (this is what preserves extra tensors such as grafted MTP heads).
_BLOCK_RULES: List[Tuple[str, str]] = [
    (r"^(?:model\.|transformer\.)?layers?\.(\d+)\.self_attn\.q_norm\.weight$", "blk.{i}.attn_q_norm.weight"),
    (r"^(?:model\.|transformer\.)?layers?\.(\d+)\.self_attn\.k_norm\.weight$", "blk.{i}.attn_k_norm.weight"),
    (r"^(?:model\.|transformer\.)?layers?\.(\d+)\.self_attn\.q_proj\.weight$", "blk.{i}.attn_q.weight"),
    (r"^(?:model\.|transformer\.)?layers?\.(\d+)\.self_attn\.k_proj\.weight$", "blk.{i}.attn_k.weight"),
    (r"^(?:model\.|transformer\.)?layers?\.(\d+)\.self_attn\.v_proj\.weight$", "blk.{i}.attn_v.weight"),
    (r"^(?:model\.|transformer\.)?layers?\.(\d+)\.self_attn\.o_proj\.weight$", "blk.{i}.attn_output.weight"),
    (r"^(?:model\.|transformer\.)?layers?\.(\d+)\.mlp\.gate_proj\.weight$", "blk.{i}.ffn_gate.weight"),
    (r"^(?:model\.|transformer\.)?layers?\.(\d+)\.mlp\.up_proj\.weight$", "blk.{i}.ffn_up.weight"),
    (r"^(?:model\.|transformer\.)?layers?\.(\d+)\.mlp\.down_proj\.weight$", "blk.{i}.ffn_down.weight"),
    (r"^(?:model\.|transformer\.)?layers?\.(\d+)\.input_layernorm\.weight$", "blk.{i}.attn_norm.weight"),
    (r"^(?:model\.|transformer\.)?layers?\.(\d+)\.post_attention_layernorm\.weight$", "blk.{i}.ffn_norm.weight"),
    # GPT-NeoX / SmolLM-style attention naming.
    (r"^(?:gpt_neox\.)?layers\.(\d+)\.attention\.query_key_value\.weight$", "blk.{i}.attn_qkv.weight"),
    (r"^(?:gpt_neox\.)?layers\.(\d+)\.attention\.dense\.weight$", "blk.{i}.attn_output.weight"),
    (r"^(?:gpt_neox\.)?layers\.(\d+)\.mlp\.dense_h_to_4h\.weight$", "blk.{i}.ffn_up.weight"),
    (r"^(?:gpt_neox\.)?layers\.(\d+)\.mlp\.dense_4h_to_h\.weight$", "blk.{i}.ffn_down.weight"),
    (r"^(?:gpt_neox\.)?layers\.(\d+)\.input_layernorm\.weight$", "blk.{i}.attn_norm.weight"),
    (r"^(?:gpt_neox\.)?layers\.(\d+)\.post_attention_layernorm\.weight$", "blk.{i}.ffn_norm.weight"),
]

# Whole-model tensors (matched against the full HF name).
_MODEL_RULES: List[Tuple[str, str]] = [
    (r"^(?:model\.)?embed_tokens\.weight$", "token_embd.weight"),
    (r"^(?:gpt_neox\.)?embed_in\.weight$", "token_embd.weight"),
    (r"^(?:model\.)?norm\.weight$", "output_norm.weight"),
    (r"^(?:gpt_neox\.)?final_layer_norm\.weight$", "output_norm.weight"),
    (r"^lm_head\.weight$", "output.weight"),
]


def map_tensor_name(hf_name: str) -> str:
    """Map one HF weight name to its GGUF canonical name (or pass through)."""
    for pattern, template in _BLOCK_RULES:
        m = re.match(pattern, hf_name)
        if m:
            return template.format(i=m.group(1))
    for pattern, target in _MODEL_RULES:
        if re.match(pattern, hf_name):
            return target
    return hf_name


def map_tensor_names(hf_names: List[str]) -> Dict[str, str]:
    """Map many names; raises on collisions (two HF tensors, one GGUF name)."""
    mapped: Dict[str, str] = {}
    seen: Dict[str, str] = {}
    for name in hf_names:
        gguf_name = map_tensor_name(name)
        if gguf_name in seen:
            raise ValueError(
                f"Name collision: {seen[gguf_name]!r} and {name!r} both map to {gguf_name!r}"
            )
        seen[gguf_name] = name
        mapped[name] = gguf_name
    return mapped


def detect_arch(hf_config: Dict[str, Any], fallback: str = "llama") -> str:
    """GGUF architecture tag from an HF config dict."""
    model_type = str(hf_config.get("model_type", "")).lower()
    if model_type in ARCH_FROM_HF:
        return ARCH_FROM_HF[model_type]
    for arch in hf_config.get("architectures", []) or []:
        a = str(arch).lower()
        if "qwen3" in a:
            return "qwen3"
        if "qwen2" in a:
            return "qwen2"
        if "mistral" in a or "mixtral" in a:
            return "llama"
        if "gemma2" in a:
            return "gemma2"
        if "gemma" in a:
            return "gemma"
        if "llama" in a:
            return "llama"
    return fallback


def load_hf_config(model_dir: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Read config.json from a local model dir (None if absent)."""
    cfg_path = Path(model_dir) / "config.json"
    if not cfg_path.is_file():
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_arch_kv(writer: Any, hf_config: Dict[str, Any], arch: str) -> List[str]:
    """Write architecture KV from an HF config; returns keys written."""
    written: List[str] = []
    # NOTE: the GGUFWriter must already be constructed with the detected arch;
    # its constructor emits general.architecture, so we only add scalars here.
    written.append("general.architecture")

    def put(fn_name: str, value: Any) -> None:
        if value is None:
            return
        getattr(writer, fn_name)(value)
        written.append(fn_name)

    n_heads = int(hf_config.get("num_attention_heads", 0) or 0)
    hidden = int(hf_config.get("hidden_size", 0) or 0)
    # head_dim is explicit in modern configs (e.g. Qwen3: 128 while
    # hidden//n_heads = 80); only fall back to the division.
    head_dim = int(hf_config.get("head_dim", 0) or 0) or (
        (hidden // n_heads) if n_heads else 0
    )

    simple = [
        ("add_context_length", hf_config.get("max_position_embeddings")),
        ("add_embedding_length", hf_config.get("hidden_size")),
        ("add_block_count", hf_config.get("num_hidden_layers")),
        ("add_feed_forward_length", hf_config.get("intermediate_size")),
        ("add_head_count", hf_config.get("num_attention_heads")),
        ("add_head_count_kv", hf_config.get("num_key_value_heads")),
        ("add_layer_norm_rms_eps", hf_config.get("rms_norm_eps")),
        ("add_rope_dimension_count", head_dim or None),
        ("add_key_length", head_dim or None),
        ("add_value_length", head_dim or None),
        ("add_rope_freq_base", hf_config.get("rope_theta")),
        ("add_vocab_size", hf_config.get("vocab_size")),
    ]
    for fn_name, value in simple:
        put(fn_name, value)

    rope_scaling = hf_config.get("rope_scaling")
    if isinstance(rope_scaling, dict) and rope_scaling:
        put("add_rope_scaling_type", rope_scaling.get("type") or rope_scaling.get("rope_type"))
        put("add_rope_scaling_factor", rope_scaling.get("factor"))
        put("add_rope_scaling_orig_ctx_len", rope_scaling.get("original_max_position_embeddings"))

    sliding = hf_config.get("sliding_window")
    put("add_sliding_window", sliding if isinstance(sliding, int) else None)
    return written


def _read_tokenizer_bundle(model_dir: Union[str, Path]) -> Tuple[List[str], List[str], Dict[str, Any], Dict[str, Any]]:
    """Return (tokens_by_id, merges, tokenizer_config, hf_config).

    HF tokenizers split the vocabulary: ``model.vocab`` holds the base BPE
    entries while ``added_tokens`` carries specials (and a few non-special
    additions) at trailing ids. Both are merged here; without the merge the
    file's token list silently drops every special token and loaders fail
    at tokenize time.
    """
    d = Path(model_dir)
    with open(d / "tokenizer.json", "r", encoding="utf-8") as f:
        tok = json.load(f)
    vocab: Dict[str, int] = tok.get("model", {}).get("vocab", {})
    if not vocab:
        raise ValueError(f"No model.vocab in {d / 'tokenizer.json'}")
    by_id: Dict[int, str] = {v: k for k, v in vocab.items()}
    added_special: set = set()
    added_plain: set = set()
    for entry in tok.get("added_tokens", []) or []:
        try:
            tid = int(entry.get("id"))
        except Exception:
            continue
        content = entry.get("content")
        if not isinstance(content, str):
            continue
        by_id[tid] = content
        (added_special if entry.get("special") else added_plain).add(tid)
    if not by_id:
        raise ValueError(f"Empty vocabulary in {d / 'tokenizer.json'}")
    top = max(by_id.keys())
    missing = [i for i in range(top + 1) if i not in by_id]
    if missing:
        raise ValueError(f"Non-contiguous token ids in {d / 'tokenizer.json'} (e.g. {missing[:5]})")
    tokens_by_id = [by_id[i] for i in range(top + 1)]
    merges: List[str] = []
    for m in tok.get("model", {}).get("merges", []) or []:
        # Newer tokenizers serialize merges as ["a", "b"] pairs; older ones
        # as pre-joined "a b" strings. GGUF wants the joined form.
        if isinstance(m, str):
            merges.append(m)
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            merges.append(f"{m[0]} {m[1]}")
    tok_cfg: Dict[str, Any] = {}
    tok_cfg_path = d / "tokenizer_config.json"
    if tok_cfg_path.is_file():
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
    hf_cfg: Dict[str, Any] = load_hf_config(d) or {}
    # Stash added-token classes for the type writer below.
    tok_cfg["__added_special_ids"] = sorted(added_special)
    tok_cfg["__added_plain_ids"] = sorted(added_plain)
    return tokens_by_id, merges, tok_cfg, hf_cfg


def write_tokenizer_kv(
    writer: Any, model_dir: Union[str, Path], model_type: str = ""
) -> List[str]:
    """Write tokenizer KV from tokenizer.json/tokenizer_config.json."""
    tokens_by_id, merges, tok_cfg, hf_cfg = _read_tokenizer_bundle(model_dir)
    written: List[str] = []

    # BPE vocabularies (merges present) decode as gpt2; SentencePiece as llama.
    # Writing "llama" for a BPE vocab breaks tokenization at runtime.
    tok_model = "gpt2" if merges else "llama"
    writer.add_tokenizer_model(tok_model)
    written.append("add_tokenizer_model")

    writer.add_token_list(tokens_by_id)
    written.append("add_token_list")
    if merges:
        writer.add_token_merges(merges)
        written.append("add_token_merges")

    # Types: CONTROL for specials, USER_DEFINED for non-special added tokens,
    # UNKNOWN for unk, NORMAL otherwise. Classes come from tokenizer.json's
    # added_tokens (authoritative) plus tokenizer_config's added_tokens_decoder.
    added = tok_cfg.get("added_tokens_decoder", {}) or {}
    special_ids = set(tok_cfg.get("__added_special_ids", []) or [])
    plain_ids = set(tok_cfg.get("__added_plain_ids", []) or [])
    for k, v in added.items():
        if isinstance(v, dict):
            try:
                tid = int(k)
            except Exception:
                continue
            if v.get("special"):
                special_ids.add(tid)
            else:
                plain_ids.add(tid)
    unk_token = tok_cfg.get("unk_token")
    if isinstance(unk_token, dict):
        unk_token = unk_token.get("content")
    try:
        unk_id = tokens_by_id.index(unk_token) if unk_token in tokens_by_id else None
    except Exception:
        unk_id = None

    types = []
    for i in range(len(tokens_by_id)):
        if i in special_ids:
            types.append(3)  # CONTROL (special wins, matching official files)
        elif i in plain_ids:
            types.append(4)  # USER_DEFINED (added but not special)
        elif unk_id is not None and i == unk_id:
            types.append(2)  # UNKNOWN
        else:
            types.append(1)  # NORMAL
    writer.add_token_types(types)
    written.append("add_token_types")

    # Scores are omitted for BPE vocabularies (official files omit them too).

    def _tok_id(*keys: str) -> Optional[int]:
        for key in keys:
            for src in (tok_cfg, hf_cfg):
                v = src.get(key)
                if isinstance(v, int):
                    return v
                if isinstance(v, dict) and isinstance(v.get("content"), str):
                    try:
                        return tokens_by_id.index(v["content"])
                    except ValueError:
                        pass
                if isinstance(v, str) and v in tokens_by_id:
                    return tokens_by_id.index(v)
        return None

    bos = _tok_id("bos_token_id", "bos_token")
    eos = _tok_id("eos_token_id", "eos_token")
    unk = unk_id if unk_id is not None else _tok_id("unk_token")
    pad = _tok_id("pad_token_id", "pad_token")
    if bos is not None:
        writer.add_bos_token_id(bos)
        written.append("add_bos_token_id")
    if eos is not None:
        writer.add_eos_token_id(eos)
        written.append("add_eos_token_id")
    if unk is not None:
        writer.add_unk_token_id(unk)
        written.append("add_unk_token_id")
    if pad is not None:
        writer.add_pad_token_id(pad)
        written.append("add_pad_token_id")

    chat = tok_cfg.get("chat_template")
    if isinstance(chat, str) and chat:
        writer.add_chat_template(chat)
        written.append("add_chat_template")

    # Tokenizer behaviour flags (affect tokenization, not compute).
    # llama.cpp dispatches BPE splitting on the pre string; it must match
    # the tokenizer family or common pieces miss the vocab at runtime
    # ("unordered_map::at: key not found").
    mt = (model_type or "").lower()
    first_merge = merges[0] if merges else ""
    if mt in ("qwen3", "qwen2", "qwen2_moe", "qwen3_moe"):
        pre = "qwen2"
    elif len(tokens_by_id) == 49152 and first_merge.startswith("Ġ "):
        pre = "smollm"
    elif mt == "llama" and len(tokens_by_id) == 128256:
        pre = "llama3"
    else:
        pre = "default"
    writer.add_tokenizer_pre(pre)
    written.append("add_tokenizer_pre")
    add_bos = tok_cfg.get("add_bos_token")
    if not isinstance(add_bos, bool):
        add_bos = tok_model != "gpt2"  # SPM/llama-style tokenizers prepend BOS
    writer.add_add_bos_token(add_bos)
    written.append("add_add_bos_token")
    add_prefix = tok_cfg.get("add_prefix_space")
    if isinstance(add_prefix, bool):
        writer.add_add_space_prefix(add_prefix)
        written.append("add_add_space_prefix")
    return written
