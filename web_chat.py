"""
web_chat.py — Flask web GUI for dynamic multi-model Seq2Seq chatbot comparison.

Version : 4.0.0
Modified: 2026-03-18
Changes : v4.0.0 — Version bump for multi-corpus project
          v3.2.0 — Flask web GUI, model registry, attention heatmaps
          v3.0.0 — Initial web interface

Provides a browser-based chat interface with model registry support:
  - Dynamic model registry (model_registry.json) for any number of models
  - Side-by-side model response comparison with selectable models
  - Per-token confidence bars (color-coded)
  - Attention weights heatmap (context x response tokens)
  - Inference metrics panel (tokens/sec, latency, confidence, etc.)
  - Confidence analysis (perplexity, distribution histogram)
  - BLEU scores 1-4 (response vs input, simple n-gram overlap)
  - Model config info panel

Usage:
    python web_chat.py
    # Opens at http://localhost:5000
"""

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template_string, request

from config import CONFIG
from gpu_utils import setup_device
from models import build_model
from phase1 import _clean_text

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------
DEVICE: torch.device = torch.device("cpu")
REGISTRY: list = []  # raw entries from model_registry.json
LOADED_MODELS: Dict[str, dict] = {}  # keyed by model id, each value has: model, tokenizer, meta, config_info

# ---------------------------------------------------------------------------
# Startup: load models from registry
# ---------------------------------------------------------------------------

def _load_models():
    """Load model registry and all models with valid checkpoints."""
    global DEVICE, REGISTRY, LOADED_MODELS

    device, gpu_info = setup_device()
    DEVICE = device

    import sentencepiece as spm_module

    # Load registry relative to this script's location
    script_dir = Path(__file__).resolve().parent
    registry_path = script_dir / "model_registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry not found: {registry_path}")

    with open(registry_path, "r", encoding="utf-8") as f:
        REGISTRY = json.load(f)

    print(f"[web_chat] Loaded registry with {len(REGISTRY)} entries")

    for entry in REGISTRY:
        model_id = entry["id"]
        architecture = entry["architecture"]
        ckpt_rel = entry["checkpoint"]
        tok_rel = entry["tokenizer"]

        ckpt_path = script_dir / ckpt_rel
        tok_path = script_dir / tok_rel

        if not ckpt_path.exists():
            print(f"[web_chat] WARNING: {ckpt_path} not found -- skipping {model_id}")
            continue

        if not tok_path.exists():
            print(f"[web_chat] WARNING: tokenizer {tok_path} not found -- skipping {model_id}")
            continue

        try:
            # Load tokenizer for this model
            sp = spm_module.SentencePieceProcessor(model_file=str(tok_path))
            print(f"[web_chat] Tokenizer loaded from {tok_path} for {model_id}")

            # Load model checkpoint
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            model = build_model(architecture, CONFIG, device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            n_params = sum(p.numel() for p in model.parameters())

            meta = {
                "epoch": ckpt.get("epoch", "?"),
                "val_loss": round(ckpt.get("val_loss", float("nan")), 4),
                "n_params": n_params,
            }

            has_attn = architecture == "attention"
            config_info = {
                "model_type": architecture,
                "encoder": f"Bidirectional {CONFIG.get('num_layers', 2)}-layer LSTM",
                "decoder": f"{'Attention' if has_attn else 'Baseline'} {CONFIG.get('num_layers', 2)}-layer LSTM",
                "total_params": f"{n_params:,}",
                "decoding": "top-p nucleus sampling",
                "hidden_dim": CONFIG.get("dec_hidden_dim", 1024),
                "embed_dim": CONFIG.get("embed_dim", 300),
                "vocab_size": CONFIG.get("vocab_size", 16000),
            }

            LOADED_MODELS[model_id] = {
                "model": model,
                "tokenizer": sp,
                "meta": {**meta, **entry},  # merge checkpoint meta + registry entry
                "config_info": config_info,
            }

            print(f"[web_chat] Loaded {model_id} (arch={architecture}, epoch={meta['epoch']}, "
                  f"params={n_params:,})")
        except Exception as e:
            print(f"[web_chat] ERROR loading {model_id}: {e}")

    if not LOADED_MODELS:
        print("[web_chat] WARNING: No models loaded! Chat will return errors.")


# ---------------------------------------------------------------------------
# Context building (mirrors chatv2.py) — accepts tokenizer parameter
# ---------------------------------------------------------------------------

def _build_context(history: List[str], tokenizer) -> torch.Tensor:
    max_ctx_tokens = CONFIG.get("max_ctx_tokens", 100)
    max_turns = CONFIG.get("max_ctx_turns", 8)
    turns = history[-max_turns:] if history else []
    cleaned = [_clean_text(t) for t in turns]
    cleaned = [t for t in cleaned if t]
    joined = " __eot__ ".join(cleaned) if cleaned else ""
    tokens = tokenizer.encode(joined, out_type=int) if joined else []
    tokens = tokens[-max_ctx_tokens:]
    if not tokens:
        tokens = [CONFIG.get("unk_idx", 1)]
    return torch.tensor([tokens], dtype=torch.long)


# ---------------------------------------------------------------------------
# Step-by-step decoding with per-token details — accepts tokenizer parameter
# ---------------------------------------------------------------------------

def decode_with_details(
    model,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    device: torch.device,
    tokenizer,
    mode: str = "greedy",
    top_p: float = 0.9,
    temperature: float = 0.8,
    ngram_block: int = 3,
) -> dict:
    """
    Decode using the proven evaluate.py functions, then do a second pass
    to extract per-token confidence and attention weights.

    Returns dict with keys:
      token_ids, pieces, confidences, attn_weights (list of lists or None),
      response_text
    """
    from evaluate import greedy_decode, top_p_decode

    model.eval()
    sos_idx = CONFIG["sos_idx"]
    eos_idx = CONFIG["eos_idx"]
    max_len = CONFIG.get("max_decode_len", 40)

    # Step 1: Get token IDs using the proven decoders from evaluate.py
    if mode == "greedy":
        with torch.no_grad():
            ids_batch = greedy_decode(model, src, src_lengths, sos_idx, eos_idx,
                                      max_len, device)
    else:
        ids_batch = top_p_decode(model, src, src_lengths, sos_idx, eos_idx,
                                  max_len, device, top_p, temperature, ngram_block)

    token_ids = ids_batch[0] if ids_batch else []

    # Step 2: Re-run the decoder step-by-step to extract confidence + attention
    confidences: List[float] = []
    attn_weights_list: List[List[float]] = []
    has_attention = hasattr(model.decoder, "attention")

    if token_ids:
        with torch.no_grad():
            src_d = src.to(device)
            src_lengths_d = src_lengths.to(device)
            encoder_outputs, (h_n, c_n) = model.encoder(src_d, src_lengths_d)
            src_mask = (src_d == model.encoder.embedding.padding_idx)
            dec_h, dec_c = model.bridge(h_n, c_n)
            input_token = torch.full((1,), sos_idx, dtype=torch.long, device=device)
            context = torch.zeros(1, encoder_outputs.size(-1), device=device)

            for tid in token_ids:
                logits, dec_h, dec_c, context, step_attn = model.decoder.forward_step(
                    input_token, dec_h, dec_c, encoder_outputs, context, src_mask
                )
                probs = F.softmax(logits[0], dim=-1)
                confidences.append(probs[tid].item())

                if has_attention and step_attn is not None:
                    attn_weights_list.append(step_attn.squeeze(0).cpu().tolist())

                input_token = torch.full((1,), tid, dtype=torch.long, device=device)

    # Get piece strings using this model's tokenizer
    pieces = []
    for tid in token_ids:
        try:
            piece = tokenizer.id_to_piece(tid)
        except Exception:
            piece = f"<{tid}>"
        pieces.append(piece)

    response_text = tokenizer.decode(token_ids) if token_ids else ""

    return {
        "token_ids": token_ids,
        "pieces": pieces,
        "confidences": confidences,
        "attn_weights": attn_weights_list if has_attention else None,
        "response_text": response_text,
    }


# ---------------------------------------------------------------------------
# Simple BLEU (response vs input, n-gram overlap)
# ---------------------------------------------------------------------------

def _compute_simple_bleu(response_ids: List[int], input_ids: List[int]) -> Dict[str, float]:
    """Compute BLEU-1 through BLEU-4 as simple n-gram precision."""
    results = {}
    for n in range(1, 5):
        if len(response_ids) < n:
            results[f"bleu{n}"] = 0.0
            continue
        # Response n-grams
        resp_ngrams = [tuple(response_ids[i:i+n]) for i in range(len(response_ids) - n + 1)]
        # Input n-grams
        inp_ngrams = Counter(tuple(input_ids[i:i+n]) for i in range(len(input_ids) - n + 1))
        if not resp_ngrams:
            results[f"bleu{n}"] = 0.0
            continue
        matches = 0
        for ng in resp_ngrams:
            if inp_ngrams[ng] > 0:
                matches += 1
                inp_ngrams[ng] -= 1
        results[f"bleu{n}"] = round(matches / len(resp_ngrams), 4)
    return results


# ---------------------------------------------------------------------------
# Confidence analysis
# ---------------------------------------------------------------------------

def _confidence_analysis(confidences: List[float]) -> dict:
    if not confidences:
        return {"perplexity": 0, "very_low": 0, "low": 0, "mid": 0, "high": 0}

    # Perplexity = exp(-mean(log(conf)))
    log_sum = sum(math.log(max(c, 1e-10)) for c in confidences)
    perplexity = math.exp(-log_sum / len(confidences))

    # Distribution buckets
    very_low = sum(1 for c in confidences if c < 0.4)
    low = sum(1 for c in confidences if 0.4 <= c < 0.6)
    mid = sum(1 for c in confidences if 0.6 <= c < 0.8)
    high = sum(1 for c in confidences if c >= 0.8)

    return {
        "perplexity": round(perplexity, 2),
        "very_low": very_low,
        "low": low,
        "mid": mid,
        "high": high,
    }


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    """Health check for OpenShift probes."""
    return jsonify({
        "status": "ok",
        "models_loaded": [
            {"id": mid, "name": LOADED_MODELS[mid]["meta"].get("name", mid)}
            for mid in LOADED_MODELS
        ],
        "device": str(DEVICE),
    })


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Main inference endpoint. Returns all data as JSON."""
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    mode = data.get("mode", "greedy")
    selected_models = data.get("selected_models", None)

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    if not LOADED_MODELS:
        return jsonify({"error": "No models loaded"}), 503

    # Determine which models to run
    if selected_models:
        model_ids = [mid for mid in selected_models if mid in LOADED_MODELS]
    else:
        model_ids = list(LOADED_MODELS.keys())

    if not model_ids:
        return jsonify({"error": "No valid models selected"}), 400

    history_with_msg = history + [user_message]

    results = {}

    for model_id in model_ids:
        entry = LOADED_MODELS[model_id]
        model = entry["model"]
        tokenizer = entry["tokenizer"]

        # Build context with this model's tokenizer
        context_tensor = _build_context(history_with_msg, tokenizer)
        src_lengths = torch.tensor([context_tensor.size(1)], dtype=torch.long)

        # Get source token info using this model's tokenizer
        src_ids = context_tensor[0].tolist()
        src_pieces = []
        for tid in src_ids:
            try:
                src_pieces.append(tokenizer.id_to_piece(tid))
            except Exception:
                src_pieces.append(f"<{tid}>")

        unk_idx = CONFIG.get("unk_idx", 1)
        unk_count = sum(1 for t in src_ids if t == unk_idx)

        t0 = time.perf_counter()
        details = decode_with_details(
            model, context_tensor, src_lengths, DEVICE,
            tokenizer=tokenizer,
            mode=mode,
            top_p=CONFIG.get("top_p", 0.9),
            temperature=CONFIG.get("temperature", 0.8),
            ngram_block=CONFIG.get("ngram_block", 3),
        )
        elapsed = time.perf_counter() - t0

        n_tokens = len(details["token_ids"])
        tokens_per_sec = n_tokens / max(elapsed, 1e-6)
        avg_conf = sum(details["confidences"]) / max(len(details["confidences"]), 1)

        # BLEU scores (response vs input)
        bleu_scores = _compute_simple_bleu(details["token_ids"], src_ids)

        # Confidence analysis
        conf_analysis = _confidence_analysis(details["confidences"])

        results[model_id] = {
            "response_text": details["response_text"],
            "pieces": details["pieces"],
            "confidences": details["confidences"],
            "attn_weights": details["attn_weights"],
            "src_pieces": src_pieces,
            "metrics": {
                "tokens_per_sec": round(tokens_per_sec, 1),
                "latency_ms": round(elapsed * 1000, 1),
                "avg_confidence": round(avg_conf, 4),
                "unk_in_input": unk_count,
                "response_len": n_tokens,
                "input_tokens": len(src_ids),
            },
            "bleu": bleu_scores,
            "confidence_analysis": conf_analysis,
            "config_info": entry["config_info"],
            "meta": {
                "id": model_id,
                "name": entry["meta"].get("name", model_id),
                "architecture": entry["meta"].get("architecture", "unknown"),
                "dataset": entry["meta"].get("dataset", ""),
                "description": entry["meta"].get("description", ""),
            },
        }

    return jsonify({
        "results": results,
        "models_used": model_ids,
    })


@app.route("/api/models")
def api_models():
    """Return loaded models and their registry metadata."""
    models_info = {}
    for mid, entry in LOADED_MODELS.items():
        models_info[mid] = {
            "loaded": True,
            "id": mid,
            "name": entry["meta"].get("name", mid),
            "architecture": entry["meta"].get("architecture", "unknown"),
            "dataset": entry["meta"].get("dataset", ""),
            "description": entry["meta"].get("description", ""),
            "epoch": entry["meta"].get("epoch", "?"),
            "val_loss": entry["meta"].get("val_loss", None),
            "n_params": entry["meta"].get("n_params", 0),
            "config": entry["config_info"],
        }
    return jsonify({
        "models": models_info,
        "device": str(DEVICE),
    })


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ubuntu Seq2Seq Chatbot v2.0</title>
<style>
:root {
    --bg: #f5f7fa;
    --card-bg: #ffffff;
    --border: #e2e8f0;
    --text: #1a202c;
    --text-secondary: #64748b;
    --primary: #3b82f6;
    --primary-dark: #2563eb;
    --accent: #8b5cf6;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
    --shadow-md: 0 4px 6px rgba(0,0,0,0.07), 0 2px 4px rgba(0,0,0,0.06);
    --radius: 12px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
}

/* Header */
.header {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    color: white;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    position: sticky;
    top: 0;
    z-index: 50;
    flex-wrap: wrap;
    gap: 12px;
}

.header h1 {
    font-size: 20px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 12px;
}

.badge {
    background: linear-gradient(135deg, var(--primary), var(--accent));
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.header-controls {
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
}

.header-controls select {
    background: rgba(255,255,255,0.15);
    color: white;
    border: 1px solid rgba(255,255,255,0.3);
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 13px;
    cursor: pointer;
}

.header-controls select option { color: #333; }

.status-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--success);
    display: inline-block;
    margin-right: 6px;
}

.status-text { font-size: 12px; opacity: 0.9; }

/* Model selector checkboxes in header */
.model-selector {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.model-selector label {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    cursor: pointer;
    background: rgba(255,255,255,0.1);
    padding: 4px 10px;
    border-radius: 6px;
    transition: background 0.2s;
}

.model-selector label:hover {
    background: rgba(255,255,255,0.2);
}

.model-selector input[type="checkbox"] {
    accent-color: var(--primary);
    cursor: pointer;
}

/* Main content area — full width, no sidebar */
.main-area {
    padding: 20px 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    max-width: 1400px;
    margin: 0 auto;
    width: 100%;
}

/* Chat input — inline */
.chat-input-area {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 16px;
    display: flex;
    gap: 12px;
}

.chat-input-area input {
    flex: 1;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 14px;
    outline: none;
    transition: border-color 0.2s;
    max-width: 1400px;
}

.chat-input-area input:focus { border-color: var(--primary); }

.chat-input-area button {
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
}

.chat-input-area button:hover { background: var(--primary-dark); }
.chat-input-area button:disabled { opacity: 0.5; cursor: not-allowed; }

/* User message bubble */
.user-msg {
    align-self: flex-end;
    background: var(--primary);
    color: white;
    padding: 10px 16px;
    border-radius: 16px 16px 4px 16px;
    max-width: 70%;
    font-size: 14px;
}

/* Metrics row — dynamic panels */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}

/* Panel styling (used inside metrics-row) */
.panel {
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 14px;
}

.panel h4 {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

.panel-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0;
    font-size: 12px;
}

.panel-row .label { color: var(--text-secondary); }
.panel-row .value { font-weight: 600; font-variant-numeric: tabular-nums; }

.panel-model-label {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 8px;
    margin-bottom: 4px;
    padding-top: 6px;
    border-top: 1px solid var(--border);
    color: var(--primary-dark);
}

/* Model response cards — dynamic grid */
.response-row {
    display: grid;
    gap: 16px;
}

.model-card {
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    overflow: hidden;
    display: flex;
    flex-direction: column;
}

.model-card-header {
    padding: 10px 16px;
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #eff6ff;
    color: #1e40af;
}

.model-card-body { padding: 14px 16px; font-size: 14px; line-height: 1.6; background: var(--primary); color: white; }

/* Confidence row — dynamic columns */
.confidence-row {
    display: grid;
    gap: 16px;
}

/* Per-token confidence — horizontal rows */
.confidence-section {
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 14px;
}

.confidence-section h3 {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
}

.token-rows {
    display: flex;
    flex-direction: column;
    gap: 3px;
}

.token-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    line-height: 1.3;
}

.token-row .piece-text {
    width: 90px;
    min-width: 90px;
    text-align: right;
    font-weight: 500;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.token-row .bar-track {
    flex: 1;
    height: 16px;
    background: #f1f5f9;
    border-radius: 4px;
    overflow: hidden;
}

.token-row .bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}

.token-row .pct-text {
    width: 40px;
    min-width: 40px;
    font-weight: 600;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
    color: var(--text);
}

.conf-very-low { background: #ef4444; }
.conf-low { background: #f59e0b; }
.conf-mid { background: #3b82f6; }
.conf-high { background: #22c55e; }

/* Confidence histogram bars */
.hist-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
    font-size: 11px;
}

.hist-label { width: 60px; text-align: right; color: var(--text-secondary); font-size: 10px; }

.hist-bar-bg {
    flex: 1;
    height: 14px;
    background: #f1f5f9;
    border-radius: 3px;
    overflow: hidden;
}

.hist-bar {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s;
}

.hist-count { width: 20px; font-size: 10px; font-weight: 600; color: var(--text-secondary); }

/* BLEU scores */
.bleu-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 6px;
    margin-top: 4px;
}

.bleu-item {
    text-align: center;
    background: var(--bg);
    border-radius: 6px;
    padding: 6px 2px;
}

.bleu-item .bleu-n { font-size: 9px; color: var(--text-secondary); font-weight: 600; }
.bleu-item .bleu-val { font-size: 14px; font-weight: 700; }

/* Attention heatmap */
.heatmap-section {
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 16px;
}

.heatmap-section h3 {
    font-size: 16px;
    font-weight: 700;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
}

.heatmap-container {
    overflow-x: auto;
    overflow-y: auto;
    max-height: 500px;
}

.heatmap-table { border-collapse: collapse; }

.heatmap-table td {
    width: 36px;
    height: 30px;
    text-align: center;
    font-size: 0;
    border: 1px solid rgba(255,255,255,0.4);
}

.heatmap-table th {
    font-size: 13px;
    padding: 4px 6px;
    color: var(--text-secondary);
    font-weight: 600;
    white-space: nowrap;
    max-width: 70px;
    overflow: hidden;
    text-overflow: ellipsis;
}

.heatmap-table th.row-header {
    text-align: right;
    padding-right: 8px;
    font-size: 13px;
}

/* No attention placeholder */
.no-attn-msg {
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 24px;
    text-align: center;
    color: var(--text-secondary);
    font-size: 13px;
    font-style: italic;
}

/* Loading spinner */
.spinner {
    display: inline-block;
    width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,0.3);
    border-radius: 50%;
    border-top-color: white;
    animation: spin 0.6s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

/* Model warning */
.model-warning {
    background: #fef3c7;
    border: 1px solid #f59e0b;
    color: #92400e;
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 13px;
    margin-bottom: 8px;
}

/* Tooltip */
.tooltip {
    position: absolute;
    background: #1e293b;
    color: white;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    pointer-events: none;
    z-index: 100;
    white-space: nowrap;
}

/* Config info compact */
.config-row {
    display: flex;
    justify-content: space-between;
    padding: 2px 0;
    font-size: 11px;
}

.config-row .label { color: var(--text-secondary); }
.config-row .value { font-weight: 600; font-size: 11px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* Responsive */
@media (max-width: 900px) {
    .metrics-row { grid-template-columns: 1fr 1fr; }
    .response-row { grid-template-columns: 1fr !important; }
    .confidence-row { grid-template-columns: 1fr !important; }
    .token-row .piece-text { width: 80px; min-width: 80px; }
}

@media (max-width: 600px) {
    .metrics-row { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<div class="header">
    <h1>
        Ubuntu Seq2Seq Chatbot v2.0
        <span class="badge" id="header-badge">Loading...</span>
    </h1>
    <div class="header-controls">
        <div class="model-selector" id="model-selector"></div>
        <select id="decode-mode">
            <option value="greedy">Greedy</option>
            <option value="topp">Top-p Sampling</option>
        </select>
        <span>
            <span class="status-dot" id="status-dot"></span>
            <span class="status-text" id="status-text">Loading...</span>
        </span>
    </div>
</div>

<div class="main-area" id="main-area">
    <div id="model-warning" class="model-warning" style="display:none;"></div>
    <div id="top-results"></div>
    <div id="user-msg-area"></div>

    <div class="chat-input-area">
        <input type="text" id="chat-input" placeholder="Type your message..." autocomplete="off" />
        <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>

    <div id="bottom-results"></div>
</div>

<script>
const topResults = document.getElementById('top-results');
const userMsgArea = document.getElementById('user-msg-area');
const bottomResults = document.getElementById('bottom-results');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
let history = [];
let lastResults = null;
let allModels = {};       // full model info from /api/models
let selectedModelIds = []; // currently checked model IDs

// Model color palette for dynamic assignment
const MODEL_COLORS = [
    {bg: '#eff6ff', text: '#1e40af'},
    {bg: '#f5f3ff', text: '#5b21b6'},
    {bg: '#f0fdf4', text: '#166534'},
    {bg: '#fef3c7', text: '#92400e'},
    {bg: '#fdf2f8', text: '#9d174d'},
    {bg: '#ecfeff', text: '#155e75'},
];

function getModelColor(index) {
    return MODEL_COLORS[index % MODEL_COLORS.length];
}

// Map model ID to a stable color index
let modelColorMap = {};

// Enter key
chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

function getSelectedModels() {
    const checkboxes = document.querySelectorAll('#model-selector input[type="checkbox"]');
    const selected = [];
    checkboxes.forEach(cb => { if (cb.checked) selected.push(cb.value); });
    return selected;
}

function updateBadge() {
    const badge = document.getElementById('header-badge');
    const selected = getSelectedModels();
    if (selected.length === 0) {
        badge.textContent = 'No models selected';
    } else if (selected.length <= 2) {
        const names = selected.map(id => allModels[id] ? allModels[id].name : id);
        badge.textContent = names.join(' + ');
    } else {
        badge.textContent = selected.length + ' models selected';
    }
    selectedModelIds = selected;
}

function buildModelSelector(models) {
    const container = document.getElementById('model-selector');
    container.innerHTML = '';
    let idx = 0;
    for (const [mid, info] of Object.entries(models)) {
        modelColorMap[mid] = idx;
        const label = document.createElement('label');
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.value = mid;
        cb.checked = true;
        cb.addEventListener('change', updateBadge);
        label.appendChild(cb);
        label.appendChild(document.createTextNode(' ' + info.name));
        container.appendChild(label);
        idx++;
    }
    updateBadge();
}

// Check health on load
fetch('/health').then(r => r.json()).then(data => {
    const dot = document.getElementById('status-dot');
    const txt = document.getElementById('status-text');
    if (data.models_loaded && data.models_loaded.length > 0) {
        dot.style.background = '#22c55e';
        const names = data.models_loaded.map(m => m.name);
        txt.textContent = names.join(' + ') + ' on ' + data.device;
    } else {
        dot.style.background = '#ef4444';
        txt.textContent = 'No models loaded';
    }

    if (data.models_loaded.length === 0) {
        const w = document.getElementById('model-warning');
        w.style.display = 'block';
        w.textContent = 'Warning: No models are loaded. Please ensure checkpoints exist in the checkpoints/ directory.';
    }
});

// Load model config
fetch('/api/models').then(r => r.json()).then(data => {
    allModels = data.models;
    buildModelSelector(allModels);
});

function gridCols(n) {
    if (n <= 1) return '1fr';
    if (n === 2) return '1fr 1fr';
    if (n === 3) return '1fr 1fr 1fr';
    return '1fr 1fr'; // 4+ -> 2 columns, wrapping
}

function sendMessage() {
    const msg = chatInput.value.trim();
    if (!msg) return;
    chatInput.value = '';
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<span class="spinner"></span>';

    // Clear previous results
    topResults.innerHTML = '';
    userMsgArea.innerHTML = '';
    bottomResults.innerHTML = '';

    const mode = document.getElementById('decode-mode').value;
    const selected = getSelectedModels();

    fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ message: msg, history: history, mode: mode, selected_models: selected })
    })
    .then(r => r.json())
    .then(data => {
        if (data.error) {
            addBotError(data.error);
            return;
        }
        lastResults = data;
        history.push(msg);

        const modelIds = data.models_used;
        const results = data.results;
        const numModels = modelIds.length;
        const cols = gridCols(numModels);

        // Metrics row (4 panels) — top area
        renderMetricsRow(modelIds, results, topResults);

        // User message bubble — just above input
        const userDiv = document.createElement('div');
        userDiv.className = 'user-msg';
        userDiv.textContent = msg;
        userMsgArea.appendChild(userDiv);

        // Per-model vertical cards (below input)
        const row = document.createElement('div');
        row.className = 'response-row';
        row.style.gridTemplateColumns = cols;

        for (const mid of modelIds) {
            const r = results[mid];
            const colorIdx = modelColorMap[mid] !== undefined ? modelColorMap[mid] : 0;
            const color = getModelColor(colorIdx);
            const modelName = r.meta.name;

            // Build a full vertical card: response + confidence + heatmap
            const card = document.createElement('div');
            card.className = 'model-card';
            card.style.display = 'flex';
            card.style.flexDirection = 'column';
            card.style.gap = '0';

            // Response header + body
            const header = document.createElement('div');
            header.className = 'model-card-header';
            header.style.background = color.bg;
            header.style.color = color.text;
            header.innerHTML = `<span>${escapeHtml(modelName)}</span>
                <span style="font-size:11px;font-weight:400;opacity:0.8;">${r.metrics.latency_ms}ms</span>`;
            card.appendChild(header);

            const body = document.createElement('div');
            body.className = 'model-card-body';
            body.textContent = r.response_text;
            card.appendChild(body);

            // Per-token confidence bars inside the card
            const confSection = document.createElement('div');
            confSection.style.padding = '14px';
            confSection.style.borderTop = '1px solid var(--border)';
            const confTitle = document.createElement('h3');
            confTitle.style.cssText = 'font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:10px;color:' + color.text;
            confTitle.textContent = 'Per-Token Confidence';
            confSection.appendChild(confTitle);

            const rowsDiv = document.createElement('div');
            rowsDiv.className = 'token-rows';
            for (let i = 0; i < r.pieces.length; i++) {
                const trow = document.createElement('div');
                trow.className = 'token-row';

                const pieceSpan = document.createElement('span');
                pieceSpan.className = 'piece-text';
                pieceSpan.textContent = r.pieces[i].replace(/^▁/, '_');

                const track = document.createElement('div');
                track.className = 'bar-track';
                const fill = document.createElement('div');
                fill.className = 'bar-fill ' + confClass(r.confidences[i]);
                fill.style.width = (r.confidences[i] * 100).toFixed(1) + '%';
                track.appendChild(fill);

                const pctSpan = document.createElement('span');
                pctSpan.className = 'pct-text';
                pctSpan.textContent = (r.confidences[i] * 100).toFixed(0) + '%';

                trow.appendChild(pieceSpan);
                trow.appendChild(track);
                trow.appendChild(pctSpan);
                rowsDiv.appendChild(trow);
            }
            confSection.appendChild(rowsDiv);
            card.appendChild(confSection);

            // Attention heatmap or "no attention" message
            if (r.attn_weights && r.attn_weights.length > 0) {
                const heatSection = document.createElement('div');
                heatSection.style.padding = '16px';
                heatSection.style.borderTop = '1px solid var(--border)';
                const heatTitle = document.createElement('h3');
                heatTitle.style.cssText = 'font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:12px;color:' + color.text;
                heatTitle.textContent = 'Attention Heatmap';
                heatSection.appendChild(heatTitle);

                const container = document.createElement('div');
                container.className = 'heatmap-container';
                const srcPieces = r.src_pieces;
                const attn = r.attn_weights;
                const respPieces = r.pieces;

                const table = document.createElement('table');
                table.className = 'heatmap-table';

                let headerRow = '<tr><th></th>';
                for (const sp of srcPieces) {
                    headerRow += '<th>' + escapeHtml(sp.replace(/^▁/, '_')) + '</th>';
                }
                headerRow += '</tr>';
                table.innerHTML = headerRow;

                for (let ri = 0; ri < attn.length && ri < respPieces.length; ri++) {
                    const tr = document.createElement('tr');
                    const th = document.createElement('th');
                    th.className = 'row-header';
                    th.textContent = respPieces[ri].replace(/^▁/, '_');
                    tr.appendChild(th);

                    for (let c = 0; c < attn[ri].length; c++) {
                        const td = document.createElement('td');
                        const w = attn[ri][c];
                        const intensity = Math.min(w * 3, 1);
                        const red = Math.round(240 - 200 * intensity);
                        const green = Math.round(245 - 175 * intensity);
                        const blue = Math.round(250 - 30 * intensity);
                        td.style.background = 'rgb(' + red + ',' + green + ',' + blue + ')';
                        td.title = srcPieces[c] + ' -> ' + respPieces[ri] + ': ' + (w*100).toFixed(1) + '%';
                        tr.appendChild(td);
                    }
                    table.appendChild(tr);
                }

                container.appendChild(table);
                heatSection.appendChild(container);
                card.appendChild(heatSection);
            } else {
                const noAttn = document.createElement('div');
                noAttn.style.cssText = 'padding:24px;text-align:center;color:var(--text-secondary);font-size:13px;font-style:italic;border-top:1px solid var(--border);';
                noAttn.textContent = 'No attention data — this model does not use attention';
                card.appendChild(noAttn);
            }

            row.appendChild(card);
        }
        bottomResults.appendChild(row);

        // Add first model response to history
        if (modelIds.length > 0) {
            history.push(results[modelIds[0]].response_text);
        }
    })
    .catch(err => {
        addBotError('Request failed: ' + err.message);
    })
    .finally(() => {
        sendBtn.disabled = false;
        sendBtn.textContent = 'Send';
    });
}

function addBotError(msg) {
    const div = document.createElement('div');
    div.style.cssText = 'background:#fef2f2;color:#991b1b;padding:10px 16px;border-radius:8px;font-size:13px;';
    div.textContent = msg;
    bottomResults.appendChild(div);
}

function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function confClass(c) {
    if (c < 0.4) return 'conf-very-low';
    if (c < 0.6) return 'conf-low';
    if (c < 0.8) return 'conf-mid';
    return 'conf-high';
}

function confColor(c) {
    if (c < 0.4) return '#ef4444';
    if (c < 0.6) return '#f59e0b';
    if (c < 0.8) return '#3b82f6';
    return '#22c55e';
}

function renderMetricsRow(modelIds, results, target) {
    const container = document.createElement('div');
    container.className = 'metrics-row';

    // 1) Inference Metrics panel
    const metricsPanel = document.createElement('div');
    metricsPanel.className = 'panel';
    let mhtml = '<h4>Inference Metrics</h4>';
    for (const mid of modelIds) {
        const m = results[mid].metrics;
        const name = results[mid].meta.name;
        mhtml += '<div class="panel-model-label">' + escapeHtml(name) + '</div>';
        mhtml += metricRow('Tokens/sec', m.tokens_per_sec);
        mhtml += metricRow('Latency', m.latency_ms + ' ms');
        mhtml += metricRow('Avg Confidence', (m.avg_confidence * 100).toFixed(1) + '%');
        mhtml += metricRow('UNK in Input', m.unk_in_input);
        mhtml += metricRow('Response Len', m.response_len);
        mhtml += metricRow('Input Tokens', m.input_tokens);
    }
    metricsPanel.innerHTML = mhtml;
    container.appendChild(metricsPanel);

    // 2) Confidence Analysis panel
    const confPanel = document.createElement('div');
    confPanel.className = 'panel';
    let chtml = '<h4>Confidence Analysis</h4>';
    for (const mid of modelIds) {
        const ca = results[mid].confidence_analysis;
        const total = ca.very_low + ca.low + ca.mid + ca.high;
        const name = results[mid].meta.name;
        chtml += '<div class="panel-model-label">' + escapeHtml(name) + '</div>';
        chtml += metricRow('Perplexity', ca.perplexity);
        chtml += '<div style="margin-top:6px;">';
        chtml += histRow('Very Low <0.4', ca.very_low, total, '#ef4444');
        chtml += histRow('Low 0.4-0.6', ca.low, total, '#f59e0b');
        chtml += histRow('Mid 0.6-0.8', ca.mid, total, '#3b82f6');
        chtml += histRow('High >=0.8', ca.high, total, '#22c55e');
        chtml += '</div>';
    }
    confPanel.innerHTML = chtml;
    container.appendChild(confPanel);

    // 3) BLEU Scores panel
    const bleuPanel = document.createElement('div');
    bleuPanel.className = 'panel';
    let bhtml = '<h4>BLEU Scores (vs Input)</h4>';
    for (const mid of modelIds) {
        const bleu = results[mid].bleu;
        const name = results[mid].meta.name;
        bhtml += '<div class="panel-model-label">' + escapeHtml(name) + '</div>';
        bhtml += '<div class="bleu-grid">';
        for (let n = 1; n <= 4; n++) {
            const val = bleu['bleu' + n];
            const color = val >= 0.3 ? '#22c55e' : val >= 0.1 ? '#f59e0b' : '#ef4444';
            bhtml += '<div class="bleu-item">' +
                '<div class="bleu-n">BLEU-' + n + '</div>' +
                '<div class="bleu-val" style="color:' + color + '">' + (val * 100).toFixed(1) + '</div>' +
            '</div>';
        }
        bhtml += '</div>';
    }
    bleuPanel.innerHTML = bhtml;
    container.appendChild(bleuPanel);

    // 4) Model Config panel
    const cfgPanel = document.createElement('div');
    cfgPanel.className = 'panel';
    let cfghtml = '<h4>Model Config</h4>';
    for (const mid of modelIds) {
        const cfg = results[mid].config_info;
        const name = results[mid].meta.name;
        cfghtml += '<div class="panel-model-label">' + escapeHtml(name) + '</div>';
        cfghtml += configRow('Type', cfg.model_type);
        cfghtml += configRow('Encoder', cfg.encoder);
        cfghtml += configRow('Decoder', cfg.decoder);
        cfghtml += configRow('Params', cfg.total_params);
        cfghtml += configRow('Decoding', cfg.decoding);
        cfghtml += configRow('Hidden', cfg.hidden_dim);
        cfghtml += configRow('Embed', cfg.embed_dim);
        cfghtml += configRow('Vocab', Number(cfg.vocab_size).toLocaleString());
    }
    cfgPanel.innerHTML = cfghtml;
    container.appendChild(cfgPanel);

    target.appendChild(container);
}

function metricRow(label, value) {
    return '<div class="panel-row"><span class="label">' + label + '</span><span class="value">' + value + '</span></div>';
}

function histRow(label, count, total, color) {
    const pct = total > 0 ? (count / total * 100) : 0;
    return '<div class="hist-row">' +
        '<span class="hist-label">' + label + '</span>' +
        '<div class="hist-bar-bg"><div class="hist-bar" style="width:' + pct + '%;background:' + color + ';"></div></div>' +
        '<span class="hist-count">' + count + '</span>' +
    '</div>';
}

function configRow(label, value) {
    return '<div class="config-row"><span class="label">' + label + '</span><span class="value">' + value + '</span></div>';
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _load_models()
    app.run(host="0.0.0.0", port=5000, debug=False)
