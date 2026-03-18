# Ubuntu Seq2Seq Chatbot v4.0

A sequence-to-sequence chatbot trained on the Ubuntu Dialogue Corpus. Supports baseline (no attention) and Bahdanau attention models. Includes a web GUI for side-by-side model comparison with per-token confidence, attention heatmaps, and inference metrics.

## Quick Start (Pre-trained)

The chatbot comes with pre-trained checkpoints and artifacts. No training needed to start chatting.

1. Install Python 3.9+
2. Install PyTorch -- see <https://pytorch.org/get-started> for GPU-specific instructions
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. **Terminal chat** (baseline + attention side by side):
   ```
   python chatv2.py
   ```
5. **Web GUI** (browser-based, model registry, attention heatmaps):
   ```
   python web_chat.py
   ```
   Then open <http://localhost:5000>

## Project Structure

| File | Purpose |
|------|---------|
| `phase1.py` | Data pipeline (8 stages: load CSV, clean, split, pairs, filter, BPE, encode, embeddings) |
| `train.py` | Training loop (baseline + attention, supports multi-GPU via DataParallel) |
| `evaluate.py` | Full evaluation (BLEU, ROUGE, BERTScore, attention heatmaps) |
| `evaluate_mini.py` | Lightweight 3-layer evaluation |
| `chatv2.py` | CLI dual-model chat (baseline + attention side by side) |
| `web_chat.py` | Flask web GUI with model registry, dynamic model comparison |
| `config.py` | Central configuration (hyperparameters, paths, special token IDs) |
| `models.py` | Model definitions (Encoder, Decoder, Bridge, Attention) |
| `dataset.py` | PyTorch Dataset/DataLoader for encoded pairs |
| `gpu_utils.py` | Auto GPU detection, A100 batch scaling, multi-GPU wrapping |
| `tokenizer_utils.py` | Unified tokenizer interface (HF Tokenizers + SentencePiece) |
| `logging_utils.py` | Timestamped run logging |
| `model_registry.json` | Model registry for web GUI (add new models here) |
| `requirements.txt` | Python dependencies |
| `analyze_data.py` | Standalone data quality analysis tool |
| `check_embedding.py` | Standalone embedding validation tool |

## Training From Scratch

If you want to retrain from scratch:

1. Clear the `artifacts/` and `checkpoints/` folders
2. Place your dataset CSV in `data/Ubuntu-dialogue-corpus/`
3. Run the data pipeline (~15-30 min):
   ```
   python phase1.py
   ```
4. Run training (~2-4 hours on A100, longer on CPU):
   ```
   python train.py
   ```
5. Run evaluation:
   ```
   python evaluate.py
   ```

## Adding New Models

The web GUI uses `model_registry.json` to discover models. To add a new model:

1. Train your model (different architecture, dataset, hyperparameters, etc.)
2. Add an entry to `model_registry.json` with the following fields:
   - `id` -- unique identifier
   - `name` -- display name
   - `architecture` -- model type (e.g., `seq2seq_attention`)
   - `checkpoint` -- path to the `.pt` checkpoint file
   - `tokenizer` -- path to the tokenizer file
   - `dataset` -- dataset name
   - `description` -- short description
3. The web GUI automatically picks up new entries -- no code changes needed.

## OpenShift AI / GPU Cluster

- The code auto-detects GPUs and scales batch size accordingly
- For A100 80GB: batch size auto-scales to 512
- Multi-GPU supported via DataParallel
- Set up Kaggle API credentials for dataset download on the cluster

## Requirements

- Python 3.9+
- PyTorch 2.1+ (CUDA recommended for training)
- See `requirements.txt` for the full dependency list

## Hardware

- **Training**: NVIDIA GPU recommended (A100 ideal, RTX 3090+ works)
- **Inference / chat**: CPU works fine (~100-300ms per response)
