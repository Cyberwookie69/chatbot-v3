"""
models.py — Seq2Seq architecture (clean from scratch).

Version : 4.0.1
Modified: 2026-03-19
Changes : v4.0.1 — Clamp logits to [-50, 50] to prevent bf16 overflow in autoregressive decoding
          v4.0.0 — Version bump for multi-corpus project
          v3.2.0 — Bahdanau attention, Bridge module, shared embeddings
          v3.0.0 — Multi-GPU DataParallel compatibility
          v2.0.0 — Initial clean-from-scratch rewrite

Components:
  1. create_pretrained_embedding  — loads Phase 1 Stage 8 .npy matrix
  2. Encoder                      — BIDIRECTIONAL 2-layer LSTM (key upgrade)
  3. EncoderDecoderBridge         — projects bidir h_n/c_n to decoder shape
  4. BahdanauAttention            — additive attention with key precomputation
  5. AttentionDecoder             — with projection bottleneck
  6. BaselineDecoder              — parameter-fair baseline (no attention)
  7. Seq2Seq                      — wraps encoder + bridge + decoder

Key architectural decisions vs old models.py:
  - Encoder is now BIDIRECTIONAL: forward + backward pass over context.
    Final hidden state = [forward_layer_n ; backward_layer_n] bridged to decoder.
    This is the single biggest quality upgrade over the prior codebase.
  - hidden_dim=512 per direction → 1024 total effective encoder output.
  - EncoderDecoderBridge merges the 4 interleaved bidir hidden states into
    2 decoder-sized states via a learned linear projection.
  - Shared embedding (weight tying): encoder and decoder share one nn.Embedding.
    Justified: single vocabulary, reduces params, regularises.
  - Projection bottleneck: [hidden+context](2048) → proj(512) → vocab(16k).
    More important now — without it the output matrix would be 2048×16k = 32M.
  - BahdanauAttention: keys_proj precomputed once by the decoder and passed
    explicitly per step, avoiding recomputing W_enc(encoder_outputs) trg_len-1
    times. Explicit passing is safe under DataParallel / gradient checkpointing.
  - No import inside forward(). All imports at module top.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from typing import Tuple, Optional


# ── Embedding utility ────────────────────────────────────────────────────────

def create_pretrained_embedding(
    matrix_path: str,
    pad_idx: int = 0,
    freeze: bool = False,
) -> nn.Embedding:
    """
    Load Phase 1 Stage 8 embedding matrix (.npy) into nn.Embedding.

    Args:
        matrix_path: Path to .npy file of shape [vocab_size, embed_dim].
        pad_idx:     Index of the <pad> token; its embedding row is zeroed.
        freeze:      If True, embeddings are fixed (requires_grad=False).
                     Default False: fine-tune during training so BPE pieces
                     adapt from general FastText init to seq2seq task.

    Returns:
        nn.Embedding with pretrained weights and padding_idx set.
    """
    matrix = np.load(matrix_path)                          # [vocab_size, embed_dim]
    vocab_size, embed_dim = matrix.shape

    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embed_dim,
        padding_idx=pad_idx,
    )

    # Load pretrained weights; wrap in nn.Parameter to control grad flag.
    embedding.weight = nn.Parameter(
        torch.FloatTensor(matrix),
        requires_grad=(not freeze),
    )

    # Guarantee the pad row is zero and stays zero (no gradient).
    embedding.weight.data[pad_idx].zero_()

    return embedding


# ── Encoder ──────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Bidirectional 2-layer LSTM encoder.

    Upgrade from old codebase: bidirectional=True.
    For a context like "how do I mount a USB drive", the forward pass
    sees 'mount' before 'USB drive'; the backward pass sees 'mount'
    already knowing 'USB drive' follows. The combined representation
    is richer for downstream attention alignment.

    Output shapes:
      encoder_outputs: [batch, src_len, hidden_dim*2]  — all timestep states
      (h_n, c_n):      each [num_layers*2, batch, hidden_dim]
                       PyTorch interleaves fwd/bwd: [fwd_L0, bwd_L0, fwd_L1, bwd_L1]

    The caller passes (h_n, c_n) to EncoderDecoderBridge before the decoder.
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        hidden_dim: int = 512,        # per direction; effective output = hidden_dim*2 = 1024
        num_layers: int = 2,
        dropout_embed: float = 0.3,   # light: preserve pretrained FastText signal
        dropout_lstm: float = 0.5,    # aggressive between LSTM layers
        dropout_out: float = 0.4,     # regularise encoder outputs seen by attention
    ):
        super().__init__()

        self.embedding = embedding
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        embed_dim = embedding.embedding_dim

        self.embed_dropout = nn.Dropout(dropout_embed)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,       # KEY UPGRADE: fwd + bwd pass
            # PyTorch applies dropout between LSTM layers (not within a layer).
            # With num_layers=2 this fires once, between layer 1 and layer 2.
            dropout=dropout_lstm if num_layers > 1 else 0.0,
        )

        self.output_dropout = nn.Dropout(dropout_out)

    def forward(
        self,
        src: torch.LongTensor,
        src_lengths: torch.LongTensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src:         [batch, src_len]  padded token IDs
            src_lengths: [batch]           actual (unpadded) sequence lengths

        Returns:
            outputs: [batch, src_len, hidden_dim*2]  — all encoder hidden states
            (h_n, c_n): each [num_layers*2, batch, hidden_dim], bridgeable to decoder
        """
        # Guard: zero-length sequences cause NaN in pack_padded_sequence.
        assert (src_lengths > 0).all(), \
            f"Zero-length sequence in batch: {src_lengths.tolist()}"

        # Step 1: embed tokens and apply embedding dropout.
        embedded = self.embed_dropout(self.embedding(src))   # [batch, src_len, embed_dim]

        # Step 2: pack for efficient LSTM computation over variable-length seqs.
        # enforce_sorted=False lets PyTorch sort internally (no manual sort needed).
        # .cpu() required by pack_padded_sequence — lengths must live on CPU (M4).
        packed = pack_padded_sequence(
            embedded,
            src_lengths.cpu(),        # M4: explicit .cpu() to avoid device mismatch
            batch_first=True,
            enforce_sorted=False,
        )

        # Step 3: bidirectional LSTM forward.
        packed_outputs, (h_n, c_n) = self.lstm(packed)

        # Step 4: unpack back to padded tensor.
        # CRITICAL (B1): total_length forces output to match src_len exactly.
        # Without it, pad_packed_sequence returns max_actual_len which can be
        # shorter than src.size(1), causing a shape mismatch with src_mask later.
        outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=src.size(1),  # B1: pin output length to padded src_len
        )

        # Step 5: apply output dropout before attention consumes encoder states.
        outputs = self.output_dropout(outputs)   # [batch, src_len, hidden_dim*2]

        return outputs, (h_n, c_n)


# ── Encoder→Decoder hidden state bridge ─────────────────────────────────────

class EncoderDecoderBridge(nn.Module):
    """
    Projects bidirectional encoder final states to decoder initial states.

    PyTorch bidirectional LSTM stores h_n with an interleaved layout:
      h_n[0] = layer 0 forward
      h_n[1] = layer 0 backward
      h_n[2] = layer 1 forward
      h_n[3] = layer 1 backward
      ...

    So for num_layers=2, shape is [4, batch, enc_hidden_dim].

    Correct extraction (B4):
      h_fwd = h_n[0::2]   → indices [0, 2] → layers [0_fwd, 1_fwd]  shape [num_layers, batch, enc_hidden_dim]
      h_bwd = h_n[1::2]   → indices [1, 3] → layers [0_bwd, 1_bwd]  shape [num_layers, batch, enc_hidden_dim]

    After concat per layer:  [num_layers, batch, enc_hidden_dim*2]
    After linear projection: [num_layers, batch, dec_hidden_dim]

    The linear projection is a learned transformation, not a fixed reshape,
    allowing the bridge to emphasise whichever directional signal is more
    useful for initialising each decoder layer.
    """

    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int, num_layers: int):
        """
        Args:
            enc_hidden_dim: Hidden dim per direction (512). Input to bridge = enc_hidden_dim*2.
            dec_hidden_dim: Decoder hidden dim (1024 = enc_hidden_dim*2).
            num_layers:     Number of LSTM layers shared by encoder and decoder.
        """
        super().__init__()

        self.num_layers = num_layers

        # Separate projections for hidden state and cell state.
        # Input: concatenated fwd+bwd = enc_hidden_dim * 2
        self.h_projection = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)
        self.c_projection = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)

    def _merge_bidir(self, states: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        """
        Merge interleaved bidirectional states using the given linear projection.

        Args:
            states: [num_layers*2, batch, enc_hidden_dim]  interleaved fwd/bwd
            proj:   Linear(enc_hidden_dim*2, dec_hidden_dim)

        Returns:
            [num_layers, batch, dec_hidden_dim]
        """
        # B4: PyTorch interleaves fwd and bwd layers — even indices are forward,
        # odd indices are backward.  Slice with step 2 to separate them cleanly.
        fwd = states[0::2]   # [num_layers, batch, enc_hidden_dim]  forward layers
        bwd = states[1::2]   # [num_layers, batch, enc_hidden_dim]  backward layers

        # Concatenate along the hidden dimension, then project.
        merged = torch.cat([fwd, bwd], dim=2)   # [num_layers, batch, enc_hidden_dim*2]
        projected = torch.tanh(proj(merged))    # [num_layers, batch, dec_hidden_dim]
        return projected

    def forward(
        self,
        h_n: torch.Tensor,
        c_n: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_n: [num_layers*2, batch, enc_hidden_dim]  encoder final hidden states
            c_n: [num_layers*2, batch, enc_hidden_dim]  encoder final cell states

        Returns:
            h0: [num_layers, batch, dec_hidden_dim]  decoder initial hidden
            c0: [num_layers, batch, dec_hidden_dim]  decoder initial cell
        """
        h0 = self._merge_bidir(h_n, self.h_projection)   # [num_layers, batch, dec_hidden_dim]
        c0 = self._merge_bidir(c_n, self.c_projection)   # [num_layers, batch, dec_hidden_dim]
        return h0, c0


# ── Bahdanau Attention ───────────────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    """
    Additive (Bahdanau) attention mechanism.

    Energy at step t:
        e_{t,i} = v^T · tanh( W_enc · h_enc_i  +  W_dec · s_t )

    Attention weights:
        α_{t,i} = softmax( e_{t,:} )   with padding positions masked to -inf

    Context vector:
        c_t = Σ_i  α_{t,i} · h_enc_i

    Optimisation: W_enc(encoder_outputs) is constant across all decoder steps.
    The caller (AttentionDecoder.forward) computes keys_proj once before the
    decode loop and passes it explicitly to each forward() call, saving
    (trg_len-1) redundant matrix multiplications over the full source sequence.
    Explicit passing (rather than caching on self) is safe under DataParallel
    and gradient checkpointing, avoiding mutable-state concurrency bugs.
    """

    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 256):
        """
        Args:
            enc_dim:  Encoder output dim (1024 for bidirectional 512-per-dir).
            dec_dim:  Decoder hidden dim (1024).
            attn_dim: Shared attention space dimensionality (256).
        """
        super().__init__()

        self.W_enc = nn.Linear(enc_dim,  attn_dim, bias=False)  # projects encoder states
        self.W_dec = nn.Linear(dec_dim,  attn_dim, bias=False)  # projects decoder hidden
        self.v     = nn.Linear(attn_dim, 1,        bias=False)  # scalar energy score

    def forward(
        self,
        encoder_outputs: torch.Tensor,           # [batch, src_len, enc_dim]
        decoder_hidden: torch.Tensor,            # [batch, dec_dim]
        src_mask: Optional[torch.Tensor] = None, # [batch, src_len] bool, True=padding
        keys_proj: Optional[torch.Tensor] = None,# [batch, src_len, attn_dim] precomputed
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention context for a single decoder step.

        Args:
            encoder_outputs: [batch, src_len, enc_dim]
            decoder_hidden:  [batch, dec_dim]   top-layer decoder hidden state
            src_mask:        [batch, src_len]   True at <pad> positions
            keys_proj:       [batch, src_len, attn_dim]  W_enc(encoder_outputs),
                             precomputed by the caller to avoid redundant work
                             across the decode loop. If None, computed here.

        Returns:
            context:      [batch, enc_dim]   attention-weighted encoder states
            attn_weights: [batch, src_len]   normalised attention distribution
        """
        # Use caller-supplied precomputed projection or compute on the fly.
        # Passing keys_proj explicitly (rather than caching on self) avoids
        # mutable-state bugs under DataParallel / gradient checkpointing.
        keys = keys_proj if keys_proj is not None else self.W_enc(encoder_outputs)

        # Project decoder hidden state and broadcast over src_len.
        query = self.W_dec(decoder_hidden).unsqueeze(1)  # [batch, 1, attn_dim]

        # Additive energy: tanh(key + query) then scalar projection.
        energy = self.v(torch.tanh(keys + query)).squeeze(2)  # [batch, src_len]

        # Mask padding positions with -inf so softmax assigns them zero weight.
        if src_mask is not None:
            energy = energy.masked_fill(src_mask, float("-inf"))

        attn_weights = F.softmax(energy, dim=1)          # [batch, src_len]

        # Weighted sum of encoder states to form context vector.
        context = torch.bmm(
            attn_weights.unsqueeze(1),   # [batch, 1, src_len]
            encoder_outputs,             # [batch, src_len, enc_dim]
        ).squeeze(1)                     # [batch, enc_dim]

        return context, attn_weights


# ── Attention Decoder ────────────────────────────────────────────────────────

class AttentionDecoder(nn.Module):
    """
    Decoder with Bahdanau attention and projection bottleneck.

    At each decode step t:
      1. Embed input token:                    embedded [batch, 1, embed_dim]
      2. Concat [embedded ; context_prev]:     LSTM input [batch, 1, embed_dim+enc_hidden_dim]
      3. LSTM step:                            output [batch, 1, dec_hidden_dim]
      4. Attend with new top-layer hidden:     context_new [batch, enc_hidden_dim]
      5. Concat [output ; context_new]:        [batch, dec_hidden_dim+enc_hidden_dim]
      6. Project through bottleneck:           [batch, projection_dim]
      7. Linear to vocab:                      logits [batch, vocab_size]

    CRITICAL (B6): LSTM input_size = embed_dim + enc_hidden_dim = 300 + 1024 = 1324.
    The context vector has the FULL bidirectional encoder dimension (1024), not
    the per-direction dimension (512).  Initial context is zeros(batch, 1024).

    Projection bottleneck (dec_hidden_dim + enc_hidden_dim → projection_dim → vocab_size):
      Without: 2048 × 16000 = 32.8M params in a single layer
      With:    2048 × 512 + 512 × 16000 = 9.2M params  (72% reduction)
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        enc_hidden_dim: int,      # FULL bidir encoder dim = 1024
        dec_hidden_dim: int,      # decoder LSTM hidden dim = 1024
        vocab_size: int,
        projection_dim: int = 512,
        num_layers: int = 2,
        dropout_embed: float = 0.3,
        dropout_lstm: float = 0.5,
        dropout_out: float = 0.4,
        pad_idx: int = 0,
        attn_dim: int = 256,
    ):
        super().__init__()

        self.embedding = embedding
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.num_layers = num_layers

        embed_dim = embedding.embedding_dim

        self.embed_dropout = nn.Dropout(dropout_embed)
        self.lstm_dropout  = nn.Dropout(dropout_lstm)
        self.out_dropout   = nn.Dropout(dropout_out)

        self.attention = BahdanauAttention(
            enc_dim=enc_hidden_dim,
            dec_dim=dec_hidden_dim,
            attn_dim=attn_dim,
        )

        # B6: input_size = embed_dim + enc_hidden_dim = 300 + 1024 = 1324.
        # The full bidirectional context (1024) is prepended to each step's input.
        self.lstm = nn.LSTM(
            input_size=embed_dim + enc_hidden_dim,   # 1324
            hidden_size=dec_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_lstm if num_layers > 1 else 0.0,
        )

        # Bottleneck: [LSTM output ; attention context] → projection → vocab.
        # Input dim = dec_hidden_dim + enc_hidden_dim = 1024 + 1024 = 2048.
        self.projection = nn.Linear(dec_hidden_dim + enc_hidden_dim, projection_dim)
        self.fc_out     = nn.Linear(projection_dim, vocab_size)

    def forward_step(
        self,
        input_token: torch.LongTensor,            # [batch]
        hidden: torch.Tensor,                     # [num_layers, batch, dec_hidden_dim]
        cell: torch.Tensor,                       # [num_layers, batch, dec_hidden_dim]
        encoder_outputs: torch.Tensor,            # [batch, src_len, enc_hidden_dim]
        context: torch.Tensor,                    # [batch, enc_hidden_dim]  from previous step
        src_mask: Optional[torch.Tensor],
        keys_proj: Optional[torch.Tensor] = None, # [batch, src_len, attn_dim] precomputed
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Single decoder step.

        Returns:
            logits:      [batch, vocab_size]
            hidden:      [num_layers, batch, dec_hidden_dim]  updated
            cell:        [num_layers, batch, dec_hidden_dim]  updated
            new_context: [batch, enc_hidden_dim]              for next step
            attn_weights:[batch, src_len]                     for visualisation (None for baseline)
        """
        # Embed input token.
        embedded = self.embed_dropout(
            self.embedding(input_token.unsqueeze(1))   # [batch, 1, embed_dim]
        )

        # Concatenate previous context with embedding as LSTM input (B6: 1324-dim).
        lstm_input = torch.cat(
            [embedded, context.unsqueeze(1)],   # [batch, 1, embed_dim + enc_hidden_dim]
            dim=2,
        )

        # LSTM step; inter-layer dropout handled internally by nn.LSTM.
        lstm_out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        lstm_out = self.lstm_dropout(lstm_out.squeeze(1))   # [batch, dec_hidden_dim]

        # Compute new attention context using the TOP-LAYER hidden state.
        # Pass keys_proj to avoid recomputing W_enc(encoder_outputs) each step.
        new_context, step_attn = self.attention(
            encoder_outputs,
            hidden[-1],    # top layer: [batch, dec_hidden_dim]
            src_mask,
            keys_proj=keys_proj,
        )

        # Concat LSTM output with new context for richer output prediction.
        combined = torch.cat([lstm_out, new_context], dim=1)  # [batch, 2048]

        # Projection bottleneck + tanh nonlinearity.
        projected = torch.tanh(self.projection(self.out_dropout(combined)))  # [batch, 512]
        logits = self.fc_out(projected)                                       # [batch, vocab_size]
        logits = logits.clamp(-50, 50)  # prevent bf16 overflow in autoregressive decoding

        return logits, hidden, cell, new_context, step_attn

    def forward(
        self,
        trg: torch.LongTensor,            # [batch, trg_len]
        encoder_outputs: torch.Tensor,    # [batch, src_len, enc_hidden_dim]
        h0: torch.Tensor,                 # [num_layers, batch, dec_hidden_dim]
        c0: torch.Tensor,                 # [num_layers, batch, dec_hidden_dim]
        src_mask: Optional[torch.Tensor], # [batch, src_len]
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """
        Full decode loop over trg_len-1 steps (step 0 is the <sos> input,
        not predicted as output).

        Returns:
            outputs: [batch, trg_len-1, vocab_size]
        """
        batch_size = trg.size(0)
        trg_len    = trg.size(1)

        hidden, cell = h0, c0

        # B6: initial context is zeros of shape [batch, enc_hidden_dim=1024].
        context = torch.zeros(batch_size, self.enc_hidden_dim, device=trg.device)

        # Precompute W_enc(encoder_outputs) once for the entire loop — avoids
        # recomputing this projection at every decoder step (saves trg_len-1 ops).
        # Keys are passed explicitly rather than cached on self to be safe under
        # DataParallel / gradient checkpointing / concurrent forward passes.
        keys_proj = self.attention.W_enc(encoder_outputs)

        logits_list = []
        input_token = trg[:, 0]   # first input is always <sos>

        for t in range(1, trg_len):
            logits, hidden, cell, context, _ = self.forward_step(
                input_token, hidden, cell, encoder_outputs, context, src_mask,
                keys_proj=keys_proj,
            )
            logits_list.append(logits)   # [batch, vocab_size]

            # Per-sample teacher forcing using torch RNG (reproducible under set_seed).
            # Fast paths avoid a random call when ratio is exactly 0 or 1.
            if teacher_forcing_ratio >= 1.0:
                input_token = trg[:, t]
            elif teacher_forcing_ratio <= 0.0:
                input_token = logits.argmax(dim=1)
            else:
                tf_mask = torch.rand(logits.size(0), device=trg.device) < teacher_forcing_ratio
                input_token = torch.where(tf_mask, trg[:, t], logits.argmax(dim=1))

        return torch.stack(logits_list, dim=1)   # [batch, trg_len-1, vocab_size]


# ── Baseline Decoder (no attention, parameter-fair) ──────────────────────────

class BaselineDecoder(nn.Module):
    """
    Standard decoder with NO attention — parameter-near baseline (F1).

    The context vector is fixed throughout decoding: it equals the last
    encoder output timestep encoder_outputs[:, -1, :], which has shape
    [batch, enc_hidden_dim=1024].  This vector is concatenated to every
    decoder LSTM input, giving the same input_size (1324) as AttentionDecoder
    so both models are parameter-near (~0.5M difference, <1.2% of total params).

    All other design choices (projection bottleneck, dropout schedule,
    shared embedding) are identical to AttentionDecoder for a fair ablation.
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        enc_hidden_dim: int,      # FULL bidir encoder dim = 1024
        dec_hidden_dim: int,
        vocab_size: int,
        projection_dim: int = 512,
        num_layers: int = 2,
        dropout_embed: float = 0.3,
        dropout_lstm: float = 0.5,
        dropout_out: float = 0.4,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.embedding = embedding
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.num_layers = num_layers

        embed_dim = embedding.embedding_dim

        self.embed_dropout = nn.Dropout(dropout_embed)
        self.lstm_dropout  = nn.Dropout(dropout_lstm)   # applied to LSTM output (top layer)
        self.out_dropout   = nn.Dropout(dropout_out)

        # F1: same input_size = embed_dim + enc_hidden_dim = 1324 as AttentionDecoder.
        # Context is fixed (last encoder output), not attention-weighted, but the
        # LSTM still receives it so parameter count remains identical.
        self.lstm = nn.LSTM(
            input_size=embed_dim + enc_hidden_dim,   # 1324  (F1: same as AttentionDecoder)
            hidden_size=dec_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_lstm if num_layers > 1 else 0.0,
        )

        # Same bottleneck as AttentionDecoder: [lstm_out ; context] → proj → vocab.
        self.projection = nn.Linear(dec_hidden_dim + enc_hidden_dim, projection_dim)
        self.fc_out     = nn.Linear(projection_dim, vocab_size)

    def forward(
        self,
        trg: torch.LongTensor,            # [batch, trg_len]
        encoder_outputs: torch.Tensor,    # [batch, src_len, enc_hidden_dim]
        h0: torch.Tensor,                 # [num_layers, batch, dec_hidden_dim]
        c0: torch.Tensor,                 # [num_layers, batch, dec_hidden_dim]
        src_mask: Optional[torch.Tensor], # accepted for API compatibility, unused
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """
        Full decode loop. Context is fixed for all steps.

        Returns:
            outputs: [batch, trg_len-1, vocab_size]
        """
        batch_size = trg.size(0)
        trg_len    = trg.size(1)

        hidden, cell = h0, c0

        # Fixed context: last encoder output timestep [batch, enc_hidden_dim].
        # Note: for a bidirectional encoder, position -1 contains the forward
        # LSTM's full-sequence representation (✓) but the backward LSTM's
        # single-step representation at the last token only (✗).  Mean pooling
        # would be more balanced; however, a fixed-context baseline is by design
        # simpler than the attention model, and the decoder's initial hidden
        # state (from the bridge) already provides full bidirectional coverage.
        context = encoder_outputs[:, -1, :]   # [batch, enc_hidden_dim]

        logits_list = []
        input_token = trg[:, 0]   # always start from <sos>

        for t in range(1, trg_len):
            # Embed and concat fixed context as LSTM input.
            embedded = self.embed_dropout(
                self.embedding(input_token.unsqueeze(1))   # [batch, 1, embed_dim]
            )
            lstm_input = torch.cat(
                [embedded, context.unsqueeze(1)],   # [batch, 1, 1324]
                dim=2,
            )

            lstm_out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
            lstm_out = self.lstm_dropout(lstm_out.squeeze(1))   # [batch, dec_hidden_dim]
            combined  = torch.cat([lstm_out, context], dim=1)          # [batch, 2048]
            projected = torch.tanh(self.projection(self.out_dropout(combined)))  # [batch, 512]
            logits    = self.fc_out(projected)                           # [batch, vocab_size]
            logits    = logits.clamp(-50, 50)  # prevent bf16 overflow in autoregressive decoding

            logits_list.append(logits)

            # Per-sample teacher forcing using torch RNG (reproducible under set_seed).
            if teacher_forcing_ratio >= 1.0:
                input_token = trg[:, t]
            elif teacher_forcing_ratio <= 0.0:
                input_token = logits.argmax(dim=1)
            else:
                tf_mask = torch.rand(logits.size(0), device=trg.device) < teacher_forcing_ratio
                input_token = torch.where(tf_mask, trg[:, t], logits.argmax(dim=1))

        return torch.stack(logits_list, dim=1)   # [batch, trg_len-1, vocab_size]

    def forward_step(
        self,
        input_token: torch.LongTensor,    # [batch]
        hidden: torch.Tensor,             # [num_layers, batch, dec_hidden_dim]
        cell: torch.Tensor,               # [num_layers, batch, dec_hidden_dim]
        encoder_outputs: torch.Tensor,    # [batch, src_len, enc_hidden_dim]
        context: torch.Tensor,            # [batch, enc_hidden_dim] — ignored; fixed context used
        src_mask: Optional[torch.Tensor], # [batch, src_len]        — ignored; no attention
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Single decoder step for inference (mirrors AttentionDecoder.forward_step API).

        Context is always the last encoder output (fixed) — the context arg is
        accepted for API compatibility but ignored.

        Returns:
            logits:      [batch, vocab_size]
            hidden:      [num_layers, batch, dec_hidden_dim]
            cell:        [num_layers, batch, dec_hidden_dim]
            fixed_ctx:   [batch, enc_hidden_dim]  (returned for API symmetry)
            None:        no attention weights (baseline has no attention mechanism)
        """
        fixed_ctx = encoder_outputs[:, -1, :]   # [batch, enc_hidden_dim]

        embedded = self.embed_dropout(
            self.embedding(input_token.unsqueeze(1))   # [batch, 1, embed_dim]
        )
        lstm_input = torch.cat(
            [embedded, fixed_ctx.unsqueeze(1)],        # [batch, 1, 1324]
            dim=2,
        )
        lstm_out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        lstm_out = self.lstm_dropout(lstm_out.squeeze(1))   # [batch, dec_hidden_dim]

        combined  = torch.cat([lstm_out, fixed_ctx], dim=1)          # [batch, 2048]
        projected = torch.tanh(self.projection(self.out_dropout(combined)))  # [batch, 512]
        logits    = self.fc_out(projected)                            # [batch, vocab_size]

        return logits, hidden, cell, fixed_ctx, None

class Seq2Seq(nn.Module):
    """
    Orchestrates the full Encoder → Bridge → Decoder pipeline.

    Responsibilities:
      - Run encoder to get all hidden states and final (h_n, c_n).
      - Build src_mask (True at <pad> positions) and pass through to decoder/attention.
      - Bridge encoder final states to decoder initial states.
      - Run decoder with teacher-forcing ratio from the training schedule.

    Works with either AttentionDecoder or BaselineDecoder — the decoder
    interface is identical; attention decoder uses encoder_outputs while
    baseline decoder only uses encoder_outputs[:, -1, :].
    """

    def __init__(
        self,
        encoder: Encoder,
        bridge: EncoderDecoderBridge,
        decoder,     # AttentionDecoder | BaselineDecoder
        pad_idx: int = 0,
    ):
        super().__init__()

        self.encoder = encoder
        self.bridge  = bridge
        self.decoder = decoder
        self.pad_idx = pad_idx

    def forward(
        self,
        src: torch.LongTensor,            # [batch, src_len]
        src_lengths: torch.LongTensor,    # [batch]
        trg: torch.LongTensor,            # [batch, trg_len]
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """
        Full forward pass.

        Returns:
            outputs: [batch, trg_len-1, vocab_size]
        """
        # Encode: all hidden states + final (h_n, c_n).
        encoder_outputs, (h_n, c_n) = self.encoder(src, src_lengths)

        # B7: build src_mask here and thread it through to attention.
        # True at positions that are <pad> tokens → will be masked to -inf.
        src_mask = (src == self.pad_idx)   # [batch, src_len]

        # Bridge: project bidir encoder finals to decoder initial states.
        h0, c0 = self.bridge(h_n, c_n)    # each [num_layers, batch, dec_hidden_dim]

        # Decode with teacher forcing.
        outputs = self.decoder(
            trg, encoder_outputs, h0, c0, src_mask, teacher_forcing_ratio,
        )

        return outputs   # [batch, trg_len-1, vocab_size]


# ── Weight initialisation ────────────────────────────────────────────────────

def _init_weights(module: nn.Module) -> None:
    """
    Xavier / orthogonal weight initialisation for Seq2Seq Linear and LSTM layers.

    Linear layers:
        weight → Xavier uniform (suited to tanh/sigmoid nonlinearities).
        bias   → zeros.

    LSTM input-hidden weights (weight_ih):
        → Xavier uniform (input projection into gate activations).

    LSTM hidden-hidden weights (weight_hh):
        → Orthogonal (Saxe et al., 2013) — preserves gradient norms through
          recurrent steps, mitigating vanishing gradients better than Xavier.

    LSTM biases:
        All biases → zeros first, then the forget-gate slice is reset to 1.0.
        PyTorch bias layout: [input | forget | cell | output], each hidden_size.
        Forget-gate bias = 1.0 (Jozefowicz et al., 2015, §3) keeps the forget
        gate open at initialisation, allowing long-range gradient flow.

    nn.Embedding is intentionally skipped: pretrained FastText vectors are
    loaded by build_model() and must not be overwritten.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget gate occupies the second quarter of the bias vector.
                hidden_size = param.size(0) // 4
                param.data[hidden_size : 2 * hidden_size].fill_(1.0)


# ── Model factory ────────────────────────────────────────────────────────────

def build_model(
    model_type: str,
    config: dict,
    device: torch.device,
) -> Seq2Seq:
    """
    Construct a complete Seq2Seq model from CONFIG and move it to device.

    Args:
        model_type: "attention" or "baseline".
        config:     CONFIG dict from config.py.
        device:     torch.device to place the model on.

    Returns:
        Initialised Seq2Seq model on the specified device.

    NOTE: Encoder and decoder share the same nn.Embedding instance (weight tying).
    Both operate on the same BPE vocabulary; tying reduces parameter count and
    regularises by coupling token representations across the encode/decode phases.
    """
    if model_type not in ("attention", "baseline"):
        raise ValueError(f"model_type must be 'attention' or 'baseline', got {model_type!r}")

    # Load pretrained FastText embedding matrix once; shared by encoder and decoder.
    embedding = create_pretrained_embedding(
        matrix_path=config["embedding_matrix_path"],
        pad_idx=config["pad_idx"],
        freeze=False,   # fine-tune embeddings during training
    )

    encoder = Encoder(
        embedding=embedding,
        hidden_dim=config["enc_hidden_dim"],       # 512 per direction
        num_layers=config["num_layers"],
        dropout_embed=config.get("dropout_embed", 0.3),
        dropout_lstm=config.get("dropout_lstm",  0.5),
        dropout_out=config.get("dropout_out",   0.4),
    )

    bridge = EncoderDecoderBridge(
        enc_hidden_dim=config["enc_hidden_dim"],   # 512
        dec_hidden_dim=config["dec_hidden_dim"],   # 1024
        num_layers=config["num_layers"],
    )

    # Full bidirectional encoder output dim = enc_hidden_dim * 2 = 1024.
    enc_hidden_total = config["enc_hidden_dim"] * 2   # 1024

    decoder_kwargs = dict(
        embedding=embedding,                       # shared with encoder (weight tying)
        enc_hidden_dim=enc_hidden_total,           # 1024
        dec_hidden_dim=config["dec_hidden_dim"],   # 1024
        vocab_size=config["vocab_size"],           # 16000
        projection_dim=config["projection_dim"],   # 512
        num_layers=config["num_layers"],
        pad_idx=config["pad_idx"],
        dropout_embed=config.get("dropout_embed", 0.3),
        dropout_lstm=config.get("dropout_lstm",  0.5),
        dropout_out=config.get("dropout_out",   0.4),
    )

    if model_type == "attention":
        decoder = AttentionDecoder(
            **decoder_kwargs,
            attn_dim=config.get("attn_dim", 256),  # thread from config; default matches original
        )
    else:
        decoder = BaselineDecoder(**decoder_kwargs)

    model = Seq2Seq(
        encoder=encoder,
        bridge=bridge,
        decoder=decoder,
        pad_idx=config["pad_idx"],
    ).to(device)

    # Apply weight initialisation AFTER moving to device.
    # nn.Embedding is skipped by _init_weights — pretrained FastText vectors
    # loaded above are preserved. All Linear and LSTM layers get Xavier /
    # orthogonal / forget-gate-bias=1.0 initialisation.
    model.apply(_init_weights)

    # Report parameter counts for each component.
    def _count(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    print(f"[build_model] model_type     : {model_type}")
    print(f"[build_model] encoder params : {_count(encoder):,}")
    print(f"[build_model] bridge  params : {_count(bridge):,}")
    print(f"[build_model] decoder params : {_count(decoder):,}")
    print(f"[build_model] total   params : {_count(model):,}")

    return model
