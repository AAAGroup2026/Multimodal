"""Adaptive interaction structures for RIMoE.

The four structures (TA, TV, AV, TAV) share one short-sequence Transformer.
Only modality features named by a structure enter its CLS encoder; a separate
interaction-aware router can inspect all modality and candidate features.
"""
import torch
import torch.nn as nn

from .encoders.transformer.transformer import TransformerEncoder


RELATION_NAMES = ("TA", "TV", "AV", "TAV")
RELATION_MODALITIES = ((0, 1), (0, 2), (1, 2), (0, 1, 2))


class AdaptiveInteractionStructure(nn.Module):
    """Encode four candidate cross-modal structures and route per sample."""

    def __init__(self, dim=256, num_heads=4, num_layers=1, dropout=0.1,
                 router_hidden=128, temperature=1.0):
        super().__init__()
        if dim <= 0 or num_heads <= 0 or dim % num_heads:
            raise ValueError("dim must be positive and divisible by num_heads")
        if num_layers <= 0 or router_hidden <= 0:
            raise ValueError("num_layers and router_hidden must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.dim = dim
        self.temperature = float(temperature)
        self.modality_embeddings = nn.Parameter(torch.empty(3, dim))
        self.cls_tokens = nn.Parameter(torch.empty(4, 1, dim))
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_tokens, mean=0.0, std=0.02)

        # One shared encoder serves all four sequence structures.
        self.encoder = TransformerEncoder(
            embed_dim=dim,
            num_heads=num_heads,
            layers=num_layers,
            attn_dropout=dropout,
            relu_dropout=dropout,
            res_dropout=dropout,
            embed_dropout=dropout,
            attn_mask=False,
        )
        self.output_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.structure_router = nn.Sequential(
            nn.Linear(4 * dim, router_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden, 1),
        )

    def _encode_structure(self, index, modalities):
        batch_size = modalities[0].shape[0]
        cls = self.cls_tokens[index].unsqueeze(0).expand(-1, batch_size, -1)
        modality_tokens = [
            modalities[modality] + self.modality_embeddings[modality].unsqueeze(0)
            for modality in RELATION_MODALITIES[index]
        ]
        tokens = torch.cat([cls, *[token.unsqueeze(0) for token in modality_tokens]], dim=0)
        # Sequence length is 3 for pair structures and 4 for TAV.
        encoded = self.encoder(tokens)
        return self.output_norms[index](encoded[0])

    def forward(self, h_t, h_a, h_v):
        modalities = (h_t, h_a, h_v)
        if any(value.ndim != 2 or value.shape[-1] != self.dim for value in modalities):
            raise ValueError(f"Expected three [B,{self.dim}] modality tensors")
        if not (h_t.shape[0] == h_a.shape[0] == h_v.shape[0]):
            raise ValueError("Modality batch sizes must match")

        relation_features = torch.stack([
            self._encode_structure(index, modalities) for index in range(4)
        ], dim=1)
        global_context = torch.cat(modalities, dim=-1)
        global_context = global_context.unsqueeze(1).expand(-1, 4, -1)
        score_inputs = torch.cat([global_context, relation_features], dim=-1)
        relation_logits = self.structure_router(score_inputs).squeeze(-1)
        relation_weights = torch.softmax(relation_logits / self.temperature, dim=1)
        interaction_feature = torch.sum(
            relation_weights.unsqueeze(-1) * relation_features, dim=1
        )
        return {
            "relation_features": relation_features,
            "relation_weights": relation_weights,
            "interaction_feature": interaction_feature,
            "relation_logits": relation_logits,
        }
