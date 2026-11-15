"""RIMoE with label-trained latent interaction-structure inference."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import ModalityBackbone
from .interaction_structure import AdaptiveInteractionStructure


class RIMoE(ModalityBackbone):
    """Predict with a prior mixture over TA, TV, AV and TAV experts."""

    def __init__(self, args):
        super().__init__(args)
        dim = self.d_l
        aism_dim = getattr(args, "aism_dim", dim)
        fusion_dim = dim if self.fusion_method == "sum" else 3 * dim
        self.aism = AdaptiveInteractionStructure(
            dim=aism_dim,
            num_heads=getattr(args, "aism_num_heads", 4),
            num_layers=getattr(args, "aism_num_layers", 1),
            dropout=getattr(args, "aism_dropout", 0.1),
            router_hidden=getattr(args, "aism_router_hidden", 128),
            temperature=getattr(args, "aism_temperature", 1.0),
        )
        self.aism_input_projections = nn.ModuleList([
            nn.Identity() if aism_dim == dim else nn.Linear(dim, aism_dim)
            for _ in range(3)
        ])
        self.aism_adapter = nn.Sequential(
            nn.LayerNorm(aism_dim),
            nn.Linear(aism_dim, aism_dim),
            nn.GELU(),
            nn.Linear(aism_dim, fusion_dim),
        )
        self.aism_scale_logit = nn.Parameter(torch.tensor(-2.1972246))
        scale = float(getattr(args, "bayes_scale", 1.0))
        if not torch.isfinite(torch.tensor(scale)) or scale <= 0:
            raise ValueError("bayes_scale must be a finite positive scalar")
        self.register_buffer("bayes_scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, text, audio, video):
        out = super().forward(text, audio, video)
        projected = [projection(value) for projection, value in zip(
            self.aism_input_projections,
            (out["modality_t"], out["modality_a"], out["modality_v"]),
        )]
        aism_out = self.aism(*projected)
        features = aism_out["relation_features"]
        raw_logits = aism_out["relation_logits"]
        log_prior = F.log_softmax(raw_logits.float() / self.aism.temperature, dim=-1)
        prior = log_prior.exp()
        beta = torch.sigmoid(self.aism_scale_logit)
        delta = self.aism_adapter(features)
        component_features = out["c_proj"].unsqueeze(1) + beta * delta
        component_means = self.out_layer_c(component_features).squeeze(-1)
        prediction = (prior * component_means.float()).sum(dim=1, keepdim=True)
        enhanced_proj = (prior.unsqueeze(-1) * component_features.float()).sum(dim=1)
        interaction_feature = (prior.unsqueeze(-1) * features.float()).sum(dim=1)
        out.update({
            "logits_c": prediction,
            "relation_logits": raw_logits,
            "structure_log_prior": log_prior,
            "structure_prior": prior,
            "relation_weight": prior,
            "component_means": component_means,
            "component_features": component_features,
            "enhanced_proj": enhanced_proj,
            "interaction_feature": interaction_feature,
            "relation_features": features,
            "aism_beta": beta,
            "bayes_scale": self.bayes_scale,
        })
        return out
