from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Tuple

import torch
import torch.nn as nn

import json

# ----------------------------
# Model definition (same as notebook)
# ----------------------------
class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        n_cat_features: int,
        cat_cardinalities: list[int],
        embed_dim: int = 32,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_num_features = int(n_num_features)
        self.n_cat_features = int(n_cat_features)

        if len(cat_cardinalities) != self.n_cat_features:
            raise ValueError(
                f"cat_cardinalities length ({len(cat_cardinalities)}) must match "
                f"n_cat_features ({self.n_cat_features})."
            )

        self.cat_embeds = nn.ModuleList(
            [
                nn.Embedding(int(card) + 1, embed_dim, padding_idx=0)
                for card in cat_cardinalities
            ]
        )
        self.num_projections = nn.ModuleList(
            [nn.Linear(1, embed_dim) for _ in range(self.n_num_features)]
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

        self.dropout_layers = [m for m in self.modules() if isinstance(m, nn.Dropout)]

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        batch_size = x_num.size(0)

        cat_tokens = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.cat_embeds)]
        num_tokens = [proj(x_num[:, i : i + 1]) for i, proj in enumerate(self.num_projections)]

        tokens = torch.stack(cat_tokens + num_tokens, dim=1)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        x = self.transformer(tokens)
        cls_output = self.ln(x[:, 0])
        return self.head(cls_output)

    def enable_mc_dropout(self) -> None:
        for m in self.dropout_layers:
            m.train()


# ----------------------------
# Config helper (optional)
# ----------------------------
@dataclass(frozen=True)
class FTTransformerConfig:
    n_num_features: int
    n_cat_features: int
    cat_cardinalities: list[int]
    embed_dim: int = 32
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FTTransformerConfig":
        return FTTransformerConfig(
            n_num_features=int(d["n_num_features"]),
            n_cat_features=int(d["n_cat_features"]),
            cat_cardinalities=[int(x) for x in d["cat_cardinalities"]],
            embed_dim=int(d.get("embed_dim", 32)),
            n_layers=int(d.get("n_layers", 3)),
            n_heads=int(d.get("n_heads", 4)),
            dropout=float(d.get("dropout", 0.1)),
        )


# ----------------------------
# Core thing you asked for: build arch only
# ----------------------------
ConfigLike = Union[FTTransformerConfig, Dict[str, Any]]

def build_fttransformer(
    config: ConfigLike,
    *,
    device: str = "cpu",
) -> FTTransformer:
    """
    Build the FTTransformer architecture ONLY (no weights loaded).
    Pass either FTTransformerConfig or a plain dict of params.
    """
    if isinstance(config, dict):
        config = FTTransformerConfig.from_dict(config)

    model = FTTransformer(
        n_num_features=config.n_num_features,
        n_cat_features=config.n_cat_features,
        cat_cardinalities=config.cat_cardinalities,
        embed_dim=config.embed_dim,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        dropout=config.dropout,
    ).to(device)

    return model


# ----------------------------
# Weight loading helper (optional, but convenient)
# ----------------------------
def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """
    Supports:
      - raw state_dict
      - {"state_dict": ...} / {"model_state_dict": ...} / {"model": ...}
    """
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        # assume it's already a state_dict
        return obj
    raise ValueError("Unsupported checkpoint format; expected a dict-like checkpoint/state_dict.")


def load_weights(
    model: nn.Module,
    weights: Union[str, Dict[str, Any]],
    *,
    map_location: str = "cpu",
    strict: bool = True,
) -> nn.Module:
    """
    Load a .pth/.pt file or an in-memory dict into an existing model.
    """
    if isinstance(weights, str):
        obj = torch.load(weights, map_location=map_location)
    else:
        obj = weights

    state_dict = _extract_state_dict(obj)
    model.load_state_dict(state_dict, strict=strict)
    return model


# ----------------------------
# Convenience: build + load (optional)
# ----------------------------
def load_fttransformer_ready(
    config: ConfigLike,
    weights: Optional[Union[str, Dict[str, Any]]] = None,
    *,
    device: str = "cpu",
    map_location: str = "cpu",
    strict: bool = True,
) -> FTTransformer:
    """
    Build model and optionally load weights.
    """
    model = build_fttransformer(config, device=device)
    if weights is not None:
        load_weights(model, weights, map_location=map_location, strict=strict)
    model.eval()
    return model

def arr_load_fttransformer_from_config_json_and_pth(
    *,
    config_json_path: str,
    weights_pth_path: str,
    device: str = "cpu",
    strict: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Build FTTransformer architecture from a config JSON and load .pth weights.

    Returns:
        model : torch.nn.Module
            Model with weights loaded and moved to `device`
        meta : dict
            Metadata about the load (missing keys, unexpected keys, config, etc.)
    """

    # ----------------------------
    # Load config
    # ----------------------------
    with open(config_json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg = cfg.get("config", {})
    required = ["n_num_features", "n_cat_features", "cat_cardinalities"]
    missing_cfg = [k for k in required if k not in cfg]
    if missing_cfg:
        raise ValueError(f"Config JSON missing required keys: {missing_cfg}")

    # ----------------------------
    # Build model architecture
    # ----------------------------
    model = FTTransformer(
        n_num_features=int(cfg["n_num_features"]),
        n_cat_features=int(cfg["n_cat_features"]),
        cat_cardinalities=[int(x) for x in cfg["cat_cardinalities"]],
        embed_dim=int(cfg.get("embed_dim", 32)),
        n_layers=int(cfg.get("n_layers", 3)),
        n_heads=int(cfg.get("n_heads", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
    ).to(device)

    # ----------------------------
    # Load weights
    # ----------------------------
    checkpoint = torch.load(weights_pth_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint  # assume raw state_dict
    else:
        raise ValueError("Unsupported .pth format (expected dict-like object).")

    load_result = model.load_state_dict(state_dict, strict=strict)

    # ----------------------------
    # Finalize
    # ----------------------------
    model.eval()

    meta = {
        "config": cfg,
        "strict": strict,
        "device": device,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }

    return model, meta
