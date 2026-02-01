import json
from typing import Any, Dict, Tuple, Optional, Union

import torch
import torch.nn as nn


# --- Your model class (keep as-is, or import it) ---
class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num_features: int,
        n_cat_features: int,
        cat_cardinalities,
        embed_dim: int = 32,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cat_embeds = nn.ModuleList([
            nn.Embedding(int(card) + 1, embed_dim, padding_idx=0)
            for card in cat_cardinalities
        ])

        self.num_projections = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(int(n_num_features))
        ])

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

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        bs = x_num.size(0)

        cat_tokens = [emb(x_cat[:, i].long()) for i, emb in enumerate(self.cat_embeds)]
        num_tokens = [proj(x_num[:, i:i+1]) for i, proj in enumerate(self.num_projections)]

        tokens = torch.stack(cat_tokens + num_tokens, dim=1)
        cls = self.cls_token.expand(bs, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        x = self.transformer(tokens)
        cls_out = self.ln(x[:, 0])
        return self.head(cls_out)


# --- helpers ---
def _extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # raw state_dict
        if any(hasattr(v, "shape") for v in ckpt.values()):
            return ckpt
    raise ValueError(
        "Unrecognized .pth format. Expected a state_dict or dict containing "
        "'state_dict' / 'model_state_dict' / 'model'."
    )

def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if sd and all(k.startswith("module.") for k in sd.keys()):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd

def _load_config(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r") as f:
        payload = json.load(f)
    if "config" not in payload or not isinstance(payload["config"], dict):
        raise ValueError("JSON must contain a top-level 'config' dict.")
    cfg = dict(payload["config"])

    # coerce types
    cfg["n_num_features"] = int(cfg["n_num_features"])
    cfg["n_cat_features"] = int(cfg["n_cat_features"])
    cfg["cat_cardinalities"] = [int(x) for x in cfg["cat_cardinalities"]]
    cfg["embed_dim"] = int(cfg.get("embed_dim", 32))
    cfg["n_layers"] = int(cfg.get("n_layers", 3))
    cfg["n_heads"] = int(cfg.get("n_heads", 4))
    cfg["dropout"] = float(cfg.get("dropout", 0.1))
    return cfg

def load_fttransformer_from_config_json_and_pth(
    config_json_path: str,
    weights_pth_path: str,
    *,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Tuple[FTTransformer, Dict[str, Any]]:
    device = torch.device(device)

    cfg = _load_config(config_json_path)

    model = FTTransformer(**cfg).to(device)

    ckpt = torch.load(weights_pth_path, map_location=device)
    sd = _strip_module_prefix(_extract_state_dict(ckpt))

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    model.eval()

    meta = {
        "used_config": cfg,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    return model, meta
