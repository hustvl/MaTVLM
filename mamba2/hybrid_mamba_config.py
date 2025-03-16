from dataclasses import dataclass, field
from typing import List

@dataclass
class MambaConfig:
    d_model: int = 2560
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm_eps: float = 1e-5
    vocab_size: int = None
    d_inner: int = None
    d_xb: int = 2560
    intermediate_size: int = 10240
    hidden_act: str = "silu"
    n_layer: int = 32
    attn_layers: List[int] = field(default_factory=list)
    bidirectional: bool = False
    is_bias: bool = False

@dataclass
class PhiMambaConfig:
    d_model: int = 2560
    ssm_cfg: dict = field(default_factory=dict)
    layer_norm_eps: float = 1e-5
    vocab_size: int = None
    d_inner: int = None
    d_xb: int = 2560
    intermediate_size: int = 10240
    hidden_act: str = "silu"
    n_layer: int = 32
    attn_layers: List[int] = field(default_factory=list)
    resid_pdrop: float = 0.1
    bidirectional: bool = False
    is_bias: bool = False