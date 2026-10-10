from dataclasses import dataclass, field

@dataclass(frozen=True)
class ViTConfig:
    patch_size: int = 16
    layers: int = 12
    dim: int = 768
    heads: int = 12
    mlp_dim: int = 3072
    input_size: int = 224
    
@dataclass(frozen=True)
class VLAConfig:
    vit: ViTConfig = field(default_factory=ViTConfig)
    layers: int = 12
    heads: int = 12
    vlm_dim: int = 768
    vlm_mlp_dim: int = 3072
    action_dim: int = 768
    action_mlp_dim: int = 3072
    state_size: int= 32
    action_size: int= 32
    action_chunk_size: int = 50