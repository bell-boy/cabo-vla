from flax import nnx
import jax.numpy as jnp

from cabo_vla.load_model import load_hf_vit
from cabo_vla.vit import ViT
from cabo_vla.config import ViTConfig

def main():
    vit_config = ViTConfig()
    vit = ViT(vit_config, rngs=nnx.Rngs(0))
    load_hf_vit(vit, repo="google/vit-base-patch16-224")
    sample_input = jnp.ones((16, 3, vit_config.input_size, vit_config.input_size))
    print(f"Sample input shape: {sample_input.shape}")
    output = vit(sample_input)
    print(f"Output shape: {output.shape}") 

if __name__ == "__main__":
    main()
