dependencies = ["torch", "natten"]

import torch

from src.model.naf import DiveUp, NAF


def naf(pretrained: bool = True, device="cpu"):
    """
    NAF (Neighborhood Attention Filtering) model for feature upsampling.
    VFM-agnostic upsampler that works with any Vision Foundation Model without retraining.

    Dependencies:
        - torch: PyTorch framework
        - natten: Neighborhood Attention Extension (required for cross-scale attention)

    Installation:
        pip install natten -f https://shi-labs.com/natten/wheels
    """
    model = NAF().to(device)
    if pretrained:
        checkpoint = "https://github.com/Xiaoqiong-Liu/DiveUp/releases/download/v1.0/model_25000steps.pth"
        model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True, map_location=device))
    return model


def diveup(
    pretrained: bool = False,
    device: str = "cpu",
    dim: int = 256,
    heads_attn: int = 4,
    heads_rope: int = 4,
    kernel_size: int = 9,
    use_encoder: bool = True,
    img_layers: int = 2,
    rope_rescale: float = 2.0,
    rope_base: float = 100.0,
    checkpoint_url: str | None = None,
    **kwargs,
):
    """
    DiveUp: NAF upsampler trained with MoT (Mixture of Teachers) boundary loss.
    Same architecture as NAF; use this entrypoint for DiveUp-trained weights.

    Args:
        pretrained: If True, load weights from checkpoint_url (or default release).
        device: Device to place the model on.
        dim: Feature dimension (default 256).
        heads_attn: Number of cross-attention heads.
        heads_rope: Number of RoPE heads.
        kernel_size: Neighborhood attention kernel size.
        use_encoder: Use image encoder in the upsampler.
        img_layers: Number of encoder layers.
        rope_rescale: RoPE coordinate rescale factor.
        rope_base: RoPE base frequency.
        checkpoint_url: Override URL for pretrained weights (used when pretrained=True).
        **kwargs: Extra arguments passed to DiveUp.

    Returns:
        DiveUp model, optionally loaded with pretrained weights.

    Dependencies:
        - torch, natten (same as NAF).
    """
    model = DiveUp(
        dim=dim,
        heads_attn=heads_attn,
        heads_rope=heads_rope,
        kernel_size=kernel_size,
        use_encoder=use_encoder,
        img_layers=img_layers,
        rope_rescale=rope_rescale,
        rope_base=rope_base,
        **kwargs,
    ).to(device)
    if pretrained:
        url = checkpoint_url or (
            "https://github.com/Xiaoqiong-Liu/DiveUp/releases/download/v1.0/model_25000steps.pth"
        )
        state = torch.hub.load_state_dict_from_url(url, progress=True, map_location=device)
        model.load_state_dict(state, strict=True)
    return model
