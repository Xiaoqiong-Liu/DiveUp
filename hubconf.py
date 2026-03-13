dependencies = ["torch", "natten"]

import torch

from src.model.naf import NAF


def diveup(pretrained=True, device="cpu", **kwargs):
    # ... 前面的实例化代码 ...
    model = NAF().to(device)
    if pretrained:
            checkpoint = "https://github.com/Xiaoqiong-Liu/DiveUp/releases/download/v1.0/model_25000steps.pth"
            # 这一步会自动从你提供的链接下载并缓存到用户的电脑里
            state_dict = torch.hub.load_state_dict_from_url(checkpoint, progress=True, map_location=device)
            model.load_state_dict(state_dict)
        
    return model
