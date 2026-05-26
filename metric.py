# import torch
# from models import build, count_params_millions

# m = build("faster_rcnn")
# m.load_state_dict(torch.load("outputs/faster_rcnn/best.pt", map_location="cpu"))
# print(f"Total params: {count_params_millions(m):.2f}M")
# for name, mod in m.named_children():
#     n = sum(p.numel() for p in mod.parameters())
#     print(f"  {name:12s}: {n / 1e6:.2f}M params")


import torch
from models import build

m = build("faster_rcnn").eval()
m.load_state_dict(torch.load("outputs/faster_rcnn/best.pt", map_location="cpu"))

# install once: pip install fvcore
from fvcore.nn import FlopCountAnalysis

# Faster R-CNN takes a list of CHW tensors; use your real input scale (~1280x386)
dummy = [torch.rand(3, 386, 1280)]
flops = FlopCountAnalysis(m, (dummy,))
print(f"{flops.total() / 1e9:.1f} GFLOPs")
