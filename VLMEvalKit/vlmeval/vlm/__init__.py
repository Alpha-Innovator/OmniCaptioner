import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)


from .omnicaptioner import Qwen2VLOmniCap
from .qwencaptioner import Qwen2VLCap
from .omnicaptioner_cot import Qwen2VLOmniCap_cot