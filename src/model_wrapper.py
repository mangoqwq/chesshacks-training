from typing import Tuple
import torch


class ModelWrapper:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    def predict(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            policy, value = self.model(input_tensor.unsqueeze(0))

        return policy.squeeze(0), value.squeeze(0)
