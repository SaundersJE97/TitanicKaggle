import torch
from jaxtyping import Float, Array
from torch import nn


class NeuralNet(nn.Module):

    def __init__(self, input_features: int, out_features: int):
        super().__init__()

        self.layer_1 = nn.Linear(input_features, 256)
        self.layer_2 = nn.Linear(256, 256)
        self.layer_3 = nn.Linear(256, out_features)
        self.to(device=torch.device('cuda'))

    def forward(self, input: Float[Array, "batch features"]):
        x = nn.functional.relu(self.layer_1.forward(input))
        x = nn.functional.relu(self.layer_2.forward(x))
        x = self.layer_3.forward(x)
        return x