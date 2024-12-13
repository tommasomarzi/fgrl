import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, input_size, output_size, layers, last_activation="tanh"):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        activations = [nn.Tanh() for _ in range(len(layers))]
        if last_activation == "tanh":
            activations.append(nn.Tanh())
        elif last_activation == "softmax":
            activations.append(nn.Softmax(dim=-1))
        elif last_activation == "none":
            activations.append(nn.Identity())

        in_feats = [self.input_size] + layers
        out_feats = layers + [self.output_size]

        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_feats[l], out_feats[l]),
                activations[l])
            for l in range(len(layers) + 1)])

    def forward(self, x):

        out = x
        for layer in self.linears:
            out = layer(out)

        return out


def create_model(input_size, output_size, layers, last_activation="tanh"):
    model = FullyConnected(input_size, output_size, layers, last_activation)
    # model.to(config.device)

    return model
