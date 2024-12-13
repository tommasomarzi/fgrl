import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, input_size, output_size, layers):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        in_feats = [self.input_size] + layers
        out_feats = layers + [self.output_size]

        self.linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_feats[l], out_feats[l]),
                nn.Tanh())
            for l in range(len(layers) + 1)])

    def forward(self, x):

        out = x
        for layer in self.linears:
            out = layer(out)

        return out


def create_model(input_size, output_size, layers):
    model = FullyConnected(input_size, output_size, layers)
    # model.to(config.device)

    return model
