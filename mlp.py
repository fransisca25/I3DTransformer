import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim=1024,
        adaptation_dims=[512, 256],
        with_norm=True,
        num_classes=2000,
        with_classification=True,
        dropout_keep_prob=0.0,
    ):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.with_norm = with_norm
        self.with_classification = with_classification
        self.dropout = nn.Dropout(dropout_keep_prob)

        self.res = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.LeakyReLU(0.2, inplace=True)
        )
        layers = []
        layers.append(nn.Linear(input_dim, adaptation_dims[0]))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        for i in range(1, len(adaptation_dims)):
            layers.append(nn.Linear(adaptation_dims[i - 1], adaptation_dims[i]))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.adaptor = nn.Sequential(*layers)
        output_dim = adaptation_dims[-1]
        self.logits = nn.Linear(output_dim, num_classes)

    def forward(self, x):
        res_output = self.res(x)
        # print("RES OUTPUT:", res_output.shape)

        x = x + self.res(x)
        # print("X + RESIDUAL MLP:", x.shape)
        
        x = self.adaptor(x)
        # print("ADAPTOR MLP:", x.shape)
        
        if self.with_norm:
            x = F.normalize(x, p=2, dim=1)
        out = {"embds": x}
        # print("OUT EMBDS SHAPE MLP:", out["embds"].shape)

        if self.with_classification:
            logits = self.logits(self.dropout(x))
            out["logits"] = logits
            # print("LOGITS MLP:", out["logits"].shape)

        return out


