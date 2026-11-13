"""gnina pose networks in PyTorch (adapted from gnina/models pytorch/*.py, Apache 2.0) and the published ensemble."""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
# the gnina 1.0 default ensemble: (weights file stem, architecture)
ENSEMBLE = (("dense", "dense"), ("general_default2018_3", "default2018"), ("dense_3", "dense"),
            ("crossdock_default2018", "default2018"), ("redock_default2018", "default2018"))
GRID_CHANNELS, GRID_POINTS, GRID_RESOLUTION, GRID_DIMENSION = 28, 48, 0.5, 23.5


class DenseBlock(nn.Module):
    def __init__(self, input_feats, level, convolutions=4, num_dense_filters=16):
        super().__init__()
        self.layers = []
        feats = input_feats
        for idx in range(convolutions):
            bn = nn.BatchNorm3d(feats)
            self.add_module(f"data_enc_level{level}_batchnorm_conv{idx}", bn)
            conv = nn.Conv3d(feats, num_dense_filters, kernel_size=3, padding=1)
            self.add_module(f"data_enc_level{level}_conv{idx}", conv)
            self.layers += [bn, conv]
            feats += num_dense_filters

    def forward(self, x):
        previous = [x]
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Conv3d):
                x = F.relu(x)
                previous.append(x)
                x = torch.cat(previous, 1)
        return x


class Dense(nn.Module):
    """DenseNet-style 3D CNN; returns (pose logits [N, 2], affinity [N, 1])."""

    def __init__(self, channels=GRID_CHANNELS, num_dense_blocks=3, num_dense_filters=16, num_dense_convs=4):
        super().__init__()
        self.layers = []
        self.data_enc_init_pool = nn.MaxPool3d(2, stride=2)
        self.data_enc_init_conv = nn.Conv3d(channels, 32, 3, padding=1)
        out_feats = 32
        for idx in range(num_dense_blocks - 1):
            block = DenseBlock(out_feats, idx, num_dense_convs, num_dense_filters)
            self.add_module(f"dense_block_{idx}", block)
            out_feats = num_dense_convs * num_dense_filters + out_feats
            bottleneck = nn.Conv3d(out_feats, out_feats, 1)
            self.add_module(f"data_enc_level{idx}_bottleneck", bottleneck)
            pool = nn.MaxPool3d(2, stride=2)
            self.add_module(f"data_enc_level{idx + 1}_pool", pool)
            self.layers += [block, bottleneck, pool]
        block = DenseBlock(out_feats, num_dense_blocks - 1, num_dense_convs, num_dense_filters)
        self.add_module(f"dense_block_{num_dense_blocks - 1}", block)
        out_feats = num_dense_convs * num_dense_filters + out_feats
        self.layers.append(block)
        self.affinity_output = nn.Linear(out_feats, 1)
        self.pose_output = nn.Linear(out_feats, 2)

    def forward(self, x):
        x = F.relu(self.data_enc_init_conv(self.data_enc_init_pool(x)))
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Conv3d):
                x = F.relu(x)
        x = F.max_pool3d(x, kernel_size=x.size()[2:]).view(*x.size()[:2])
        return self.pose_output(x), self.affinity_output(x)


class Default2018(nn.Module):
    """gnina default2018: three conv/pool stages; returns (pose logits [N, 2], affinity [N, 1])."""

    def __init__(self, channels=GRID_CHANNELS, points=GRID_POINTS):
        super().__init__()
        self.layers = []
        spec = [("avgpool_0", nn.AvgPool3d(2, stride=2)), ("unit1_conv", nn.Conv3d(channels, 32, 3, padding=1)),
                ("unit2_conv", nn.Conv3d(32, 32, 1)), ("avgpool_1", nn.AvgPool3d(2, stride=2)),
                ("unit3_conv", nn.Conv3d(32, 64, 3, padding=1)), ("unit4_conv", nn.Conv3d(64, 64, 1)),
                ("avgpool_2", nn.AvgPool3d(2, stride=2)), ("unit5_conv", nn.Conv3d(64, 128, 3, padding=1))]
        for name, layer in spec:
            self.add_module(name, layer)
            self.layers.append(layer)
        last = 128 * (points // 8) ** 3
        self.affinity_output = nn.Linear(last, 1)
        self.pose_output = nn.Linear(last, 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Conv3d):
                x = F.relu(x)
        x = x.view(x.size(0), -1)
        return self.pose_output(x), self.affinity_output(x)


def load_ensemble(device, weights_dir=WEIGHTS_DIR):
    models = []
    for stem, arch in ENSEMBLE:
        model = Dense() if arch == "dense" else Default2018()
        state = torch.load(os.path.join(weights_dir, f"{stem}.pt"), map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)
        models.append(model.to(device).eval())
    return models
