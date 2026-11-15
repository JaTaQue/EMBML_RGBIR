"""Make a copy of a pretrained YOLO model whose first layer takes 4 channels (R, G, B, IR).

    python tools/make_4ch_weights.py --model weights/yolo26n.pt --out weights/yolo26n_4ch.pt

A pretrained model reads 3-channel pictures, but the fusion pictures have 4 channels. This keeps the
three pretrained RGB filters of the first layer as they are, and starts the new IR filter as the mean of
the three RGB filters, so the pretrained features are not thrown away. All other layers are unchanged.
Train with it on a data yaml that has `channels: 4` (tools/train.py checks that the two match).

Pass a weights file that exists: if the name is not found, Ultralytics downloads it into the current folder.
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def widen_first_layer(model, channels):
    """Give the first convolution `channels` inputs. The first 3 keep their weights, the new ones get the mean of those."""
    net = model.model
    conv = net.model[0].conv
    if conv.in_channels != 3:
        raise SystemExit(f"the first layer takes {conv.in_channels} channels, expected 3: is this model already widened?")
    new = torch.nn.Conv2d(channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                          bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3:] = conv.weight.mean(dim=1, keepdim=True)
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    net.model[0].conv = new
    net.yaml["ch"] = channels
    return tuple(conv.weight.shape), tuple(new.weight.shape)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="3-channel weights file that exists, e.g. weights/yolo26n.pt")
    p.add_argument("--out", required=True, help="where to write the 4-channel copy")
    p.add_argument("--channels", type=int, default=4, help="input channels of the new model (default 4)")
    a = p.parse_args()
    if not Path(a.model).is_file():
        raise SystemExit(f"{a.model} not found (it would be downloaded into the current folder, so pass a real file)")
    if a.channels < 4:
        raise SystemExit("--channels must be 4 or more")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} already exists, delete it or pick another name")
    out.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(a.model)
    before, after = widen_first_layer(model, a.channels)
    model.save(out)
    print(f"first layer {before} -> {after}, written to {out}")


if __name__ == "__main__":
    main()
