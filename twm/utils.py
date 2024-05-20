import math

import torch
from torch import nn, optim


def random_choice(n, num_samples, replacement=False, device=None):
    if replacement:
        return torch.randint(0, n, (num_samples,), device=device)

    weights = torch.ones(n, device=device)
    return torch.multinomial(weights, num_samples, replacement=False)


def windows(x, window_size, window_stride=1):
    x = x.unfold(1, window_size, window_stride)
    dims = list(range(x.ndim))[:-1]
    dims.insert(2, x.ndim - 1)
    x = x.permute(dims)
    return x


def same_batch_shape(tensors, ndim=2):
    batch_shape = tensors[0].shape[:ndim]
    assert all(t.ndim >= ndim for t in tensors)
    return all(tensors[i].shape[:ndim] == batch_shape for i in range(1, len(tensors)))


def same_batch_shape_time_offset(a, b, offset):
    assert a.ndim >= 2 and b.ndim >= 2
    return a.shape[:2] == (b.shape[0], b.shape[1] + offset)


def check_no_grad(*tensors):
    return all((t is None or not t.requires_grad) for t in tensors)


class AdamOptim:

    def __init__(self, parameters, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, grad_clip=0):
        self.parameters = list(parameters)
        self.grad_clip = grad_clip
        self.optimizer = optim.Adam(self.parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    def step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
        self.optimizer.step()



@torch.no_grad()
def make_grid(tensor, nrow, padding, pad_value=0):
    # modified version of torchvision.utils.make_grid that supports different paddings for x and y
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding[0]), int(tensor.size(3) + padding[1])
    num_channels = tensor.size(1)
    grid = tensor.new_full(
        (num_channels, height * ymaps + padding[0], width * xmaps + padding[1]), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding[0], height - padding[0]) \
                .narrow(2, x * width + padding[1], width - padding[1]) \
                .copy_(tensor[k])
            k = k + 1
    return grid


def to_image(tensor):
    from PIL import Image
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8)
    if tensor.shape[2] == 1:
        tensor = tensor.squeeze(2)
    return Image.fromarray(tensor.numpy()).convert('RGB')
