import torch
import torchvision.transforms as T
from pathlib import Path

from shared import utils
from .dataset import DATASETS
from torch.utils.data import random_split


def build(args, rng):
    Dataset = DATASETS[args.foreground_type]

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    transform = T.Compose([
        T.GaussianBlur(kernel_size=3, sigma=1)
    ])
    tp = Path(args.root).joinpath('data/textures')
    textures, _ = utils.load_textures(tp)
    MNIST_root = Path(args.root).joinpath('data/generated/')
    MNIST_root.mkdir(exist_ok=True)
    train_set = Dataset(
        root=MNIST_root,
        textures=textures,
        im_size=(args.size, args.size),
        foreground=args.foreground_type,
        shadow_px=args.shadow_px,
        transform=transform,
        rng=rng,
        match_bg_fg=args.match_bg_fg,
        batch_size=args.batch_size,
        fg_size=args.fg_size
    )
    test_set = Dataset(
        root=MNIST_root,
        textures=textures,
        foreground=args.foreground_type,
        shadow_px=args.shadow_px,
        train=False,
        transform=transform,
        rng=rng,
        match_bg_fg=args.match_bg_fg,
        batch_size=args.batch_size,
        fg_size=args.fg_size
    )
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=True, **test_kwargs)

    return train_loader, test_loader


def val_batch_from_train(args, train_loader):
    train_set = train_loader.dataset
    dataset_len = len(train_set)
    val_size = int(dataset_len * args.val_frac)
    train_size = dataset_len - val_size

    train_set, val_set = random_split(
        train_set, [train_size, val_size])

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    train_kwargs = {'batch_size': args.batch_size}
    val_kwargs = {'batch_size': args.val_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, **val_kwargs)

    return train_loader, val_loader
