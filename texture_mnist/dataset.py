import torch
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms.functional as F
import numpy as np
from shared import utils
from PIL import Image
from ._textureMNIST_ops import fuse, extract_patch
import torchvision.transforms as T


class TextureForeground():
    """ Custom Pytorch Dataset for Texture MNIST

    root:
        Where the MNIST data is downloaded and saved
    textures:
        A (r, m, m) tensor. These are the source texture images from which
        smaller random patches are extracted to blend with the MNIST digits.
        `r` is the number of distinct textures, and `m` is the size of each
        image. It is expected that `m` > `im_size`.
    im_size:
        Size of the texture MNIST images in pixels. Assumes square images.


    Here we modify the `data` and `targets` attribute of the MNIST Pytorch
    Dataset. For `data`, we add texture to the MNIST digits and add texture-only
    images. `targets` is modified to add the labels of the the texture-only
    images (in {10,...,10+r}).
    """

    def __init__(self,
                 root,
                 textures,
                 foreground='MNIST-V1',
                 im_size=(90, 90),
                 shadow_px=None,
                 train=True,
                 rng=None,
                 transform=None,
                 target_transform=None,
                 download=True,
                 match_bg_fg=False,
                 batch_size=0,
                 fg_size=None
                 ):
        if foreground not in DATASETS.keys():
            raise ValueError(
                f'{foreground} is not a valid option for foreground datasets, the options are {DATASETS.keys()}')
        if foreground == 'MNIST-V1':
            instance = MNIST(root, train, transform, target_transform, download=download)
        elif foreground == 'FashionMNIST':
            instance = FashionMNIST(root, train, transform, target_transform, download=download)
        else:
            raise NotImplementedError(f'{foreground} foregound dataset is not implemented.')
        self.data = instance.data
        self.transform = instance.transform
        self.target_transform = instance.target_transform
        self.shadow_px = shadow_px
        self.images = F.resize(self.data, size=im_size, interpolation=T.InterpolationMode.NEAREST)
        self.n_texture_only = self.images.shape[0]
        self.rng = utils.check_rng(rng)
        self.textures = textures
        self.train = train
        self.im_size = im_size
        self.match_bg_fg = match_bg_fg
        self.fg_size = fg_size
        self.images, self.masks, self.texture_fg_labels = self._blend_MNIST_with_texture()

        textures, texture_labels = self._generate_texture_only()
        self.images = torch.cat((self.images, textures))
        self.targets = torch.cat((instance.targets, texture_labels))
        if self.match_bg_fg:
            if batch_size == 0:
                raise ValueError(
                    "batch size needs to be passed with a value larger than zero if `match_bg_fg` is set to true")
            self.batch_size = batch_size
            self.batch_counter = 0
            self.target_texture = []
            self.index_sampler = []

    def _blend_MNIST_with_texture(self):
        masks = torch.zeros(self.images.shape, dtype=torch.uint8)
        images = torch.zeros(self.images.shape, dtype=torch.uint8)
        texture_fg_labels = torch.zeros(len(self.images), dtype=torch.uint8)
        for i in range(len(self.images)):
            i_texture = self.rng.choice(len(self.textures))
            texture_fg_labels[i] = i_texture
            images[i], masks[i] = fuse(
                self.textures[i_texture], self.images[i], rng=self.rng, shadow_px=self.shadow_px, fg_size=self.fg_size)
        return images, masks, texture_fg_labels

    def _generate_texture_only(self):
        im_size = self.images.shape[1]
        textures = torch.zeros(
            (self.n_texture_only, im_size, im_size),
            dtype=torch.uint8
        )
        texture_labels = torch.zeros(self.n_texture_only, dtype=torch.int8)
        for i in range(self.n_texture_only):
            if self.match_bg_fg:
                idx = self.texture_fg_labels[i].item()
            else:
                idx = self.rng.choice(len(self.textures))
            textures[i] = extract_patch(self.textures[idx], im_size, self.rng)
            texture_labels[i] = idx + 10
        return textures, texture_labels

    def __getitem__(self, index: int):

        if self.targets[index] <= 9:
            mask = (F.resize(self.data[index][None, ::], self.im_size,
                             interpolation=T.InterpolationMode.NEAREST) == 0).long()
        else:
            mask = torch.ones((1,) + self.im_size).long()
        if self.match_bg_fg and self.train:
            # ignore index. the variable that is passed. Need to ensure that each batch is composed of the same texture.
            self.batch_counter %= self.batch_size
            if self.batch_counter == 0:
                self.target_texture = self.texture_fg_labels[torch.randint(len(self.texture_fg_labels), (1,))[0]].item()
                # texture_fg_labels: texture labels for the fg images. targets: labels for fg images (like digit 1,2,
                # etc) and labels for bg images which are the same of the texture labels.
                self.index_sampler = torch.concat((torch.where(self.texture_fg_labels == self.target_texture)[0], \
                                                   torch.where(self.targets[
                                                               len(self.texture_fg_labels):] == self.target_texture + 10)[
                                                       0] + len(self.texture_fg_labels)))
            self.batch_counter += 1
            index = torch.randint(len(self.index_sampler), (1,)).item()
            index = self.index_sampler[index]

        img = self.images[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = torch.repeat_interleave(img[..., np.newaxis], 3, -1).permute(2, 0, 1)
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def __len__(self):
        return len(self.images)


class TextureMNIST_V2(MNIST):
    """ Custom Pytorch Dataset for Texture MNIST
    TODO this should be deleted.
    root:
        Where the MNIST data is downloaded and saved
    textures:
        A (r, m, m) tensor. These are the source texture images from which
        smaller random patches are extracted to blend with the MNIST digits.
        `r` is the number of distinct textures, and `m` is the size of each
        image. It is expected that `m` > `im_size`.
    im_size:
        Size of the texture MNIST images in pixels. Assumes square images.


    Here we modify the `data` and `targets` attribute of the MNIST Pytorch
    Dataset. For `data`, we add texture to the MNIST digits and add texture-only
    images. `targets` is modified to add the labels of the the texture-only
    images (in {10,...,10+r}).
    """

    def __init__(self,
                 root,
                 textures,
                 im_size=90,
                 shadow_px=None,
                 train=True,
                 rng=None,
                 transform=None,
                 target_transform=None,
                 download=True
                 ):

        super(TextureMNIST_V2, self).__init__(root, train, transform,
                                              target_transform, download=download)

        self.data = F.resize(self.mnist, size=im_size, interpolation=T.InterpolationMode.NEAREST)
        self.n_texture_only = self.data.shape[0]
        self.rng = utils.check_rng(rng)
        self.texture_sources = textures

        self._blend_MNIST_with_texture()
        self.textures, self.texture_labels = self._generate_texture_only()

    def _blend_MNIST_with_texture(self):
        self.masks = torch.zeros(self.data.shape, dtype=torch.uint8)
        self.texture_labels = torch.zeros(self.data.shape[0], dtype=torch.uint8)
        for i in range(len(self.data)):
            i_texture = self.rng.choice(len(self.texture_sources))
            self.data[i], self.masks[i] = fuse(
                self.texture_sources[i_texture], self.data[i], rng=self.rng)
            self.texture_labels[i] = i_texture

    # XXX: Fix bug with self.texture_labels
    def _generate_texture_only(self):
        im_size = self.data.shape[1]
        textures = torch.zeros(
            (self.n_texture_only, im_size, im_size),
            dtype=torch.uint8
        )
        texture_labels = torch.zeros(
            self.n_texture_only,
            dtype=self.targets.dtype
        )
        for i in range(self.n_texture_only):
            idx = self.rng.choice(len(self.texture_sources))
            textures[i] = extract_patch(self.texture_sources[idx], im_size, self.rng)
            texture_labels[i] = idx + 10
        return textures, texture_labels

    def __getitem__(self, index: int):

        img = self.data[index]
        mask = self.masks[index]
        img = torch.repeat_interleave(img[..., np.newaxis], 3, -1).permute(2, 0, 1)
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)
            mask = mask.unsqueeze(0).float()

        return img, mask

    def __len__(self):
        return len(self.data)


DATASETS = {
    'MNIST-V1': TextureForeground,
    'MNIST-V2': TextureMNIST_V2,
    'FashionMNIST': TextureForeground
}
