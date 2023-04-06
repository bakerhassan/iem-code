import numpy as np
import numbers
from pathlib import Path
from PIL import Image
import re
import torchvision.transforms.functional as F
import torch


def check_rng(seed):
    """Turn seed into a np.random.Generator instance

    Parameters
    ----------
    seed : None, int or instance of Generator
        If seed is None, return the Generator using the OS entropy.
        If seed is an int, return a new Generator instance seeded with seed.
        If seed is already a Generator instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None:
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        seed = np.random.SeedSequence(seed)
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.SeedSequence):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed

    raise ValueError('%r cannot be used to seed a numpy.random.Generator'
                     ' instance' % seed)


def get_id(path: Path):
    """
    Get texture id from file name

    Parameters
    ----------
    paths:
        A Path object (from pathlib) for a texture file. The file
        name is supposed to follow the pattern 'D<id><ext>', where `id` is an
        integer and `ext` is the extension or suffix.
    """

    pat = 'D(?P<id>\d+)' + path.suffix
    p = re.compile(pat)

    m = p.search(path.name)
    id = m.group('id')
    id = np.uint8(id)

    return id


def load_textures(root: Path):
    """ Load source textures

    Load the texture images that will be used as background for the MNIST
    digits.

    Args
    ----
    root: Path
        Path object of folder with texture images.

    Returns
    -------
    textures: torch.Tensor
        An list of PIL images.
    ids: list[int]
        A list with the id of each texture image.

    Notes
    -----
    The name of each texture image file is supposed to have a pattern with a
    unique numeric id. The id of the `i`-th image is saved in `ids[i]`.
    """

    textures, ids = [], []

    for path in Path(root).iterdir():
        with Image.open(path) as im:
            im = im.convert(mode='L')
            textures.append(F.pil_to_tensor(im))
            ids.append(get_id(path))

    textures = torch.cat(textures)
    ids = torch.tensor(ids)

    return textures, ids
