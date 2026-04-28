# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
from numpy import random as rand
from PIL import Image
import cv2

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


# ---------------------------------------------------------------------------
# Photometric distortion (B5) - ported from datasets/transforms_multi.py.
# Inner ops operate on np.float32 arrays in BGR-ordered channels (matches the
# multi-frame pipeline's convention). The PhotometricDistort wrapper at the
# bottom converts PIL <-> np for a single image.
# ---------------------------------------------------------------------------

class _NumpyContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        assert upper >= lower and lower >= 0
        self.lower, self.upper = lower, upper

    def __call__(self, image, target):
        if rand.randint(2):
            image = image * rand.uniform(self.lower, self.upper)
        return image, target


class _NumpyBrightness(object):
    def __init__(self, delta=32):
        assert 0.0 <= delta <= 255.0
        self.delta = delta

    def __call__(self, image, target):
        if rand.randint(2):
            image = image + rand.uniform(-self.delta, self.delta)
        return image, target


class _NumpySaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        assert upper >= lower and lower >= 0
        self.lower, self.upper = lower, upper

    def __call__(self, image, target):
        if rand.randint(2):
            image[:, :, 1] = image[:, :, 1] * rand.uniform(self.lower, self.upper)
        return image, target


class _NumpyHue(object):
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image, target):
        if rand.randint(2):
            image[:, :, 0] = image[:, :, 0] + rand.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, target


class _SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        return image[:, :, self.swaps]


class _RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, target):
        if rand.randint(2):
            swap = self.perms[rand.randint(len(self.perms))]
            image = _SwapChannels(swap)(image)
        return image, target


class _ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform, self.current = transform, current

    def __call__(self, image, target):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, target


class _ComposeNumpy(object):
    """Local mini-Compose for numpy-array transforms used inside PhotometricDistort."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class PhotometricDistort(object):
    """Single-image photometric jitter. PIL in -> PIL out; target unchanged."""
    def __init__(self):
        self.pd = [
            _NumpyContrast(),
            _ConvertColor(transform='HSV'),
            _NumpySaturation(),
            _NumpyHue(),
            _ConvertColor(current='HSV', transform='BGR'),
            _NumpyContrast(),
        ]
        self.rand_brightness = _NumpyBrightness()
        self.rand_light_noise = _RandomLightingNoise()

    def __call__(self, img, target):
        arr = np.asarray(img).astype('float32')
        arr, target = self.rand_brightness(arr, target)
        if rand.randint(2):
            distort = _ComposeNumpy(self.pd[:-1])
        else:
            distort = _ComposeNumpy(self.pd[1:])
        arr, target = distort(arr, target)
        arr, target = self.rand_light_noise(arr, target)
        arr = np.clip(arr, 0, 255).astype('uint8')
        return Image.fromarray(arr), target


# ---------------------------------------------------------------------------
# Mosaic (B2) and CopyPaste (B3).
#
# Both transforms need access to additional samples beyond the one currently
# being processed. The dataset wrapper (MosaicCocoDetection in
# datasets/coco.py) injects those siblings into the target dict under the
# private keys '_mosaic_siblings' and '_copy_paste_donor', then strips them
# back out before the rest of the pipeline runs. Keeping the sibling-fetching
# concern inside the dataset wrapper means these classes stay pure transforms.
# ---------------------------------------------------------------------------

def _filter_zero_area(target, fields=("labels", "area", "iscrowd", "boxes")):
    if "boxes" not in target or target["boxes"].numel() == 0:
        return target
    boxes = target["boxes"]
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    out = target.copy()
    for f in fields:
        if f in out and out[f].numel() > 0:
            out[f] = out[f][keep]
    return out


def _resize_pil_and_target(img, target, new_w, new_h):
    """Resize a single (PIL, target) to exact (new_w, new_h)."""
    ow, oh = img.size
    img = img.resize((new_w, new_h), Image.BILINEAR)
    target = target.copy()
    if "boxes" in target and target["boxes"].numel() > 0:
        rw, rh = new_w / float(ow), new_h / float(oh)
        scale = torch.as_tensor([rw, rh, rw, rh], dtype=torch.float32)
        target["boxes"] = target["boxes"] * scale
    if "area" in target and target["area"].numel() > 0:
        rw, rh = new_w / float(ow), new_h / float(oh)
        target["area"] = target["area"] * (rw * rh)
    target["size"] = torch.tensor([new_h, new_w])
    return img, target


class Mosaic(object):
    """Tiles 4 images on a 2x2 canvas of side `output_size`. Quadrant size
    is `output_size // 2`. Each sample is resized to the quadrant size, then
    boxes are translated by the quadrant offset. Final image is returned at
    `output_size x output_size`. Probability `p` controls whether mosaic
    fires; when it doesn't, the base sample is returned unchanged.

    The base call signature stays `(img, target)` for compatibility with
    `Compose`; the three sibling samples are read from `target['_mosaic_siblings']`,
    a list of `(PIL.Image, dict)` injected by the dataset wrapper.
    """
    def __init__(self, output_size=640, p=0.5):
        self.output_size = int(output_size)
        self.p = p

    def __call__(self, img, target):
        siblings = target.pop('_mosaic_siblings', None) if isinstance(target, dict) else None
        if siblings is None or len(siblings) < 3 or random.random() >= self.p:
            return img, target

        q = self.output_size // 2
        canvas = Image.new('RGB', (self.output_size, self.output_size))
        all_boxes, all_labels, all_areas, all_iscrowd = [], [], [], []

        # Quadrant offsets: top-left, top-right, bottom-left, bottom-right.
        quadrants = [(0, 0), (q, 0), (0, q), (q, q)]
        samples = [(img, target)] + list(siblings)[:3]

        for (off_x, off_y), (qimg, qtgt) in zip(quadrants, samples):
            qimg, qtgt = _resize_pil_and_target(qimg, qtgt, q, q)
            canvas.paste(qimg, (off_x, off_y))
            if "boxes" in qtgt and qtgt["boxes"].numel() > 0:
                shift = torch.as_tensor([off_x, off_y, off_x, off_y], dtype=torch.float32)
                all_boxes.append(qtgt["boxes"] + shift)
                all_labels.append(qtgt["labels"])
                if "area" in qtgt and qtgt["area"].numel() > 0:
                    all_areas.append(qtgt["area"])
                else:
                    all_areas.append(torch.zeros(qtgt["boxes"].shape[0]))
                if "iscrowd" in qtgt and qtgt["iscrowd"].numel() > 0:
                    all_iscrowd.append(qtgt["iscrowd"])
                else:
                    all_iscrowd.append(torch.zeros(qtgt["boxes"].shape[0], dtype=torch.int64))

        merged = target.copy() if isinstance(target, dict) else {}
        merged["size"] = torch.tensor([self.output_size, self.output_size])
        merged["orig_size"] = merged.get("orig_size", torch.tensor([self.output_size, self.output_size]))
        if all_boxes:
            merged["boxes"] = torch.cat(all_boxes, dim=0)
            merged["labels"] = torch.cat(all_labels, dim=0)
            merged["area"] = torch.cat(all_areas, dim=0)
            merged["iscrowd"] = torch.cat(all_iscrowd, dim=0)
        else:
            merged["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            merged["labels"] = torch.zeros((0,), dtype=torch.int64)
            merged["area"] = torch.zeros((0,), dtype=torch.float32)
            merged["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        merged = _filter_zero_area(merged)
        return canvas, merged


class CopyPaste(object):
    """Pastes up to `n_paste` random object crops from a donor sample onto
    the base image at random locations. Bbox-only (no masks) - safe for COCO
    detection. Donor is read from `target['_copy_paste_donor']`, a tuple
    `(PIL.Image, dict)` injected by the dataset wrapper.

    Pasted instances are appended to the target's boxes/labels/area/iscrowd.
    Random rescale in `scale_range`. Pasted patches are clipped to the base
    image bounds; if the resulting box has zero area, the instance is dropped.
    """
    def __init__(self, p=0.3, n_paste=3, scale_range=(0.5, 1.5)):
        self.p = p
        self.n_paste = int(n_paste)
        self.scale_range = scale_range

    def __call__(self, img, target):
        donor = target.pop('_copy_paste_donor', None) if isinstance(target, dict) else None
        if donor is None or random.random() >= self.p:
            return img, target

        donor_img, donor_tgt = donor
        if "boxes" not in donor_tgt or donor_tgt["boxes"].numel() == 0:
            return img, target

        base_arr = np.array(img.convert('RGB'))
        H, W = base_arr.shape[:2]
        donor_arr = np.array(donor_img.convert('RGB'))
        DH, DW = donor_arr.shape[:2]

        donor_boxes = donor_tgt["boxes"].numpy()
        donor_labels = donor_tgt["labels"].numpy()
        n_avail = donor_boxes.shape[0]
        n_to_paste = min(self.n_paste, n_avail)
        idxs = np.random.permutation(n_avail)[:n_to_paste]

        new_boxes, new_labels, new_areas = [], [], []
        for i in idxs:
            x1, y1, x2, y2 = donor_boxes[i]
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(DW, x2)), int(min(DH, y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            patch = donor_arr[y1:y2, x1:x2].copy()
            ph, pw = patch.shape[:2]

            scale = float(rand.uniform(self.scale_range[0], self.scale_range[1]))
            new_w, new_h = max(1, int(pw * scale)), max(1, int(ph * scale))
            if new_w >= W or new_h >= H:
                continue
            patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            px = int(rand.randint(0, max(1, W - new_w)))
            py = int(rand.randint(0, max(1, H - new_h)))
            base_arr[py:py + new_h, px:px + new_w] = patch
            new_boxes.append([px, py, px + new_w, py + new_h])
            new_labels.append(int(donor_labels[i]))
            new_areas.append(float(new_w * new_h))

        if not new_boxes:
            return img, target

        merged = target.copy()
        nb = torch.as_tensor(new_boxes, dtype=torch.float32)
        nl = torch.as_tensor(new_labels, dtype=torch.int64)
        na = torch.as_tensor(new_areas, dtype=torch.float32)
        ni = torch.zeros((len(new_boxes),), dtype=torch.int64)
        if "boxes" in merged and merged["boxes"].numel() > 0:
            merged["boxes"] = torch.cat([merged["boxes"], nb], dim=0)
            merged["labels"] = torch.cat([merged["labels"], nl], dim=0)
            merged["area"] = torch.cat([merged.get("area", torch.zeros(0)), na], dim=0)
            merged["iscrowd"] = torch.cat([merged.get("iscrowd", torch.zeros(0, dtype=torch.int64)), ni], dim=0)
        else:
            merged["boxes"] = nb
            merged["labels"] = nl
            merged["area"] = na
            merged["iscrowd"] = ni

        return Image.fromarray(base_arr), merged
