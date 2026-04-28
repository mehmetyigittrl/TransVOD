# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import random
from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from util.misc import get_local_rank, get_local_size
import datasets.transforms_single as T


class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def _get_prepared(self, idx):
        """Return (PIL image, target dict) without the train transform stack.
        Used by MosaicCocoDetection to fetch sibling samples cheaply.
        """
        img, anns = TvCocoDetection.__getitem__(self, idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': anns}
        img, target = self.prepare(img, target)
        return img, target


class MosaicCocoDetection(CocoDetection):
    """Wraps CocoDetection so the training-time transform stack can perform
    Mosaic and CopyPaste. Injects up to 3 sibling samples and 1 donor sample
    into the target dict under private keys; the Mosaic / CopyPaste transforms
    pop those keys and use them.

    Probabilities `mosaic_prob` and `copy_paste_prob` are checked here so we
    don't waste decode cost when the transforms won't fire.
    """
    def __init__(self, img_folder, ann_file, transforms, return_masks,
                 mosaic_prob=0.0, copy_paste_prob=0.0,
                 cache_mode=False, local_rank=0, local_size=1):
        super().__init__(img_folder, ann_file, transforms, return_masks,
                         cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.mosaic_prob = float(mosaic_prob)
        self.copy_paste_prob = float(copy_paste_prob)

    def _random_sibling(self, exclude_idx):
        n = len(self.ids)
        if n <= 1:
            return self._get_prepared(exclude_idx)
        j = random.randrange(n)
        if j == exclude_idx:
            j = (j + 1) % n
        return self._get_prepared(j)

    def __getitem__(self, idx):
        img, anns = TvCocoDetection.__getitem__(self, idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': anns}
        img, target = self.prepare(img, target)

        if self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            target['_mosaic_siblings'] = [self._random_sibling(idx) for _ in range(3)]
        if self.copy_paste_prob > 0 and random.random() < self.copy_paste_prob:
            target['_copy_paste_donor'] = self._random_sibling(idx)

        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if isinstance(target, dict):
            target.pop('_mosaic_siblings', None)
            target.pop('_copy_paste_donor', None)
        return img, target


def build_small_object_sampler(dataset, threshold=32 * 32, repeat=3.0):
    """Returns a torch.utils.data.WeightedRandomSampler that oversamples
    images containing at least one annotation with `area < threshold` by
    factor `repeat` (relative to the rest). Walks `dataset.coco.anns` once
    so it is O(num_annotations) at startup, free at sample time.
    """
    coco = dataset.coco
    flagged_image_ids = set()
    for ann in coco.anns.values():
        if ann.get('iscrowd', 0):
            continue
        if ann.get('area', 0) < threshold:
            flagged_image_ids.add(ann['image_id'])

    weights = torch.ones(len(dataset), dtype=torch.double)
    for i, image_id in enumerate(dataset.ids):
        if image_id in flagged_image_ids:
            weights[i] = float(repeat)

    n_flagged = int((weights > 1.0).sum().item())
    print(f"[small-object sampler] flagged {n_flagged}/{len(dataset)} "
          f"images (area < {threshold}px^2), repeat={repeat}x")
    return torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(dataset), replacement=True)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, wider_scales=False, photometric=False,
                         mosaic_prob=0.0, copy_paste_prob=0.0, mosaic_size=640):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # B1: optionally extend scales above 800 so small objects see higher-res
    # input during training (matched to higher inference resolution).
    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    if wider_scales:
        scales = scales + [832, 864, 896, 928, 960, 992, 1024]
    max_size = 1600 if wider_scales else 1333

    if image_set == 'train':
        ops = [T.RandomHorizontalFlip()]
        # B5: photometric color jitter (RandomBrightness + HSV-space distort)
        if photometric:
            ops.append(T.PhotometricDistort())
        # B2/B3: Mosaic and CopyPaste read sibling/donor from target dict
        # injected by MosaicCocoDetection. They are no-ops if the keys are
        # missing or the per-call probability check fails.
        if mosaic_prob > 0:
            ops.append(T.Mosaic(output_size=mosaic_size, p=1.0))  # outer prob is handled in dataset wrapper
        if copy_paste_prob > 0:
            ops.append(T.CopyPaste(p=1.0))                         # outer prob is handled in dataset wrapper

        ops.append(
            T.RandomSelect(
                T.RandomResize(scales, max_size=max_size),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                ])
            )
        )
        ops.append(normalize)
        return T.Compose(ops)

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]

    wider_scales = bool(getattr(args, 'wider_scales', False))
    photometric = bool(getattr(args, 'photometric_distort', False))
    mosaic_prob = float(getattr(args, 'mosaic_prob', 0.0)) if image_set == 'train' else 0.0
    copy_paste_prob = float(getattr(args, 'copy_paste_prob', 0.0)) if image_set == 'train' else 0.0
    mosaic_size = int(getattr(args, 'mosaic_size', 640))

    transforms = make_coco_transforms(
        image_set,
        wider_scales=wider_scales,
        photometric=photometric and image_set == 'train',
        mosaic_prob=mosaic_prob,
        copy_paste_prob=copy_paste_prob,
        mosaic_size=mosaic_size,
    )

    use_mosaic_wrapper = image_set == 'train' and (mosaic_prob > 0 or copy_paste_prob > 0)
    if use_mosaic_wrapper:
        dataset = MosaicCocoDetection(
            img_folder, ann_file, transforms=transforms, return_masks=args.masks,
            mosaic_prob=mosaic_prob, copy_paste_prob=copy_paste_prob,
            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    else:
        dataset = CocoDetection(
            img_folder, ann_file, transforms=transforms, return_masks=args.masks,
            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())
    return dataset
