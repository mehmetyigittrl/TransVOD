# ------------------------------------------------------------------------
# Single-frame inference for Deformable DETR / TransVOD baseline.
#
# Edit the CONFIG block below, then:  python inference.py
# Writes an annotated mp4 next to the input video.
# ------------------------------------------------------------------------

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import datasets.transforms_single as T
from main import get_args_parser, load_config_file
from models import build_model
from util.misc import nested_tensor_from_tensor_list


# ============================== CONFIG ==================================
CONFIG = 'configs/custom_small.yaml'
CHECKPOINT = 'exps/our_models/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth'
VIDEO = '531307120-b85b661c-319c-4de3-96c0-833c5a53c01c.mp4'
BATCH_SIZE = 2
THRESHOLD = 0.8
DEVICE = 'cuda'                 # 'cuda' or 'cpu' ('' = use config file value)
INPUT_RESOLUTION = None         # short-side; None = use val default (800 coco / 600 vid_single)
MAX_SIZE = None                 # long-side cap; None = use val default (1333 coco / 1000 vid_single)
# ========================================================================


def build_model_args():
    """Build the same args namespace main.py builds, but with no CLI.

    Starts from get_args_parser() defaults and overlays the YAML config so
    build_model() sees an identically-shaped namespace.
    """
    parser = get_args_parser()
    args = parser.parse_args([])

    cfg = load_config_file(CONFIG)
    valid_dests = {a.dest for a in parser._actions}
    unknown = [k for k in cfg.keys() if k not in valid_dests]
    if unknown:
        raise ValueError(f"Unknown keys in config {CONFIG}: {unknown}")
    for k, v in cfg.items():
        setattr(args, k, v)

    if DEVICE:
        args.device = DEVICE
    args.distributed = False
    return args


def build_val_transforms(args):
    # Matches datasets/coco.py::make_coco_transforms('val') for dataset_file='coco'
    # and datasets/vid_single.py::make_coco_transforms('val') for vid_single.
    if args.dataset_file == 'vid_single':
        default_size, default_max = 600, 1000
    else:
        default_size, default_max = 800, 1333

    size = INPUT_RESOLUTION if INPUT_RESOLUTION is not None else default_size
    max_size = MAX_SIZE if MAX_SIZE is not None else default_max

    return T.Compose([
        T.RandomResize([size], max_size=max_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _warn_config_checkpoint_mismatch(args, skipped_keys, unexpected_keys):
    """Raise a loud, actionable warning when the YAML config and the checkpoint
    describe different architectures. We key off the state_dict diff produced
    by load_checkpoint() — the same signals that cause the real failure modes.
    """
    reasons = []  # (category, detail, remedy)

    # class_embed head shape mismatch => num_classes disagrees with checkpoint.
    cls_keys = [k for k in skipped_keys if 'class_embed' in k]
    if cls_keys:
        ckpt_n = _extract_first_dim(skipped_keys, 'class_embed', index_in_ckpt=True)
        model_n = _extract_first_dim(skipped_keys, 'class_embed', index_in_ckpt=False)
        reasons.append((
            'num_classes',
            f'config num_classes={args.num_classes} (model head={model_n}) '
            f'but checkpoint head expects {ckpt_n} classes',
            f"set num_classes: {ckpt_n} in {CONFIG} "
            f"(or fine-tune on your {args.num_classes}-class data first)",
        ))

    # query_embed shape mismatch => num_queries disagrees.
    if any('query_embed' in k for k in skipped_keys):
        ckpt_n = _extract_first_dim(skipped_keys, 'query_embed', index_in_ckpt=True)
        model_n = _extract_first_dim(skipped_keys, 'query_embed', index_in_ckpt=False)
        reasons.append((
            'num_queries',
            f'config num_queries={args.num_queries} (model={model_n}) '
            f'but checkpoint was trained with {ckpt_n} queries',
            f"set num_queries: {ckpt_n} in {CONFIG}",
        ))

    # input_proj.*.weight shape mismatch => num_feature_levels or dilation mismatch.
    if any(k.startswith('input_proj.') for k in skipped_keys):
        reasons.append((
            'backbone / feature levels',
            'input_proj.* shape differs — num_feature_levels and/or dilation '
            'do not match the checkpoint',
            f"check backbone/dilation/num_feature_levels in {CONFIG} against "
            "the checkpoint's training recipe",
        ))

    # level_embed + attention shapes scale with num_feature_levels.
    if any('level_embed' in k or 'sampling_offsets' in k or 'attention_weights' in k
           for k in skipped_keys):
        reasons.append((
            'num_feature_levels',
            'transformer attention shapes differ — num_feature_levels does not '
            'match the checkpoint',
            f"set num_feature_levels to match the checkpoint in {CONFIG}",
        ))

    # transformer.* in unexpected_keys => checkpoint has components the current model lacks
    # (e.g. temporal layers from vid_multi checkpoint loaded into single-frame).
    temporal_unexpected = [k for k in unexpected_keys if 'temp' in k.lower()]
    if temporal_unexpected:
        reasons.append((
            'single vs multi-frame',
            'checkpoint contains temporal parameters but this inference script '
            'builds a single-frame model',
            'use a single-frame (vid_single / coco) checkpoint, '
            'or run with the multi-frame pipeline',
        ))

    if not reasons:
        return

    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for r in reasons:
        if r[0] not in seen:
            seen.add(r[0])
            unique.append(r)

    bar = '!' * 78
    print()
    print(bar)
    print('!! WARNING: config / checkpoint architecture mismatch')
    print('!! The checkpoint could not be fully loaded. Parts of the model are')
    print('!! randomly initialized, so predictions on this video WILL NOT be meaningful.')
    print(bar)
    for i, (cat, detail, remedy) in enumerate(unique, 1):
        print(f"  {i}. [{cat}] {detail}")
        print(f"     fix: {remedy}")
    print(bar)
    print()


def _extract_first_dim(skipped_keys, name, index_in_ckpt):
    """Pull the first dim from the 'ckpt (A, B) vs model (C, D)' formatted lines.
    index_in_ckpt=True returns A (checkpoint), False returns C (current model).
    Returns '?' if not parseable.
    """
    import re
    for line in skipped_keys:
        if name not in line:
            continue
        m = re.search(r'ckpt \((\d+)[,)].*?model \((\d+)[,)]', line)
        if m:
            return m.group(1) if index_in_ckpt else m.group(2)
    return '?'


def load_checkpoint(model, checkpoint_path, device, args):
    """Load a checkpoint tolerantly: skip keys whose shape disagrees with the
    current model (e.g. num_classes / num_queries differ from the pretrained
    weights). Mirrors the coco_pretrain branch in main.py:305-320.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    src = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in src.items():
        target = model_sd.get(k)
        if target is None:
            continue
        if hasattr(target, 'shape') and target.shape != v.shape:
            skipped.append(f"{k}: ckpt {tuple(v.shape)} vs model {tuple(target.shape)}")
            continue
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    unexpected = [k for k in unexpected if not (k.endswith('total_params') or k.endswith('total_ops'))]
    if skipped:
        print(f"Skipped {len(skipped)} shape-mismatched keys (will stay randomly initialized):")
        for line in skipped:
            print(f"  - {line}")
    if missing:
        print(f"Missing keys: {missing}")
    if unexpected:
        print(f"Unexpected keys: {unexpected}")

    _warn_config_checkpoint_mismatch(args, skipped, unexpected)

    model.to(device)
    model.eval()
    return model


def preprocess_frame(bgr_frame, transform):
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    tensor, _ = transform(pil, None)
    return tensor


def _color_for_label(label_id):
    rng = np.random.default_rng(int(label_id) * 2654435761 % (2**32))
    c = rng.integers(64, 256, size=3)
    return int(c[0]), int(c[1]), int(c[2])


def draw_detections(bgr_frame, boxes, scores, labels, threshold):
    for box, score, label in zip(boxes, scores, labels):
        s = float(score)
        if s < threshold:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
        lid = int(label)
        color = _color_for_label(lid)
        cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), color, 2)
        text = f"{lid}:{s:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_text_top = max(0, y1 - th - baseline - 2)
        cv2.rectangle(bgr_frame,
                      (x1, y_text_top),
                      (x1 + tw + 2, y_text_top + th + baseline + 2),
                      color, -1)
        cv2.putText(bgr_frame, text, (x1 + 1, y_text_top + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return bgr_frame


@torch.no_grad()
def run_batch(model, postprocessors, frames_bgr, transform, device):
    tensors = [preprocess_frame(f, transform) for f in frames_bgr]
    samples = nested_tensor_from_tensor_list(tensors).to(device)
    outputs = model(samples)

    h, w = frames_bgr[0].shape[:2]
    orig_target_sizes = torch.as_tensor(
        [[h, w]] * len(frames_bgr), dtype=torch.int64, device=device
    )
    return postprocessors['bbox'](outputs, orig_target_sizes)


def run_inference():
    args = build_model_args()

    if args.dataset_file not in ('coco', 'vid_single'):
        raise ValueError(
            f"inference.py supports single-frame models only "
            f"(dataset_file in {{coco, vid_single}}); got '{args.dataset_file}'."
        )

    video_path = Path(VIDEO)
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Dataset file (model branch): {args.dataset_file}")

    model, _criterion, postprocessors = build_model(args)
    model = load_checkpoint(model, CHECKPOINT, device, args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    transform = build_val_transforms(args)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path.name}  {width}x{height} @ {fps:.2f} fps  ~{total_frames} frames")

    out_path = video_path.with_name(f"{video_path.stem}_inference.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video writer: {out_path}")

    print(f"Writing: {out_path}")
    print(f"Batch size: {BATCH_SIZE}  Threshold: {THRESHOLD}")

    batch_frames = []
    processed = 0
    t0 = time.time()

    def flush(batch):
        nonlocal processed
        if not batch:
            return
        results = run_batch(model, postprocessors, batch, transform, device)
        for f, r in zip(batch, results):
            annotated = draw_detections(f, r['boxes'], r['scores'], r['labels'], THRESHOLD)
            writer.write(annotated)
        processed += len(batch)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            batch_frames.append(frame)
            if len(batch_frames) == BATCH_SIZE:
                flush(batch_frames)
                batch_frames = []
                if processed % (BATCH_SIZE * 10) == 0:
                    elapsed = time.time() - t0
                    fps_eff = processed / elapsed if elapsed > 0 else 0.0
                    print(f"  processed {processed} frames ({fps_eff:.1f} fps)")

        flush(batch_frames)
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t0
    fps_eff = processed / elapsed if elapsed > 0 else 0.0
    print(f"Done. {processed} frames in {elapsed:.1f}s ({fps_eff:.1f} fps) -> {out_path}")


if __name__ == '__main__':
    run_inference()
