# TransVOD port — session handoff

## TL;DR

Single-frame Deformable DETR on COCO 2017 is verified working on
**Python 3.13 + PyTorch 2.10 + CUDA 13.0 + RTX 5060**.

- **Eval** on `val2017` reproduces the published baseline: **AP 0.414 / AP50 0.618**.
- **Train** loop runs cleanly; loss descends, no NaNs, ~2:40/epoch at `batch_size=2`.

Just hit **F5** in VSCode on `main.py` — defaults to `configs/custom_coco.yaml`.
Or run `python main.py`.

## Pipeline modes

Toggle `eval:` in `configs/custom_coco.yaml`:

- `eval: true`  → run COCO evaluator on `val2017` and exit.
- `eval: false` → run training loop for `epochs: 50`.

For a short training smoke-test use `configs/smoke_test_train.yaml`.

## What was changed vs. upstream

### Python fixes for modern torchvision (0.25+)

- `util/misc.py` — removed the `float(torchvision.__version__[:3]) < 0.5`
  version check which mis-parses "0.25.0" as 0.2 and imports long-removed
  symbols (`_NewEmptyTensorOp`, `_new_empty_tensor`). `interpolate()` now
  calls native `torch.nn.functional.interpolate`.
- `util/misc_multi.py` — same treatment.

### CUDA extension fixes for PyTorch 2.x API (MSDeformAttn)

Based on GroundingDINO PR #415:

- `models/ops/src/ms_deform_attn.h` — `.type().is_cuda()` → `.is_cuda()` (×2).
- `models/ops/src/cuda/ms_deform_attn_cuda.cu`
  - `.type().is_cuda()` → `.is_cuda()` on 11 `AT_ASSERTM` guards.
  - `AT_DISPATCH_FLOATING_TYPES(value.type(), ...)` → `value.scalar_type()` (×2).
- Rebuilt extension — produces a Python 3.13 `.so` in `models/ops/`.

### numpy 2.x compat

- `datasets/coco_video_parser.py` and `datasets/parsers/coco_video_parser.py`:
  `np.int` → `np.int64`. Only used by the VID multi-frame pipeline, but
  prevents import failures when switching `dataset_file` later.

### Config-file support in `main.py`

- New `--config` CLI flag that loads a YAML/JSON file of defaults. Default
  value is `configs/custom_coco.yaml` so zero-arg runs just work.
- Two-pass parse: file values override argparse defaults, explicit CLI flags
  still win.
- Routes `dataset_file: coco` to the single-frame engine (`engine_single`)
  and picks the `train` image_set (vs. VID-specific `train_joint`).

### Checkpoint loading

- `torch.load(..., weights_only=False)` at the two `main.py` call sites. The
  checkpoint embeds `argparse.Namespace` which can't be loaded under
  PyTorch 2.6+'s default `weights_only=True`.

### Wiring

- `models/__init__.py` — routes `dataset_file == "coco"` to `build_single`.
- `models/deformable_detr_single.py` — `num_classes` now reads from
  `args.num_classes` (defaults to 31 for back-compat).
- `.gitignore` — ignores `data/`, `exps/`, `.pth`, build/editor junk.
- `.vscode/launch.json` + `main.py`'s `--config` default → F5 runs with zero
  args.

## Gotchas to remember

1. **`with_box_refine` must match the checkpoint.** The shipped
   `r50_deformable_detr_single_scale_dc5-checkpoint.pth` was trained with
   `with_box_refine=False`. Enabling box-refine at inference time with that
   checkpoint produces mAP ≈ 0 (the decoder math path differs). Keep the
   config flag aligned with `ckpt['args'].with_box_refine`.

2. **`num_classes: 91` for COCO 2017** — category IDs are sparse (1–90 with
   gaps). 91 covers all sparse IDs plus the "no-object" slot.

3. **`coco_pretrain` semantics:**
   - `true`  → strip `class_embed.*` on load (use when num_classes differs).
   - `false` → load full state_dict including the classifier (use for plain
     COCO eval/train where num_classes matches).

4. **The `.so` is now `cpython-313` not `cpython-37m`.** Do not reuse the
   old Python 3.7 `.so`. To rebuild:
   ```bash
   cd models/ops
   rm -rf build MultiScaleDeformableAttention*.so MultiScaleDeformableAttention.egg-info
   TORCH_CUDA_ARCH_LIST='12.0' python setup.py build_ext --inplace
   ```

## Quick verification commands

```bash
conda activate transvod

# Eval (~5 min on val2017):
python -u main.py                                           # defaults to eval: true

# Quick training smoke-test (kill with Ctrl+C after a minute of iters):
python -u main.py --config configs/smoke_test_train.yaml
```

## Switching to a custom COCO-format dataset

1. Arrange data as `your_data/{train2017,val2017}/` and
   `your_data/annotations/instances_{train,val}2017.json` (folder names are
   hardcoded in `datasets/coco.py`).
2. In `configs/custom_coco.yaml`:
   - `coco_path: /path/to/your_data`
   - `num_classes: <max(category_id) + 1>`
   - `coco_pretrain: true` (reinitializes the classifier for your classes)
   - `eval: false` (run training)

## Outstanding / nice-to-haves (not blocking)

- `.data<T>()` in the CUDA source is deprecated in favor of `.data_ptr<T>()`;
  still compiles fine in PyTorch 2.10 but may need updating in future.
- `engine_single.py` has leftover print-noise (`"------...!!!!"`) that's
  harmless but ugly.
- `benchmark.py` was not touched — same `torch.load` + `_NewEmptyTensorOp`
  patterns would need the same fixes if you run it.
