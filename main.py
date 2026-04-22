# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets

import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
from util import mlflow_logger


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr_drop', default=5, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    parser.add_argument('--num_ref_frames', default=3, type=int, help='number of reference frames')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')


    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--n_temporal_decoder_layers', default=1, type=int)
    parser.add_argument('--interval1', default=20, type=int)
    parser.add_argument('--interval2', default=60, type=int)

    parser.add_argument("--fixed_pretrained_model", default=False, action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='vid_multi')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--vid_path', default='./data/vid', type=str)
    parser.add_argument('--coco_pretrain', default=False, action='store_true')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--num_classes', default=31, type=int,
                        help='number of object classes (max category_id + 1 for COCO)')
    parser.add_argument('--config', default='configs/custom_coco.yaml', type=str,
                        help='Path to a YAML or JSON config file. Values override argparse defaults; explicit CLI flags still win. '
                             'Defaults to configs/custom_coco.yaml so `python main.py` runs with no args.')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--max_iters', default=0, type=int,
                        help='If > 0, cap each training epoch to this many iterations. 0 = full epoch.')
    parser.add_argument('--num_workers', default=0, type=int)

    # MLflow tracking
    parser.add_argument('--mlflow_enabled', default=False, action='store_true',
                        help='Enable MLflow logging of params, metrics, and artifacts.')
    parser.add_argument('--mlflow_tracking_uri', default='http://127.0.0.1:5000', type=str,
                        help='URI of the MLflow tracking server.')
    parser.add_argument('--mlflow_experiment_name', default='TransVOD', type=str,
                        help='MLflow experiment name.')
    parser.add_argument('--mlflow_run_name', default='', type=str,
                        help='Optional MLflow run name. Defaults to the output_dir basename.')
    parser.add_argument('--mlflow_log_every', default=50, type=int,
                        help='Log per-iteration training loss every N steps.')
    parser.add_argument('--mlflow_log_checkpoint', default=False, action='store_true',
                        help='Upload the final checkpoint as an MLflow artifact (expensive, ~400 MB).')
    parser.add_argument('--mlflow_log_system_metrics', default=False, action='store_true',
                        help='Enable MLflow system-metrics logging (CPU/RAM/GPU util/mem/power). '
                             'Requires psutil and pynvml (or nvidia-ml-py).')
    parser.add_argument('--mlflow_system_metrics_interval', default=10.0, type=float,
                        help='Seconds between system-metrics samples.')
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    return parser


def main(args):
    print(args.dataset_file, 11111111)
    if args.dataset_file in ("vid_single", "coco"):
        from engine_single import evaluate, train_one_epoch
        import util.misc as utils

    else:
        from engine_multi import evaluate, train_one_epoch
        import util.misc_multi as utils

    print(args.dataset_file)
    device = torch.device(args.device)
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    # MLflow init happens from rank 0 only. For non-distributed runs this is
    # the sole process; distributed workers skip all logging calls since
    # init_mlflow returns False there.
    if utils.is_main_process():
        run_name = args.mlflow_run_name or (Path(args.output_dir).name if args.output_dir else None)
        mlflow_logger.init_mlflow(
            enabled=args.mlflow_enabled,
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment_name,
            run_name=run_name,
            tags={'dataset_file': args.dataset_file, 'backbone': args.backbone,
                  'mode': 'eval' if args.eval else 'train'},
            log_system_metrics=args.mlflow_log_system_metrics,
            system_metrics_interval=args.mlflow_system_metrics_interval,
        )
        mlflow_logger.log_params(vars(args))
        if args.config:
            mlflow_logger.log_artifact(args.config, artifact_path='config')


    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    train_image_set = 'train' if args.dataset_file == 'coco' else 'train_joint'
    dataset_train = build_dataset(image_set=train_image_set, args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    print(args.lr_drop_epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_drop_epochs)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu', weights_only=False)
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)

        if args.eval:
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        else:
            tmp_dict = model_without_ddp.state_dict().copy()
            if args.coco_pretrain: # singleBaseline
                for k, v in checkpoint['model'].items():
                    if ('class_embed' not in k) :
                        tmp_dict[k] = v 
                    else:
                        print('k', k)
            else:
                tmp_dict = checkpoint['model']
                # Only freeze non-temporal params for the TransVOD multi-frame recipe.
                # Single-frame models have no `temp.*` params, so this freeze used to
                # zero requires_grad for the whole model and break training.
                if args.dataset_file == 'vid_multi':
                    for name, param in model_without_ddp.named_parameters():
                        if ('temp' in name):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(tmp_dict, strict=False)

        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
    try:
        if args.eval:
            test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                                  data_loader_val, base_ds, device, args.output_dir)
            if args.output_dir:
                utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            if utils.is_main_process():
                _log_eval_metrics_to_mlflow(test_stats, step=0)
            return

        print("Start training")
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm,
                max_iters=args.max_iters, mlflow_log_every=args.mlflow_log_every)
            lr_scheduler.step()
            print('args.output_dir', args.output_dir)
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 5 epochs
                # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
                if (epoch + 1) % 1 == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            if utils.is_main_process():
                mlflow_logger.log_metrics(
                    {f'epoch/{k}': v for k, v in train_stats.items()
                     if isinstance(v, (int, float))},
                    step=epoch,
                )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        if utils.is_main_process() and args.output_dir:
            log_txt = output_dir / 'log.txt'
            if log_txt.exists():
                mlflow_logger.log_artifact(str(log_txt))
            if args.mlflow_log_checkpoint:
                final_ckpt = output_dir / 'checkpoint.pth'
                if final_ckpt.exists():
                    mlflow_logger.log_artifact(str(final_ckpt), artifact_path='checkpoints')
    finally:
        if utils.is_main_process():
            mlflow_logger.end_run()


def _log_eval_metrics_to_mlflow(test_stats, step):
    """Flatten COCO evaluator stats and push to MLflow under eval/*."""
    flat = {}
    for k, v in test_stats.items():
        if isinstance(v, (int, float)):
            flat[f'eval/{k}'] = v
        elif isinstance(v, (list, tuple)) and k == 'coco_eval_bbox' and len(v) >= 12:
            # Standard COCO 12-value bbox stats ordering.
            names = ['AP', 'AP50', 'AP75', 'AP_small', 'AP_medium', 'AP_large',
                     'AR_1', 'AR_10', 'AR_100', 'AR_small', 'AR_medium', 'AR_large']
            for name, val in zip(names, v[:12]):
                flat[f'eval/{name}'] = float(val)
    mlflow_logger.log_metrics(flat, step=step)


def load_config_file(path):
    import json as _json
    import os as _os
    ext = _os.path.splitext(path)[1].lower()
    with open(path, 'r') as f:
        if ext in ('.yaml', '.yml'):
            try:
                import yaml
            except ImportError as e:
                raise ImportError("PyYAML is required for YAML configs. Install with `pip install pyyaml`.") from e
            data = yaml.safe_load(f)
        elif ext == '.json':
            data = _json.load(f)
        else:
            raise ValueError(f"Unsupported config extension '{ext}'. Use .yaml, .yml, or .json.")
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a top-level mapping.")
    return data


if __name__ == '__main__':
    # Two-pass parsing so file-provided values override argparse defaults
    # but explicit command-line flags still win.
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', default='configs/custom_coco.yaml', type=str)
    pre_args, _remaining = pre_parser.parse_known_args()

    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])

    if pre_args.config:
        cfg = load_config_file(pre_args.config)
        valid_dests = {a.dest for a in parser._actions}
        unknown = [k for k in cfg.keys() if k not in valid_dests]
        if unknown:
            raise ValueError(f"Unknown keys in config {pre_args.config}: {unknown}")
        parser.set_defaults(**cfg)

    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
