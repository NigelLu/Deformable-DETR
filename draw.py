# ------------------------------------------------------------------------------------
# FSOD-TOPG Codebase (https://github.com/NigelLu/FSOD-TOPG)
# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Originated from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------

import os
import pdb
import cv2
import torch
import shutil
import random
import argparse
import numpy as np

from pathlib import Path
from einops import rearrange
from torch.utils.data import DataLoader

import util.misc as utils

from models import build_model
from datasets import build_dataset


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names',
                        default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names',
                        default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine',
                        default=False, action='store_true')
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
    parser.add_argument('--num_feature_levels', default=4,
                        type=int, help='number of feature levels')

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
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='/scratch/xl3139/Deformable-DETR/checkpoints/checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--cache_mode', default=False,
                        action='store_true', help='whether to cache images on memory')

    # * draw
    parser.add_argument('--num_to_draw', default=10, type=int)
    parser.add_argument('--save_path', default="/scratch/xl3139/Deformable-DETR/demo", type=str)
    parser.add_argument('--box_per_img', default=2, type=int)

    return parser


def main(args):

    print(args, '\n')

    assert os.path.isdir(args.coco_path), f'--coco_path expects a folder path, {args.coco_path} is not a folder'
    assert os.path.isdir(args.save_path), f'--save_path expects a folder path, {args.save_path} is not a folder'
    assert os.path.isfile(args.resume), f'--resume expects path to a .pth weight file, {args.resume} is not a file'

    if len(os.listdir(args.save_path)) > 0:
        print(f"Warning: the save path '{args.save_path}' you specified is NOT empty\nIt contains")

        dir_content = os.listdir(args.save_path)
        dir_content.sort()
        print('\n'.join(dir_content))
        should_proceed = ''
        while should_proceed != 'yes' and should_proceed != 'no':
            should_proceed = input("Would you like us to empty the directory for you and proceed? (yes or no)\n> ").lower()
            if should_proceed == 'no':
                return
            if should_proceed == 'yes':
                shutil.rmtree(args.save_path, ignore_errors=True)
                os.mkdir(args.save_path)

    device = torch.device(args.device)

    # * fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # * build model
    model, criterion, _ = build_model(args)
    model.to(device)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # * build dataset
    dataset_train = build_dataset(image_set='train', args=args)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)

    # * load model weight
    assert args.resume, "Expect --resume argument for model weight file, got None"

    print(f"Loading weight from {args.resume}")
    checkpoint = torch.load(args.resume, map_location='cpu')

    missing_keys, unexpected_keys = model.load_state_dict(
        checkpoint['model'], strict=False)
    unexpected_keys = [k for k in unexpected_keys if not (
        k.endswith('total_params') or k.endswith('total_ops'))]
    if len(missing_keys) > 0:
        print('Missing Keys: {}'.format(missing_keys))
    if len(unexpected_keys) > 0:
        print('Unexpected Keys: {}'.format(unexpected_keys))

    # * start model forwarding and drawing
    model.eval()
    criterion.eval()

    counter = 1

    for samples, targets in data_loader_train:
        img_info = dataset_train.coco.imgs[targets[0]['image_id'].item()]
        # * (1, 3, h, w)
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # img1 = rearrange(samples.tensors[0], 'c h w -> h w c').cpu().detach().numpy()
        # cv2.imwrite('/scratch/xl3139/FSOD-TOPG/hello1.jpg', img1)

        # * dict_keys(['pred_logits', 'pred_boxes', 'aux_outputs'])
        # * (1, 300, 21) -- (1, 300, 4)
        outputs = model(samples)

        pred_logits, pred_boxes = outputs['pred_logits'], outputs['pred_boxes']

        # * (1, 300, 21) -> (1, 300, 21)
        pred_logits = torch.nn.Softmax(dim=2)(pred_logits)
        # * (1, 300, 21) -> (1, 300), both
        target_box_confidence, target_box_cls = [ele.squeeze(0) for ele in torch.max(pred_logits, dim=2)]
        _, topk_indices = torch.topk(target_box_confidence, args.box_per_img, dim=0)

        target_boxes = pred_boxes[0, topk_indices, :]

        # * drawing preparation
        h, w = img_info['height'], img_info['width']
        img = cv2.imread(
            f'{args.coco_path}/train2017/{img_info["file_name"]}')

        for box_idx in range(args.box_per_img):
            box = target_boxes[box_idx]
            x1, y1, x2, y2 = int(box[0].item()*w), int(box[1].item()*h), int(
                box[2].item()*w), int(box[3].item()*h)

            img = cv2.rectangle(img, (x1, y1), (x2, y2),
                                color=(0, 255, 0), thickness=2)
        cv2.imwrite(f'{args.save_path}/{counter}.png', img)
        print(f"Saved to {args.save_path}/{counter}.png")
        counter += 1

        if counter > int(args.num_to_draw):
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
