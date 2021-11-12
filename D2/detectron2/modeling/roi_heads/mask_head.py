# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
import copy

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, get_norm
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.structures.masks import PolygonMasks
import pycocotools.mask as mask_utils


ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")
ROI_MASK_HEAD_REGISTRY.__doc__ = """
Registry for mask heads, which predicts instance masks given
per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""
    
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge_flat(logits, labels, reduction='mean'):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    logits, labels = logits.view(-1), labels.view(-1)#.int()
    logits = torch.log(logits) - torch.log(1 - logits)
    logits[torch.isnan(logits)] = 0.
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)#Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    signs.requires_grad = True
    grad.requires_grad = True
    loss = torch.dot(F.relu(errors_sorted), grad)#Variable(grad))
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

def crop(polygons: List[List[np.ndarray]], boxes: torch.Tensor) -> "PolygonMasks":
    boxes = boxes.to(torch.device("cpu")).numpy()
    results = [
        _crop(polygon, box) for polygon, box in zip(polygons, boxes)
    ]

    return PolygonMasks(results)


def _crop(polygons: np.ndarray, box: np.ndarray) -> List[np.ndarray]:
    w, h = box[2] - box[0], box[3] - box[1]

    polygons = copy.deepcopy(polygons)
    for p in polygons:
        p[0::2] = p[0::2] - box[0]  # .clamp(min=0, max=w)
        p[1::2] = p[1::2] - box[1]  # .clamp(min=0, max=h)

    return polygons


def mask_rcnn_loss(pred_mask_logits, instances, vis_period=0, maskiou_on=False):
    """
    Compute the mask prediction loss defined in the Mask R-CNN paper.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
            ...) associated with each instance are stored in fields.
        vis_period (int): the period (in steps) to dump visualization.

    Returns:
        mask_loss (Tensor): A scalar tensor containing the loss.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1
    total_num_masks = pred_mask_logits.size(0)
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"

    gt_classes = []
    gt_masks = []
    mask_ratios = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue

        if not cls_agnostic_mask:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        if maskiou_on:
            cropped_mask = crop(instances_per_image.gt_masks.polygons, instances_per_image.proposal_boxes.tensor)
            cropped_mask = torch.tensor(
                [mask_utils.area(mask_utils.frPyObjects([p for p in obj], box[3]-box[1], box[2]-box[0])).sum().astype(float)
                for obj, box in zip(cropped_mask.polygons, instances_per_image.proposal_boxes.tensor)]
                )
                
            mask_ratios.append(
                (cropped_mask / instances_per_image.gt_masks.area())
                .to(device=pred_mask_logits.device).clamp(min=0., max=1.)
            )
        
        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)
        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    if len(gt_masks) == 0:
        gt_classes = torch.LongTensor(gt_classes)
        if maskiou_on:
            selected_index = torch.arange(pred_mask_logits.shape[0], device=pred_mask_logits.device)
            if cls_agnostic_mask:
                selected_mask = pred_mask_logits[:, 0]
            else:
                # gt_classes = torch.LongTensor(gt_classes)
                selected_mask = pred_mask_logits[selected_index, gt_classes]
            mask_num, mask_h, mask_w = selected_mask.shape
            selected_mask = selected_mask.reshape(mask_num, 1, mask_h, mask_w)
            return pred_mask_logits.sum() * 0, selected_mask, gt_classes, None
        
        else:
            return pred_mask_logits.sum() * 0

    gt_masks = cat(gt_masks, dim=0)

    if cls_agnostic_mask:
        pred_mask_logits = pred_mask_logits[:, 0]
        gt_classes = torch.zeros(total_num_masks, dtype=torch.int64)
    else:
        indices = torch.arange(total_num_masks)
        gt_classes = cat(gt_classes, dim=0)
        pred_mask_logits = pred_mask_logits[indices, gt_classes] # (num_mask, Hmask, Wmask)

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask_rcnn/accuracy", mask_accuracy)
    storage.put_scalar("mask_rcnn/false_positive", false_positive)
    storage.put_scalar("mask_rcnn/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], axis=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, axis=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    
# Mask Area Loss
# =============================================================================
#     _pred_mask_logits = pred_mask_logits.sigmoid().squeeze().reshape(pred_mask_logits.shape[0],-1)
#     _pred_mask_logits =  (_pred_mask_logits > 0.5).sum(1).type(torch.cuda.FloatTensor)
#     _gt_masks = gt_masks.squeeze().reshape(gt_masks.shape[0],-1)
#     _gt_masks = _gt_masks.sum(1).type(torch.cuda.FloatTensor)
#     mask_count_loss = F.l1_loss(_pred_mask_logits,_gt_masks)*0.1
# =============================================================================
    
    
    
    # mask_loss = lovasz_hinge_flat(pred_mask_logits, gt_masks, reduction="mean")
# =============================================================================
#     alpha=0.25
#     gamma=2
#     BCE_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction='none')
#     pt = torch.exp(-BCE_loss) # prevents nans when probability 0
#     F_loss = alpha * (1-pt)**gamma * BCE_loss
#     mask_loss =  F_loss.mean()
#     
# =============================================================================
    
    
    if maskiou_on:
        mask_ratios = cat(mask_ratios, dim=0)

        value_eps = 1e-10 * torch.ones(gt_masks.shape[0], device=gt_masks.device).float()
        mask_ratios = torch.max(mask_ratios, value_eps)

        pred_masks = pred_mask_logits > 0

        mask_targets_full_area = gt_masks.sum(dim=[1,2]) / mask_ratios

        mask_ovr_area = (pred_masks * gt_masks).sum(dim=[1,2]).float()
        mask_union_area = pred_masks.sum(dim=[1,2]) + mask_targets_full_area - mask_ovr_area
        value_1 = torch.ones(pred_masks.shape[0], device=gt_masks.device).float()
        value_0 = torch.zeros(pred_masks.shape[0], device=gt_masks.device)
        mask_union_area = torch.max(mask_union_area, value_1)
        mask_ovr_area = torch.max(mask_ovr_area, value_0)
        maskiou_targets = mask_ovr_area / mask_union_area
        mask_num, mask_h, mask_w = pred_mask_logits.shape
        selected_mask = pred_mask_logits.reshape(mask_num, 1, mask_h, mask_w)
        selected_mask = selected_mask.sigmoid()
        return mask_loss, selected_mask, gt_classes, maskiou_targets.detach()
    else:
        return mask_loss
        # return mask_loss, mask_count_loss


def mask_rcnn_inference(pred_mask_logits, pred_instances):
    """
    Convert pred_mask_logits to estimated foreground probability masks while also
    extracting only the masks for the predicted classes in pred_instances. For each
    predicted box, the mask of the same class is attached to the instance by adding a
    new "pred_masks" field to pred_instances.

    Args:
        pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
            for class-specific or class-agnostic, where B is the total number of predicted masks
            in all images, C is the number of foreground classes, and Hmask, Wmask are the height
            and width of the mask predictions. The values are logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
            Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
            masks the resolution predicted by the network; post-processing steps, such as resizing
            the predicted masks to the original image resolution and/or binarizing them, is left
            to the caller.
    """
    cls_agnostic_mask = pred_mask_logits.size(1) == 1

    if cls_agnostic_mask:
        mask_probs_pred = pred_mask_logits.sigmoid()
    else:
        # Select masks corresponding to the predicted classes
        num_masks = pred_mask_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_masks, device=class_pred.device)
        mask_probs_pred = pred_mask_logits[indices, class_pred][:, None].sigmoid()
    # mask_probs_pred.shape: (B, 1, Hmask, Wmask)

    num_boxes_per_image = [len(i) for i in pred_instances]
    mask_probs_pred = mask_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(mask_probs_pred, pred_instances):
        instances.pred_masks = prob  # (1, Hmask, Wmask)


class BaseMaskRCNNHead(nn.Module):
    """
    Implement the basic Mask R-CNN losses and inference logic.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        self.vis_period = cfg.VIS_PERIOD

    def forward(self, x, instances: List[Instances], fusion_out=None, maskiou_on=False):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        mask_features = x
        x = self.layers(x)
        
        '''By Pawn maskpoint'''
        if(fusion_out is not None):
            x = x * fusion_out[:,None,:,:]
        
        if self.training:
            if(maskiou_on):
                loss, selected_mask, labels, maskiou_targets = mask_rcnn_loss(x, instances, maskiou_on=maskiou_on)
                return {"loss_mask": loss}, mask_features, selected_mask, labels, maskiou_targets
            else:
                return {"loss_mask": mask_rcnn_loss(x, instances, self.vis_period)}
                # mask_loss, mask_count_loss = mask_rcnn_loss(x, instances, self.vis_period)
                # return {"loss_mask": mask_loss, "loss_mask_count":mask_count_loss}
        else:
            mask_rcnn_inference(x, instances)
            if(maskiou_on):
                return instances, mask_features
            else:
                return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from input features.
        """
        raise NotImplementedError


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNConvUpsampleHead(BaseMaskRCNNHead):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv: the number of conv layers
            conv_dim: the dimension of the conv layers
            norm: normalization for the conv layers
        """
        super().__init__(cfg, input_shape)

        # fmt: off
        num_classes       = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        conv_dims         = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.norm         = cfg.MODEL.ROI_MASK_HEAD.NORM
        num_conv          = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        input_channels    = input_shape.channels
        cls_agnostic_mask = cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK
        # fmt: on

        self.conv_norm_relus = []

        for k in range(num_conv):
            conv = Conv2d(
                input_channels if k == 0 else conv_dims,
                conv_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not self.norm,
                norm=get_norm(self.norm, conv_dims),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)

        self.deconv = ConvTranspose2d(
            conv_dims if num_conv > 0 else input_channels,
            conv_dims,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = Conv2d(conv_dims, num_mask_classes, kernel_size=1, stride=1, padding=0)

        for layer in self.conv_norm_relus + [self.deconv]:
            weight_init.c2_msra_fill(layer)
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    def layers(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = F.relu(self.deconv(x))
        return self.predictor(x)


def build_mask_head(cfg, input_shape):
    """
    Build a mask head defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_MASK_HEAD.NAME
    return ROI_MASK_HEAD_REGISTRY.get(name)(cfg, input_shape)
