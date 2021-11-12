# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ConvTranspose2d, ShapeSpec, cat, interpolate
from detectron2.structures import Instances, heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

_TOTAL_SKIPPED = 0

ROI_KEYPOINT_HEAD_REGISTRY = Registry("ROI_KEYPOINT_HEAD")
ROI_KEYPOINT_HEAD_REGISTRY.__doc__ = """
Registry for keypoint heads, which make keypoint predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

import numpy as np
import cv2
import time

def make_heatmaps_from_joints(keypoints, hw=(256,256), guass_variance= 1):

#    keypoints: [num_mask, num_keypoint, 3]
#    
#    output:
#        result: [num_mask, num_keypoint, hw[0], hw[1]]
    result_heatmap_np= []
#    invert_heatmap_np = np.ones(shape=(hw[0], hw[1]))
    for i in range(keypoints.shape[0]):
        gt_heatmap_np = []
        for j in range(keypoints.shape[1]):
            cur_joint_heatmap = make_gaussian_beliefMap(hw,
                                          guass_variance,
                                          center=(keypoints[i][j][0],keypoints[i][j][1]))
            gt_heatmap_np.append(cur_joint_heatmap)
#            invert_heatmap_np -= cur_joint_heatmap
#        gt_heatmap_np.append(invert_heatmap_np)
        result_heatmap_np.append(gt_heatmap_np)
    return np.array(result_heatmap_np)

def make_gaussian_beliefMap(hw, fwhm=1, center=(0,0)):
    center_map_size = fwhm*6
    center_map = create_center_gaussian(center_map_size, fwhm)
    final_map = np.zeros((hw[0], hw[1]))
    if center[0]>=0 and center[1]>=0:
        centerX = center[0]
        centerY = center[1]

        centerMap_y1 = 0
        centerMap_x1 = 0
        centerMap_y2 = center_map_size
        centerMap_x2 = center_map_size
        radius_center_map = center_map_size // 2
        map_y1 = int(centerY) - radius_center_map
        map_x1 = int(centerX) - radius_center_map
        map_y2 = int(centerY) + radius_center_map 
        map_x2 = int(centerX) + radius_center_map 

        if map_y1 < 0:
            centerMap_y1 = -map_y1
            map_y1 = 0
        if map_x1 < 0:
            centerMap_x1 = -map_x1
            map_x1 = 0
        if map_y2 > hw[0]:
            centerMap_y2 = center_map_size - (map_y2 - (hw[0]))
            map_y2 = hw[0]
        if map_x2 > hw[1]:
            centerMap_x2 = center_map_size - (map_x2 - (hw[1]))
            map_x2 = hw[1]

        final_map[map_y1:map_y2, map_x1:map_x2] = center_map[centerMap_y1:centerMap_y2, centerMap_x1:centerMap_x2]
    return final_map

def create_center_gaussian(size, fwhm=1):
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    x0 = y0 = size // 2

    r = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / fwhm / fwhm)
    return r

def cropResize_gt_heatmap_byROI(gt_heatmap, proposals):
    proposals = proposals.cpu().numpy()
    new_heatmap = np.zeros((gt_heatmap.shape[0], gt_heatmap.shape[1], 56, 56))
    i=0
    for gt_H, proposal in zip(gt_heatmap, proposals):
        new_gt = gt_H[:, int(proposal[1]):int(proposal[3]), int(proposal[0]):int(proposal[2])]
        new_gts = np.zeros((new_gt.shape[0], 56, 56))
        
        for p in range(new_gt.shape[0]):
            new_gts[p] = cv2.resize(new_gt[p], (56,56))
        new_heatmap[i] = new_gts
        i+=1
    return new_heatmap


def make_heatmaps_from_joints_cuda(keypoints, hw=(256,256), guass_variance= 1):
    result_heatmap= torch.zeros((keypoints.shape[0], keypoints.shape[1], hw[0], hw[1]), device='cuda')
    for i in range(keypoints.shape[0]):
        for j in range(keypoints.shape[1]):
            cur_joint_heatmap = make_gaussian_beliefMap_cuda(hw,
                                          guass_variance,
                                          center=(keypoints[i][j][0],keypoints[i][j][1]))
            result_heatmap[i,j]=cur_joint_heatmap
    return result_heatmap

def make_gaussian_beliefMap_cuda(hw, fwhm=1, center=(0,0)):
    center_map_size = fwhm*6
    center_map = create_center_gaussian_cuda(center_map_size, fwhm)
    final_map = torch.zeros((hw[0], hw[1]), device='cuda')
    if center[0]>=0 and center[1]>=0:
        centerX = center[0]
        centerY = center[1]

        centerMap_y1 = 0
        centerMap_x1 = 0
        centerMap_y2 = center_map_size
        centerMap_x2 = center_map_size
        radius_center_map = center_map_size // 2
        map_y1 = int(centerY) - radius_center_map
        map_x1 = int(centerX) - radius_center_map
        map_y2 = int(centerY) + radius_center_map 
        map_x2 = int(centerX) + radius_center_map 

        if map_y1 < 0:
            centerMap_y1 = -map_y1
            map_y1 = 0
        if map_x1 < 0:
            centerMap_x1 = -map_x1
            map_x1 = 0
        if map_y2 > hw[0]:
            centerMap_y2 = center_map_size - (map_y2 - (hw[0]))
            map_y2 = hw[0]
        if map_x2 > hw[1]:
            centerMap_x2 = center_map_size - (map_x2 - (hw[1]))
            map_x2 = hw[1]
            
        final_map[map_y1:map_y2, map_x1:map_x2] = center_map[centerMap_y1:centerMap_y2, centerMap_x1:centerMap_x2]

    return final_map

def create_center_gaussian_cuda(size, fwhm=1):
    x = torch.arange(0, size, 1, device='cuda', dtype= float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    r = torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / 2.0 / fwhm / fwhm)
    return r

def cropResize_gt_heatmap_byROI_cuda(gt_heatmap, proposals):
    new_heatmap = torch.zeros((gt_heatmap.shape[0], gt_heatmap.shape[1], 56, 56), device='cuda')
    i=0
    for gt_H, proposal in zip(gt_heatmap, proposals):
        new_gt = gt_H[:, int(proposal[1]):int(proposal[3]), int(proposal[0]):int(proposal[2])]
        # new_gts = torch.zeros((new_gt.shape[0], 56, 56), device='cuda')
        # for p in range(new_gt.shape[0]):
        #     # new_gts[p] = new_gt[p].resize_(56,56)
        #     # new_gts[p] = torch.tensor(cv2.resize(new_gt[p].cpu().numpy(), (56,56)), device='cuda')
        
        new_gts = F.interpolate(new_gt.unsqueeze(1), size=(56, 56)).squeeze(1)
        
        new_heatmap[i] = new_gts
        i+=1
    return new_heatmap


                                                                   
        
        

def build_keypoint_head(cfg, input_shape):
    """
    Build a keypoint head from `cfg.MODEL.ROI_KEYPOINT_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_KEYPOINT_HEAD.NAME
    return ROI_KEYPOINT_HEAD_REGISTRY.get(name)(cfg, input_shape)


# =============================================================================
# def keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer):
#     """
#     Arguments:
#         pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
#             of instances in the batch, K is the number of keypoints, and S is the side length
#             of the keypoint heatmap. The values are spatial logits.
#         instances (list[Instances]): A list of M Instances, where M is the batch size.
#             These instances are predictions from the model
#             that are in 1:1 correspondence with pred_keypoint_logits.
#             Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
#             instance.
#         normalizer (float): Normalize the loss by this amount.
#             If not specified, we normalize by the number of visible keypoints in the minibatch.
# 
#     Returns a scalar tensor containing the loss.
#     """
#     heatmaps = []
#     valid = []
#     
# # =============================================================================
# #     # Pawn
# #     gt_keypoints_ori = instances[0].gt_keypoints.tensor.cpu().numpy()
# #     # for i in range(3):
# #     #     im = np.zeros((750,1333, 3), dtype=np.uint8)
# #     #     color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
# #     #     for j in range(gt_keypoints_ori.shape[1]):
# #     #         x = gt_keypoints_ori[i][j][0]
# #     #         y = gt_keypoints_ori[i][j][1]
# #     #         cv2.circle(im,(x, y), 10, color, 3)
# #     #     cv2.imwrite('E:\\temp_detect\\'+'ORI'+str(i)+'.jpg', im)
# #     
# #     Time0=time.time()
# #     sTime = time.time()
# #     gt_heatmap = make_heatmaps_from_joints(gt_keypoints_ori, (750,1333), guass_variance=5)
# #     # for i in range(3):
# #     #     im = np.zeros_like(gt_heatmap[i][0])
# #     #     for j in range(gt_heatmap.shape[1]):
# #     #         im = im + gt_heatmap[i][j]
# #     #     cv2.imwrite('E:\\temp_detect\\NewGT'+str(i)+'.jpg', (im*255).astype(np.uint8))
# #     print(f'make_heatmaps_from_joints : {time.time()-sTime:.2f}')
# #     
# #     sTime = time.time()         
# #     new_gt_heatmap = cropResize_gt_heatmap_byROI(gt_heatmap, instances[0].gt_boxes.tensor)
# #     # for i in range(3):
# #     #     im = np.zeros_like(new_gt_heatmap[i][0])
# #     #     for j in range(new_gt_heatmap.shape[1]):
# #     #         im = im + new_gt_heatmap[i][j]
# #     #     cv2.imwrite('E:\\temp_detect\\NewGTS'+str(i)+'.jpg', (im*255).astype(np.uint8))
# #     print(f'cropResize_gt_heatmap_byROI : {time.time()-sTime:.2f}')
# #     print(f'TOTAL : {time.time()-Time0:.2f}')
# # =============================================================================
#     
#     try:
#         gt_keypoints_ori = instances[0].gt_keypoints.tensor
#         # for i in range(3):
#         #     im = np.zeros((750,1333, 3), dtype=np.uint8)
#         #     color = (np.random.randint(256), np.random.randint(256), np.random.randint(256))
#         #     for j in range(gt_keypoints_ori.cpu().numpy().shape[1]):
#         #         x = gt_keypoints_ori[i][j][0].cpu().numpy()
#         #         y = gt_keypoints_ori[i][j][1].cpu().numpy()
#         #         cv2.circle(im,(x, y), 10, color, 3)
#         #     cv2.imwrite('E:\\temp_detect\\'+'ORI'+str(i)+'.jpg', im)
#         # Time0=time.time()
#         # sTime = time.time()
#         gt_heatmap = make_heatmaps_from_joints_cuda(gt_keypoints_ori, (750,1333), guass_variance=5)
#     
#         # for i in range(3):
#         #     im = np.zeros_like(gt_heatmap[i][0].cpu().numpy())
#         #     for j in range(gt_heatmap.shape[1]):
#         #         im = im + gt_heatmap[i][j].cpu().numpy()
#         #     cv2.imwrite('E:\\temp_detect\\NewGT'+str(i)+'.jpg', (im*255).astype(np.uint8))
#         # print(f'make_heatmaps_from_joints_cuda : {time.time()-sTime:.2f}')
#         # sTime = time.time()         
#         new_gt_heatmap = cropResize_gt_heatmap_byROI_cuda(gt_heatmap, instances[0].gt_boxes.tensor)
#         # for i in range(3):
#         #     im = np.zeros_like(new_gt_heatmap[i][0].cpu().numpy())
#         #     for j in range(new_gt_heatmap.shape[1]):
#         #         im = im + new_gt_heatmap[i][j].cpu().numpy()
#         #     cv2.imwrite('E:\\temp_detect\\NewGTS'+str(i)+'.jpg', (im*255).astype(np.uint8))
#         # print(f'cropResize_gt_heatmap_byROI_cuda : {time.time()-sTime:.2f}')
#         # print(f'TOTAL CUDA : {time.time()-Time0:.2f}\n')
#         
#     
#         for i in range(3):
#             im0 = np.zeros_like(F.sigmoid(pred_keypoint_logits[i][0]).cpu().detach().numpy())
#             im = np.zeros_like(new_gt_heatmap[i][0].cpu().numpy())
#             for j in range(pred_keypoint_logits.shape[1]):
#                 im0 = im0 + F.sigmoid(pred_keypoint_logits[i][j]).cpu().detach().numpy()
#                 im = im + new_gt_heatmap[i][j].cpu().numpy()
#             cv2.imwrite('E:\\temp_detect\\Pred'+str(i)+'.jpg', (im0*255).astype(np.uint8))
#             cv2.imwrite('E:\\temp_detect\\NewGTS'+str(i)+'.jpg', (im*255).astype(np.uint8))
#             
#         print(f'{pred_keypoint_logits.max():.2f} {pred_keypoint_logits.min():.2f} {pred_keypoint_logits.mean():.2f}')
#         # print(f'{new_gt_heatmap.max():.2f}')
#         # print(f'{new_gt_heatmap.min():.2f}')
#         # print(f'{new_gt_heatmap.mean():.2f}')
#         # print()
#     except:
#         pass
#     
# 
#     keypoint_side_len = pred_keypoint_logits.shape[2]
#     for instances_per_image in instances:
#         if len(instances_per_image) == 0:
#             continue
#         keypoints = instances_per_image.gt_keypoints
#         heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
#             instances_per_image.proposal_boxes.tensor, keypoint_side_len
#         )
#         heatmaps.append(heatmaps_per_image.view(-1))
#         valid.append(valid_per_image.view(-1))
# 
#     if len(heatmaps):
#         keypoint_targets = cat(heatmaps, dim=0)
#         valid = cat(valid, dim=0).to(dtype=torch.uint8)
#         valid = torch.nonzero(valid).squeeze(1)
# 
#     # torch.mean (in binary_cross_entropy_with_logits) doesn't
#     # accept empty tensors, so handle it separately
#     if len(heatmaps) == 0 or valid.numel() == 0:
#         global _TOTAL_SKIPPED
#         _TOTAL_SKIPPED += 1
#         storage = get_event_storage()
#         storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
#         return pred_keypoint_logits.sum() * 0
# 
#     N, K, H, W = pred_keypoint_logits.shape
#     pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)
# 
# # =============================================================================
# #     keypoint_loss = F.cross_entropy(
# #         pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
# #     )
# #     print(f'Ori KP loss :            {keypoint_loss:.2f}')
# # =============================================================================
#     # Pawn
# # =============================================================================
# #     new_gt_heatmap = new_gt_heatmap.view(N * K, H * W)
# #     keypoint_loss = F.smooth_l1_loss(
# #         F.sigmoid(pred_keypoint_logits[valid]), new_gt_heatmap[valid], reduction="sum"
# #         # pred_keypoint_logits[valid], new_gt_heatmap[valid], reduction="sum"
# #     ) 
# #     print(f'smooth_l1_loss KP loss : {keypoint_loss:.2f}')
# # =============================================================================
# # =============================================================================
# #     new_gt_heatmap = new_gt_heatmap.view(N * K, H * W)
# #     keypoint_loss = F.binary_cross_entropy(
# #         F.sigmoid(pred_keypoint_logits[valid]), new_gt_heatmap[valid], reduction="sum"
# #     ) 
# #     print(f'binary_cross_entropy KP loss : {keypoint_loss:.2f}')
# # =============================================================================
#     new_gt_heatmap = new_gt_heatmap.view(N * K, H * W)
#     keypoint_loss = F.kl_div(
#         F.log_softmax(pred_keypoint_logits[valid], dim=1), new_gt_heatmap[valid], reduction="sum"
#     ) /100
#     print(f'kl_div KP loss : {keypoint_loss:.2f}')
#     # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
#     if normalizer is None:
#         normalizer = valid.numel()
#     keypoint_loss /= normalizer
# 
#     return keypoint_loss
# =============================================================================

def keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer):
    """
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.

    Returns a scalar tensor containing the loss.
    """
    heatmaps = []
    valid = []
    
    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_keypoints
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0

    N, K, H, W = pred_keypoint_logits.shape
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(
        pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer

    return keypoint_loss

def keypoint_rcnn_inference(pred_keypoint_logits, pred_instances):
    """
    Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score)
        and add it to the `pred_instances` as a `pred_keypoints` field.

    Args:
        pred_keypoint_logits (Tensor): A tensor of shape (R, K, S, S) where R is the total number
           of instances in the batch, K is the number of keypoints, and S is the side length of
           the keypoint heatmap. The values are spatial logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images.

    Returns:
        None. Each element in pred_instances will contain an extra "pred_keypoints" field.
            The field is a tensor of shape (#instance, K, 3) where the last
            dimension corresponds to (x, y, score).
            The scores are larger than 0.
    """
    # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
    bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)

    keypoint_results = heatmaps_to_keypoints(pred_keypoint_logits.detach(), bboxes_flat.detach())
    num_instances_per_image = [len(i) for i in pred_instances]
    keypoint_results = keypoint_results[:, :, [0, 1, 3]].split(num_instances_per_image, dim=0)

    for keypoint_results_per_image, instances_per_image in zip(keypoint_results, pred_instances):
        # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score)
        instances_per_image.pred_keypoints = keypoint_results_per_image


class BaseKeypointRCNNHead(nn.Module):
    """
    Implement the basic Keypoint R-CNN losses and inference logic.
    """

    def __init__(self, cfg, input_shape):
        super().__init__()
        # fmt: off
        self.loss_weight                    = cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT
        self.normalize_by_visible_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS  # noqa
        self.num_keypoints                  = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        batch_size_per_image                = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        positive_sample_fraction            = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        # fmt: on
        self.normalizer_per_img = (
            self.num_keypoints * batch_size_per_image * positive_sample_fraction
        )

    def forward(self, x, instances: List[Instances]):
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
            A dict of losses if in training. The predicted "instances" if in inference.
        """
        x = self.layers(x)
        if self.training:
            num_images = len(instances)
            normalizer = (
                None
                if self.normalize_by_visible_keypoints
                else num_images * self.normalizer_per_img
            )
            return {
                "loss_keypoint": keypoint_rcnn_loss(x, instances, normalizer=normalizer)
                * self.loss_weight
            }
        else:
            keypoint_rcnn_inference(x, instances)
            return instances

    def layers(self, x):
        """
        Neural network layers that makes predictions from regional input features.
        """
        raise NotImplementedError


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class KRCNNConvDeconvUpsampleHead(BaseKeypointRCNNHead):
    """
    A standard keypoint head containing a series of 3x3 convs, followed by
    a transpose convolution and bilinear interpolation for upsampling.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            conv_dims: an iterable of output channel counts for each conv in the head
                         e.g. (512, 512, 512) for three convs outputting 512 channels.
            num_keypoints: number of keypoint heatmaps to predicts, determines the number of
                           channels in the final output.
        """
        super().__init__(cfg, input_shape)

        # fmt: off
        # default up_scale to 2 (this can eventually be moved to config)
        up_scale      = 2
        conv_dims     = cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS
        num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS
        in_channels   = input_shape.channels
        # fmt: on

        self.blocks = []
        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1)
            self.add_module("conv_fcn{}".format(idx), module)
            self.blocks.append(module)
            in_channels = layer_channels

        deconv_kernel = 4
        self.score_lowres = ConvTranspose2d(
            in_channels, num_keypoints, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.up_scale = up_scale

        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def layers(self, x):
        for layer in self.blocks:
            x = F.relu(layer(x))
        x = self.score_lowres(x)
        x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x
