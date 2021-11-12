# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec
from detectron2.structures import ImageList
from detectron2.utils.registry import Registry

from ..backbone import build_backbone
from ..postprocessing import sem_seg_postprocess
from .build import META_ARCH_REGISTRY

__all__ = ["SemanticSegmentor", "SEM_SEG_HEADS_REGISTRY", "SemSegFPNHead", "build_sem_seg_head"]


SEM_SEG_HEADS_REGISTRY = Registry("SEM_SEG_HEADS")
"""
Registry for semantic segmentation heads, which make semantic segmentation predictions
from feature maps.
"""

from torch.autograd import Variable


# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha,(float,int,float)): self.alpha = torch.Tensor([alpha,1-alpha])
#         if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type()!=input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0,target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()


def focal_loss(input, target, gamma=0, alpha=None, size_average=True):
        if input.dim()>2:
            input = input.reshape(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().reshape(-1,input.size(2))   # N,H*W,C => N*H*W,C
            # input = input.contiguous().reshape(-1,input.size(1))
        
        target = target.reshape(-1,1)
        
        # weight= torch.tensor([[0.1]+[1]*input.shape[1]]*input.shape[0]).cuda()
        # weight = weight.gather(1, target)
        # weight = weight.reshape(-1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.reshape(-1)
        pt = Variable(logpt.data.exp())

        if alpha is not None:
            if alpha.type()!=input.data.type():
                alpha = alpha.type_as(input.data)
            at = alpha.gather(0,target.data.reshape(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**gamma * logpt
        # loss = -1 * (1-pt)**gamma * logpt * weight
        if size_average: 
            return loss.mean()
        else: 
            return loss.sum()
    

@META_ARCH_REGISTRY.register()
class SemanticSegmentor(nn.Module):
    """
    Main class for semantic segmentation architectures.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.sem_seg_head = build_sem_seg_head(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.

                For now, each item in the list is a dict that contains:

                   * "image": Tensor, image in (C, H, W) format.
                   * "sem_seg": semantic segmentation ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model, used in inference.
                     See :meth:`postprocess` for details.

        Returns:
            list[dict]:
              Each dict is the output for one input image.
              The dict contains one key "sem_seg" whose value is a
              Tensor of the output resolution that represents the
              per-pixel segmentation prediction.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        features = self.backbone(images.tensor)

        if "sem_seg" in batched_inputs[0]:
            targets = [x["sem_seg"].to(self.device) for x in batched_inputs] #aaa = targets[0].cpu().numpy()
            targets = ImageList.from_tensors(
                targets, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            targets = None
        results, losses = self.sem_seg_head(features, targets)

        if self.training:
            return losses

# =============================================================================
#         processed_results = []
#         for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
#             height = input_per_image.get("height")
#             width = input_per_image.get("width")
#             r = sem_seg_postprocess(result, image_size, height, width)
#             processed_results.append({"sem_seg": r})
#         return processed_results
# =============================================================================
        processed_results = []
        if(len(results)==0): 
            return processed_results
        else:
            height = batched_inputs[0].get("height")
            width = batched_inputs[0].get("width")
            r = sem_seg_postprocess(results[0], images.image_sizes[0], height, width)
            processed_results.append({"sem_seg": r,"class_out":results[1]})            
        return processed_results

def build_sem_seg_head(cfg, input_shape):
    """
    Build a semantic segmentation head from `cfg.MODEL.SEM_SEG_HEAD.NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.NAME
    return SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)


@SEM_SEG_HEADS_REGISTRY.register()
class SemSegFPNHead(nn.Module):
    """
    A semantic segmentation head described in detail in the Panoptic Feature Pyramid Networks paper
    (https://arxiv.org/abs/1901.02446). It takes FPN features as input and merges information from
    all levels of the FPN into single output.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

# =============================================================================
#         '''By Pawn'''
#         try:
#             from .init import Init
#             self.tfControl = Init()
# #            self.tfControl.getLossName('Total_acc')
#             print('Successful in SolVision2')
#         except:
#             print('Not in SolVision or fails')
# =============================================================================
        
        # fmt: off
        self.in_features      = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        self.ignore_value     = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes           = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims             = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride    = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        norm                  = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight      = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)
        
        self.d_net = DecisionNet(num_classes=num_classes).to('cuda')

    def forward(self, features, targets=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        f=x.clone().detach()
        x = self.predictor(x)
        class_out = self.d_net(f, x.clone().detach())
        
        # class_out = None
        
        x = F.interpolate(x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        if self.training:
            '''Pawn'''
# =============================================================================
#             prediction = torch.argmax(x, dim=1)
#             correct_matrix = torch.eq(prediction, targets)
#             total_count = (targets!=255).sum()
#             correct_count = correct_matrix.sum()
#             acc = torch.div(correct_count.type(torch.float), total_count.type(torch.float)).cpu().numpy()
#             print(f'Acc       : {acc:.3f} ({correct_count.cpu().numpy()}/{total_count.cpu().numpy()})')
#             
#             ignore_bg_mask = torch.where(targets==0, torch.full_like(targets, 0), torch.full_like(targets, 1))
#             correct_matrix_nobg = correct_matrix*ignore_bg_mask
#             total_count_nobg = targets.view(-1).shape[0] - ((targets==255).sum() + (targets==0).sum())
#             correct_count_nobg = correct_matrix_nobg.sum()
#             acc_nobg = torch.div(correct_count_nobg.type(torch.float), total_count_nobg.type(torch.float)).cpu().numpy()
#             print(f'Acc no bg : {acc_nobg:.3f} ({correct_count_nobg.cpu().numpy()}/{total_count_nobg.cpu().numpy()})')
#             self.tfControl.getLoss('Total_acc', acc_nobg)
# =============================================================================
            
            # targets[(targets!=0) & (targets!=255)] = 1
            # targets[targets==255]=0
            # targets[targets!=0]=1
            
            class_gt = ((targets!=0) & (targets!=255)).sum()==0
            class_gt = class_gt.unsqueeze(0).unsqueeze(0)
            
            x_limit= (targets[0,0,:]!=255).sum()
            y_limit= (targets[0,:,0]!=255).sum()
            x = x[:,:,:y_limit,:x_limit]
            targets=targets[:,:y_limit,:x_limit]
            
            
            # losses = {}
            # a = F.cross_entropy(x, targets.type(torch.LongTensor).cuda(), reduce=False)
            # aloss = torch.exp(-a).cuda()
            # losses["loss_sem_seg"] = (0.25*(1-aloss)**2.0* aloss).mean() * self.loss_weight  #Pawn temp
            # # bbb = focal_loss(x, targets.type(torch.LongTensor).cuda() , gamma=2)
            
            
            losses = {}
            losses["loss_sem_seg"] = (
#                F.cross_entropy(x, targets, reduction="mean", ignore_index=self.ignore_value)
                # F.cross_entropy(x, targets.type(torch.LongTensor).cuda(), reduction="mean", ignore_index=self.ignore_value) #Pawn temp
                F.cross_entropy(x, targets.type(torch.LongTensor).cuda(), weight=torch.tensor([0.1]+[1]*(x.shape[1]-1)).cuda(), reduction="mean", ignore_index=self.ignore_value) #Pawn temp
                # focal_loss(x, targets.type(torch.LongTensor).cuda() , gamma=2)
                # F.binary_cross_entropy(F.sigmoid(x), targets.type(torch.FloatTensor).cuda(), reduction="mean") #Pawn temp
                * self.loss_weight
            )
            
            
            losses["loss_class"]=(F.binary_cross_entropy(class_out, class_gt.type(torch.FloatTensor).to('cuda'), reduction="mean"))
            
            return [], losses
        else:
            return [x, class_out], {}
            # return x, {}

class DecisionNet(nn.Module):
    
    def __init__(self, init_weights=True, num_classes=2):
        super(DecisionNet, self).__init__()

        self.layer1 = nn.Sequential(
                            nn.MaxPool2d(2),
                            nn.Conv2d(128+num_classes, 8, 5, stride=1, padding=2),
                            nn.BatchNorm2d(8),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(2),
                            nn.Conv2d(8, 16, 5, stride=1, padding=2),
                            nn.BatchNorm2d(16),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(16, 32, 5, stride=1, padding=2),
                            nn.BatchNorm2d(32),
                            nn.ReLU(inplace=True)
                        )

        self.fc =  nn.Sequential(
                            nn.Linear(64+num_classes*2, 1, bias=False),
                            nn.Sigmoid()
                        )
        
    def forward(self, f, s):
        xx = torch.cat((f, s), 1)
        x1 = self.layer1(xx)
        x2 = x1.reshape(x1.size(0), x1.size(1), -1)
        s2 = s.reshape(s.size(0), s.size(1), -1)

        x_max, x_max_idx = torch.max(x2, dim=2)
        x_avg = torch.mean(x2, dim=2)
        s_max, s_max_idx = torch.max(s2, dim=2)
        s_avg = torch.mean(s2, dim=2)

        y = torch.cat((x_max, x_avg, s_avg, s_max), 1)
        y = y.reshape(y.size(0), -1)

        return self.fc(y)