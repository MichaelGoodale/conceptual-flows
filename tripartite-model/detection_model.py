from typing import List, Optional, Dict, Tuple
from collections import OrderedDict

import torch
from torch import nn, Tensor

from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

class EntityHeads(RoIHeads):
    ''' Take CNN features from RCNN model and returns the box regressions and entity vector
        Adapted from torchvision.models.detection.roi_heads

    '''

    def __init__(self,
          box_roi_pool: nn.Module=None,
          box_head: nn.Module = None,
          box_predictor: nn.Module = None,
          out_channels: int = 128,
          score_thresh=0.05,
          nms_thresh=0.5,
          detections_per_img=100,
          fg_iou_thresh=0.5,
          bg_iou_thresh=0.5,
          batch_size_per_image=512,
          positive_fraction=0.25,
          bbox_reg_weights: Optional[Tuple[float]]=None,) -> None:

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            representation_size = 1024
            box_head = TwoMLPHead(out_channels * resolution**2, representation_size)

        if box_predictor is None:
            representation_size = 1024
            box_predictor = nn.Sequential(
                    nn.Linear(representation_size, 4))

        super().__init__(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=fg_iou_thresh,
            bg_iou_thresh=bg_iou_thresh,
            batch_size_per_image=batch_size_per_image,
            positive_fraction=positive_fraction,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
            bbox_reg_weights=bbox_reg_weights
            )

    def forward(self, features, proposals, image_shapes, targets=None):
        '''
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        '''

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(features.flatten(start_dim=1))
        detections = self.box_predictor(box_features.flatten(start_dim=1))

        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}

        results = {'boxes': detections, 'entity_features': box_features}
        return results, losses



class EntityRCNN(nn.Module):
    '''
    Faster-RCNN based model which provides vector representations of entities in an image.
    Based off off torchvision.models.detection.faster_rcnn
    Args:
        backbone (nn.Module):
        rpn_anchor_generator (AnchorGenerator): module that generates the anchors for a set of feature
            maps.
        rpn_head (nn.Module): module that computes the objectness and regression deltas from the RPN
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals
        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_batch_size_per_image (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors in a mini-batch during training
            of the RPN
        rpn_score_thresh (float): during inference, only return proposals with a classification score
            greater than rpn_score_thresh
    '''

    def __init__(self, backbone: nn.Module, 
                    out_channels: int = 128,
                    # RPN parameters
                    rpn_anchor_generator=None,
                    rpn_head=None,
                    rpn_pre_nms_top_n_train=2000,
                    rpn_pre_nms_top_n_test=1000,
                    rpn_post_nms_top_n_train=2000,
                    rpn_post_nms_top_n_test=1000,
                    rpn_nms_thresh=0.7,
                    rpn_fg_iou_thresh=0.7,
                    rpn_bg_iou_thresh=0.3,
                    rpn_batch_size_per_image=256,
                    rpn_positive_fraction=0.5,
                    rpn_score_thresh=0.0,
                 ) -> None:
        super().__init__()
        self.backbone = backbone
        self.transform = GeneralizedRCNNTransform(
                min_size=800,
                max_size=1333,
                image_mean = [0.485, 0.456, 0.406],
                image_std = [0.229, 0.224, 0.225])

        if not isinstance(rpn_anchor_generator, (AnchorGenerator, type(None))):
            raise TypeError(
                f"rpn_anchor_generator should be of type AnchorGenerator or None instead of {type(rpn_anchor_generator)}"
            )

        if rpn_anchor_generator is None:
            rpn_anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )

        self.roi_heads = EntityHeads(out_channels=out_channels)

    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors) #TODO: Handle IMAGE LIST
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses, detections

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Using pretrained weights:
backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
model = EntityRCNN(backbone, out_channels=1280)
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 250, 650)]
predictions = model(x)
print([(k, v.shape) for k, v in predictions[1].items()])


