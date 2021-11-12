# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Common data processing utilities that are used in a
typical object detection data pipeline.
"""
import logging
import numpy as np
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from PIL import Image, ImageOps

from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

from . import transforms as T
from .catalog import MetadataCatalog
import random

class SizeMismatchError(ValueError):
    """
    When loaded image has difference width/height compared with annotation.
    """


def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"

    Returns:
        image (np.ndarray): an HWC image in the given format.
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            # for orientation in ExifTags.TAGS.keys():
            #     if ExifTags.TAGS[orientation]=='Orientation':
            #         break
            try:
                orientation = 274
                if (image._getexif() is not None):
                    exif=dict(image._getexif().items())
                if exif[orientation] == 3:
                    image=image.transpose(Image.ROTATE_180)
                elif exif[orientation] == 6:
                    image=image.transpose(Image.ROTATE_270)
                elif exif[orientation] == 8:
                    image=image.transpose(Image.ROTATE_90)
            except Exception:
                pass
            # pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image


def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            raise SizeMismatchError(
                "Mismatched (W,H){}, got {}, expect {}".format(
                    " for image " + dataset_dict["file_name"]
                    if "file_name" in dataset_dict
                    else "",
                    image_wh,
                    expected_wh,
                )
            )

    # To ensure bbox always remap to original image size
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]


def transform_proposals(dataset_dict, image_shape, transforms, min_box_side_len, proposal_topk):
    """
    Apply transformations to the proposals in dataset_dict, if any.

    Args:
        dataset_dict (dict): a dict read from the dataset, possibly
            contains fields "proposal_boxes", "proposal_objectness_logits", "proposal_bbox_mode"
        image_shape (tuple): height, width
        transforms (TransformList):
        min_box_side_len (int): keep proposals with at least this size
        proposal_topk (int): only keep top-K scoring proposals

    The input dict is modified in-place, with abovementioned keys removed. A new
    key "proposals" will be added. Its value is an `Instances`
    object which contains the transformed proposals in its field
    "proposal_boxes" and "objectness_logits".
    """
    if "proposal_boxes" in dataset_dict:
        # Transform proposal boxes
        boxes = transforms.apply_box(
            BoxMode.convert(
                dataset_dict.pop("proposal_boxes"),
                dataset_dict.pop("proposal_bbox_mode"),
                BoxMode.XYXY_ABS,
            )
        )
        boxes = Boxes(boxes)
        objectness_logits = torch.as_tensor(
            dataset_dict.pop("proposal_objectness_logits").astype("float32")
        )

        boxes.clip(image_shape)
        keep = boxes.nonempty(threshold=min_box_side_len)
        boxes = boxes[keep]
        objectness_logits = objectness_logits[keep]

        proposals = Instances(image_shape)
        proposals.proposal_boxes = boxes[:proposal_topk]
        proposals.objectness_logits = objectness_logits[:proposal_topk]
        dataset_dict["proposals"] = proposals


def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # Note that bbox is 1d (per-instance bounding box)
    annotation["bbox"] = transforms.apply_box([bbox])[0]
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )

    if "keypoints" in annotation:
        keypoints = transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["keypoints"] = keypoints

    if "maskpoints" in annotation:
        keypoints = transform_keypoint_annotations(
            annotation["maskpoints"], transforms, image_size, keypoint_hflip_indices
        )
        annotation["maskpoints"] = keypoints

    return annotation

def transform_instance_annotations_rotated(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):

    # bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYWHA_ABS)
    bbox= annotation["bbox"]
    bbox.append(annotation['angle'])
    from detectron2.data.transforms.transform import Resize_rotated_box
    annotation["bbox"] = Resize_rotated_box(transforms.transforms[0], np.array([bbox]))[0]
    annotation["bbox_mode"] = BoxMode.XYWHA_ABS

# =============================================================================
#     if "segmentation" in annotation:
#         # each instance contains 1 or more polygons
#         segm = annotation["segmentation"]
#         if isinstance(segm, list):
#             # polygons
#             polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
#             annotation["segmentation"] = [
#                 p.reshape(-1) for p in transforms.apply_polygons(polygons)
#             ]
#         elif isinstance(segm, dict):
#             # RLE
#             mask = mask_util.decode(segm)
#             mask = transforms.apply_segmentation(mask)
#             assert tuple(mask.shape[:2]) == image_size
#             annotation["segmentation"] = mask
#         else:
#             raise ValueError(
#                 "Cannot transform segmentation of type '{}'!"
#                 "Supported types are: polygons as list[list[float] or ndarray],"
#                 " COCO-style RLE as a dict.".format(type(segm))
#             )
# =============================================================================

    return annotation


def transform_keypoint_annotations(keypoints, transforms, image_size, keypoint_hflip_indices=None):
    """
    Transform keypoint annotations of an image.

    Args:
        keypoints (list[float]): Nx3 float in Detectron2 Dataset format.
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
    """
    # (N*3,) -> (N, 3)
    keypoints = np.asarray(keypoints, dtype="float64").reshape(-1, 3)
    keypoints[:, :2] = transforms.apply_coords(keypoints[:, :2])

    # This assumes that HorizFlipTransform is the only one that does flip
    do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.transforms) % 2 == 1

    # Alternative way: check if probe points was horizontally flipped.
    # probe = np.asarray([[0.0, 0.0], [image_width, 0.0]])
    # probe_aug = transforms.apply_coords(probe.copy())
    # do_hflip = np.sign(probe[1][0] - probe[0][0]) != np.sign(probe_aug[1][0] - probe_aug[0][0])  # noqa

    # If flipped, swap each keypoint with its opposite-handed equivalent
    if do_hflip:
        assert keypoint_hflip_indices is not None
        keypoints = keypoints[keypoint_hflip_indices, :]

    # Maintain COCO convention that if visibility == 0, then x, y = 0
    # TODO may need to reset visibility for cropped keypoints,
    # but it does not matter for our existing algorithms
    keypoints[keypoints[:, 2] == 0] = 0
    keypoints[keypoints[:, 2] == 1] = 0 # Pawn temp
    return keypoints


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes
    
    #Pawn temp
    angles = [obj["angle"] for obj in annos]
    angles = torch.tensor(angles, dtype=torch.int64)
    target.gt_angles = angles
    ###

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)
    if len(annos) and "maskpoints" in annos[0]:
        mkpts = []
        for obj in annos:
            maskpoints = obj.get("maskpoints")
            # np.random.shuffle(maskpoints[1:])
            mkpts.append(maskpoints)
        target.gt_maskpoints = Keypoints(mkpts)

    return target


def annotations_to_instances_iterdet(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    
    random.shuffle(annos)
    old_n = np.random.randint(0, len(annos))
    new_n = len(annos) - old_n
    
    annos_old = annos[:old_n]
    annos_new = annos[old_n:]
    
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos_new]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)
    
    if(old_n==0):
        target._history = torch.zeros(1, image_size[0], image_size[1], device='cuda')
    else:
        height, width = image_size
        boxes_old = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos_old]
        boxes_old = np.array(boxes_old)
        np.clip(boxes_old[:,0], 0, width, out=boxes_old[:,0])
        np.clip(boxes_old[:,1], 0, height, out=boxes_old[:,1])
        np.clip(boxes_old[:,2], 0, width, out=boxes_old[:,2])
        np.clip(boxes_old[:,3], 0, height, out=boxes_old[:,3])
        history = np.zeros((1, height, width), dtype=np.uint8)
        for bbox in boxes_old:
            x_min = max(int(round(bbox[0])), 0)
            y_min = max(int(round(bbox[1])), 0)
            x_max = min(int(round(bbox[2])), width - 1)
            y_max = min(int(round(bbox[3])), height - 1)
            history[0, y_min: y_max + 1, x_min: x_max + 1] += 1
        target._history = torch.tensor(history, dtype=torch.float, device='cuda')


    

    classes = [obj["category_id"] for obj in annos_new]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes
    
    #Pawn temp
    angles = [obj["angle"] for obj in annos_new]
    angles = torch.tensor(angles, dtype=torch.int64)
    target.gt_angles = angles
    ###

    if len(annos_new) and "segmentation" in annos_new[0]:
        segms = [obj["segmentation"] for obj in annos_new]
        if mask_format == "polygon":
            masks = PolygonMasks(segms)
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a full-image segmentation mask "
                        "as a 2D ndarray.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos_new) and "keypoints" in annos_new[0]:
        kpts = [obj.get("keypoints", []) for obj in annos_new]
        target.gt_keypoints = Keypoints(kpts)
    if len(annos_new) and "maskpoints" in annos_new[0]:
        mkpts = []
        for obj in annos_new:
            maskpoints = obj.get("maskpoints")
            # np.random.shuffle(maskpoints[1:])
            mkpts.append(maskpoints)
        target.gt_maskpoints = Keypoints(mkpts)

    return target


def annotations_to_instances_rotated(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Compared to `annotations_to_instances`, this function is for rotated boxes only

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            Containing fields "gt_boxes", "gt_classes",
            if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [obj["bbox"] for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = RotatedBoxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes
    return target

def filter_empty_instances(instances, by_box=True, by_mask=True):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks

    Returns:
        Instances: the filtered instances.
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty())
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.nonempty())

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        m = m & x
    return instances[m]


def create_keypoint_hflip_indices(dataset_names):
    """
    Args:
        dataset_names (list[str]): list of dataset names
    Returns:
        ndarray[int]: a vector of size=#keypoints, storing the
        horizontally-flipped keypoint indices.
    """

    check_metadata_consistency("keypoint_names", dataset_names)
    check_metadata_consistency("keypoint_flip_map", dataset_names)

    meta = MetadataCatalog.get(dataset_names[0])
    names = meta.keypoint_names
    # TODO flip -> hflip
    flip_map = dict(meta.keypoint_flip_map)
    flip_map.update({v: k for k, v in flip_map.items()})
    flipped_names = [i if i not in flip_map else flip_map[i] for i in names]
    flip_indices = [names.index(i) for i in flipped_names]
    return np.asarray(flip_indices)


def gen_crop_transform_with_instance(crop_size, image_size, instance):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    crop_size = np.asarray(crop_size, dtype=np.int32)
    bbox = BoxMode.convert(instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5
    assert (
        image_size[0] >= center_yx[0] and image_size[1] >= center_yx[1]
    ), "The annotation bounding box is outside of the image!"
    assert (
        image_size[0] >= crop_size[0] and image_size[1] >= crop_size[1]
    ), "Crop size is larger than image size!"

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)
    return T.CropTransform(x0, y0, crop_size[1], crop_size[0])


def check_metadata_consistency(key, dataset_names):
    """
    Check that the datasets have consistent metadata.

    Args:
        key (str): a metadata key
        dataset_names (list[str]): a list of dataset names

    Raises:
        AttributeError: if the key does not exist in the metadata
        ValueError: if the given datasets do not have the same metadata values defined by key
    """
    if len(dataset_names) == 0:
        return
    logger = logging.getLogger(__name__)
    entries_per_dataset = [getattr(MetadataCatalog.get(d), key) for d in dataset_names]
    for idx, entry in enumerate(entries_per_dataset):
        if entry != entries_per_dataset[0]:
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(key, dataset_names[idx], str(entry))
            )
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(
                    key, dataset_names[0], str(entries_per_dataset[0])
                )
            )
            raise ValueError("Datasets have different metadata '{}'!".format(key))


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    logger = logging.getLogger(__name__)
    tfm_gens = []
    
    if(not cfg.INPUT.IS_ABSOLUTE_SIZE):
        tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    else:
        tfm_gens.append(T.Resize((cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH)))
        print(f"[Logging] Fixed resize [{cfg.INPUT.HEIGHT},{cfg.INPUT.WIDTH}]")
        
    print("\n\n.......TransformGens is used now......\n\n")
    if is_train:
        if(cfg.get('DATASETS').get('Flip_ON')): tfm_gens.append(T.RandomFlip()) # Pawn temp
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens

