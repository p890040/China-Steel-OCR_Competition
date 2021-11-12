# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pickle
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm

from .c2_model_loading import align_and_update_state_dicts
import os

class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron & detectron2
    model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )
        

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
                
# =============================================================================
#             # Pawn temp. Load Both maskrcnn and keypoint weight. 
#             import os
#             if(os.path.exists(os.path.join(os.path.dirname(filename), 'k.pkl'))):
#                 with PathManager.open(os.path.join(os.path.dirname(filename), 'k.pkl'), "rb") as f:
#                     keypoint_data = pickle.load(f, encoding="latin1") 
#                 self.logger.info("Reading a file from '{}'".format(data["__author__"]))
#                 
#                 for k in list(keypoint_data['model'].keys()):
#                     if 'keypoint_head' in k:
#                         data['model'][k] = keypoint_data['model'][k]
#                 return data
#             # end
# =============================================================================
            
            
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}
        loaded = super()._load_file(filename)  # load native pth checkpoint
        

        if "model" not in loaded:
            loaded = {"model": loaded}
        # Pawn temp. Delete optimizer and lr_schedule and so on.
        return {"model": loaded['model']}
    
    
    
        # '''Pawn temp'''
        # if(len(loaded['model']) ==585):
        #     assert  len(loaded['optimizer']['state']) == 153
        #     assert  len(loaded['optimizer']['param_groups']) == 153
        #     assert  len(loaded['scheduler']['base_lrs']) == 153
        # elif(len(loaded['model']) ==567):
        #     assert  len(loaded['optimizer']['state']) == 135
        #     assert  len(loaded['optimizer']['param_groups']) == 135
        #     assert  len(loaded['scheduler']['base_lrs']) == 135
        # ''''''
        
        
        # self.checkpointables['optimizer']['param']
        # times = 18
        # param_group = loaded['optimizer']['param_groups'][-1]
        # state = loaded['optimizer']['state'][list(loaded['optimizer']['state'].keys())[0]]
        # for i in range(times):
        #     stateKey = i
        #     loaded['optimizer']['param_groups'].append(param_group)
        #     loaded['optimizer']['param_groups'][-1]['params'] = [stateKey]
        #     loaded['optimizer']['state'][str(stateKey)] = state

        
        

        return loaded

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            model_state_dict = self.model.state_dict()
            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict
        # for non-caffe2 models, use standard ways to load it
        super()._load_model(checkpoint)
