# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
import time
import weakref
import torch

import detectron2.utils.comm as comm
from detectron2.utils.events import EventStorage

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer"]


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:

    .. code-block:: python

        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        hook.after_train()

    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self):
        self._hooks = []
        '''By Pawn'''
        try:
            from .init import Init
            self.tfControl = Init()
            self.tfControl.getLossName('Total_Loss')
            # self.tfControl.getLossName(['Total_Loss'])
#            self.tfControl.getLossName(['Total_Loss', 'Total_acc'])
            print('Successful in SolVision')
        except:
            print('Not in SolVision or fails')
            
        
    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    '''By Pawn'''
                    try:
                        # print('{0} {1}'.format(self.iter, max_iter))
#                        if(self.iter %100 == 0):
#                            print('{0} {1}'.format(self.iter, max_iter))
#                            print(time.time()-s_time)
#                            s_time=time.time()
                        self.tfControl.getEpoch(int(self.iter/max_iter*100))
                        self.tfControl.getLoss(self.storage._history['total_loss']._data[-1][0]*100) #Single Loss pass
                        # print(self.storage._history['total_loss']._data[-1][0])

#                        self.tfControl.getLoss('Total_Loss', self.storage._history['total_loss']._data[-1][0]*100) #Multi Loss pass
                        
                        if(self.tfControl.NeedToSaveModel()):
                            self.iter=max_iter-1
                            self.after_step()
                            break 
                        if(self.tfControl.NeedToStopTrain()):
                            break     
                    except:
                        pass
                    ''''''
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        # this guarantees, that in each hook's after_step, storage.iter == trainer.iter
        self.storage.step()

    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer, optimizer_2=None, data_loader_er=None):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer
        self.optimizer_2 = optimizer_2
        
        # OPTION ER
        if data_loader_er is not None:    
            self.data_loader_er = data_loader_er
            self._data_loader_er_iter = iter(data_loader_er)
            print('[ER INFO] data loader for ER has been initialized...')
        # print(model)
        
        params=[]
        for name, param in model.named_parameters():
            params.append([name, param.requires_grad])
        print()
        
# =============================================================================
#         params=[]
#         for name, param in model.named_parameters():
#             if('angle_head' in name):
#                 pass
#             else:
#                 param.requires_grad = False
#             params.append(name)
#         print()
# =============================================================================

    # OPTION ER
    def add_er(self, data):
        # TODO: ADD dataloader here to be combined with the original training images
        if hasattr(self, '_data_loader_er_iter'):
            # ref: https://github.com/pytorch/pytorch/issues/1917
            data_er = next(self._data_loader_er_iter)
            print('[ER INFO] call next er data...')
            return data + data_er
        return data

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start
        
        # OPTION ER
        data = self.add_er(data)
        """
        If you want to do something with the losses, you can wrap the model.
        """
        if data[0].get('annotations', False) != False:
            if len(data[0]['annotations']) == 0:
                return
        loss_dict = self.model(data) 
# =============================================================================
#         for k in loss_dict.keys():
#             if not torch.isfinite(loss_dict[k]):
#                 loss_dict[k] = torch.zeros_like(loss_dict[k], requires_grad=True)  
# =============================================================================
            
            
        losses = sum(loss_dict.values())
        
# =============================================================================
#         if not torch.isfinite(losses).all():
#             print(f"\n\nSkip loss nan iter {self.iter}\n\n")
#             return
# =============================================================================
        
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need to accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()
        
# =============================================================================
#     def run_step(self):
#         """
#         Implement the standard training logic described above.
#         """
#         assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
#         start = time.perf_counter()
#         """
#         If you want to do something with the data, you can wrap the dataloader.
#         """
#         data = next(self._data_loader_iter)
#         data_time = time.perf_counter() - start
# 
#         """
#         If you want to do something with the losses, you can wrap the model.
#         """
#         loss_dict = self.model(data)
#         #Pawn temp
#         loss_dict_ori = {'loss_box_reg' : loss_dict['loss_box_reg'],
#                          'loss_cls' : loss_dict['loss_cls'],
#                          'loss_mask': loss_dict['loss_mask'],
#                          'loss_rpn_cls': loss_dict['loss_rpn_cls'],
#                          'loss_rpn_loc': loss_dict['loss_rpn_loc']}
#         
#         loss_dict_angle = {'loss_ag1' : loss_dict['loss_ag1'],
#                            'loss_ag2' : loss_dict['loss_ag2'],
#                            'loss_ag3' : loss_dict['loss_ag3'],
#                            'loss_ag4' : loss_dict['loss_ag4'],
#                            'loss_ag5' : loss_dict['loss_ag5'],
#                            'loss_ag6' : loss_dict['loss_ag6'],
#                            'loss_ag7' : loss_dict['loss_ag7'],
#                            'loss_ag8' : loss_dict['loss_ag8'],
#                            'loss_ag9' : loss_dict['loss_ag9']} 
#         
#         losses = sum(loss_dict.values())
#         self._detect_anomaly(losses, loss_dict)
# 
#         metrics_dict = loss_dict
#         metrics_dict["data_time"] = data_time
#         self._write_metrics(metrics_dict)
# 
#         """
#         If you need to accumulate gradients or something similar, you can
#         wrap the optimizer with your custom `zero_grad()` method.
#         """
#         
#         losses_ori = sum(loss_dict_ori.values())
#         losses_angle = sum(loss_dict_angle.values())
#         
#         self.optimizer.zero_grad()
#         losses_ori.backward(retain_graph=True)
#         self.optimizer.step()
#         
#         self.optimizer_2.zero_grad()
#         losses_angle.backward()
#         self.optimizer_2.step()
# =============================================================================


    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)
