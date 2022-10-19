#! /usr/bin/env python3

import json
import os
import random

import cv2
import detectron2.utils.comm as comm
import matplotlib.pyplot as plt
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import transforms as T
from detectron2.data.datasets.coco import (load_coco_json,
                                           register_coco_instances)
from detectron2.engine import (DefaultPredictor, DefaultTrainer, HookBase,
                               ValidationLoss)
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode, Visualizer

classes = ['Lamps']
base_path = '../../Lamps.v3i.coco/'

register_coco_instances("train_set",{}, base_path + "train/_annotations.coco.json", base_path + "train")
register_coco_instances("valid_set",{}, base_path + "valid/_annotations.coco.json", base_path +"valid")
register_coco_instances("test_set",{}, base_path +  "test/_annotations.coco.json", base_path + "test")

microcontroller_metadata = MetadataCatalog.get("train_set")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train_set",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
# cfg.INPUT.MIN_SIZE_TRAIN=(416)

cfg.DATASETS.VAL = ("valid_set",)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 


val_loss = ValidationLoss(cfg)  
trainer.register_hooks([val_loss])
trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

trainer.resume_or_load(resume=False)

trainer.train()

print("Train finished")




