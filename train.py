from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import os

register_coco_instances("train", {}, './baobabs_dataset/train/train.json', './baobabs_dataset/train')
register_coco_instances("val", {}, './baobabs_dataset/val/val.json', './baobabs_dataset/val')
register_coco_instances("test", {}, './baobabs_dataset/test/test.json', './baobabs_dataset/test')

my_dataset_train_metadata = MetadataCatalog.get("train")
dataset_dicts = DatasetCatalog.get("train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("train",)
cfg.DATASETS.TEST = ("val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001 # base learning rate
cfg.SOLVER.WARMUP_ITERS = 1000 # number of iteration
cfg.SOLVER.MAX_ITER = 1501
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64 # batch size per image
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 #  your number of classes (Number of foreground classes)
cfg.TEST.EVAL_PERIOD = 500

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "test")
inference_on_dataset(trainer.model, val_loader, evaluator)

test_metadata = MetadataCatalog.get("test")
dataset_dicts = DatasetCatalog.get("test")

for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                metadata=test_metadata,
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    filepath = './output_images/'+d["file_name"].split('/')[-1]
    cv2.imwrite(filepath, out.get_image()[:, :, ::-1])