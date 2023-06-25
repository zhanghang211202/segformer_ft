from SegformerFinetuner import SegformerFinetuner
from transformers import SegformerFeatureExtractor
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
import os
from utils.get_dataset_colormap import create_cityscapes_label_colormap
import numpy as np
from transformers import SegformerImageProcessor

image_root = "D:/WorkSpace/Carla1/carla_images_ft/images/town07/00047948.png"
# image_root = "D:/WorkSpace/dataset/L&R/leftImg8bit/leftImg8bit/train/01_Hanns_Klemm_Str_45/01_Hanns_Klemm_Str_45_000000_000270_leftImg8bit.png"
image = Image.open(image_root)

data_dir = "D:/WorkSpace/Carla1/carla_images_ft"
classes_csv_file = os.path.join(data_dir, "_classes.csv")
with open(classes_csv_file, 'r') as fid:
    data = [l.split(',') for i, l in enumerate(fid) if i != 0]
id2label = {x[0]: x[1] for x in data}
label2id = {v: k for k, v in id2label.items()}

model = SegformerFinetuner.load_from_checkpoint("./check_points/epoch=18-step=3040.ckpt",id2label=id2label)
model.eval()
feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
feature_extractor.do_reduce_labels = False
inputs = feature_extractor(images=image, return_tensors="pt")
inputs.to(model.device)

with torch.no_grad():
    outputs = model.predict((inputs['pixel_values']))

logits = outputs[0]
upsampled_logits = nn.functional.interpolate(
    logits,
    size=image.size[::-1],
    mode="bilinear",
    align_corners=True,
)

pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
max_logits = upsampled_logits.max(dim=1)[0].squeeze(0)
max_logits_array = max_logits.cpu().numpy()

# pred_seg to color mask using get_dataset_colormap
palette = create_cityscapes_label_colormap()
color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
for label, color in enumerate(palette):
    color_seg[pred_seg == label, :] = color


print(1)
