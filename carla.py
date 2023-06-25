import os
from PIL import Image
import torch
from torch.utils.data import Dataset


class CarlaImagesDataset(Dataset):
    def __init__(self, data_dir, feature_extractor):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.image_dir = os.path.join(data_dir, 'images')
        self.label_dir = os.path.join(data_dir, 'labels')
        self.classes_csv_file = os.path.join(self.data_dir, "_classes.csv")
        with open(self.classes_csv_file, 'r') as fid:
            data = [l.split(',') for i, l in enumerate(fid) if i != 0]
        self.id2label = {x[0]: x[1] for x in data}
        self.towns = ['town01', 'town02', 'town07', 'town10']
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for town in self.towns:
            image_path = os.path.join(self.image_dir, town)
            label_path = os.path.join(self.label_dir, town+'_cs_id')
            image_files = os.listdir(image_path)
            for img_file in image_files:
                img_name = img_file.split('.')[0]
                label_file = img_name + '.png'
                sample = {
                    'image': os.path.join(image_path, img_file),
                    'label': os.path.join(label_path, label_file)
                }
                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self._load_image(sample['image'])
        label = self._load_label(sample['label'])
        encoded_inputs = self.feature_extractor(image, label, return_tensors="pt")
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()
        return encoded_inputs

    def _load_image(self, image_path):
        with Image.open(image_path) as img:
            img = img.convert('RGB')
        return img

    def _load_label(self, label_path):
        with Image.open(label_path) as label:
            # Preprocess the label image if needed
            # e.g., convert to grayscale, apply normalization
            label = label.convert('L')
        return label
