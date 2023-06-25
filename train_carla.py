import random

from transformers import SegformerImageProcessor
from SegformerFinetuner import SegformerFinetuner
from carla import CarlaImagesDataset
from torch.utils.data import Dataset, DataLoader
import torch

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
from multiprocessing import freeze_support

def main():
    seed = torch.Generator().manual_seed(42)
    data_root = "D:/WorkSpace/Carla1/carla_images_ft"

    feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    feature_extractor.do_reduce_labels = False

    dataset = CarlaImagesDataset(data_root, feature_extractor)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
                                                               [int(len(dataset) * 0.8), int(len(dataset) * 0.2)],
                                                               generator=seed)

    batch_size = 2
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, prefetch_factor=8)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=1, prefetch_factor=8)
    id2label = dataset.id2label
    num_classes = len(id2label)
    label2id = {v: k for k, v in id2label.items()}
    torch.set_float32_matmul_precision('high')

    # jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

    segformer_finetuner = SegformerFinetuner(
        dataset.id2label,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        # test_dataloader=test_dataloader,
        metrics_interval=10,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")
    trainer = pl.Trainer(
        num_nodes=1,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=50,
        val_check_interval=len(train_dataloader),
    )

    trainer.fit(segformer_finetuner)

if __name__ == '__main__':
    # freeze_support()
    main()

