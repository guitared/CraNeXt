import json
import argparse

import torch
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    ToDeviced,
    Orientationd,
    Resized,
    RandFlipd,
    EnsureTyped,
)

from monai.utils import set_determinism
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.data import DataLoader, PersistentDataset, decollate_batch


from model import CraNeXt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="define filename")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    name = args.name

    set_determinism(seed=2024)

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    torch.cuda.set_device(0)

    model = CraNeXt()

    image_size = (176, 224, 144)
    batch_size = 1
    n_epochs = 300
    model_path = f"model/{name}.pth"
    train_dataset = "train_dataset.json"
    validation_dataset = "validation_dataset.json"
    # e.g. [{"image": "skullbreak/001_frontoorbital_defective.nii.gz",
    #        "label": "skullbreak/001_complete.nii.gz"}, ...]

    with open(train_dataset) as f:
        train_files = json.load(f)

    with open(validation_dataset) as f:
        val_files = json.load(f)

    train_transforms = Compose(
        [
            LoadImaged(
                keys=["image", "label"],
                ensure_channel_first=True,
                image_only=True,
                dtype=torch.float,
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ToDeviced(keys=["image", "label"], device=device),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                margin=25,
                allow_smaller=False,
            ),
            Resized(
                keys=["image", "label"],
                spatial_size=image_size,
                mode=["area", "nearest"],
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            EnsureTyped(
                keys=["image", "label"],
                dtype=torch.float,
                device=device,
                track_meta=False,
            ),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(
                keys=["image", "label"],
                ensure_channel_first=True,
                image_only=True,
                dtype=torch.float,
            ),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ToDeviced(keys=["image", "label"], device=device),
            CropForegroundd(
                keys=["image", "label"],
                source_key="image",
                margin=25,
                allow_smaller=False,
            ),
            Resized(
                keys=["image", "label"],
                spatial_size=image_size,
                mode=["area", "nearest"],
            ),
            EnsureTyped(
                keys=["image", "label"],
                dtype=torch.float,
                device=device,
                track_meta=False,
            ),
        ]
    )

    train_ds = PersistentDataset(
        data=train_files, transform=train_transforms, cache_dir="cache"
    )
    val_ds = PersistentDataset(
        data=val_files, transform=val_transforms, cache_dir="cache"
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_function = DiceCELoss()
    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )

    best_metric = -1

    post_pred = Compose([AsDiscrete(threshold=0.5)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        for _, item in enumerate(train_loader):
            step += 1
            inputs = item["image"]
            labels = item["label"]
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            outputs = model(inputs)
            train_outputs = [post_pred(i) for i in decollate_batch(outputs)]
            train_labels = [post_label(i) for i in decollate_batch(labels)]
            dice_metric(y_pred=train_outputs, y=train_labels)
            dice = dice_metric.aggregate().item()
            train_loss = loss_function(outputs, labels)
            train_loss.backward()
            optimizer.step()
            epoch_loss += train_loss.item()
        train_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        epoch_loss /= step

        model.eval()
        torch.save(model.state_dict(), model_path + ".checkpoint")
        with torch.no_grad():
            for i, item in enumerate(val_loader):
                inputs = item["image"]
                labels = item["label"]
                outputs = model(inputs)
                val_outputs = [post_pred(i) for i in decollate_batch(outputs)]
                val_labels = [post_label(i) for i in decollate_batch(labels)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice = dice_metric.aggregate().item()
                val_loss = loss_function(outputs, labels)
        val_dice = dice_metric.aggregate().item()
        dice_metric.reset()
        if val_dice > best_metric:
            best_metric = val_dice
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
