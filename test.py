import json
import argparse

import torch
import numpy as np
import nibabel as nib
from skimage.measure import label
from skimage.morphology import binary_erosion, binary_dilation

from monai.transforms import (
    AsDiscreted,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    Resized,
    Invertd,
    ToDeviced,
    EnsureTyped,
)
from monai.data import DataLoader, PersistentDataset, decollate_batch

from model import CraNeXt


def merge_batch(batch_pred: list, post_pred: Compose):
    post_pred_batch = [post_pred(i) for i in decollate_batch(batch_pred)]
    pred_img = post_pred_batch[0]["pred"][0, :, :, :].detach().cpu().numpy()
    return pred_img


def largest_connected_component(segmentation):
    labels = label(segmentation)
    largest_cc = labels == np.argmax(
        np.bincount(labels.flat, weights=segmentation.flat)
    )
    return largest_cc


def save_prediction_array_to_nifti(
    prediction_data: np.array, orig_defective_nii_path: str, output_path: str
):
    orig_defect_img = nib.load(orig_defective_nii_path)
    orig_defect_data = orig_defect_img.get_fdata()

    implant_data = prediction_data - orig_defect_data

    implant_data[implant_data < 0] = 0
    implant_data = binary_erosion(implant_data).astype(int)
    implant_data = binary_erosion(implant_data).astype(int)
    implant_data = largest_connected_component(implant_data)
    implant_data = binary_dilation(implant_data).astype(int)
    implant_data = binary_dilation(implant_data).astype(int)

    defect_skull_data = orig_defect_data.copy()
    defect_skull_data[implant_data > 0] = 2
    defect_skull_data = defect_skull_data.clip(0, 2)
    ni_img = nib.Nifti1Image(
        defect_skull_data.astype(np.uint8),
        affine=orig_defect_img.affine,
        header=orig_defect_img.header,
    )
    ni_img.header.set_data_dtype(np.uint8)
    nib.save(ni_img, output_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="define filename")
    parser.add_argument("--model_path", type=str, help="define input model path")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model_path = args.model_path
    name = args.name

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    model = CraNeXt()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    model_path = f"model/{name}.pth"
    test_dataset = "test_dataset.json"
    # e.g. [{"image": "skullbreak/001_frontoorbital_defective.nii.gz"}, ...]

    with open(test_dataset) as f:
        test_files = json.load(f)

    for skull in test_files:
        output_path = (
            skull["image"]
            .replace("/data/", "/pred/")
            .replace("_defective.nii.gz", f"_{name}.nii.gz")
        )
        skull["defective_path"] = skull["image"]
        skull["output_path"] = output_path

    test_transforms = Compose(
        [
            LoadImaged(
                keys="image",
                ensure_channel_first=True,
                image_only=True,
                dtype=torch.float,
            ),
            Orientationd(keys="image", axcodes="RAS"),
            ToDeviced(keys="image", device=device),
            CropForegroundd(
                keys="image", source_key="image", margin=25, allow_smaller=False
            ),
            Resized(
                keys="image",
                spatial_size=(176, 224, 144),
                mode="area",
            ),
            EnsureTyped(keys="image", dtype=torch.float, device=device),
        ]
    )

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=test_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred", threshold=0.5),
        ]
    )

    test_ds = PersistentDataset(
        data=test_files, transform=test_transforms, cache_dir="cache"
    )
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

    for batch_data in test_loader:
        inputs = batch_data["image"].to(device)
        batch_data["pred"] = model(inputs)
        bd = [x for x in decollate_batch(batch_data)]
        image_output = merge_batch(batch_data, post_transforms)
        save_prediction_array_to_nifti(
            image_output, bd[0]["defective_path"], bd[0]["output_path"]
        )


if __name__ == "__main__":
    main()
