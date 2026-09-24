"""Evaluate a trained model on an explicit spatial split.

The default split is ``test``. The test set must not be used for checkpoint,
threshold, or hyperparameter selection.
"""

import argparse
import json
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CDDataset
from train import ALL_MODELS, PRIOR_MODELS, build_model, set_global_seed
from utils import MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=str, required=True, choices=ALL_MODELS)
    parser.add_argument('--pth', type=str, required=True)
    parser.add_argument(
        '--data_root', type=str,
        default=r"D:/yoyu/SA_Identification/dataset_patches_2020_2024")
    parser.add_argument('--split_manifest', type=str, default=None)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--prior_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = args.split_manifest or os.path.join(
        args.data_root, 'spatial_split_manifest.csv')

    dataset = CDDataset(
        args.data_root,
        split=args.split,
        transform=False,
        prior_dir_name=args.prior_dir,
        manifest_path=manifest_path,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = build_model(args.model, device)
    state_dict = torch.load(args.pth, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    tracker = MetricTracker()
    tracker.reset()
    with torch.no_grad():
        for img_a, img_b, label, prior in tqdm(loader, desc=f"Evaluating {args.split}"):
            img_a = img_a.to(device)
            img_b = img_b.to(device)
            label = label.to(device)
            prior = prior.to(device)
            output = (
                model(img_a, img_b, prior)
                if args.model in PRIOR_MODELS
                else model(img_a, img_b)
            )
            if isinstance(output, list):
                output = output[-1]
            tracker.update(output, label)

    metrics = tracker.get_metrics()
    result = {
        'model': args.model,
        'checkpoint': os.path.abspath(args.pth),
        'split': args.split,
        'split_manifest': os.path.abspath(manifest_path),
        'sample_count': len(dataset),
        'seed': args.seed,
        'metrics': metrics,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as handle:
            json.dump(result, handle, indent=2)


if __name__ == '__main__':
    main()
