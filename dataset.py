"""
dataset.py — Change Detection Dataset
======================================================================
目录结构（root_dir 下）：
    A/                  T1 影像切片（.npy，[C, H, W]）
    B/                  T2 影像切片（.npy，[C, H, W]）
    label/              变化标签（.npy 或 .png，[H, W]）
    spatial_prior/      先验切片（通用名）
  或
    spatial_prior_gwr/  GWR 先验切片
    spatial_prior_gwda/ GWDA 先验切片

先验文件夹查找优先级：
    spatial_prior → spatial_prior_gwr → spatial_prior_gwda → prior
找不到时不报错，返回全零占位（供 BiT_Online 纯在线模式使用）。

返回：(imgA, imgB, label, spatial_prior)
    imgA / imgB     : FloatTensor [C, H, W]，取前 8 通道
    label           : FloatTensor [1, H, W]，二值 {0, 1}
    spatial_prior   : FloatTensor [1, H, W]，值域 [0, 1]
======================================================================
"""

import csv
import hashlib
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CDDataset(Dataset):

    _LABEL_DIRS = ['label', 'Label', 'labels', 'Labels']

    def __init__(self, root_dir: str, split: str = 'train',
                 split_ratio: float = 0.85, transform: bool = True,
                 prior_dir_name: str = None, manifest_path: str = None,
                 allow_random_split: bool = False,
                 prior_control: str = 'none', control_seed: int = 42):
        """
        Args:
            prior_dir_name: 先验文件夹名称，明确指定用哪个先验。
                            例如 'spatial_prior_gwr' 或 'spatial_prior_gwda'。
                            若为 None，则按优先级自动查找。
                            若文件夹不存在，返回全零占位（在线先验模式）。
            manifest_path: 空间独立划分清单。CSV 至少包含 filename 和 split。
            allow_random_split: 仅用于复现旧实验；不适合作为空间独立验证。
        """
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f"split 必须是 train、val 或 test，收到：{split}")

        self.root_dir  = root_dir
        self.transform = transform
        valid_prior_controls = {
            'none', 'shuffled', 'random', 'zero', 'constant',
            'random_per_epoch',
        }
        if prior_control not in valid_prior_controls:
            raise ValueError(
                f"prior_control 必须是 {sorted(valid_prior_controls)} 之一")
        self.prior_control = prior_control
        self.control_seed = control_seed
        self.epoch = 0

        # 标签文件夹（必须存在）
        self.label_dir = self._find_dir(self._LABEL_DIRS)
        if self.label_dir is None:
            raise FileNotFoundError(
                f"找不到 label 文件夹，已搜索：{self._LABEL_DIRS}\n"
                f"根目录：{root_dir}")

        # 先验文件夹（明确指定或自动查找）
        if prior_dir_name is not None:
            candidate = os.path.join(root_dir, prior_dir_name)
            self.prior_dir = candidate if os.path.exists(candidate) else None
            if self.prior_dir is None:
                print(f"[{split.upper()}] 指定的先验文件夹 '{prior_dir_name}' 不存在，"
                      "使用全零占位")
            else:
                print(f"[{split.upper()}] 先验文件夹（指定）：{prior_dir_name}")
        else:
            fallback = ['spatial_prior_gwr', 'spatial_prior_gwda',
                        'spatial_prior', 'prior']
            self.prior_dir = self._find_dir(fallback)
            if self.prior_dir is None:
                print(f"[{split.upper()}] 未找到先验文件夹，使用全零占位（在线先验模式）")
            else:
                print(f"[{split.upper()}] 先验文件夹（自动）：{os.path.basename(self.prior_dir)}")

        # 文件列表（支持 .npy / .png）
        all_files = os.listdir(self.label_dir)
        npy_files = [f for f in all_files if f.endswith('.npy')]
        png_files = [f for f in all_files if f.endswith('.png')]

        if npy_files:
            self.files  = npy_files
            self.is_npy = True
        elif png_files:
            self.files  = png_files
            self.is_npy = False
        else:
            raise ValueError(
                f"在 {self.label_dir} 下未找到 .npy 或 .png 文件")

        if manifest_path is not None:
            self.file_list = self._load_manifest(manifest_path, split)
        elif allow_random_split:
            if split == 'test':
                raise ValueError("旧的随机切片方案没有独立 test 集")
            print(
                "[WARNING] 正在使用旧的随机 patch 划分。相邻重叠切片可能跨越 "
                "train/val，仅可用于复现旧结果。"
            )
            files = np.array(sorted(self.files))
            np.random.default_rng(42).shuffle(files)
            n_train = int(len(files) * split_ratio)
            self.file_list = (files[:n_train] if split == 'train'
                              else files[n_train:]).tolist()
        else:
            raise ValueError(
                "必须提供 manifest_path 以使用空间独立划分。若仅需复现旧实验，"
                "请显式设置 allow_random_split=True。"
            )

        missing = [name for name in self.file_list
                   if not os.path.exists(os.path.join(self.label_dir, name))]
        if missing:
            raise FileNotFoundError(
                f"划分清单中的 {len(missing)} 个标签文件不存在，例如：{missing[:3]}")

        self.prior_name_map = {name: name for name in self.file_list}
        if self.prior_control == 'shuffled':
            rng = np.random.default_rng(control_seed)
            shuffled = np.asarray(self.file_list, dtype=object).copy()
            if len(shuffled) > 1:
                for _ in range(100):
                    rng.shuffle(shuffled)
                    if all(left != right for left, right in
                           zip(self.file_list, shuffled)):
                        break
                else:
                    shuffled = np.roll(np.asarray(self.file_list, dtype=object), 1)
            self.prior_name_map = dict(zip(self.file_list, shuffled.tolist()))

        print(f"[{split.upper()}] 共 {len(self.file_list)} 个切片；"
              f"prior_control={self.prior_control}")

    # ── 工具 ──────────────────────────────────────────────────────────
    def _find_dir(self, candidates: list):
        for name in candidates:
            path = os.path.join(self.root_dir, name)
            if os.path.exists(path):
                return path
        return None

    def _load_manifest(self, manifest_path: str, split: str) -> list:
        if not os.path.isabs(manifest_path):
            manifest_path = os.path.join(self.root_dir, manifest_path)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"找不到空间划分清单：{manifest_path}")

        selected = []
        seen = set()
        with open(manifest_path, newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            required = {'filename', 'split'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"划分清单必须包含 {sorted(required)} 列，"
                    f"实际列为：{reader.fieldnames}")
            for row in reader:
                filename = row['filename'].strip()
                row_split = row['split'].strip().lower()
                if filename in seen:
                    raise ValueError(f"划分清单包含重复文件：{filename}")
                seen.add(filename)
                if row_split == split:
                    selected.append(filename)

        if not selected:
            raise ValueError(f"划分清单中没有 split={split} 的样本")
        return sorted(selected)

    # ── Dataset 接口 ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.file_list)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by deterministic epoch-varying controls."""
        self.epoch = int(epoch)

    def __getitem__(self, idx: int):
        fname     = self.file_list[idx]
        npy_fname = fname if self.is_npy else fname.replace('.png', '.npy')

        # 标签
        label_path = os.path.join(self.label_dir, fname)
        if self.is_npy:
            label = np.load(label_path).astype(np.float32)
        else:
            label = (np.array(Image.open(label_path)).astype(np.float32)
                     / 255.0)

        # 双时相影像（取前 8 通道）
        img_A = np.load(
            os.path.join(self.root_dir, 'A', npy_fname)
        ).astype(np.float32)[:8]
        img_B = np.load(
            os.path.join(self.root_dir, 'B', npy_fname)
        ).astype(np.float32)[:8]

        # 先验图（文件夹或文件不存在时返回全零）
        if self.prior_control == 'zero':
            prior = np.zeros(label.shape[-2:], dtype=np.float32)
        elif self.prior_control == 'constant':
            prior = np.full(label.shape[-2:], 0.5, dtype=np.float32)
        elif self.prior_control in {'random', 'random_per_epoch'}:
            epoch = self.epoch if self.prior_control == 'random_per_epoch' else 0
            digest = hashlib.blake2b(
                f"{self.control_seed}:{epoch}:{fname}".encode('utf-8'),
                digest_size=8).digest()
            sample_seed = int.from_bytes(digest, 'little')
            prior = np.random.default_rng(sample_seed).random(
                label.shape[-2:], dtype=np.float32)
        elif self.prior_dir is not None:
            prior_name = self.prior_name_map[fname]
            prior_name = (prior_name if self.is_npy
                          else prior_name.replace('.png', '.npy'))
            prior_path = os.path.join(self.prior_dir, prior_name)
            prior = (np.load(prior_path).astype(np.float32)
                     if os.path.exists(prior_path)
                     else np.zeros(label.shape[-2:], dtype=np.float32))
        else:
            prior = np.zeros(img_A.shape[-2:], dtype=np.float32)

        # 数据增强
        if self.transform:
            img_A, img_B, label, prior = self._augment(
                img_A, img_B, label, prior)

        # 统一维度 → [1, H, W]
        if label.ndim == 2:
            label = label[np.newaxis]
        if prior.ndim == 2:
            prior = prior[np.newaxis]

        # 二值化标签
        label = (label > 0.5).astype(np.float32)

        return (torch.from_numpy(img_A),
                torch.from_numpy(img_B),
                torch.from_numpy(label),
                torch.from_numpy(prior))

    def _augment(self, img_A, img_B, label, prior):
        """同步随机翻转 + 旋转（影像 / 标签 / 先验保持一致）。"""
        # 水平翻转
        if np.random.rand() > 0.5:
            img_A = np.flip(img_A, axis=2)
            img_B = np.flip(img_B, axis=2)
            label = np.flip(label, axis=1)
            prior = np.flip(prior, axis=1)

        # 垂直翻转
        if np.random.rand() > 0.5:
            img_A = np.flip(img_A, axis=1)
            img_B = np.flip(img_B, axis=1)
            label = np.flip(label, axis=0)
            prior = np.flip(prior, axis=0)

        # 随机旋转 0° / 90° / 180° / 270°
        k = np.random.randint(0, 4)
        if k > 0:
            img_A = np.rot90(img_A, k, axes=(1, 2))
            img_B = np.rot90(img_B, k, axes=(1, 2))
            label = np.rot90(label, k, axes=(0, 1))
            prior = np.rot90(prior, k, axes=(0, 1))

        return (img_A.copy(), img_B.copy(),
                label.copy(), prior.copy())
