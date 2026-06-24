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
                 prior_dir_name: str = None):
        """
        Args:
            prior_dir_name: 先验文件夹名称，明确指定用哪个先验。
                            例如 'spatial_prior_gwr' 或 'spatial_prior_gwda'。
                            若为 None，则按优先级自动查找。
                            若文件夹不存在，返回全零占位（在线先验模式）。
        """
        self.root_dir  = root_dir
        self.transform = transform

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

        # 训练 / 验证划分（固定随机种子保证可复现）
        files = np.array(sorted(self.files))
        np.random.default_rng(42).shuffle(files)
        n_train = int(len(files) * split_ratio)
        self.file_list = (files[:n_train] if split == 'train'
                          else files[n_train:]).tolist()

        print(f"[{split.upper()}] 共 {len(self.file_list)} 个切片")

    # ── 工具 ──────────────────────────────────────────────────────────
    def _find_dir(self, candidates: list):
        for name in candidates:
            path = os.path.join(self.root_dir, name)
            if os.path.exists(path):
                return path
        return None

    # ── Dataset 接口 ──────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.file_list)

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
        if self.prior_dir is not None:
            prior_path = os.path.join(self.prior_dir, npy_fname)
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