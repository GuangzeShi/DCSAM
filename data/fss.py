import os
import glob
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np

# MAX_SAMPLES_PER_CLASS = 40   # 每个类别只读取前 10 张；改成 None 表示不限制

class DatasetFSSModified(Dataset):
    def __init__(self, datapath, dataset_name, transform, support_indices=[0], resize=(224, 224)):
        self.datapath = datapath
        self.dataset_name = dataset_name
        self.transform = transform
        self.support_indices = support_indices
        self.resize = resize  # (H, W)

        # 构造所有类别路径
        self.class_folders = sorted(glob.glob(os.path.join(self.datapath, dataset_name, '*')))
        self.img_metadata = self.build_img_metadata()

    def build_img_metadata(self):
        metadata = []
        for class_path in self.class_folders:
            class_name = os.path.basename(class_path)

            # 获取 target 图像和 mask 列表
            target_img_list = sorted(glob.glob(os.path.join(class_path, 'target_images', '*.jpg')))
            target_mask_list = sorted(glob.glob(os.path.join(class_path, 'target_masks', '*.png')))

            # if MAX_SAMPLES_PER_CLASS is not None:
            #     target_img_list = target_img_list[:MAX_SAMPLES_PER_CLASS]
            #     target_mask_list = target_mask_list[:MAX_SAMPLES_PER_CLASS]

        # print('num_classes =', len(self.class_folders))
        # print('num_samples =', len(metadata))
        # print('first_10_class_names =', sorted(list(set([x['class_name'] for x in metadata])))[:10])
        # print('all_class_num_in_metadata =', len(set([x['class_name'] for x in metadata])))

            # 获取 reference 图像和 mask 列表（后面按索引使用）
            reference_img_list = sorted(glob.glob(os.path.join(class_path, 'reference_images', '*.jpg')))
            reference_mask_list = sorted(glob.glob(os.path.join(class_path, 'reference_masks', '*.png')))

            # 组合所有 target 样本，每个记录包含必要路径和 reference 列表
            for query_img, query_mask in zip(target_img_list, target_mask_list):
                metadata.append({
                    'class_path': class_path,
                    'class_name': class_name,
                    'query_img': query_img,
                    'query_mask': query_mask,
                    'reference_imgs': reference_img_list,
                    'reference_masks': reference_mask_list
                })
        return metadata

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        sample = self.img_metadata[idx]
        class_name = sample['class_name']
        query_img_path = sample['query_img']
        query_mask_path = sample['query_mask']
        ref_img_list = sample['reference_imgs']
        ref_mask_list = sample['reference_masks']

        # 加载 query 图像和 mask
        query_img = Image.open(query_img_path).convert('RGB')
        query_mask = Image.open(query_mask_path).convert('L')

        # 加载支持图像和 mask（通过索引）
        support_imgs, support_masks = [], []
        for i in self.support_indices:
            support_img = Image.open(ref_img_list[i]).convert('RGB')
            support_mask = Image.open(ref_mask_list[i]).convert('L')
            support_imgs.append(support_img)
            support_masks.append(support_mask)

        # 图像 transform
        query_img = self.transform(query_img)
        support_imgs = torch.stack([self.transform(img) for img in support_imgs])

        # mask resize（最近邻），并二值化
        query_mask = self.process_mask(query_mask, query_img.shape[-2:])
        support_masks = torch.stack([
            self.process_mask(mask, support_imgs.shape[-2:]) for mask in support_masks
        ])

        batch = {
            'query_img': query_img,
            'query_mask': query_mask,
            'query_name': query_img_path,

            'support_imgs': support_imgs,
            'support_masks': support_masks,
            'support_names': [ref_img_list[i] for i in self.support_indices],

            'class_name': class_name
        }

        return batch

    def process_mask(self, mask_img, target_size):
        mask = torch.tensor(np.array(mask_img), dtype=torch.uint8)
        mask = (mask >= 128).float()
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=target_size, mode='nearest').squeeze()
        return mask
