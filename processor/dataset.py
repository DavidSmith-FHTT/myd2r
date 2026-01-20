import random
import os
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class MSDProcessor(object):
    """
    处理MSD数据集的类，用于加载和处理数据集。

    Args:
        data_path: 数据集路径
        bert_name: 预训练模型名称
        clip_processor: CLIP处理器
    """

    def __init__(self, data_path, bert_name, clip_processor):
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.clip_processor = clip_processor

    def load_from_file(self, mode="train"):
        """
        解析 JSON 格式的数据集，从里面提取信息。
        returns:
            dict: 包含文本、标签和图片 ID 的字典
        """

        logger.info("Loading data from {}".format(self.data_path[mode]))

        with open(self.data_path[mode], "r", encoding="utf-8") as f:
            # 解析 json 格式数据集
            dataset = json.load(f)  # 加载整个数据集

            raw_texts, raw_labels, imgs = [], [], []

            for index in range(0, len(dataset)):  # 一条一条地读取数据
                sample = dataset[index]

                id = sample['id']
                img_id = id + '.jpg'
                text = sample['text']
                label = sample['emotion_label']

                # 将所有数据分别放到对应的列表中
                raw_texts.append(text)
                raw_labels.append(label)
                imgs.append(img_id)

        assert len(raw_texts) == len(raw_labels) == len(imgs), "{}, {}, {}".format(len(raw_texts), len(raw_labels),
                                                                                   len(imgs))

        return {"texts": raw_texts, "labels": raw_labels, "imgs": imgs}


class MSDDataset(Dataset):
    """
    用于加载和处理数据集。它会从 MSDProcessor 加载数据，对文本进行分词、编码和填充操作，对图像进行读取和预处理，最终返回可供模型训练或评估使用的数据样本。
    """

    def __init__(self, processor, img_path, max_seq=128, mode="train"):
        self.processor = processor
        self.img_path = img_path
        # 分词器
        self.tokenizer = self.processor.tokenizer
        self.data_dict = self.processor.load_from_file(mode)
        self.clip_processor = self.processor.clip_processor
        self.max_seq = max_seq

    def __len__(self):
        return len(self.data_dict['texts'])

    def __getitem__(self, idx):
        """
        用于获取数据集中指定位置的样本。
        """
        text, label, img = self.data_dict['texts'][idx], self.data_dict['labels'][idx], self.data_dict['imgs'][idx]

        tokens_text = self.tokenizer.tokenize(text)

        if len(tokens_text) > self.max_seq - 2:
            tokens_text = tokens_text[:self.max_seq - 2]

        tokens = ["[CLS]"] + tokens_text + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding = [0] * (self.max_seq - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq
        assert len(input_mask) == self.max_seq
        assert len(segment_ids) == self.max_seq

        # 对图片进行处理
        if self.img_path is not None:
            # image process
            try:
                img_path = os.path.join(self.img_path, img)
                image = Image.open(img_path).convert('RGB')
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
            except:
                # 原始代码
                # img_path = os.path.join(self.img_path, 'inf.png')
                # image = Image.open(img_path).convert('RGB')
                # image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()

                # 如果图像文件不存在，创建黑色占位图像
                image = Image.new('RGB', (224, 224), (0, 0, 0))
                image = self.clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()

        img_mask = [1] * 50

        return torch.tensor(input_ids), torch.tensor(input_mask), torch.tensor(segment_ids), \
            torch.tensor(img_mask), torch.tensor(label), image
