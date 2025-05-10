import os
import math
import logging
from time import time
from tqdm import tqdm

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

import torch.distributed as dist
import random
import numpy as np
import traceback

from copy import deepcopy
import transformers
from transformers import get_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


from accelerate import Accelerator
from accelerate.utils import DeepSpeedEngineWrapper, DeepSpeedOptimizerWrapper, DeepSpeedSchedulerWrapper,set_seed, DummyOptim, DummyScheduler
from accelerate.logging import get_logger
from accelerate.scheduler import AcceleratedScheduler
from accelerate.optimizer import AcceleratedOptimizer

from mamba2.hybrid_wrapper_internvl import InternVLMambaTransformerHybridModelWrapper
from mamba2.hybrid_mamba_config import MambaConfig

from train_configs import DistillConfig
from train_configs import DistillArgumentParser
# from dataset import TextDataset

from util import load_safetensors_to_dict, construct_language_layer_dict

from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    check_conversations_repetition,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm,
                                    preprocess_internvl2_5, preprocess_mpt,
                                    preprocess_phi3)

from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN)

from internvl.model.internvl_chat import HybridInternVLChatModel, InternVLChatConfig
import dataclasses

from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


from torch.utils.data import Dataset
import json
from typing import Dict, Optional, Sequence, List,Literal
import copy

from dataclasses import dataclass, field

import tokenizers

from internvl.patch import concat_pad_data_collator

logger = get_logger(__name__)


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    mm_use_im_start_end: bool = False
    model_max_length: int = 2048

    vision_select_layer: int = field(
        default=-1,
        metadata={'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'},
    )

    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={'help': 'Specify the version of pixel shuffle implementation. Default is v2.'}
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'}
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'},
    )
    pad2square: bool = field(
        default=False,
        metadata={'help': 'Pad the image to a square shape if set to True. Default is False.'},
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={'help': 'Set to True to use dynamic high resolution strategy. Default is False.'},
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={'help': 'The maximum number of frames for video data. Default is 32.'},
    )
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={'help': 'Whether to use packed dataset for efficient training. Default is False.'},
    )
    num_images_expected: int = field(
        default=40,
        metadata={'help': 'The maximum number of images per packed sample. Default is 40.'},
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={'help': 'The required token length of per packed sample. Default is 8192.'},
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=True,
        metadata={'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'},
    )
    replacement: bool = field(
        default=False,
        metadata={'help': 'Whether to restart the dataset after it is exhausted. Default is False.'},
    )
    allow_overflow: bool = field(
        default=False,
        metadata={'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'},
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={'help': 'Whether to gather all during loss reduction. Default is False.'},
    )
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}')

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        # TODO: quick resume
        self._state_dict = {}

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith('jsonl'), f'annotation must be jsonl, but got {meta["annotation"]}'

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[:int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = {}  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item['length']  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations, return_tensors='pt', padding=False, truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = token_length + num_image_token * (
                                    max_dynamic_patch + use_thumbnail)
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(is_train=self.is_train, input_size=self.image_size,
                                    pad2square=self.pad2square, normalize_type=self.normalize_type)
        return transform

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<image>\n' + data_item['conversations'][0]['value']

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=self.max_dynamic_patch,
                                        image_size=self.image_size, use_thumbnail=self.use_thumbnail)
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches],
                                  group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=max(1, self.max_dynamic_patch // num_image),
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_image)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == num_image, f'image tokens are truncated, this dataset is {self.ds_name}'

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = '<video>\n' + data_item['conversations'][0]['value']

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
        image_list = self.tcs_loader(
            video_path,
            image_type='video',
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=data_item.get('clip', None))

        # Generate special tokens for each video frame
        special_tokens = '\n'.join(['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))])
        data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace(
            '<video>\n', special_tokens + '\n')

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, num_image_tokens, group_by_length=self.group_by_length,
                                  use_packed_ds=self.use_packed_ds, ds_name=self.ds_name, num_image=num_patches)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(image, min_num=self.min_dynamic_patch, max_num=1,
                                    image_size=self.image_size, use_thumbnail=self.use_thumbnail)

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert num_patches == 1, f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(self.template_name, [deepcopy(data_item['conversations'])],
                                  self.tokenizer, [self.num_image_token * num_patches], text_only=True,
                                  group_by_length=self.group_by_length, use_packed_ds=self.use_packed_ds,
                                  ds_name=self.ds_name)

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long)
        )
        return ret

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id::self.num_workers]
            logger.info(f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}')

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif 'video' in data_item and data_item['video'] is not None and data_item['video'] != '':
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, flush=True)
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(f'Failed to load image: {images}, the dataset is: {self.ds_name}')
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(f'Failed to load image: {data_path}, the dataset is: {self.ds_name}')
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(f'Failed to load video: {data_path}, the dataset is: {self.ds_name}')
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if self.worker_state_key in self._state_dict and len(self._state_dict[self.worker_state_key]) > 0:
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            yield self[i]


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    min_num_frame=8,
    max_num_frame=32,
    normalize_type='imagenet',
):
    datasets = []
    lengths = []
    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(f'max_dynamic_patch is set to {max_num} according to the meta file')
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style, ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            min_num_frame=min_num_frame,
            max_num_frame=max_num_frame,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def get_parameter_groups(model, weight_decay=1e-5, skip_list=()):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        if group_name not in parameter_group_names:

            parameter_group_names[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
            }
            parameter_group_vars[group_name] = {
                "weight_decay": this_weight_decay,
                "params": [],
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    return list(parameter_group_vars.values()), list(parameter_group_names.values())


def create_optimizer(training_args, model, optimizer_cls, filter_bias_and_bn=True, skip_list=None):
    weight_decay = training_args.weight_decay
    if filter_bias_and_bn:
        skip = {}
        if skip_list is not None:
            skip = skip_list
        elif hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        parameters, parameter_names = get_parameter_groups(model, weight_decay, skip)
        weight_decay = 0.
    else:
        parameters = model.parameters()
        
    optimizer = optimizer_cls(parameters, lr=training_args.learning_rate, betas=(training_args.adam_beta1, training_args.adam_beta2))

    return optimizer, parameter_names

class MyAccelerator(Accelerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _prepare_deepspeed(self, *args):
        import deepspeed

        deepspeed_plugin = self.state.deepspeed_plugin

        is_dataloader_present = any(isinstance(obj, torch.utils.data.DataLoader) for obj in args)
        result = [
            self._prepare_one(obj, first_pass=True) if isinstance(obj, torch.utils.data.DataLoader) else obj
            for obj in args
        ]

        if deepspeed_plugin.is_auto("train_micro_batch_size_per_gpu"):
            if is_dataloader_present:
                batch_sizes = [obj.batch_size for obj in args if hasattr(obj, "batch_size")]
                if any(bs is None for bs in batch_sizes):
                    raise ValueError(
                        "At least one of the dataloaders passed to `accelerate.prepare()` has `None` as batch size. "
                        "Please set an integer value in `train_micro_batch_size_per_gpu` in the deepspeed config file "
                        "or assign integer value to `AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']`."
                    )
                if self.split_batches:
                    batch_sizes = [batch_size // self.num_processes for batch_size in batch_sizes]

                batch_size_per_device = min(batch_sizes) if deepspeed_plugin.is_train_batch_min else max(batch_sizes)
                if len(batch_sizes) > 1:
                    logger.info(
                        "Since you passed both train and evaluation dataloader, `is_train_batch_min` (here "
                        f"{deepspeed_plugin.is_train_batch_min} will decide the `train_batch_size` ({batch_size_per_device})."
                    )
            else:
                raise ValueError(
                    "When using DeepSpeed, `accelerate.prepare()` requires you to pass at least one of training or evaluation dataloaders "
                    "with `batch_size` attribute returning an integer value "
                    "or alternatively set an integer value in `train_micro_batch_size_per_gpu` in the deepspeed config file "
                    "or assign integer value to `AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']`."
                )
        else:
            batch_size_per_device = deepspeed_plugin.get_value("train_micro_batch_size_per_gpu")

        # handle `gradient_accumulation_steps` when the value is `auto`
        deepspeed_plugin.fill_match(
            "gradient_accumulation_steps",
            must_match=False,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        config_kwargs = {
            "train_micro_batch_size_per_gpu": batch_size_per_device,
            "train_batch_size": batch_size_per_device
            * deepspeed_plugin.get_value("gradient_accumulation_steps")
            * self.num_processes,
            "gradient_clipping": 1.0,
            "zero_optimization.stage3_gather_16bit_weights_on_model_save": False,
        }

        model = None
        optimizer = None
        scheduler = None
        for obj in result:
            if isinstance(obj, torch.nn.Module):
                model = obj
            elif isinstance(obj, (torch.optim.Optimizer, DummyOptim)):
                optimizer = obj
            elif (isinstance(obj, (LRScheduler, DummyScheduler))) or (
                type(obj).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
            ):
                scheduler = obj

        if optimizer is not None:
            if "optimizer" in deepspeed_plugin.deepspeed_config and not isinstance(optimizer, (DummyOptim)):
                raise ValueError(
                    "You cannot specify an optimizer in the config file and in the code at the same time. "
                    "Please remove the optimizer from the config file or "
                    "create `accelerate.utils.DummyOptim` in the code."
                )
            elif "optimizer" not in deepspeed_plugin.deepspeed_config and isinstance(optimizer, (DummyOptim)):
                raise ValueError(
                    "You cannot create a `DummyOptim` without specifying an optimizer in the config file."
                )

            if isinstance(optimizer, (torch.optim.Optimizer)):
                deepspeed_plugin.deepspeed_config["zero_allow_untested_optimizer"] = True

        if scheduler is not None:
            if "scheduler" in deepspeed_plugin.deepspeed_config and not isinstance(scheduler, (DummyScheduler)):
                raise ValueError(
                    "You cannot specify a scheduler in the config file and in the code at the same time. "
                    "Please remove the scheduler from the config file or "
                    "create `accelerate.utils.DummyScheduler` in the code."
                )
            elif (
                "scheduler" not in deepspeed_plugin.deepspeed_config
                and isinstance(scheduler, (DummyScheduler))
                and scheduler.lr_scheduler_callable is None
            ):
                raise ValueError(
                    "Either specify a scheduler in the config file or "
                    "pass in the `lr_scheduler_callable` parameter when using `accelerate.utils.DummyScheduler`."
                )

        if optimizer is not None and scheduler is not None:
            if isinstance(optimizer, (DummyOptim)) and not isinstance(scheduler, (DummyScheduler)):
                raise ValueError(
                    "You can only specify `accelerate.utils.DummyScheduler` in the code when using "
                    "`accelerate.utils.DummyOptim`."
                )

        if model is not None:
            # if the model is an MOE, set the appropriate MOE layers as leaf Z3 modules
            deepspeed_plugin.set_moe_leaf_modules(model)
            # deal with config keys that use `auto` value and rely on model's hidden_size
            hidden_size_based_keys = [
                "zero_optimization.reduce_bucket_size",
                "zero_optimization.stage3_prefetch_bucket_size",
                "zero_optimization.stage3_param_persistence_threshold",
            ]
            hidden_size_auto_keys = [x for x in hidden_size_based_keys if deepspeed_plugin.is_auto(x)]
            if len(hidden_size_auto_keys) > 0:
                reasoning = (
                    "therefore it's not possible to automatically fill out the following `auto` entries "
                    + f"in the DeepSpeed config file: {hidden_size_auto_keys}. You can fix that by replacing "
                    + "`auto` values for these keys with an integer value of your choice."
                )
                if not hasattr(model, "config"):
                    raise ValueError("Can't find `model.config` entry, " + reasoning)

                if hasattr(model.config, "hidden_size"):
                    hidden_size = model.config.hidden_size
                elif hasattr(model.config, "hidden_sizes"):
                    # if there are many hidden sizes pick the largest one
                    hidden_size = max(model.config.hidden_sizes)
                else:
                    raise ValueError(
                        "Can find neither `model.config.hidden_size` nor `model.config.hidden_sizes`, " + reasoning
                    )

                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": int(0.9 * hidden_size * hidden_size),
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    }
                )

            if isinstance(optimizer, (DummyOptim)):
                config_kwargs.update(
                    {"optimizer.params.lr": optimizer.lr, "optimizer.params.weight_decay": optimizer.weight_decay}
                )
            if isinstance(scheduler, (DummyScheduler)) and scheduler.lr_scheduler_callable is None:
                max_lr = (
                    getattr(scheduler.optimizer, "lr", None)
                    if getattr(scheduler.optimizer, "defaults", None) is None
                    else scheduler.optimizer.defaults["lr"]
                )
                config_kwargs.update(
                    {
                        "scheduler.params.warmup_min_lr": 0,
                        "scheduler.params.warmup_max_lr": max_lr,
                        "scheduler.params.warmup_num_steps": scheduler.warmup_num_steps,
                    }
                )
                if scheduler.total_num_steps is not None:
                    config_kwargs["scheduler.params.total_num_steps"] = (
                        math.ceil(scheduler.total_num_steps / self.num_processes)
                        if not self.split_batches
                        else scheduler.total_num_steps
                    )
            deepspeed_plugin.deepspeed_config_process(must_match=False, **config_kwargs)
            self.deepspeed_config = deepspeed_plugin.deepspeed_config
            kwargs = dict(model=model, config_params=self.deepspeed_config)
            if optimizer is not None:
                if isinstance(optimizer, (DummyOptim)):
                    kwargs["model_parameters"] = optimizer.params
                    if isinstance(scheduler, (DummyScheduler)) and scheduler.lr_scheduler_callable is not None:
                        kwargs["lr_scheduler"] = scheduler.lr_scheduler_callable
                else:
                    if self.deepspeed_config["zero_optimization"].get("offload_optimizer", {}).get(
                        "device", "none"
                    ) != "none" and self.deepspeed_config.get("zero_force_ds_cpu_optimizer", True):
                        from deepspeed.ops.adam import DeepSpeedCPUAdam

                        defaults = {k: v for k, v in optimizer.defaults.items() if k in ["lr", "weight_decay", "betas"]}
                        if isinstance(optimizer, torch.optim.Adam):
                            optimizer = DeepSpeedCPUAdam(optimizer.param_groups, **defaults, adamw_mode=False)
                        else:
                            optimizer = DeepSpeedCPUAdam(optimizer.param_groups, **defaults)
                    kwargs["optimizer"] = optimizer
                    if scheduler is not None:
                        if type(scheduler).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES:
                            kwargs["lr_scheduler"] = scheduler

            engine, optimizer, _, lr_scheduler = deepspeed.initialize(**kwargs)
            if optimizer is not None:
                optimizer = DeepSpeedOptimizerWrapper(optimizer)
            if scheduler is not None:
                if lr_scheduler is None:
                    scheduler = AcceleratedScheduler(
                        scheduler,
                        optimizer,
                        step_with_optimizer=self.step_scheduler_with_optimizer,
                        split_batches=self.split_batches,
                    )
                else:
                    scheduler = DeepSpeedSchedulerWrapper(lr_scheduler, optimizer)

            for i in range(len(result)):
                if isinstance(result[i], torch.nn.Module):
                    result[i] = engine
                elif isinstance(result[i], (torch.optim.Optimizer, DummyOptim)):
                    result[i] = optimizer
                elif (isinstance(result[i], (LRScheduler, DummyScheduler))) or (
                    type(result[i]).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
                ):
                    result[i] = scheduler
            # pointing for deepspeed_engine_wrapped.backward()
            self.deepspeed_engine_wrapped = DeepSpeedEngineWrapper(engine)
            self._models.append(engine)
            if optimizer is not None:
                self._optimizers.append(optimizer)
            if scheduler is not None:
                self._schedulers.append(scheduler)
            if len(self._models) > 1:
                raise AssertionError(
                    "You can't use same `Accelerator()` instance with multiple models when using DeepSpeed"
                )
        return tuple(result)


def main():
    parser = DistillArgumentParser((DistillConfig, DataArguments))
    training_args, data_args = parser.parse()

    accelerator = (
        MyAccelerator(log_with="wandb")
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    set_seed(training_args.seed)

    if accelerator.is_main_process:
        if training_args.output_dir is not None:
            os.makedirs(training_args.output_dir, exist_ok=True)
            # save training arguments and data arguments
            with open(os.path.join(training_args.output_dir, "training_args.json"), "w") as f:
                json.dump(training_args.to_dict(), f)
            with open(os.path.join(training_args.output_dir, "data_args.json"), "w") as f:
                data_args_dict = dataclasses.asdict(data_args)
                json.dump(data_args_dict, f)

    accelerator.wait_for_everyone()

    model_name = training_args.model_name
    dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
    
    if "gemma" in model_name:
        attn_implementation = "sdpa"


    logger.info('Loading InternVLChatModel...')
    config = InternVLChatConfig.from_pretrained(model_name)

    config.vision_config.drop_path_rate = data_args.drop_path_rate
    if config.llm_config.model_type == 'internlm2':
        config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
        logger.info('Using flash_attention_2 for InternLM')
    else:
        config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
        logger.info('Using flash_attention_2 for LLaMA')
    config.template = data_args.conv_style
    config.select_layer = data_args.vision_select_layer
    config.dynamic_image_size = data_args.dynamic_image_size
    config.use_thumbnail = data_args.use_thumbnail
    config.ps_version = data_args.ps_version
    config.min_dynamic_patch = data_args.min_dynamic_patch
    config.max_dynamic_patch = data_args.max_dynamic_patch
    teacher_model = HybridInternVLChatModel.from_pretrained(
        model_name, torch_dtype=dtype, config=config)

    # Freeze all parameters in teacher model by setting requires_grad to False
    for param in teacher_model.parameters():
        param.requires_grad = False

    teacher_model = teacher_model.to(accelerator.device)
    teacher_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=False, trust_remote_code=True, model_max_length=data_args.model_max_length, use_fast=data_args.use_fast_tokenizer)

    teacher_model.config.tokenizer_model_max_length = tokenizer.model_max_length
    
    text_config = config.llm_config
    if not hasattr(text_config, 'head_dim'):
        d_xb = text_config.num_key_value_heads * \
            (text_config.hidden_size // text_config.num_attention_heads)
        d_inner = text_config.hidden_size
        d_state = text_config.hidden_size//text_config.num_attention_heads
    else:
        # to handle gemma2
        d_xb = text_config.num_key_value_heads * text_config.head_dim
        d_inner = text_config.num_attention_heads * text_config.head_dim
        d_state = text_config.head_dim

    ssm_layers = training_args.ssm_layers
    attn_layers = [i for i in range(text_config.num_hidden_layers) if i not in ssm_layers]
    
    mamba_config = MambaConfig(
        text_config.hidden_size,
        {"expand": 1, "ngroups":text_config.num_attention_heads, "d_state": d_state},
        text_config.rms_norm_eps,
        d_inner=d_inner,
        d_xb=d_xb,
        intermediate_size=text_config.intermediate_size,
        hidden_act=text_config.hidden_act,
        n_layer=text_config.num_hidden_layers,
        attn_layers=attn_layers
    )
    student_model = InternVLMambaTransformerHybridModelWrapper.init_distillation(
            None, model_name, mamba_config, attn_layers=attn_layers, init_with_kqvo=training_args.init_with_kqvo, attn_implementation=attn_implementation)
    student_model.config.tokenizer_model_max_length = tokenizer.model_max_length

    student_model.model.vision_model.gradient_checkpointing = True
    student_model.model.vision_model.encoder.gradient_checkpointing = True
    if training_args.gradient_checkpointing:
        student_model.model.language_model._set_gradient_checkpointing()

    if training_args.prev_checkpoint_path is not None:
        # this is for progressive distillation,
        # override ssm layers using the previous weights
        prev_checkpoint = load_safetensors_to_dict(
            training_args.prev_checkpoint_path)
        prev_checkpoint_layers, is_mamba_layer = construct_language_layer_dict(prev_checkpoint, config.num_hidden_layers)
        print(is_mamba_layer)
        for (layer_id, layer_checkpoint) in prev_checkpoint_layers.items():
            if is_mamba_layer[layer_id]:
                # override weights of that layer
                student_model.model.language_model.model.layers[layer_id].load_state_dict(layer_checkpoint)
                

    # Freeze all non mamba parameters in student  model
    for name, param in student_model.named_parameters():
        if f"mamba" not in name and (f"mlp1" not in name or not training_args.tune_mlp):
            param.requires_grad = False

    if accelerator.is_main_process:
        print("teacher_model:", teacher_model)
        total_params = sum(p.numel() for p in teacher_model.parameters())
        total_trainable_params = sum(
            p.numel() for p in teacher_model.parameters() if p.requires_grad)
        print(f"number of total params:{total_params}")
        print(f"number of total trainable params:{total_trainable_params}")

        print("student_model:", student_model)
        total_params = sum(p.numel() for p in student_model.parameters())
        total_trainable_params = sum(
            p.numel() for p in student_model.parameters() if p.requires_grad)
        print(f"number of total params:{total_params}")
        print(f"number of total trainable params:{total_trainable_params}")
        
        for name, param in student_model.named_parameters():
            if param.requires_grad:
                print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        student_model.save_config(training_args.output_dir)

    # default v1
    patch_size = teacher_model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {teacher_model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(f'model.config.vision_config.image_size: {teacher_model.config.vision_config.image_size}')
    if teacher_model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(f'Resizing position embedding from '
                    f'{teacher_model.config.vision_config.image_size} '
                    f'to {data_args.force_image_size}...')
        teacher_model.vision_model.resize_pos_embeddings(old_size=teacher_model.config.vision_config.image_size,
                                                 new_size=data_args.force_image_size,
                                                 patch_size=patch_size)
        teacher_model.config.vision_config.image_size = data_args.force_image_size
    teacher_model.config.force_image_size = data_args.force_image_size
    teacher_model.num_image_token = int((data_args.force_image_size // patch_size) ** 2 * (data_args.down_sample_ratio ** 2))

    token_list = [IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
                  QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN,
                  REF_END_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    teacher_model.img_context_token_id = img_context_token_id
    student_model.model.img_context_token_id = img_context_token_id

    train_dataset = build_datasets(
        data_args, tokenizer, None, teacher_model, group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size, use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch, max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type, min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame)
    
    train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=concat_pad_data_collator, shuffle=True)
    if accelerator.is_main_process:
        print("length of dataset:", len(train_dataset))

    
    optimizer_cls = (
     torch.optim.AdamW if training_args.optim == "adamw_torch" else torch.optim.Adam
     if accelerator.state.deepspeed_plugin is None
     or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
     else DummyOptim
    )
    # optimizer = optimizer_cls(filter(lambda p: p.requires_grad, student_model.parameters()), lr=training_args.learning_rate, betas=(0.9, 0.98))
    optimizer, parameter_names = create_optimizer(training_args, student_model, optimizer_cls, filter_bias_and_bn=True, skip_list=None)

    if accelerator.is_main_process:
        print("parameter names:", parameter_names)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    training_args.gradient_accumulation_steps = training_args.gradient_accumulation_steps // accelerator.num_processes
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if training_args.max_steps is None or training_args.max_steps < 0:
        training_args.max_steps = training_args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    training_args.warmup_steps = training_args.warmup_steps // training_args.gradient_accumulation_steps
    if training_args.lr_scheduler_type == "warmup_stable_decay":
        training_args.decay_steps = training_args.decay_steps // training_args.gradient_accumulation_steps
        training_args.stable_steps = training_args.max_steps - training_args.warmup_steps - training_args.decay_steps
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.max_steps,
            scheduler_specific_kwargs={"num_stable_steps": training_args.stable_steps, "num_decay_steps": training_args.decay_steps} if training_args.lr_scheduler_type == "warmup_stable_decay" else None
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=training_args.max_steps, warmup_num_steps=training_args.warmup_steps
        )

    if training_args.do_eval:
        # Prepare everything with our `accelerator`.
        student_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            student_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    else:
        # Prepare everything with our `accelerator`.
        student_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            student_model, optimizer, train_dataloader, lr_scheduler
        )

    
    if accelerator.is_main_process:
        print("length of dataloader:", len(train_dataloader))

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        training_args.max_train_steps = training_args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(training_args.max_train_steps / num_update_steps_per_epoch)

    save_steps = None
    # Figure out how many steps we should save the Accelerator states
    if training_args.save_steps is not None:
        save_steps = training_args.save_steps
    
    # We need to initialize the trackers we use, and also store our configuration.
    # We initialize the trackers only on main process because `accelerator.log`
    # only logs on main process and we don't want empty logs/runs on other processes.
    if accelerator.is_main_process:
        experiment_config = vars(training_args)
        experiment_config["lr_scheduler_type"] = "cosine"
        accelerator.init_trackers("mamba_distill", init_kwargs={"wandb": {"name": training_args.output_dir.split("/")[-2]}})

    # Train!
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        if training_args.resume_from_checkpoint is not None or training_args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {training_args.resume_from_checkpoint}")
            accelerator.load_state(training_args.resume_from_checkpoint)
            path = os.path.basename(training_args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    curr_loss = 0.0
    curr_kl_loss = 0.0
    curr_teacher_loss = 0.0
    curr_student_loss = 0.0
    curr_teacher_input_loss = 0.0
    curr_student_input_loss = 0.0

    # training
    for epoch in range(starting_epoch, training_args.num_train_epochs):
        start_time = time()
        student_model.train()
        
        for step, batch in enumerate(train_dataloader):
            # We need to skip steps until we reach the resumed step
            if training_args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and step < resume_step:
                    completed_steps += 1
                    continue

            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            pixel_values = batch["pixel_values"].to(dtype)
            image_flags = batch["image_flags"]
            position_ids = batch["position_ids"]
            if training_args.tl_weight > 0:    
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, pixel_values=pixel_values, image_flags=image_flags, position_ids=position_ids,
                    use_cache=False, output_hidden_states=training_args.output_hidden_states)
                    teacher_hidden_states, _ = teacher_outputs.hidden_states
                student_outputs = student_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels, pixel_values=pixel_values, image_flags=image_flags, position_ids=position_ids, use_cache=False, output_hidden_states=training_args.output_hidden_states, teacher_outputs=teacher_hidden_states)
                student_hidden_states, teacher_input_hidden_states = student_outputs.hidden_states

                teacher_input_loss = 0
                if training_args.tl_loss_type == "kl":
                    for i in range(1, len(teacher_input_hidden_states)):
                        teacher_input_loss += F.kl_div(F.log_softmax(teacher_input_hidden_states[i], dim=-1), F.softmax(teacher_hidden_states[i], dim=-1), reduction='batchmean')
                elif training_args.tl_loss_type == "l2":
                    for i in range(1, len(teacher_input_hidden_states)):
                        if i - 1 not in mamba_config.attn_layers:
                            teacher_input_loss += torch.norm(teacher_input_hidden_states[i] - teacher_hidden_states[i], p=2)
                    
                loss = training_args.tl_weight * teacher_input_loss
                curr_teacher_input_loss += teacher_input_loss.detach().float()
            else:
                student_outputs = student_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels, pixel_values=pixel_values, image_flags=image_flags, position_ids=position_ids, use_cache=False, output_hidden_states=training_args.output_hidden_states)
                student_hidden_states, _ = student_outputs.hidden_states

                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels, pixel_values=pixel_values, image_flags=image_flags, position_ids=position_ids, use_cache=False, output_hidden_states=training_args.output_hidden_states, teacher_outputs=student_hidden_states)
                    teacher_hidden_states, student_input_hidden_states = teacher_outputs.hidden_states
                if training_args.sl_weight > 0:
                    student_input_loss = 0
                    for i in range(1, len(student_input_hidden_states)):
                        student_input_loss += F.kl_div(F.log_softmax(student_hidden_states[i], dim=-1), F.softmax(student_input_hidden_states[i], dim=-1), reduction='batchmean')
                    loss = training_args.sl_weight * student_input_loss
                    curr_student_input_loss += student_input_loss.detach().float()
                else:
                    loss = 0

            teacher_logits = teacher_outputs.logits
            teach_cross_entropy_loss = teacher_outputs.loss
            student_logits = student_outputs.logits
            student_cross_entropy_loss = student_outputs.loss

            if training_args.kl_weight > 0:
                targets = F.softmax(teacher_logits / training_args.temperature, dim=-1)
                if training_args.loss_type == "kl":
                    kl_loss = F.kl_div(F.log_softmax(student_logits / training_args.temperature, dim=-1), targets, reduction='batchmean') * (training_args.temperature ** 2)
                elif training_args.loss_type == "jsd":
                    student_probs = F.softmax(student_logits / training_args.temperature, dim=-1)
                    m_probs = 0.5 * (targets + student_probs)
                    kl_student_m = F.kl_div(F.log_softmax(student_logits / training_args.temperature, dim=-1), m_probs, reduction='batchmean') * (training_args.temperature ** 2)
                    kl_teacher_m = F.kl_div(F.log_softmax(teacher_logits / training_args.temperature, dim=-1), m_probs, reduction='batchmean') * (training_args.temperature ** 2)
                    
                    # 计算最终 JSD
                    kl_loss = 0.5 * (kl_student_m + kl_teacher_m)
                elif training_args.loss_type == "reverse":
                    kl_loss = F.kl_div(F.log_softmax(teacher_logits / training_args.temperature, dim=-1), F.softmax(student_logits / training_args.temperature, dim=-1), reduction='batchmean') * (training_args.temperature ** 2)
                elif training_args.loss_type == "ce":
                    kl_loss = torch.nn.CrossEntropyLoss()(student_logits.view(-1, student_logits.shape[-1]), torch.argmax(targets, dim=-1).view(-1))
                elif training_args.loss_type == "l2":
                    kl_loss = torch.norm(teacher_logits - student_logits, p=2)
                loss += training_args.kl_weight * kl_loss
                curr_kl_loss += kl_loss.detach().float()
            
            if training_args.ce_weight > 0:
                loss += training_args.ce_weight * student_cross_entropy_loss
                curr_student_loss += student_cross_entropy_loss.detach().float()
            # loss = training_args.kl_weight * kl_loss + training_args.ce_weight * student_cross_entropy_loss
            # curr_kl_loss += kl_loss.detach().float()
            curr_teacher_loss += teach_cross_entropy_loss.detach().float()

            # loss = student_cross_entropy_loss

            curr_loss += loss.detach().float()

            # loss = loss / training_args.gradient_accumulation_steps
            accelerator.backward(loss)

            if (step > 0 and (step + 1) % training_args.gradient_accumulation_steps == 0) or step == len(train_dataloader) - 1:
                # torch.nn.utils.clip_grad_norm_(
                #     student_model.parameters(), training_args.max_grad_norm)
                # optimizer.step()
                lr_scheduler.step()
                # optimizer.zero_grad()
                # log loss
                accelerator.print(
                    f'training loss: {curr_loss / training_args.gradient_accumulation_steps:.5f}')
                accelerator.log({'train loss': curr_loss / training_args.gradient_accumulation_steps,
                                'teacher kl loss': curr_kl_loss / training_args.gradient_accumulation_steps,
                                'teacher ce loss': curr_teacher_loss / training_args.gradient_accumulation_steps,
                                'student ce loss': curr_student_loss / training_args.gradient_accumulation_steps,
                                'teacher input loss': curr_teacher_input_loss / training_args.gradient_accumulation_steps,
                                'student input loss': curr_student_input_loss / training_args.gradient_accumulation_steps,
                                'lr': lr_scheduler.get_last_lr()[0], 'step': completed_steps})
                curr_loss = 0
                curr_kl_loss = 0
                curr_teacher_loss = 0
                curr_student_loss = 0
                curr_teacher_input_loss = 0
                curr_student_input_loss = 0
                completed_steps += 1
                progress_bar.update(1) 

            if isinstance(save_steps, int):
                if completed_steps > 0 and completed_steps % save_steps == 0:
                    accelerator.wait_for_everyone()
                    # save checkpoint
                    output_dir = f"step_{completed_steps}"
                    if training_args.output_dir is not None:
                        output_dir = os.path.join(training_args.output_dir, output_dir)
                    # save model weight
                    unwrapped_model = accelerator.unwrap_model(student_model)
                    unwrapped_model.model.save_pretrained(
                        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                    )
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)
                    accelerator.save_state(output_dir)

        end_time = time()
        logger.info(f"Epoch {epoch} training took {end_time-start_time} seconds")

        if training_args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(student_model)
            unwrapped_model.model.save_pretrained(
                training_args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(training_args.output_dir)
        
        if training_args.do_eval:
            # run evaluation
            student_model.eval()
            total_eval_loss = 0
            total_eval_kl_loss = 0
            total_eval_student_ce_loss = 0
            total_eval_teacher_ce_loss = 0
            for step, batch in enumerate(eval_dataloader):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch["attention_mask"]
                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
                    teacher_logits = teacher_outputs.logits
                    teach_cross_entropy_loss = teacher_outputs.loss
                targets = F.softmax(teacher_logits, dim=-1)
                student_outputs = student_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
                student_logits = student_outputs.logits
                student_cross_entropy_loss = student_outputs.loss
                kl_loss = F.kl_div(F.log_softmax(
                    student_logits, dim=-1), targets, reduction='batchmean')
                loss = training_args.kl_weight * kl_loss + training_args.ce_weight * student_cross_entropy_loss
                total_eval_loss += loss.detach().float()
                total_eval_kl_loss += kl_loss.detach().float()
                total_eval_student_ce_loss += student_cross_entropy_loss.detach().float()
                total_eval_teacher_ce_loss += teach_cross_entropy_loss.detach().float()

            avg_eval_loss = total_eval_loss / len(train_dataloader)
            avg_eval_kl_loss = total_eval_kl_loss / len(train_dataloader)
            avg_eval_teacher_ce_loss = total_eval_teacher_ce_loss / len(train_dataloader)
            avg_eval_student_ce_loss = total_eval_student_ce_loss / len(train_dataloader)

            avg_eval_loss = accelerator.gather(torch.tensor(avg_eval_loss).to(accelerator.device)).mean().item()
            avg_eval_kl_loss = accelerator.gather(torch.tensor(avg_eval_kl_loss).to(accelerator.device)).mean().item()
            avg_eval_teacher_ce_loss = accelerator.gather(torch.tensor(avg_eval_teacher_ce_loss).to(accelerator.device)).mean().item()
            avg_eval_student_ce_loss = accelerator.gather(torch.tensor(avg_eval_student_ce_loss).to(accelerator.device)).mean().item()

            accelerator.log({'eval loss': avg_eval_loss,
                    'eval kl loss': avg_eval_kl_loss,
                    'eval teacher ce loss': avg_eval_teacher_ce_loss,
                    'eval student ce loss': avg_eval_student_ce_loss,
                    'step': completed_steps})

if __name__ == "__main__":
    main()
