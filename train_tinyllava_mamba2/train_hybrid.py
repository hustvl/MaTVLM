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

import transformers
from transformers import get_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


from accelerate import Accelerator
from accelerate.utils import DeepSpeedEngineWrapper, DeepSpeedOptimizerWrapper, DeepSpeedSchedulerWrapper,set_seed, DummyOptim, DummyScheduler
from accelerate.logging import get_logger
from accelerate.scheduler import AcceleratedScheduler
from accelerate.optimizer import AcceleratedOptimizer

from mamba2.hybrid_wrapper_tinyllava import TinyllavaMambaTransformerHybridModelWrapper
from mamba2.hybrid_mamba_config import PhiMambaConfig

from train_configs import DistillConfig
from train_configs import DistillArgumentParser
# from dataset import TextDataset

from util import load_safetensors_to_dict, construct_language_layer_dict

from tinyllava.data.template import TemplateFactory

from tinyllava.utils import *

from tinyllava.model.modeling_tinyllava import HybridTinyLlavaForConditionalGeneration

import dataclasses

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


from torch.utils.data import Dataset
import json
from typing import Dict, Optional, Sequence, List
import copy

from dataclasses import dataclass, field

import tokenizers


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
    conv_version: str = 'phi'
    model_max_length: int = 2048


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.template = TemplateFactory(data_args.conv_version)()

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.template.encode(copy.deepcopy(sources["conversations"]), self.tokenizer)
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            image_aspect_ratio = getattr(self.data_args, 'image_aspect_ratio', None)
            image_grid_pinpoints = getattr(self.data_args, 'image_grid_pinpoints', None)
            if image_aspect_ratio == 'pad':
                image = self.expand2square(image, tuple(int(x * 255) for x in self.data_args.image_processor.image_mean))
            elif image_aspect_ratio == "anyres":
                image = self.process_anyres_image(image, self.data_args.image_processor, image_grid_pinpoints)
            image = self.data_args.image_processor(image, return_tensors='pt')['pixel_values'][0]
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])

        return data_dict

    @classmethod
    def expand2square(cls, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    @classmethod
    def process_anyres_image(cls, image, processor, grid_pinpoints):
        """
        Process an image with variable resolutions.

        Args:
            image (PIL.Image.Image): The input image to be processed.
            processor: The image processor object.
            grid_pinpoints (str): A string representation of a list of possible resolutions.

        Returns:
            torch.Tensor: A tensor containing the processed image patches.
        """
        if type(grid_pinpoints) is list:
            possible_resolutions = grid_pinpoints
        else:
            possible_resolutions = ast.literal_eval(grid_pinpoints)
        best_resolution = select_best_resolution(image.size, possible_resolutions)
        image_padded = resize_and_pad_image(image, best_resolution)

        patches = divide_to_patches(image_padded, processor.crop_size['height'])

        image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

        image_patches = [image_original_resize] + patches
        image_patches = [processor(image_patch, return_tensors='pt')['pixel_values'][0]
                        for image_patch in image_patches]
        return torch.stack(image_patches, dim=0)
    

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


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

    teacher_model = HybridTinyLlavaForConditionalGeneration.from_pretrained(
        model_name)

    teacher_model = teacher_model.to(dtype)
    # Freeze all parameters in teacher model by setting requires_grad to False
    for param in teacher_model.parameters():
        param.requires_grad = False


    teacher_model = teacher_model.to(accelerator.device)
    teacher_model.eval()

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, dtype=dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=data_args.model_max_length, padding_side = config.tokenizer_padding_side, use_fast = config.tokenizer_use_fast)

    teacher_model.config.tokenizer_model_max_length = tokenizer.model_max_length
    
    text_config = AutoConfig.from_pretrained(config.llm_model_name_or_path)    
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
    
    mamba_config = PhiMambaConfig(
        config.hidden_size,
        {"expand": 1, "ngroups":text_config.num_attention_heads, "d_state": d_state},
        text_config.layer_norm_eps,
        d_inner=d_inner,
        d_xb=d_xb,
        intermediate_size=text_config.intermediate_size,
        hidden_act=text_config.hidden_act,
        n_layer=text_config.num_hidden_layers,
        attn_layers=attn_layers,
        resid_pdrop=text_config.resid_pdrop,
        bidirectional=training_args.bid_mode,
        is_bias=training_args.is_bias
    )
    student_model = TinyllavaMambaTransformerHybridModelWrapper.init_distillation(
            None, model_name, mamba_config, attn_layers=attn_layers, init_with_kqvo=training_args.init_with_kqvo, attn_implementation=attn_implementation)
    student_model.config.tokenizer_model_max_length = tokenizer.model_max_length

    if training_args.gradient_checkpointing:
        student_model.model.gradient_checkpointing_enable()

    if training_args.prev_checkpoint_path is not None:
        # this is for progressive distillation,
        # override ssm layers using the previous weights
        prev_checkpoint = load_safetensors_to_dict(
            training_args.prev_checkpoint_path)
        prev_checkpoint_layers, is_mamba_layer = construct_language_layer_dict(prev_checkpoint, text_config.num_hidden_layers)
        print(is_mamba_layer)
        for (layer_id, layer_checkpoint) in prev_checkpoint_layers.items():
            if is_mamba_layer[layer_id]:
                # override weights of that layer
                student_model.model.language_model.model.layers[layer_id].load_state_dict(layer_checkpoint)
                

    # Freeze all non mamba parameters in student  model
    for name, param in student_model.named_parameters():
        if f"mamba" not in name and (f"connector" not in name or not training_args.tune_mlp):
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
    data_args.image_processor = student_model.model.vision_tower._image_processor
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    train_dataset = data_module["train_dataset"]
    train_dataloader = DataLoader(train_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=data_module["data_collator"], shuffle=True)
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
            images = batch["images"].to(dtype)

            if training_args.tl_weight > 0:    
                with torch.no_grad():
                    teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, images=images, use_cache=False, output_hidden_states=training_args.output_hidden_states)
                    teacher_hidden_states, _ = teacher_outputs.hidden_states
                student_outputs = student_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels, images=images, use_cache=False, output_hidden_states=training_args.output_hidden_states, teacher_outputs=teacher_hidden_states)
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
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels, images=images, use_cache=False, output_hidden_states=training_args.output_hidden_states)
                student_hidden_states, _ = student_outputs.hidden_states

                with torch.no_grad():
                    teacher_outputs = teacher_model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels, images=images, use_cache=False, output_hidden_states=training_args.output_hidden_states, teacher_outputs=student_hidden_states)
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
