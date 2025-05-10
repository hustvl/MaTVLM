import torch
import os
import json
from tqdm import tqdm
import shortuuid
from torch.utils.data import Dataset, DataLoader
import math
import argparse
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
import re
from PIL import Image
from questions import QUESTIONS, DEPTH_QUESTIONS

from PIL import Image
import math
import inflect
import glob

import random

p = inflect.engine()


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def load_image(image_file, use_thumbnail, input_size=224):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(is_train=False, input_size=input_size)
    if args.dynamic:
        images = dynamic_preprocess(image, image_size=input_size,
                                    use_thumbnail=use_thumbnail,
                                    max_num=args.max_num)
    else:
        images = [image]
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def post_processing(response):
    response = response.replace('\n', '').replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    response = re.sub(pattern, '', response)
    return response


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


def eval_model(args, task):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.checkpoint)
    model_name = get_model_name_from_path(model_path)
    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    questions = QUESTIONS[task] if task != "depth" else DEPTH_QUESTIONS
    answers_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, 'w') as f:
        f.write("")

    if not args.use_image:
        image_folder = None
    else:
        image_folder = args.image_folder
    if not args.use_seg:
        seg_image_folder = None
    else:
        seg_image_folder = args.seg_image_folder
        
    images = glob.glob(os.path.join(args.image_folder, '*.jpg'))
    images = get_chunk(images, args.num_chunks, args.chunk_idx)


    if seg_image_folder is not None:
        if task == "depth":
            meta = json.load(open(os.path.join(seg_image_folder, "panoptic.json")))
        else:
            meta = json.load(open(os.path.join(seg_image_folder, f"{task}.json")))
    else:
        meta = None
    # data_loader = create_data_loader(questions, args, task, image_folder, seg_image_folder, tokenizer, image_processor, model.config)

    for image_file in tqdm(images, total=len(images)):
        image_filename = os.path.basename(image_file)
        qs = random.choice(questions)

        meta_data = {}
        if meta is not None:
            segments = meta[image_filename]                    
            if len(segments) > 0:
                for seg in segments:
                    category = seg["category"].split(",")[0].split("-")[0]
                    if category not in meta_data.keys():
                        meta_data[category] = []
                    if "depth" in seg:
                        meta_data[category].append(seg["bbox"] + [seg['depth']])
                    else:
                        meta_data[category].append(seg["bbox"])
                if task == "depth" or task == "panoptic":
                    prompt = "Provide the positional information for the mentioned objects, specifying their x and y coordinates, width, and height, with the origin at the top-left corner (0,0), as well as their depth value: " 
                else:
                    prompt = "Provide the positional information for the mentioned objects, specifying their x and y coordinates, width, and height, with the origin at the top-left corner (0,0): "

                if len(meta_data) > 0:
                    qs = prompt + str(meta_data) + ". " + qs

        if task == "depth":
            qs = qs +' Return answer in the paragraph format: "The depth order for the objects present in the image is: ..." and then list the objects with their order number (if greater than 1) separated by a hyphen like "person-2". For example, an acceptable response is "The depth order for objects present in the image is: bicycle, bicycle-2, bicycle-3, pavement, road, bus, tree, sky, building."'
        else:
            qs = qs + " Return the answer in the paragraph format: 'The objects present in the image are: ...' and then list the objects with their count in word format (if greater than 1) in front of them, like 'two people'."
        
        pixel_values = load_image(image_file, use_thumbnail, image_size).cuda().to(torch.bfloat16)

        generation_config = dict(
            do_sample=args.sample,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=qs,
            generation_config=generation_config,
            verbose=True
        )
        response = post_processing(response)

        with open(f'{answers_file}', 'a') as f:
            f.write(f'Image: {image_file.split("/")[-1]}\n')
            f.write(f'<<QUESTION>>: {qs}\n')
            f.write(f'<<ANSWER>>: {response}\n')
            f.write('-------------------------------------------------------\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--seg-image-folder", type=str, default=None)
    parser.add_argument("--output-file", type=str, default="output")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--task", type=str, default="panoptic")
    parser.add_argument("--use-prompt", action="store_true")
    parser.add_argument("--use-image", action="store_true", default=True)
    parser.add_argument("--no-use-image", dest="use_image", action="store_false")
    parser.add_argument("--use-seg", action="store_true")
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    args = parser.parse_args()

    eval_model(args, args.task)
