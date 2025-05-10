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
    
def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.checkpoint)
    model_name = get_model_name_from_path(model_path)
    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if args.meta_file is not None:
        meta = json.load(open(args.meta_file, "r"))
    else:
        meta = None
    for line in tqdm(questions, total=len(questions)):
        image_file = line["image_path"]
        qs = line["question"]
        meta_data = {}
        if meta is not None:
            assert image_file in meta, f"Image {image_file} not found in the meta data."
            meta_prompt = meta[image_file]
            qs = "The texts detected in the image with the position: " + str(meta_prompt) + ".\n" + qs
        
        pixel_values = load_image(os.path.join(args.image_folder, image_file), use_thumbnail, image_size).cuda().to(torch.bfloat16)

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

        line["predict"] = response
        line['qs'] = qs
        ans_file.write(json.dumps(line) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="facebook/opt-350m")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--meta-file", type=str, default=None)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    args = parser.parse_args()

    eval_model(args)
