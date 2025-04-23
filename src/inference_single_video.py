
import os
import random
from datetime import datetime
import json
import torch
import PIL
from PIL import Image, ImageOps
from typing import Union, Optional
import argparse
import gc
import sys
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

SYSTEM_MESSAGE_VIDEO_ALL = 'You are a multimodal assistant designed to analyze video content and its associated captions. Your goal is to provide insightful feedback on the quality, relevance, and semantic alignment of the captions with the video content.'

QUESTION_video_woStyle = "How would you describe this video in detail from the perspective of Short Caption, Background Caption, Main Object Caption, Reference Caption, Standard Summary, and Key Tags?"
QUESTION_video = "How would you describe this video in detail from the perspective of Short Caption, Background Caption, Main Object Caption, Reference Caption, Standard Summary, Key Tags and Style Tags?"

system_prompt_map_dict = {
                          'video': ['Video_all', 'Video_all_wostyle' ],
   }

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "U4R/OmniCaptioner_Video",
    attn_implementation ="flash_attention_2",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(
    "U4R/OmniCaptioner_Video",
    min_pixels=6 * 28 * 28,
    max_pixels=8100 * 28 * 28
)

def map_system_prompt_image(selected_prompt):
    if selected_prompt=='Video_all':
        return SYSTEM_MESSAGE_VIDEO_ALL , QUESTION_video  
    if selected_prompt=='Video_all_wostyle':
        return SYSTEM_MESSAGE_VIDEO_ALL , QUESTION_video_woStyle

def save_output_to_jsonl(data, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')



def process_video(video_path, image_type):
    responses = {'video': video_path.split('/')[-1]}
    system_type_lists = system_prompt_map_dict['video']

    for system_prompt_type in system_type_lists:
        system_prompt, question = map_system_prompt_image(system_prompt_type)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "video","video": video_path, 'max_pixels':8100 * 28 * 28},
                    {"type": "text", "text": question},
                ],
            },
        ]
        from visual_utils_qwen25vl import process_vision_info
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages,return_video_kwargs=True)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", **video_kwargs)
        inputs = inputs.to('cuda')
        torch.random.manual_seed(random.randint(0, 2**32 - 1))
        
        try:
            output_ids = model.generate(**inputs, 
                  top_p = 0.8,
                  temperature = 0.2,
                  do_sample = True,
                  max_new_tokens=2048,
                  top_k = 10,
                  repetition_penalty=1.2)
            #breakpoint()
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            responses[system_prompt_type] = output_text[0]
        except Exception as e:
            print(f"Inference error for {system_prompt_type}: {e}")
            responses[system_prompt_type] = "Error"
        
        gc.collect()
        torch.cuda.empty_cache()
    return responses

def main(input_dir, output_file, cap_type):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"The specified folder does not exist: {input_dir}")

    video_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv', '.webm'))]
    print(video_files)
    for video_path in video_files:
        print(f"Processing video: {video_path}")
        data = process_video(video_path, cap_type)
        save_output_to_jsonl(data, output_file)
        print(f"Saved output for image: {video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate captions for images in a folder.')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the folder containing videos')
    parser.add_argument('--input_type', type=str, required=True, help='caption style')
    parser.add_argument('--output_file', type=str, required=True, help='Name of the output JSONL file')
    args = parser.parse_args()
    main(args.input_dir, args.output_file, args.input_type)

