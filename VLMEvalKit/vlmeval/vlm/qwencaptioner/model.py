from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch

from ..base import BaseModel
from .prompt import Qwen2VLPromptMixin
from ...smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag
import json
from vllm import LLM, SamplingParams

def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def split_model():
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    # + 8 is virtual layers for the memory of visual
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['visual'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = last_gpu
    device_map['model.rotary_emb'] = last_gpu
    device_map['lm_head'] = last_gpu
    return device_map


class Qwen2VLCap(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        model_path1: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        #system_prompt_chart: str | None = None,
        verbose: bool = False,
        use_vllm:   bool = False,
        tensor_parallel_size = 2,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        print('system_prompt',system_prompt)
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        self.use_vllm = use_vllm

        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        rank, world_size = get_rank_and_world_size()

        assert model_path is not None
        self.model_path = model_path
        self.processor = Qwen2VLProcessor.from_pretrained(model_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        # If only one process and GPU memory is less than 40GB
        
        #if not self.use_vllm: 
        if auto_split_flag():
            assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
            # Will Use All GPUs to run one model
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
            )
        elif '72b' not in self.model_path.lower():
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype='auto', device_map='cpu', attn_implementation='flash_attention_2'
            )
            self.model.cuda().eval()
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype='auto', device_map=split_model(), attn_implementation='flash_attention_2'
            )
            self.model.eval()
        #else:
        #    self.model = LLM( model=model_path, limit_mm_per_prompt={"image": 10, "video": 10},)
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print('load for summarizer')
        if not self.use_vllm: 
            self.summarizer = AutoModelForCausalLM.from_pretrained(
                    model_path1,
                    torch_dtype=torch.bfloat16,
                    device_map="auto", attn_implementation='flash_attention_2'
                )
        else:
            self.summarizer = LLM(model=model_path1,
             #device="cuda:1",
             #tensor_parallel_size=2,          # 根据GPU数量调整（如4*A100-40G）
             tensor_parallel_size=tensor_parallel_size,  
             gpu_memory_utilization=0.6,       # 显存利用率
             dtype="bfloat16",
             max_model_len = 113168,#12000,
             )
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained(model_path1)


        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None, function_type: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                if dataset == 'ChartQA_TEST': #OCR_chart_caption
                    
                
                    item = {'type': 'text', 'text': 'Describe this chart in detail.'}#s['value']
                elif function_type == 'OCR_table':
                    p = 0.5
                    
                    item = {'type': 'text', 'text': "Please analyze this table in detail."}#s['value']
                elif function_type == 'chem':
                    item = {'type': 'text', 'text': 'What is the isomeric SMILES notation for the chemical structure shown in the image?'}#s['value']
                else:
                    item = {'type': 'text', 'text': 'Describe this image in detail.'}#s['value']
                
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content


    def _prepare_summarizer_content(self, inputs: list[dict[str, str]], image_caption_batch: list, dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            
            if s['type'] == 'text':
                item =  f'The detailed caption of the provided image: {image_caption_batch}' + s['value']
            
                content.append(item)
        #print(content)
        print(len(content))
        return content[0]

    def generate_inner(self, message, dataset=None,question_type=None,answer=None, question=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err
        
        
                
        messages = []

        from .build_sys_prompt import build_sys_prompt
        _, function_type = build_sys_prompt(dataset,question_type)
       
        print('self.system_prompt',self.system_prompt)
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset,function_type=function_type)})

        print('messages',messages)
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        ##Omni_caption
        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs = inputs.to('cuda')
    
        max_retries = 6  # Maximum number of retries
        retry_count = 0
        #image_caption_batch = ""
        # Define the output file
        output_file = f'./outputs/qwencaptioner-instruct{dataset}_caption.jsonl'
        caption_data=[]

        # Check if the output file exists
        if not os.path.exists(output_file):
            print(f"文件 {output_file} 不存在，正在创建一个新的文件.")
            # 如果文件不存在，创建一个空的 jsonl 文件
            with open(output_file, 'w', encoding='utf-8') as file:
                pass 

        
        with open(output_file, 'r', encoding='utf-8') as file:
                for line in file:
                    entry = json.loads(line)
                    caption_data.append(entry)
        # Extract the image filename
        #breakpoint()
        if dataset == 'OlympiadBench':
          image_filename = messages[1]['content'][1]['image'].split('/')[-1]
        else:
          image_filename = messages[1]['content'][0]['image'].split('/')[-1]
        print('image_filename',image_filename)
        

        # Check if the image already exists in the JSONL file
        def image_exists_in_jsonl(image_filename, output_file):
            with open(output_file, 'r', encoding='utf-8') as file:
                for line in file:
                    entry = json.loads(line)
                    caption_first = list(entry.keys())[1]
                    if entry.get('image') == image_filename:
                        return entry.get(caption_first)
            return False
        image_caption_batch = ''
        # If the image already exists in the JSONL file, skip caption generation
        if image_exists_in_jsonl(image_filename, output_file):
            print(f"Caption for {image_filename} already exists. Skipping generation.")
            image_caption_batch = image_exists_in_jsonl(image_filename, output_file)
        
        if len(image_caption_batch.split(' ')) > 10:
            image_caption_batch=image_caption_batch
        else:
            image_caption_batch = ''
            while len(image_caption_batch.split(' ')) < 10 and retry_count < max_retries:
                # Perform the generation as usual
                if retry_count>1:
                    messages = []

                    from .build_sys_prompt import build_sys_prompt
                    system_prompt, function_type = build_sys_prompt(dataset,question_type)
                    self.system_prompt =system_prompt
                    print('self.system_prompt',self.system_prompt)
                    if self.system_prompt is not None:
                        messages.append({'role': 'system', 'content': self.system_prompt})
                    messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})

                    text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
                    images, videos = process_vision_info([messages])
                    inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
                    inputs = inputs.to('cuda')

                #print('self.generate_kwargs',self.generate_kwargs)
                self.generate_kwargs['top_p'] = 0.85
                self.generate_kwargs['temperature'] = 0.7
                self.generate_kwargs['repetition_penalty'] =1.2
                self.generate_kwargs['top_k'] =20
                self.generate_kwargs['do_sample'] =True
                
                generated_ids = self.model.generate(
                    **inputs,
                    **self.generate_kwargs,
                )

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                out = self.processor.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                # Get the generated caption (first element in the batch)
                image_caption_batch = out[0]
            
                # If the length is too short, increment retry_count and try again
                retry_count += 1

            if dataset == 'OlympiadBench':
                #image_filename = messages[1]['content'][1]['image'].split('/')[-1]
                with open(output_file, "a", encoding="utf-8") as file:      
                    json.dump({"image": messages[1]['content'][1]['image'].split('/')[-1],f"{function_type}": image_caption_batch,'answer':answer,'question':question}, file, ensure_ascii=False)
                    file.write("\n")
            else:
                #image_filename = messages[1]['content'][1]['image'].split('/')[-1]
                with open(output_file, "a", encoding="utf-8") as file:      
                    json.dump({"image": messages[1]['content'][0]['image'].split('/')[-1],f"{function_type}": image_caption_batch,'answer':answer,'question':question}, file, ensure_ascii=False)
                    file.write("\n")
        
        if len(image_caption_batch) < 10:
            print("Warning: Generated caption is still too short after 3 attempts.")
        
        ##LLM summarizer
        messages = []
        
        messages.append({'role': 'system', 'content': "You are a helpful assistant."})
        messages.append({'role': 'user', 'content': self._prepare_summarizer_content(message,image_caption_batch, dataset=dataset)})

        #text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        print('messages',messages)
        text = self.summarizer_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        summarizer_inputs = self.summarizer_tokenizer([text], return_tensors="pt").to('cuda')#.to(device)

        if not self.use_vllm:
            summarizer_generated_ids = self.summarizer.generate(
                **summarizer_inputs,
                max_new_tokens=4096,
            )
            summarizer_ggenerated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(summarizer_inputs.input_ids, summarizer_generated_ids)
            ]
            out =  self.summarizer_tokenizer.batch_decode(
                summarizer_ggenerated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            response = out[0]
        else:
            from vllm import LLM, SamplingParams
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.05,
                max_tokens=8192,
                stop_token_ids=[],
            )
            outputs = self.summarizer.generate(text, sampling_params)
            response = outputs[0].outputs[0].text

        print('=====================================')
        print(response)
        if self.verbose:
            print(f'\033[32m{response}\033[0m')

        return response
