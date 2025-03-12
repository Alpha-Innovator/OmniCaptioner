import copy
import os
from dataclasses import dataclass, field
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from .visual_utils import process_vision_info
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode
from params import DataArguments
from .constants import *
import torch.distributed as dist
from pathlib import Path
import warnings
import h5py
from .dataset_from_xllmx import FinetuneConversationDataset

def truncate_sequence(input_ids, labels, max_length, eos_token_id):
    if input_ids.size(0) > max_length:
        input_ids = input_ids[:max_length-1]
        labels = labels[:max_length-1]

    if eos_token_id is not None:
        input_ids = torch.cat([input_ids, torch.tensor([eos_token_id])])
        labels = torch.cat([labels, torch.tensor([eos_token_id])])

    return input_ids, labels

def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

def get_image_info(image_path, min_pixel, max_pixel):
    # Using this because of process_vision_info function
    # Need to fix this in the future    
    
    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "image", 
                "image": image_path,
                "min_pixels": min_pixel,
                "max_pixels": max_pixel

            }
            ]
        }
    ]

    image_input, _ = process_vision_info(messages)

    return image_input[0]

def get_video_info(video_path, max_pixels, fps):
    # Using this because of process_vision_info function
    # Need to fix this in the future

    messages = [
        {"role": "user", 
         "content": [
             {
                "type": "video", 
                "video": video_path,
                "max_pixels": max_pixels,
                "fps": fps
            }
            ]
        }
    ]

    _, video_input = process_vision_info(messages)

    return video_input[0]

class QwenItemProcessor:
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        padding=True,
    ):
        self.flag=None

        self.processor = processor
        self.data_args = data_args
        self.padding = padding
        self.min_pixel = data_args.min_pixels
        self.max_pixel = data_args.max_pixels
        self.max_length = data_args.max_length
        self.fps = data_args.fps
       
                
    def predict_item_token_length(self,data_item):
       
        if 'token_length' in data_item:
            token_length = data_item['token_length']  # Use precomputed length if available
            if 'image_token_num' in data_item:
                if isinstance(data_item['image_token_num'], int):
                    #list_data_dict = json.load(open(data_path, "r"))
                    #data_path
                    data_item['image_token_num'] = [data_item['image_token_num']]
                image_token_num = [size_item for size_item in data_item['image_token_num']]
                #image_token_num = [min(data_args.max_pixels, size_item) for size_item in data_item['image_token_num']]
                image_token_num = sum(image_token_num)
            else:
                image_token_num = 0
            #self.conv2length[str_length] = token_length + image_token_num
            token_length = token_length +image_token_num
            #if token_length>25000:
            #   print('data_item',data_item)
        else:
            # Compute token length using the tokenizer
            conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
            str_length = len(conversations)
            if str_length not in self.conv2length:
                token_length = processor.tokenizer(
                    conversations, return_tensors='pt', padding=False, truncation=False,
                ).input_ids.size(1)
                if 'image_token_num' in data_item:
                    if isinstance(data_item['image_token_num'], int):
                        #list_data_dict = json.load(open(data_path, "r"))
                        #data_path
                        data_item['image_token_num'] = [data_item['image_token_num']]
                    image_token_num = [size_item for size_item in data_item['image_token_num']]
                    #image_token_num = [min(data_args.max_pixels, size_item) for size_item in data_item['image_token_num']]
                    image_token_num = sum(image_token_num)
                else:
                    image_token_num = 0
                #self.conv2length[str_length] = token_length + image_token_num
                token_length = token_length +image_token_num 
            else:
                token_length = self.conv2length[str_length]
            #self.length.append(token_length)
        return token_length

    #def __len__(self):
    #    return len(self.list_data_dict)

    def process_item(self, data_item,training_mode) -> Dict[str, torch.Tensor]:
        
        
        sources = data_item
        if 'task_type' in data_item.keys():
           self.flag =  data_item['task_type']
        else:
            self.flag =  data_item['tasktype']
        
        is_video = False
        processor = self.processor

        if "image" in sources:

            image_files = sources["image"]
            image_folder = self.data_args.image_folder

            if isinstance(image_files, str):
                image_files = [image_files]
            num_images = len(image_files)   # 获取 image 列表中的元素个数

            # 根据 image 的数量生成相应数量的 <image> 标签，并用 \n 分隔
            #image_tags = "<image>\n" * num_images
            #sources["conversations"][0]["value"] = image_tags + sources["conversations"][0]["value"]
            # 根据 image 的数量生成相应数量的 <image> 标签，并用 \n 分隔
            no_image_flag = True
            for con_item in sources["conversations"]:
                if '<image>' in con_item["value"]:
                    no_image_flag = False
            if no_image_flag == True:
                image_tags = "<image>\n" * num_images
                sources["conversations"][0]["value"] = image_tags + sources["conversations"][0]["value"]

            videos = None
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"
            images = []
            
            min_pixel = self.min_pixel // num_images
            max_pixel = self.max_pixel // num_images

            for image_file in image_files:
                #if not os.path.exists(image_file):
                    #if not image_file.startswith("s3://"):
                    #    image_file = os.path.join(image_folder, image_file)
                    #else:
                    #    image_file = image_file
                    #image_file = os.path.join(image_folder, image_file)
                images.append(get_image_info(image_file, min_pixel, max_pixel))

        elif "video" in sources:
            is_video = True
            images=None
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

            video_files = sources["video"]
            video_folder = self.data_args.image_folder

            if isinstance(video_files, str):
                video_files = [video_files]
            
            num_videos = len(video_files)   # 获取 image 列表中的元素个数
            # 根据 image 的数量生成相应数量的 <image> 标签，并用 \n 分隔
            #video_tags = "<video>\n" * num_videos
            #sources["conversations"][0]["value"] = video_tags + sources["conversations"][0]["value"]
            no_video_flag = True
            for con_item in sources["conversations"]:
                if '<video>' in con_item["value"]:
                    no_video_flag = False
            if no_video_flag == True:
                video_tags = "<video>\n" * num_videos
                sources["conversations"][0]["value"] = video_tags + sources["conversations"][0]["value"]


            videos = []
            for video_file in video_files:
                #if not os.path.exists(video_file):
                    #if not video_file.startswith("s3://"):
                    #    video_file = os.path.join(video_folder, video_file)
                    #else:
                #   print(f'{video_file} is not exist')
                videos.append(get_video_info(video_file, self.max_pixel, self.data_args.fps))
            
        else:
            images = None
            videos = None

        sources = copy.deepcopy(llava_to_openai(sources['conversations'], is_video=is_video))

        all_input_ids = [] 
        all_labels = []
        all_pixel_values = []
        all_image_grid_thw = []

        # Qwen2-VL uses a default system message so I've added this.
        system_prompt=''
        if self.flag=='long':
            system_prompt =SYSTEM_MESSAGE_Detailed
        if self.flag=='medium':
            system_prompt =SYSTEM_MESSAGE_Medium   
        if self.flag=='short':
            system_prompt =SYSTEM_MESSAGE_Short   
        if self.flag=='tag':
            system_prompt =SYSTEM_MESSAGE_Tag   
        if self.flag=='long_CN':
            system_prompt =SYSTEM_MESSAGE_Detailed_CN   
        if self.flag=='medium_CN':
            system_prompt =SYSTEM_MESSAGE_Medium_CN   
        if self.flag=='short_CN':
            system_prompt =SYSTEM_MESSAGE_Short_CN 
        if self.flag=='tag_CN':
            system_prompt =SYSTEM_MESSAGE_Tag_CN


        if self.flag=='blip3long' or self.flag=='densefusionlong':
            system_prompt =SYSTEM_MESSAGE_Detailed_Natural
        if self.flag=='blip3medium' or self.flag=='densefusionmedium':
            system_prompt =SYSTEM_MESSAGE_Medium_Natural   
        if self.flag=='blip3short' or self.flag== 'densefusionshort':
            system_prompt =SYSTEM_MESSAGE_Short_Natural   
        if self.flag=='blip3tag' or self.flag=='densefusiontag':
            system_prompt =SYSTEM_MESSAGE_Tag_Natural   
        if self.flag=='blip3long_CN' or self.flag=='densefusionlong_CN':
            system_prompt =SYSTEM_MESSAGE_Detailed_CN_Natural   
        if self.flag=='blip3medium_CN' or self.flag=='densefusionmedium_CN':
            system_prompt =SYSTEM_MESSAGE_Medium_CN_Natural   
        if self.flag=='blip3short_CN' or self.flag== 'densefusionshort_CN':
            system_prompt =SYSTEM_MESSAGE_Short_CN_Natural 
        if self.flag=='blip3tag_CN' or self.flag=='densefusiontag_CN':
            system_prompt =SYSTEM_MESSAGE_Tag_CN_Natural
        
        if self.flag=='dense_ocr_data' or  self.flag=='chartx' or self.flag == 'TinyChart' or self.flag == 'TincyChart':
            system_prompt = SYSTEM_MESSAGE_OCR_chart_math #SYSTEM_MESSAGE_OCR_chart_math  (11k+152k)
        
        if self.flag=='MMtab' or  self.flag=='vrdu_table_chinese_2_w_caption' or self.flag=='vrdu_table_chinese_2_wo_caption' or self.flag=='vrdu_table_large_table' \
         or self.flag=='vrdu_table' :
            system_prompt = SYSTEM_MESSAGE_OCR_table_math #SYSTEM_MESSAGE_OCR_chart_math  (1M+136k+13k)
        if self.flag=='vrdu_table_final' :
            system_prompt = SYSTEM_MESSAGE_OCR_table_math #SYSTEM_MESSAGE_OCR_vrdu_table_final  

        if self.flag=='TincyChart_CN' or  self.flag=='vrdu_table_CN' or self.flag=='vrdu_equation_CN': 
            system_prompt = SYSTEM_MESSAGE_OCR_chart_math_CN #SYSTEM_MESSAGE_OCR_chart_math 
            
        if self.flag=='vrdu_equation' :
            system_prompt = SYSTEM_MESSAGE_OCR_equation_math #SYSTEM_MESSAGE_OCR_vrdu_equation  (5M) 
        if self.flag=='mathgeo' or self.flag=='AutoGeo' or self.flag == 'Mavis_geo' :
            system_prompt = SYSTEM_MESSAGE_OCR_mathgeo_math #SYSTEM_MESSAGE_OCR_vrdu_equation  (102k)
        if self.flag=='chemdata' :
            system_prompt =SYSTEM_MESSAGE_chemdata 

        if self.flag=='poster0322_10k' or   self.flag=='poster0408_4k':
            system_prompt =SYSTEM_MESSAGE_OCR_textqa #SYSTEM_MESSAGE_OCR_poster  
        if self.flag=='vrdu_texteq' :
            system_prompt =SYSTEM_MESSAGE_OCR_textqa  
        if self.flag=='infographics' :
            system_prompt =SYSTEM_MESSAGE_OCR_textqa
        if self.flag=='ocr_v2_35k_v3_20k' :
            system_prompt =SYSTEM_MESSAGE_OCR_textqa  
        if self.flag=='pdf_pageocr':
            system_prompt =SYSTEM_MESSAGE_OCR_textqa  
        
        if self.flag=='wokong':
            system_prompt =SYSTEM_MESSAGE_OCR_Image #SYSTEM_MESSAGE_OCR_poster  
        if self.flag=='wokong_CN':
            system_prompt =SYSTEM_MESSAGE_OCR_Image_CN #SYSTEM_MESSAGE_OCR_poster 
        if self.flag=='poster_CN':
            system_prompt =SYSTEM_MESSAGE_OCR_Image_CN #SYSTEM_MESSAGE_OCR_poster  
        
        if self.flag=='Camera_Caption':
            system_prompt =SYSTEM_MESSAGE_VIDEO_ALL #SYSTEM_MESSAGE_OCR_poster  
        
        if self.flag == 'guiact':
            system_prompt = SYSTEM_MESSAGE_UI
        if self.flag == 'guiact_CN':
            system_prompt = SYSTEM_MESSAGE_UI_CN


        if len(system_prompt) > 0:
            system_message = f"{DEFAULT_IM_START_TOKEN}system\n{system_prompt}\n{DEFAULT_IM_END_TOKEN}\n"
            system_message_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors='pt')['input_ids']
            system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX) 
            
            all_input_ids.append(system_message_input_ids.squeeze(0))
            all_labels.append(system_labels.squeeze(0))
        
        #assert('sorry' not in sources[1])

        last_conversation_image_count = 0 
        last_conversation_video_count = 0 
        #print('enter tokenizer')
        for idx, j in enumerate(range(0, len(sources), 2)):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            assert isinstance(gpt_response['content'], str), "gpt_response must be a string"
         
            user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}\n{DEFAULT_IM_END_TOKEN}\n"
            gpt_response = f"{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n{gpt_response['content']}\n{DEFAULT_IM_END_TOKEN}\n"
            
            #if idx == 0:

            if '<|image_pad|>' in user_input or '<|video_pad|>' in user_input:


                # 统计 <|image_pad|> 和 <|video_pad|> 出现的次数
                image_pad_count = user_input.count('<|image_pad|>')
                video_pad_count = user_input.count('<|video_pad|>')

                begin_img_idx = last_conversation_image_count
                end_img_idx = last_conversation_image_count + image_pad_count

                begin_video_idx = last_conversation_video_count
                end_video_idx = last_conversation_video_count + video_pad_count

                images_each_conversation = None
                videos_each_conversation = None
                if images is not None: 
                    images_each_conversation = images[begin_img_idx:end_img_idx]
                if videos is not None:
                    videos_each_conversation = videos[begin_video_idx:end_video_idx]

                last_conversation_image_count  = end_img_idx
                last_conversation_video_count  = end_video_idx

                inputs = processor(text=[user_input], images=images_each_conversation, videos=videos_each_conversation, padding=False, return_tensors='pt')
                prompt_input_ids = inputs['input_ids']
                all_pixel_values.append(inputs[pixel_key])
                all_image_grid_thw.append(inputs[grid_key])

            else:
                prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            response_input_ids = processor.tokenizer(gpt_response, add_special_tokens=False, padding=False, return_tensors='pt')['input_ids']

            input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
            labels = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),  
                    response_input_ids.squeeze(0),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        
        # There is no need for eos or bos tokens in the input_ids
        # Qwen2-VL doees not use them
        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)

        eos_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        input_ids, labels = truncate_sequence(input_ids, labels, self.max_length, eos_token_id)

        if images is not None:
            pixel_values = torch.cat(all_pixel_values, dim=0)
            image_thw = torch.cat(all_image_grid_thw, dim=0)

        attention_mask = (input_ids > -1000000).to(torch.long)

        data_dict = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        data_dict[pixel_key] = pixel_values
        data_dict[grid_key] = image_thw
        #print('finish process_item')
        return data_dict     
             

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_image_thw = []

        sample = examples[0]

        if "pixel_values_videos" in sample:
            grid_key = "video_grid_thw"
            pixel_key = "pixel_values_videos"

        else:
            grid_key = "image_grid_thw"
            pixel_key = "pixel_values"

        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])

            if pixel_key in example:
                batch_pixel_values.append(example[pixel_key])
                batch_image_thw.append(example[grid_key])
       
        
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        if len(batch_pixel_values)  == 0:
            pixel_values=None
            image_thw=None
        else:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
        
        print('input_ids',input_ids.shape)
       
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            pixel_key: pixel_values,
            grid_key: image_thw,
        }
    

def replace_image_tokens(input_string, is_video=False):

    if is_video:
        if LLAVA_VIDEO_TOKEN + '\n' in input_string:
           input_string = input_string.replace(LLAVA_VIDEO_TOKEN+'\n', VISION_START_TOKEN+DEFAULT_VIDEO_TOKEN+VISION_END_TOKEN)
        else:
            if LLAVA_VIDEO_TOKEN  in input_string:
               input_string = input_string.replace(LLAVA_VIDEO_TOKEN, VISION_START_TOKEN+DEFAULT_VIDEO_TOKEN+VISION_END_TOKEN)
    else:
        if LLAVA_IMAGE_TOKEN + '\n' in input_string:
            input_string = input_string.replace(LLAVA_IMAGE_TOKEN+'\n', VISION_START_TOKEN+DEFAULT_IMAGE_TOKEN+VISION_END_TOKEN)
        else: 
            if LLAVA_IMAGE_TOKEN  in input_string:
               input_string = input_string.replace(LLAVA_IMAGE_TOKEN, VISION_START_TOKEN+DEFAULT_IMAGE_TOKEN+VISION_END_TOKEN)
    return input_string


def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []

    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)

    return transformed_data



def make_supervised_data_module(processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    
    datasets = []
    lengths = []
   
   
    QwenItemProcessor1 = QwenItemProcessor( processor=processor, data_args=data_args,)
    
    sft_dataset =  FinetuneConversationDataset(data_args.meta_path, QwenItemProcessor1, cache_on_disk=True)

    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)
    
    
    return dict(train_dataset=sft_dataset,
                eval_dataset=None,
                data_collator=data_collator)

