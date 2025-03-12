import os
import random
import torch
from PIL import Image, ImageOps
from typing import Union, Optional
import argparse
import gc

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


SYSTEM_MESSAGE_Detailed = "You are a helpful assistant focused on providing detailed descriptions and background information for images. Analyze the given image and generate a comprehensive caption that includes the visual style, spatial relationships between elements, texture details, descriptions of the main objects, and relevant world knowledge to enhance understanding."
SYSTEM_MESSAGE_Medium = "You are a helpful assistant specialized in creating medium-length captions for images. Analyze the provided image and generate a caption that captures the key visual elements, while maintaining clarity and coherence."
SYSTEM_MESSAGE_Short = "You are a helpful assistant focused on creating short captions for images. Analyze the provided image and generate a concise caption that highlights the main subject."
SYSTEM_MESSAGE_Tag = "You are a helpful assistant specialized in generating key tags for images. Analyze the provided image and create a list of relevant tags that capture the main subjects, themes, and notable elements."

SYSTEM_MESSAGE_Detailed_CN = "你是一位专注于提供详细描述和背景信息的助手。分析给定的图像，生成一个全面的描述，包含视觉风格、元素之间的空间关系、纹理细节、主要对象的描述，以及增强理解的相关背景知识。"
SYSTEM_MESSAGE_Medium_CN = "你是一位专注于创建中等长度图像描述的助手。分析所提供的图像，生成一个描述，捕捉关键视觉元素，保持清晰和连贯。"
SYSTEM_MESSAGE_Short_CN = "你是一位专注于创建简短图像描述的助手。分析提供的图像，生成一个简洁的描述，突出主要主体。"
SYSTEM_MESSAGE_Tag_CN = "你是一位专注于为图像生成关键词标签的助手。分析提供的图像，创建一个相关标签列表，捕捉主要主题、元素和显著特点。"



SYSTEM_MESSAGE_Detailed_Natural = "You are a helpful natural image captioner. Provide a comprehensive description of the natural image, including the main subject, background elements, lighting conditions, color distribution, textures, spatial arrangement, and any potential dynamic context."
SYSTEM_MESSAGE_Medium_Natural = "You are a helpful natural image captioner. Describe the main content, background in the medium-length text."
SYSTEM_MESSAGE_Short_Natural = "You are a helpful natural image captioner. Describe the main content, background in the short-length text."

SYSTEM_MESSAGE_Detailed_CN_Natural = "您是一位乐于助人的自然图片分析助手。请提供自然图像的全面描述，包括主要主题、背景元素、光照条件、颜色分布、纹理、空间排列以及任何潜在的动态背景。"
SYSTEM_MESSAGE_Medium_CN_Natural = "您是一位乐于助人的自然图片分析助手。请用中等长度的文本描述主要内容和背景。"
SYSTEM_MESSAGE_Short_CN_Natural = "您是一位乐于助人的自然图片分析助手。请用短文本描述主要内容和背景。"

SYSTEM_MESSAGE_OCR_Image_CN = '你是一个精确分析文本内容图像的先进助手。你可以详细描述图像中的文本信息和视觉信息，包括字体样式、大小、颜色、背景、文本布局和其他视觉对象'
SYSTEM_MESSAGE_OCR_textqa = 'You are an advanced OCR model designed to accurately extract text from images. Your task is to analyze the provided image and return the text in a clear, readable format.'
SYSTEM_MESSAGE_OCR_chart_math = 'You are an advanced model designed to analyze and interpret charts, graphs, and data visualizations. Your task is to extract relevant information from the provided chart and return a detailed analysis.'
SYSTEM_MESSAGE_chemdata = "You are an advanced model designed to interpret and analyze SMILES (Simplified Molecular Input Line Entry System) strings, which represent chemical structures. Your task is to extract key information about the chemical structure described by the provided SMILES string"
SYSTEM_MESSAGE_OCR_chart_math_CN = '你是一个高级模型，旨在分析和解释图表、图形、表格、数学公式、数学几何图和数据可视化。你的任务是从提供的图像中提取相关信息，并以机器可读格式或结构化格式返回文本。'
SYSTEM_MESSAGE_OCR_table_math = 'You are a data conversion and extraction expert. Given a table image, you can convert it into CSV, HTML, Markdown or LaTeX formats, then extract and summarize the key relationships or insights from the data.'
SYSTEM_MESSAGE_OCR_mathgeo_math = 'You are a geometry analysis expert. Given a geometric figure, you can convert it into a corresponding LaTeX (e.g., TikZ) representation, then provide insights or interpretations about the structure or properties.'
SYSTEM_MESSAGE_OCR_equation_math = 'You are an equation analysis expert. Given an equation image, you can convert it into proper LaTeX format, then summarize any key mathematical properties, patterns, or insights it conveys.'

SYSTEM_MESSAGE_UI = "You are analyzing a UI webpage layout. Provide a detailed caption describing the layout's structure, including the arrangement, style, and functionality of key components such as buttons, navigation bars, input fields, and visual elements."


# Corresponding questions for each prompt
QUESTION_Detailed = "Describe this image in detail."

QUESTION_Equation_Detailed = "Describe this image in detail and analyze this equation step by step."

QUESTION_Medium = "Can you describe this image with a medium-length caption?"
QUESTION_Short = "Can you provide a brief caption for this image?"
QUESTION_Tag = "What are the key tags associated with this image?"


QUESTION_Detailed_CN= "详细描述这张图片。"
QUESTION_Medium_CN = "你能用中等长度的描述来描述这张图片吗？"
QUESTION_Short_CN = "你能为这张图片提供一个简短的描述吗？"
QUESTION_Tag_CN = "这张图片的主要标签是什么？"

QUESTION_chem = "What is the isomeric SMILES notation for the chemical structure shown in the image?"
QUESTION_charttablemarkdwon = "Convert this chart into a table in Markdown format."
QUESTION_tablelatex = "Convert this table into a table in latex format."
QUESTION_mathgeo = 'How can I create LaTeX-compatible versions of geometric and math visuals?'
QUESTION_equation = 'Transform the equations in this image into LaTeX format.'
QUESTION_table = 'Transform this chart into a table in LaTeX format.'


QUESTION_chart_caption = "Describe this chart in detail."
QUESTION_chart_caption_CN = "请详细分析这个图表"
QUESTION_mathgeo_caption = 'Describe this image in detail.'
QUESTION_mathgeo_caption_mavis = 'Describe how the geometric diagram is designed step by step and highlight its key properties.'
QUESTION_mathgeo_caption_autogeo = 'Render a clear and concise description of a image about geometric shapes.'

QUESTION_equation_caption = 'Describe this image in detail.'
QUESTION_table_caption = 'Please analyze this table in detail.'
QUESTION_table_caption_CN = '请详细分析这个表格'
QUESTION_UI_caption = "Describe this image in detail."

QUESTION_poster_ocr = "Extract all prominent text from the poster, ensuring clear recognition of titles, subtitles, and key information."
QUESTION_generalOCR = "Extract all readable text content from the image, ensuring accurate character recognition."
QUESTION_pdfocr = "As a smart PDF to Markdown conversion tool, can you process the given PDF and output it as Markdown?"

QUESTION_video = "How would you describe this video in detail from the perspective of Short Caption, Background Caption, Main Object Caption, Reference Caption, Standard Summary, and Key Tags?"

system_prompt_map_dict = {'aigc': ['Detailed', 'Medium' , 'Short', 'Tag', 'Detailed_CN', 'Medium_CN' , 'Short_CN', 'Tag_CN'],
                          'natural': ['Detailed_Natural', 'Medium_Natural' , 'Short_Natural', 'Tag_Natural', 'Detailed_CN_Natural', 'Medium_CN_Natural' , 'Short_CN_Natural', 'Tag_CN_Natural'],
                          'mathgeo': ['mathgeo_tikz', 'mathgeo_caption' , 'mathgeo_caption_mavis', 'mathgeo_caption_autogeo'],
                          'chart': ['chart_markdwon', 'chart_caption' , 'chart_caption_CN'],
                          'equation': ['equation_latex', 'equation_caption' , 'equation_caption_CN'],
                          'ocr': ['poster_ocr', 'general_ocr' , 'poster_caption','poster_caption_CN', 'pdf_ocr'],
                          'gui': ['gui'],
                          'table': ['table_latex', 'table_caption', 'table_caption_CN']
   }

def map_system_prompt_image(selected_prompt):
    if selected_prompt == "Detailed":
        return SYSTEM_MESSAGE_Detailed, QUESTION_Detailed
    elif selected_prompt == "Medium":
        return SYSTEM_MESSAGE_Medium, QUESTION_Medium
    elif selected_prompt == "Short":
        return SYSTEM_MESSAGE_Short, QUESTION_Short
    elif selected_prompt == "Tag":
        return SYSTEM_MESSAGE_Tag, QUESTION_Tag
    elif selected_prompt == "Detailed_CN":
        return SYSTEM_MESSAGE_Detailed_CN, QUESTION_Detailed_CN
    elif selected_prompt == "Medium_CN":
        return SYSTEM_MESSAGE_Medium_CN, QUESTION_Medium_CN
    elif selected_prompt == "Short_CN":
        return SYSTEM_MESSAGE_Short_CN, QUESTION_Short_CN
    elif selected_prompt == "Tag_CN":
        return SYSTEM_MESSAGE_Tag_CN, QUESTION_Tag_CN

    if selected_prompt == "Detailed_Natural":
        return SYSTEM_MESSAGE_Detailed_Natural, QUESTION_Detailed
    elif selected_prompt == "Medium_Natural":
        return SYSTEM_MESSAGE_Medium_Natural, QUESTION_Medium
    elif selected_prompt == "Short_Natural":
        return SYSTEM_MESSAGE_Short_Natural, QUESTION_Short
   
    elif selected_prompt == "Detailed_CN_Natural":
        return SYSTEM_MESSAGE_Detailed_CN_Natural, QUESTION_Detailed_CN
    elif selected_prompt == "Medium_CN_Natural":
        return SYSTEM_MESSAGE_Medium_CN_Natural, QUESTION_Medium_CN
    elif selected_prompt == "Short_CN_Natural":
        return SYSTEM_MESSAGE_Short_CN_Natural, QUESTION_Short_CN
    
    if  selected_prompt=='mathgeo_tikz':
        return SYSTEM_MESSAGE_OCR_mathgeo_math, QUESTION_mathgeo
    elif  selected_prompt=='mathgeo_caption':
        return SYSTEM_MESSAGE_OCR_mathgeo_math, QUESTION_mathgeo_caption
    elif  selected_prompt=='mathgeo_caption_mavis':
        return SYSTEM_MESSAGE_OCR_mathgeo_math, QUESTION_mathgeo_caption_mavis
    elif  selected_prompt=='mathgeo_caption_autogeo':
        return SYSTEM_MESSAGE_OCR_mathgeo_math, QUESTION_mathgeo_caption_autogeo

    if selected_prompt=='chart_markdwon':
        return SYSTEM_MESSAGE_OCR_chart_math, QUESTION_charttablemarkdwon
    elif selected_prompt=='chart_caption' :
        return SYSTEM_MESSAGE_OCR_chart_math ,QUESTION_chart_caption
    elif selected_prompt=='chart_caption_CN' :
        return SYSTEM_MESSAGE_OCR_chart_math_CN ,QUESTION_chart_caption_CN

    if selected_prompt=='table_latex':
        return SYSTEM_MESSAGE_OCR_table_math, QUESTION_tablelatex
    elif selected_prompt=='table_caption' :
        return SYSTEM_MESSAGE_OCR_table_math ,QUESTION_table_caption
    elif selected_prompt=='table_caption_CN' :
        return SYSTEM_MESSAGE_OCR_chart_math_CN ,QUESTION_table_caption_CN

    elif selected_prompt=='equation_latex' :
        return SYSTEM_MESSAGE_OCR_equation_math ,QUESTION_equation
    elif selected_prompt=='equation_caption' :
        return SYSTEM_MESSAGE_OCR_equation_math ,QUESTION_Detailed
    elif selected_prompt=='equation_caption_CN' :
        return SYSTEM_MESSAGE_OCR_chart_math_CN ,QUESTION_Detailed_CN

    if selected_prompt=='poster_ocr':
        return SYSTEM_MESSAGE_OCR_textqa , QUESTION_poster_ocr
    elif selected_prompt=='pdf_ocr':
        return SYSTEM_MESSAGE_OCR_textqa ,QUESTION_pdfocr
    elif selected_prompt=='general_ocr' :
        return SYSTEM_MESSAGE_OCR_textqa  , QUESTION_generalOCR 
    elif selected_prompt=='poster_caption' :
        return SYSTEM_MESSAGE_Detailed  , QUESTION_Detailed 
    elif selected_prompt=='poster_caption_CN' :
        return SYSTEM_MESSAGE_OCR_Image_CN  , QUESTION_Detailed_CN 
 
    if selected_prompt=='gui':
        return SYSTEM_MESSAGE_UI , QUESTION_UI_caption
    


def load_image(image_path: str) -> "PIL.Image.Image":
    """加载图像并转换为RGB格式"""
    try:
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def inference_single_image(model_path: str, image_path: str, image_type: str):
    """单张图像推理函数"""
    # 加载模型和处理器（只加载一次）
    if not hasattr(inference_single_image, "model"):
        inference_single_image.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        inference_single_image.processor = AutoProcessor.from_pretrained(
            model_path,
            min_pixels=2*28*28,
            max_pixels=6400*28*28
        )

    pixel_values = load_image(image_path)
    if pixel_values is None:
        return {}

    responses = {'image': os.path.basename(image_path)}
    
    # 确定要使用的系统提示类型
    type_mapping = {
        'aigc': system_prompt_map_dict['aigc'],
        'natural': system_prompt_map_dict['natural'],
        'mathgeo': system_prompt_map_dict['mathgeo'],
        'chart': system_prompt_map_dict['chart'],
        'equation': system_prompt_map_dict['equation'],
        'ocr': system_prompt_map_dict['ocr'],
        'gui': system_prompt_map_dict['gui'],
        'table': system_prompt_map_dict['table']
    }
    
    system_type_lists = type_mapping.get(image_type.lower(), [])

    for system_prompt_type in system_type_lists:
        system_prompt, question = map_system_prompt_image(system_prompt_type)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
        ]
        
        try:
            text = inference_single_image.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = inference_single_image.processor(
                text=[text], 
                images=[pixel_values], 
                padding=True, 
                return_tensors="pt"
            ).to('cuda')

            torch.random.manual_seed(random.randint(0, 2**32-1))
            
            output_ids = inference_single_image.model.generate(
                **inputs,
                top_p=0.8,
                temperature=0.2,
                do_sample=True,
                max_new_tokens=2048,
                top_k=10,
                repetition_penalty=1.2
            )
            
            generated_ids = [output_ids[len(input_ids):] 
                           for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = inference_single_image.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            
            responses[system_prompt_type] = output_text[0]
            print(f"\n=== {system_prompt_type} ===")
            print(output_text[0])

        except Exception as e:
            print(f"Error processing {system_prompt_type}: {e}")
            responses[system_prompt_type] = "Error"

        gc.collect()
        torch.cuda.empty_cache()

    return responses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Single image captioning')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--image_type', type=str, required=True, 
                        help='Image type from: aigc, natural, mathgeo, chart, equation, ocr, gui, table')
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        exit(1)

    print(f"\nProcessing image: {args.image_path}")
    results = inference_single_image(args.model_path, args.image_path, args.image_type)
    
    print("\nFinal Results:")
    for k, v in results.items():
        if k != 'image':
            print(f"\n=== {k} ===")
            print(v)

 