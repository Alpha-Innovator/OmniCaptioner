

<div align="center">
<h1> OmniCaptioner: One Captioner to Rule Them All </h1>

</div>
<div align="center">

<p align="center">
💜 <a href="https://alpha-innovator.github.io/OmniCaptioner-project-page/"><b>HomePage</b></a>&nbsp&nbsp | &nbsp&nbsp🤗 <a href="https://huggingface.co/U4R/OmniCaptioner">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2504.07089">Paper</a>&nbsp&nbsp
</p>
</div>


## 📰 News

We propose OmniCaptioner, a versatile visual captioning framework for generating fine-grained textual descriptions across a wide variety of visual domains. Unlike prior methods limited to specific image types (e.g., natural images or geometric visuals), our framework provides a unified solution for captioning natural images, visual text (e.g., posters, UIs, textbooks), and structured visuals (e.g., documents, tables, charts). By converting low-level pixel information into semantically rich textual representations, our framework bridges the gap between visual and textual modalities. Our results highlight three key advantages: (i) Enhanced Visual Reasoning with LLMs, where long-context captions of visual modalities empower LLMs, particularly the DeepSeek-R1 series, to reason effectively in multimodal scenarios; (ii) Improved Image Generation, where detailed captions improve tasks like text-to-image generation and image transformation; and (iii) Efficient Supervised Fine-Tuning (SFT), which enables faster convergence with less data. We believe the versatility and adaptability of OmniCaptioner can offer a new perspective for bridging the gap between language and visual modalities.

![Demo Visualization](assets/demo.jpg)


## 📊 Quantatitive Performance
![Quantitative Results](assets/quantitative.jpg)



## 💻 Finetuning Code
### 1. Create a conda environment and install PyTorch
```bash
conda create -n OmniCap python=3.9
conda activate OmniCap
```
### 2.Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Install flash-attn
```bash
pip install flash-attn --no-build-isolation
```
### 4. Prepare data

Step1: Compute the image visual token and text token in advance to group together features of the dataset of roughly the same length

You can loop through all 32 splits (for example) using the following bash script:
process_splits.sh
```bash
# Define your arguments
OUTPUT_DIR="/path/to/output/directory"
DATA_FILE="/path/to/input/data.jsonl"
DATA_NAME="dataset_name"
TASKTYPE="task_name"
SPLIT_NUM=32

# Loop through all splits (from 0 to 31)
for SPLIT_INDEX in $(seq 0 31); do
    python pre_token_compute_add_tasktype.py \
      --split_index $SPLIT_INDEX \
      --split_num $SPLIT_NUM \
      --output_dir $OUTPUT_DIR \
      --data_file $DATA_FILE \
      --data_name $DATA_NAME \
      --tasktype $TASKTYPE
done
```
Merge the Processed JSONL Files
merge_splits.sh
```bash
#!/bin/bash

# Define output file to merge into
FINAL_OUTPUT_FILE="/path/to/output/merged_data.jsonl"

# Concatenate all the splits into the final output file
cat /path/to/output/directory/*_split*.jsonl > $FINAL_OUTPUT_FILE
```

Step2: You can place the links to your data files in ``data/omni_caption_pretrain.yaml".

The processed data in the output JSONL file will have the following format:
```bash
{
  "id": "film.jpg",
  "image": "images/film.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "Explain the key features of this picture."
    },
    {
      "from": "gpt",
      "value": "Close-up shot of a scratched symbol resembling an eye or fish, etched into a worn, dark wooden surface. The scratches are uneven and look hastily carved, occupying the center of the frame. Minimalist background with emphasis on the raw texture of aged wood, dimly lit with shadows toward the top right corner, suggesting a single light source above. The grain of the wood is visibly highlighted by soft lighting, creating an intimate and mysterious mood. Dark, moody, high contrast, shallow depth of field, film noir aesthetic, focus on texture and detail, rustic and enigmatic atmosphere."
    }
  ],
  "token_length": 127,
  "image_token_num": 549,
  "task_type": "detailed"
}

```
### 5. Start finetuning
```bash
bash scripts/finetune_caption_slurm.sh
```
## 🚀 Inference Code for Image Caption

You can caption the image with AIGC style using the following command:


```python   
CUDA_VISIBLE_DEVICES=0 python src/inference_single_image.py \
    --model_path your_model_path \
    --image_path your_image_path  \
    --image_type aigc 
``` 

You can caption the image with OCR style using the following command:

```python   
CUDA_VISIBLE_DEVICES=0 python src/inference_single_image.py \
    --model_path U4R/OmniCaptioner \
    --image_path your_image_path  \
    --image_type ocr 
```

## 🚀 Inference Code for Video Caption

You can caption the video data using the following command:


```python   
CUDA_VISIBLE_DEVICES=0 python src/inference_single_video.py \
    --model_path U4R/OmniCaptioner_Video \
    --image_path your_video_path  \
    --image_type video 
```

## 🚀 Evaluation Code with LLM

```python   

cd VLMEvalkit
conda create -n VLMEvalkit python=3.9
conda activate VLMEvalkit
pip install -e .

CUDA_VISIBLE_DEVICES=0  nohup python run.py --data MMMU_DEV_VAL --model Omnicaptioner-qwen2-5-3B --verbose > output_omnicap_qwen2-5-3B_MMMU_DEV_VAL.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1  nohup python run.py --data MMMU_DEV_VAL --model Omnicaptioner-qwen2-5-7B --verbose > output_omnicap_qwen2-5-7B_MMMU_DEV_VAL.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3  nohup python run.py --data MMMU_DEV_VAL --model Omnicaptioner-qwen2-5-32B --verbose > output_omnicap_qwen2-5-32B_MMMU_DEV_VAL.log 2>&1 &

CUDA_VISIBLE_DEVICES=0  nohup python run.py --data MMMU_DEV_VAL --model Omnicaptioner-deepseek-distill-7B --verbose > output_omnicap_deepseek_distill_3B_MMMU_DEV_VAL.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1  nohup python run.py --data MMMU_DEV_VAL --model Omnicaptioner-deepseek-distill-32B --verbose > output_omnicap_deepseek_distill_32B_MMMU_DEV_VAL.log 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3  nohup python run.py --data MMMU_DEV_VAL --model Omnicaptioner-deepseek-distill-70B --verbose > output_omnicap_deepseek_distill_70B_MMMU_DEV_VAL.log 2>&1 &

``` 


## Citation

If you find the provided code or models useful for your research, consider citing them as:
```
@article{omnicaptioner,
  title={OmniCaptioner: One Captioner to Rule Them All},
  author={Lu, Yiting and Yuan, Jiakang and Li, Zhen and Zhao, Shitian and Qin, Qi and Li, Xinyue and Zhuo, Le and Wen, Licheng and Liu, Dongyang and Cao, Yuewen and others},
  journal={arXiv preprint arXiv:2504.07089},
  year={2025}
}
```


