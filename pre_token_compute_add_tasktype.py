import json
import math
import argparse
from PIL import Image
from pathlib import Path
from transformers import AutoProcessor
from data_reader import read_general

class DataProcessor:
    def __init__(self, raw_data, processor, conv2length={}):
        self.raw_data = raw_data
        self.processor = processor
        self.conv2length = conv2length
        self.processed_images = set()

    def load_existing_data(self, output_file):
        """Load already processed entries from the output JSONL file if it exists."""
        if Path(output_file).exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data_item = json.loads(line)
                    if 'image' in data_item:
                        self.processed_images.add(data_item['image'])

    def save_data_item(self, output_file, data_item):
        """Append a single data item to the output JSONL file."""
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data_item, ensure_ascii=False) + '\n')

    def compute_token_and_image_length(self, output_dir, split_index, data_name, tasktype, num_splits):
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 计算每个 split 的大小
        split_size = math.ceil(len(self.raw_data) / num_splits)
        split_data = self.raw_data[split_index * split_size:(split_index + 1) * split_size]

        # 定义输出文件路径
        output_file = Path(output_dir) / f'{data_name}_split{split_index + 1}.jsonl'
        
        # 加载已存在的 JSONL 文件
        self.load_existing_data(output_file)
        print(f"Loaded {len(self.processed_images)} already processed images for split {split_index + 1}")

        for data_item in split_data:
            data_item = json.loads(data_item)

            # 检查是否已处理
            image_path = data_item.get('image')
            if image_path and image_path in self.processed_images:
                print(f"Skipping already processed image: {image_path}")
                continue

            # 获取 conversations 并计算 token 长度
            conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
            str_length = len(conversations)
            
            # 计算 token 长度
            token_length = self.processor.tokenizer(
                conversations, return_tensors='pt', padding=False, truncation=False
            ).input_ids.size(1)

            # 检查并计算 image_token_num
            image_token_num = 0
            if image_path:
                #if 'size' in data_item:
                #    if 'lc2' not in image_path:
                #        file_path =  'lc2:'+image_path
                #    else:
                #        file_path =  image_path
                #    width, height = data_item['size']
                #    if isinstance(width, str):
                #        width = int(width)
                #    if isinstance(height, str):
                #        height = int(height)
                #    image_token_num += (width * height) // (28 * 28)
                #else:
                try:
                    if 'lc2' not in image_path:
                        file_path =  'lc2:'+image_path
                    else:
                        file_path =  image_path
                    with Image.open(read_general(file_path)) as img:
                        width, height = img.size
                        image_token_num += (width * height) // (28 * 28)
                        print(f"Processed image {image_path}: {width}x{height}, tokens: {image_token_num}")
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue

            if image_token_num == 0:
                continue

            # 更新数据项并保存
            data_item['token_length'] = token_length
            data_item['image_token_num'] = image_token_num
            data_item['image'] = file_path
            data_item['tasktype'] = tasktype

            # 保存当前处理的条目
            self.save_data_item(output_file, data_item)
            self.processed_images.add(image_path)
            print(f"Saved processed image: {image_path}")

        print(f"Completed processing split {split_index + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL files by split index")
    parser.add_argument("--split_index", type=int, help="Index of the split to process (0-31)")
    parser.add_argument("--split_num", type=int, help="Index of the split to process (0-31)")
    parser.add_argument("--output_dir", type=str, help="Directory to save the processed split")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the input JSONL data file")
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--tasktype", type=str, required=True, help="Name of the dataset")
    args = parser.parse_args()

    # 读取原始 JSONL 数据
    with open(args.data_file, 'r', encoding='utf-8') as f:
        raw_data = f.readlines()

    # 加载处理器
    processor = AutoProcessor.from_pretrained("/mnt/petrelfs/luyiting/ckt/Qwen2-VL-7B-Instruct/")

    # 实例化并处理指定分区
    data_processor = DataProcessor(raw_data, processor)
    data_processor.compute_token_and_image_length(args.output_dir, args.split_index, args.data_name, args.tasktype, num_splits = args.split_num)
