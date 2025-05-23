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
        # make sure the output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # calculate length of each split
        split_size = math.ceil(len(self.raw_data) / num_splits)
        split_data = self.raw_data[split_index * split_size:(split_index + 1) * split_size]

        # define the path of the output file
        output_file = Path(output_dir) / f'{data_name}_split{split_index + 1}.jsonl'
        
        # load existing jsonl file
        self.load_existing_data(output_file)
        print(f"Loaded {len(self.processed_images)} already processed images for split {split_index + 1}")

        for data_item in split_data:
            data_item = json.loads(data_item)

            # check whether the image has been processed
            image_path = data_item.get('image')
            if image_path and image_path in self.processed_images:
                print(f"Skipping already processed image: {image_path}")
                continue

            # calculate token length
            conversations = '\n'.join([temp['value'] for temp in data_item['conversations']])
            str_length = len(conversations)
            
            token_length = self.processor.tokenizer(
                conversations, return_tensors='pt', padding=False, truncation=False
            ).input_ids.size(1)

            # calculate image_token_num
            image_token_num = 0
            if image_path:
                
                try:
                
                    with Image.open(read_general(file_path)) as img:
                        width, height = img.size
                        image_token_num += (width * height) // (28 * 28)
                        print(f"Processed image {image_path}: {width}x{height}, tokens: {image_token_num}")
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue

            if image_token_num == 0:
                continue

            data_item['token_length'] = token_length
            data_item['image_token_num'] = image_token_num
            data_item['image'] = file_path
            data_item['task_type'] = tasktype

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

    with open(args.data_file, 'r', encoding='utf-8') as f:
        raw_data = f.readlines()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct/")
    data_processor = DataProcessor(raw_data, processor)
    data_processor.compute_token_and_image_length(args.output_dir, args.split_index, args.data_name, args.tasktype, num_splits = args.split_num)
