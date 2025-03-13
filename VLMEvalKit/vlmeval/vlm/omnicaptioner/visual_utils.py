from __future__ import annotations

import base64
import math
from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode
from decord import VideoReader
from decord import cpu

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return max(math.floor(number / factor),1) * factor


def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    #print('max_pixels',max_pixels)
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    #breakpoint()
    if h_bar * w_bar > max_pixels:
        #breakpoint()
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        #breakpoint()
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif image.startswith("http://") or image.startswith("https://"):
        image_obj = Image.open(requests.get(image, stream=True).raw)
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif "s3://" in image:
        from data_reader import read_general
        #print(image)
        image_obj = Image.open(read_general(image))
    elif image.startswith("data:image"):
        data = image.split(";", 1)[1]
        if data.startswith("base64,"):
            data = base64.b64decode(data[7:])
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    if image_obj.mode in ('P', 'LA', 'L'):
       image_obj = image_obj.convert('RGBA')
       image = image_obj.convert('RGB') 
    else:
        image = image_obj.convert("RGB")
    ## resize
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)
        max_pixels = ele.get("max_pixels", MAX_PIXELS)
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))
   

    return image


def process_video_info(video_path):
    """
    Process video and return its properties and processed dimensions, including pixel values.
    """
    #try:
    # Check if the video is from S3 or local
    if 's3://' in video_path:
        from data_reader import read_general
        video_bytes = read_general(video_path) # Assuming `client.get` fetches the S3 video as bytes
        video_reader = VideoReader(video_bytes, num_threads=1)
    else:
        video_reader = VideoReader(video_path, num_threads=1)

    vlen = len(video_reader)  # Number of frames
    fps = video_reader.get_avg_fps()  # Frames per second
    duration = vlen / float(fps)  # Duration in seconds

    # Get height and width from the first frame
    first_frame = video_reader[0].asnumpy()
    height, width, _ = first_frame.shape

    # Get pixel values for the entire video
    pixel_values = torch.stack([torch.from_numpy(frame.asnumpy()) for frame in video_reader])  # Shape: [T, H, W, C]

    return {
        "frame_count": vlen,
        "height": height,
        "width": width,
        "video_fps": fps,
        "duration": duration,
    
    }, pixel_values


def fetch_video(ele: dict, image_factor: int = IMAGE_FACTOR) -> torch.Tensor | list[Image.Image]:
    if isinstance(ele["video"], str):
        # TODO: support http url

        video = ele["video"]
        if video.startswith("file://"):
            video = video[7:]
        #elif "s3://" in image:
        #    from data_reader import read_general
        #    video = read_general(image)

        '''video, audio, info = io.read_video(
            video,
            start_pts=ele.get("video_start", 0.0),
            end_pts=ele.get("video_end", None),
            pts_unit="sec",
            output_format="TCHW",
        )'''
        info ,video = process_video_info(video) #TCHW
        video = video.permute(0,3,1,2) # TCHW
        assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
        if "nframes" in ele:
            nframes = round_by_factor(ele["nframes"], FRAME_FACTOR)
        else:
            fps = ele.get("fps", FPS)
            min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
            max_frames = floor_by_factor(ele.get("max_frames", min(FPS_MAX_FRAMES, video.size(0))), FRAME_FACTOR)
            nframes = video.size(0) / info["video_fps"] * fps
            nframes = min(max(nframes, min_frames), max_frames)
            nframes = round_by_factor(nframes, FRAME_FACTOR)
        if not (FRAME_FACTOR <= nframes and nframes <= video.size(0)):
            raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {video.size(0)}], but got {nframes}.")

        idx = torch.linspace(0, video.size(0) - 1, nframes).round().long()
        height, width = video.shape[2:]
        video = video[idx]
        video_length = idx.size(0)
        #print('video_length',video_length)
        min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
        total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
        video_max_pixels =  ele.get("max_pixels", VIDEO_MAX_PIXELS)
        max_pixels = max(video_max_pixels, int(min_pixels * 1.05)) #max(min(video_max_pixels, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        #max_pixels = max(min(video_max_pixels, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = max_pixels // video_length
        min_pixels = min_pixels // video_length
        #max_pixels = ele.get("max_pixels", max_pixels)
        if "resized_height" in ele and "resized_width" in ele:
            resized_height, resized_width = smart_resize(
                ele["resized_height"],
                ele["resized_width"],
                factor=image_factor,
            )
        else:
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=image_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        #breakpoint()
        print('video_max_pixels:',video_max_pixels,'max_pixels:',max_pixels,'before:',(height,width),'after:',(resized_height, resized_width))
        video = transforms.functional.resize(
            video,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

       # print('after:',video.size)
        return video
    else:
        assert isinstance(ele["video"], (list, tuple))
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)
        images = [
            fetch_image({"image": video_element, **process_info}, size_factor=image_factor)
            for video_element in ele["video"]
        ]
        nframes = ceil_by_factor(len(images), FRAME_FACTOR)
        if len(images) < nframes:
            images.extend([images[-1]] * (nframes - len(images)))
        return images


def extract_vision_info(conversations: list[dict] | list[list[dict]]) -> list[dict]:
    vision_infos = []
    if isinstance(conversations[0], dict):
        conversations = [conversations]
    for conversation in conversations:
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if (
                        "image" in ele
                        or "image_url" in ele
                        or "video" in ele
                        or ele["type"] in ("image", "image_url", "video")
                    ):
                        vision_infos.append(ele)
    return vision_infos


def process_vision_info(
    conversations: list[dict] | list[list[dict]],
) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None]:
    vision_infos = extract_vision_info(conversations)
    ## Read images or videos
    #breakpoint()
    image_inputs = []
    video_inputs = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info))
        elif "video" in vision_info:
            video_inputs.append(fetch_video(vision_info))
        else:
            raise ValueError("image, image_url or video should in content.")
    if len(image_inputs) == 0:
        image_inputs = None
    if len(video_inputs) == 0:
        video_inputs = None
    return image_inputs, video_inputs
