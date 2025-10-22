import argparse
import os
import json
import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from modelscope import MllamaForConditionalGeneration, AutoProcessor


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

class TestModel:
    def __init__(self,
                 model,
                 input_dir: str = './annotation/normal',
                 output_dir: str = './response',
                 thinking: bool = False
                 ):
        
        self.input_dir = input_dir
        self.video_meta_info_path = './annotation/video_meta_info.json'
        self.prompt_input_path = os.path.join(input_dir, 'prompt.json')
        
        self.model_name = model
        
        self.output_dir = output_dir
        self.thinking = thinking
        if self.thinking:
            self.response_output_path = os.path.join(output_dir, f'{self.model_name}_thinking_response.json')
        else:
            self.response_output_path = os.path.join(output_dir, f'{self.model_name}_nothinking_response.json')
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.meta_prompt_file = "meta_prompt/test_vlm_meta_prompt.txt"
        
        # 加载Llama-3.2-Vision模型
        # model_path = "LLM-Research/Llama-3.2-11B-Vision-Instruct"
        model_path = f"./models/Llama/LLM-Research/{model}"
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        print(f"成功加载模型: {model}")
        print(f"检测到 {torch.cuda.device_count()} 个GPU")
        
        # 生成配置
        if self.thinking:
            self.temperature = 0.6
            self.max_new_tokens = 8192
        else:
            self.temperature = 0.1
            self.max_new_tokens = 2048
        
        # 视频处理配置
        self.input_size = 448  # 分辨率处理为448*448
        self.fps = 1.0  # 按1fps采帧
        self.max_num = 1  # 每帧的最大块数
        
    def process_video_with_Llama(self, video_path: str, meta_prompt: str, prompt: str):
        """
        使用 Llama-3.2-Vision 处理视频。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件未找到: {video_path}")
        
        # 读取视频并获取帧
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        video_fps = float(vr.get_avg_fps())
        total_frames = len(vr) - 1
        
        # 根据1fps计算采样帧数
        duration = total_frames / video_fps
        num_segments = max(1, int(duration * self.fps))
        
        # 加载视频帧
        pixel_values, num_patches_list = load_video(
            video_path, 
            bound=None, 
            input_size=self.input_size, 
            max_num=self.max_num, 
            num_segments=num_segments
        )
        
        # 构建消息格式
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": f"{meta_prompt}\n{prompt}\n"},
            ]}
        ]
        
        # 添加图像帧到消息中
        for i in range(len(num_patches_list)):
            # 获取对应的图像张量
            start_idx = sum(num_patches_list[:i])
            end_idx = start_idx + num_patches_list[i]
            frame_tensor = pixel_values[start_idx:end_idx]
            
            # 将张量转换回PIL图像
            frame_images = []
            for img_tensor in frame_tensor:
                # 反归一化
                img_tensor = img_tensor * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
                img_tensor = img_tensor.clamp(0, 1)
                
                # 转换为PIL图像
                img = T.ToPILImage()(img_tensor)
                frame_images.append(img)
            
            # 如果有多张图像，使用第一张作为代表
            if frame_images:
                messages[0]["content"].append({"type": "image"})
                # 这里我们只使用第一张图像，实际可以根据需要处理多张图像
        
        # 应用聊天模板
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # 使用第一帧图像作为输入（简化处理）
        first_frame_idx = 0
        start_idx = sum(num_patches_list[:first_frame_idx])
        end_idx = start_idx + num_patches_list[first_frame_idx]
        first_frame_tensor = pixel_values[start_idx:end_idx]
        
        if len(first_frame_tensor) > 0:
            # 将第一张图像转换为PIL格式
            first_img_tensor = first_frame_tensor[0]
            first_img_tensor = first_img_tensor * torch.tensor(IMAGENET_STD).view(3, 1, 1) + torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            first_img_tensor = first_img_tensor.clamp(0, 1)
            first_img = T.ToPILImage()(first_img_tensor)
            
            # 准备输入
            inputs = self.processor(
                first_img,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)
        else:
            # 如果没有图像，只使用文本
            inputs = self.processor(
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)
        
        # 使用模型进行推理
        try:
            output = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.001,
                repetition_penalty=1.05
            )
            
            response = self.processor.decode(output[0], skip_special_tokens=True)
            
            # 提取生成的文本（去掉输入部分）
            if input_text in response:
                response = response.replace(input_text, "").strip()
            
            if self.thinking:
                # 提取最终答案（如果包含思考过程）
                if '</think>' in response:
                    response = response.split('</think>', 1)[-1].strip()
                else:
                    response = response.strip()
                    
            return response
        except Exception as e:
            print(f"模型推理过程中发生错误: {str(e)}")
            raise e
    
    def process_video(self, video_path: str, prompt: str):
        """
        调用处理函数。
        """
        return self.process_video_with_Llama(video_path, self.meta_prompt, prompt)
    
    def read_data_file(self):
        """读取输入数据文件"""
        # 检查输入文件
        if not os.path.exists(self.video_meta_info_path):
            raise FileNotFoundError(f"视频元信息文件未找到: {self.video_meta_info_path}")
        if not os.path.exists(self.prompt_input_path):
            raise FileNotFoundError(f"Prompt文件未找到: {self.prompt_input_path}")
        
        # 读取文件
        with open(self.video_meta_info_path, 'r', encoding='utf-8') as f:
            video_meta_info = json.load(f)
        print(f"成功从 '{self.video_meta_info_path}' 加载视频元信息")
        
        with open(self.prompt_input_path, 'r', encoding='utf-8') as f:
            prompt_dict = json.load(f)
        print(f"成功从 '{self.prompt_input_path}' 加载prompt数据")
        
        # 检查已有的输出文件
        if os.path.exists(self.response_output_path):
            try:
                with open(self.response_output_path, 'r', encoding='utf-8') as f:
                    response_dict = json.load(f)
                print(f"找到已有response文件 '{self.response_output_path}'，将从断点继续处理。")
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"无法读取response文件 '{self.response_output_path}'，将重新开始处理。")
                response_dict = {}
        else:
            print("未找到已有response文件，将重新开始处理。")
            response_dict = {}
            
        return video_meta_info, prompt_dict, response_dict
    
    def _save_sorted_dict(self, data_dict, file_path):
        """保存排序后的字典到文件"""
        def get_video_sort_key(video_id):
            parts = video_id.split('_')
            video_part = parts[0]
            video_num = int(parts[-1]) if parts[-1].isdigit() else 0
            part_order = {'clip': 0, 'short': 1, 'long': 2}
            part_idx = part_order.get(video_part, 999)
            return (part_idx, video_num)
        
        sorted_dict = dict(sorted(data_dict.items(), key=lambda x: get_video_sort_key(x[0])))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_dict, f, ensure_ascii=False, indent=4)
    
    def get_response(self):
        # 1. 加载核心指令 (Meta-Prompt)
        try:
            with open(self.meta_prompt_file, 'r', encoding='utf-8') as f:
                self.meta_prompt = f.read()
            print(f"成功从 '{self.meta_prompt_file}' 加载元指令。")
        except FileNotFoundError:
            print(f"错误: 元指令文件 '{self.meta_prompt_file}' 未找到。请确保文件存在于正确路径。")
            return
        
        # 2. 读取数据文件
        video_meta_info, prompt_dict, response_dict = self.read_data_file()
        
        # 3. 验证数据一致性
        video_ids = set(prompt_dict.keys())
        print(f"找到 {len(video_ids)} 个可处理的视频")
        
        # 统计已处理和待处理的视频
        processed_count = 0
        skipped_count = 0
        total_videos = len(video_ids)
        
        # 统计处理状态
        fully_completed = 0
        need_response = 0
        
        for video_id in video_ids:
            num_prompts = len(prompt_dict[video_id])
            num_responses = len(response_dict.get(video_id, []))
            
            if num_responses >= num_prompts:
                fully_completed += 1
            else:
                need_response += 1
        
        print(f"处理状态统计:")
        print(f"- 完全完成: {fully_completed}")
        print(f"- 需要生成response: {need_response}")
        
        # 4. 遍历每个视频并生成response
        # 定义排序函数
        def get_video_sort_key(video_id):
            parts = video_id.split('_')
            video_part = parts[0]
            video_num = int(parts[-1]) if parts[-1].isdigit() else 0
            part_order = {'clip': 0, 'short': 1, 'long': 2}
            part_idx = part_order.get(video_part, 999)
            return (part_idx, video_num)
        
        sorted_video_ids = sorted(video_ids, key=get_video_sort_key)
        
        for video_idx, video_id in enumerate(sorted_video_ids):
            if video_id not in video_meta_info:
                print(f"警告: 视频 {video_id} 在元信息中未找到，跳过")
                continue
            
            # 获取视频信息
            video_info = video_meta_info[video_id]
            video_prompts = prompt_dict[video_id]
            existing_responses = response_dict.get(video_id, [])
            
            # 检查是否已经完全处理过这个视频
            if len(existing_responses) >= len(video_prompts):
                print(f"跳过已完全处理的视频: {video_id}")
                skipped_count += 1
                continue
            
            # 构建视频路径
            video_path = os.path.normpath(
                os.path.join('.', video_info['processed_path'])
            ).replace('\\', '/')
            
            if not os.path.exists(video_path):
                print(f"错误: 视频文件 '{video_path}' 未找到，跳过")
                continue
                
            print(f"\n--- 正在处理视频: {video_id} ({video_path}) [{video_idx + 1}/{total_videos}] ---")
            print(f"  需要处理 {len(video_prompts) - len(existing_responses)} 个新prompt")
            
            # 重试机制
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    # 处理每个prompt
                    new_responses = []
                    for i, prompt_info in enumerate(video_prompts):
                        # 检查是否已有响应
                        if i < len(existing_responses):
                            print(f"  跳过已处理的prompt {i+1}/{len(video_prompts)} (field: {prompt_info['field']}, prompt_id: {prompt_info['prompt_id']})")
                            continue
                        
                        print(f"  处理第 {i+1}/{len(video_prompts)} 个提示...")
                        prompt = prompt_info['generated_prompt']
                        response = self.process_video(video_path, prompt)
                        
                        response_item = {
                            'field': prompt_info['field'],
                            'prompt_id': prompt_info['prompt_id'],
                            'response': response
                        }
                        new_responses.append(response_item)
                        
                        # 立即保存新响应（断点保存）
                        if video_id not in response_dict:
                            response_dict[video_id] = []
                        response_dict[video_id].append(response_item)
                        
                        # 保存到文件
                        self._save_sorted_dict(response_dict, self.response_output_path)
                        print(f"    ✓ 已保存响应 (field: {prompt_info['field']}, prompt_id: {prompt_info['prompt_id']})")
                    
                    if new_responses:
                        processed_count += 1
                        print(f"✓ 视频 {video_id} 处理完成，生成了 {len(new_responses)} 个新响应")
                    success = True
                    
                except Exception as e:
                    print(f"✗ 处理视频 '{video_id}' 时发生错误: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"正在重试 {retry_count}/{max_retries}...")
                    
            if not success:
                print(f"视频 '{video_id}' 处理失败，已达到最大重试次数。")
        
        print(f"\n--- 所有视频处理完毕 ---")
        print(f"总计视频数: {total_videos}")
        print(f"新处理: {processed_count}")
        print(f"跳过(已存在): {skipped_count}")
        print(f"任务完成！结果已保存到 '{self.response_output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Llama model type selection script")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specify the Llama model to use for processing."
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        default="./annotation/normal",
        help="Input directory containing prompt.json"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="./response",
        help="Output directory for response files"
    )
    parser.add_argument(
        "-t", "--thinking",
        action="store_true",
        help="Enable thinking mode."
    )
    
    args = parser.parse_args()
    
    # Initialize the TestModel class with the selected model
    test_model = TestModel(
        model=args.model, 
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        thinking=args.thinking
    )
    
    # Get the response from the model
    test_model.get_response()

if __name__ == "__main__":
    main()
    
    '''
    conda activate Llama
    python get_response_Llama.py --model Llama-3.2-11B-Vision-Instruct
    python get_response_Llama.py --model Llama-3.2-90B-Vision-Instruct
    
    # 使用自定义输入输出目录
    python get_response_Llama.py --model Llama-3.2-11B-Vision-Instruct -i ./annotation/test
    '''