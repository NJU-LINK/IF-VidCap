import argparse
import os
import json
import math
import numpy as np
import torch
import re
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import ARCHunyuanVideoProcessor, ARCHunyuanVideoForConditionalGeneration

def calculate_frame_indices(vlen: int, fps: float, duration: float) -> list:
    frames_per_second = fps

    if duration <= 150:
        interval = 1
        intervals = [
            (int(i * interval * frames_per_second), int((i + 1) * interval * frames_per_second))
            for i in range(math.ceil(duration))
        ]
        sample_fps = 1
    else:
        num_segments = 150
        segment_duration = duration / num_segments
        intervals = [
            (int(i * segment_duration * frames_per_second), int((i + 1) * segment_duration * frames_per_second))
            for i in range(num_segments)
        ]
        sample_fps = 1 / segment_duration

    frame_indices = []
    for start, end in intervals:
        if end > vlen:
            end = vlen
        frame_indices.append((start + end) // 2)

    return frame_indices, sample_fps

def load_video_frames(video_path: str):
    video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=4)
    vlen = len(video_reader)
    input_fps = video_reader.get_avg_fps()
    duration = vlen / input_fps

    frame_indices, sample_fps = calculate_frame_indices(vlen, input_fps, duration)

    return [Image.fromarray(video_reader[idx].asnumpy()) for idx in frame_indices], sample_fps

def build_prompt(question: str, num_frames: int, task: str = "summary"):
    video_prefix = "<image>" * num_frames

    return f"<|startoftext|>{video_prefix}\n{question}\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags, i.e., <think> reasoning process here </think><answer> answer here </answer>.<sep>"


def prepare_inputs(question: str, video_path: str, task: str = "summary"):
    video_frames, sample_fps = load_video_frames(video_path)
    
    # 创建静音音频
    duration = len(video_frames) / sample_fps
    sr = 16000
    audio = np.zeros(int(duration * sr), dtype=np.float32)

    prompt = build_prompt(question, len(video_frames), task)

    video_inputs = {
        "video": video_frames,
        "video_metadata": {
            "fps": sample_fps,
        },
    }

    audio_inputs = {
        "audio": audio,
        "sampling_rate": sr,
        "duration": float(duration),   
    }

    return prompt, video_inputs, audio_inputs

class TestModel:
    def __init__(self,
                 model,
                 input_dir: str = './annotation/normal',
                 output_dir: str = './response',
                 ):
        
        self.input_dir = input_dir
        self.video_meta_info_path = './annotation/video_meta_info.json'
        self.prompt_input_path = os.path.join(input_dir, 'prompt.json')
        self.model_name = model
        
        self.output_dir = output_dir
        self.response_output_path = os.path.join(output_dir, f'{self.model_name}_response.json')
        self.device = torch.device("cuda")

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.meta_prompt_file = "/cpfs01/user/liujiaheng/workspace/shihao/IFEval-Caption/meta_prompt/test_vlm_meta_prompt.txt"
        
        # 加载HunyuanVideo模型
        model_path = f"/cpfs01/user/liujiaheng/workspace/shihao/IFEval-Caption/models/Hunyuan/{self.model_name}"
        self.model = ARCHunyuanVideoForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).eval()

        self.model.to(self.device)
        
        self.processor = ARCHunyuanVideoProcessor.from_pretrained(
            model_path,
            font_path="/cpfs01/user/liujiaheng/workspace/shihao/IFEval-Caption/models/Hunyuan/ARC-Hunyuan-Video-7B/ARIAL.TTF"
        )
        
        print(f"成功加载模型: {model}")
        print(f"检测到 {torch.cuda.device_count()} 个GPU")
            
        self.generation_config = dict(
            max_new_tokens=1024, 
            do_sample=False,
        )
        
    def process_video_with_hunyuan(self, video_path: str, meta_prompt: str, prompt: str, task: str = "summary"):
        """
        使用 HunyuanVideo 处理视频。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件未找到: {video_path}")
        
        try:
            # 准备输入
            prompt_text, video_inputs, audio_inputs = prepare_inputs(prompt, video_path, task)
            
            # 处理输入
            inputs = self.processor(
                text=prompt_text,
                **video_inputs,
                **audio_inputs,
                return_tensors="pt",

            )

            # 确保duration是整数类型
            if 'duration' in inputs:
                inputs['duration'] = inputs['duration'].long()

            inputs = {
                k: (v.to(self.device, dtype=self.model.dtype) if v.dtype.is_floating_point else v.to(self.device))
                for k, v in inputs.items()
            }
            
            # 生成响应
            outputs = self.model.generate(**inputs, **self.generation_config)
            output_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return output_text
        except Exception as e:
            print(f"模型推理过程中发生错误: {str(e)}")
            raise e
    
    def process_video(self, video_path: str, prompt: str, task: str = "summary"):
        """
        调用处理函数。
        """
        return self.process_video_with_hunyuan(video_path, self.meta_prompt, prompt, task)
    
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
        
        with open(self.prompt_input_path, 'r', encoding='utf-8') as f:
            prompt_dict = json.load(f)
        
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

            def extract_answer(text):
                """
                从给定文本中提取 <answer> 标签内的内容。
                """
                pattern = r"<answer>(.*?)</answer>"
                match = re.search(pattern, text, re.DOTALL)  # DOTALL 让 . 匹配换行符
                if match:
                    return match.group(1).strip()  # 去掉首尾空白字符
                return None
            
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
                        response = self.process_video(video_path, prompt, prompt_info['field'])
                        response = extract_answer(response)

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
                        print(f"✓ 已保存响应 (field: {prompt_info['field']}, prompt_id: {prompt_info['prompt_id']})")
                    
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
    parser = argparse.ArgumentParser(description="HunyuanVideo model processing script")
    parser.add_argument(
        "--model",
        type=str,
        default="ARC-Hunyuan-Video-7B",
        help="Specify the model name for output file naming."
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
    
    args = parser.parse_args()

    
    # Initialize the TestModel class with the selected model
    test_model = TestModel(
        model=args.model, 
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    
    # Get the response from the model
    test_model.get_response()

if __name__ == "__main__":
    main()

# 运行命令   "conda activate hunyuan_video && python get_response_hunyuan.py --model ARC-Hunyuan-Video-7B"