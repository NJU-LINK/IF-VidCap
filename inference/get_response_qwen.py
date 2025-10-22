import argparse
import os
import json
import glob
import argparse
from pathlib import Path
from natsort import natsorted
import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

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
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.meta_prompt_file = "meta_prompt/test_vlm_meta_prompt.txt"
        
        self.tensor_parallel_size = torch.cuda.device_count()


        # 使用vLLM加载模型
        self.model = LLM(
            model=f"./models/Qwen/{model}",
            tensor_parallel_size=self.tensor_parallel_size,
            limit_mm_per_prompt={"image": 1024, "video": 10}  # 设置多模态限制
        )
        
        print(f"检测到 {torch.cuda.device_count()} 个GPU")
        print(f"使用 {self.tensor_parallel_size} 个GPU进行张量并行")

        # 设置采样参数
        self.sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=51200,
            stop_token_ids=[],
        )
        
        self.processor = AutoProcessor.from_pretrained(f"./models/Qwen/{model}", use_fast=True, trust_remote_code=True)
        
        # 配置
        self.fps = 2.0  # 视频处理的帧率
        
        
    
    def process_video_with_qwen(self, video_path: str, meta_prompt: str, prompt: str):
        """
        使用 vLLM 处理视频。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件未找到: {video_path}")
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": meta_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "video", 
                        "video": video_path,
                        "total_pixels": 20480 * 28 * 28, 
                        "min_pixels": 64 * 28 * 28 ,
                        "fps": self.fps,
                    },
                ],
            },
        ]
        
        # 使用processor处理消息
        prompt_text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 处理视觉信息
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, 
            return_video_kwargs=True
        )
        print(json.dumps(video_kwargs, indent=2, ensure_ascii=False))
        # 准备多模态数据
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        
        # 准备vLLM输入
        llm_inputs = {
            "prompt": prompt_text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,  # FPS等参数在这里传递
        }
        
        # 使用vLLM生成响应
        outputs = self.model.generate([llm_inputs], self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        return generated_text
    
    def process_video(self, video_path: str, prompt: str):
        """
        调用处理函数。
        """
        return self.process_video_with_qwen(video_path, self.meta_prompt, prompt)
    
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
    parser = argparse.ArgumentParser(description="Model type selection script")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specify the model to use for processing."
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
        output_dir=args.output_dir
    )
    
    # Get the response from the model
    test_model.get_response()

if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()
    
    '''
    python get_response_qwen.py --model Qwen2.5-VL-7B-Instruct
    python get_response_qwen.py --model Qwen2.5-VL-32B-Instruct
    python get_response_qwen.py --model Qwen2.5-VL-72B-Instruct
    
    # 使用自定义输入输出目录
    python get_response_qwen.py --model Qwen2.5-VL-7B-Instruct -i ./annotation/test
    '''

