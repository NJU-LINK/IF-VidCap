import argparse
import os
import json
import torch
from transformers import AutoModel, AutoProcessor
from keye_vl_utils import process_vision_info
from decord import VideoReader, cpu
from PIL import Image
import numpy as np

class TestModel_KwaiKeye:
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
        
        # 加载Kwai-Keye模型
        model_path = f"./models/Kwai-Keye/{model}"
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        ).eval()

        self.model.to("cuda")
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
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
        self.fps = 2.0  # Kwai-Keye 默认使用2fps
        self.max_frames = 1024  # Kwai-Keye 默认最大帧数
        
    def process_video_with_kwaikeye(self, video_path: str, meta_prompt: str, prompt: str):
        """
        使用 Kwai-Keye 处理视频。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件未找到: {video_path}")
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "fps": self.fps,
                        "max_frames": self.max_frames
                    },
                    {"type": "text", "text": f"{meta_prompt}\n{prompt}"},
                ],
            }
        ]
        
        # 处理think模式，官方给的是在末尾加上think和no_think标识
        # {"type": "text", "text": "Describe this image./no_think"},
        # {"type": "text", "text": "Describe this image./think"}
        # video_messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "video",
        #                 "video": "http://s2-11508.kwimgs.com/kos/nlav11508/MLLM/videos_caption/98312843263.mp4",
        #             },
        #             {"type": "text", "text": "Describe this video./think"},
        #         ],
        #     },
        # ]

        if self.thinking:
            # 添加think指令到文本内容
            messages[0]["content"][1]["text"] = messages[0]["content"][1]["text"] + "/think"
        else:
            messages[0]["content"][1]["text"] = messages[0]["content"][1]["text"] + "/no_think"

        # 准备推理
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **mm_processor_kwargs
        )
        inputs = inputs.to("cuda")

        # 推理：生成输出
        try:
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True if self.temperature > 0 else False,
                top_p=0.001,
                repetition_penalty=1.05
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            response = output_text[0] if output_text else ""
            
            if self.thinking:
                # 提取最终答案
                if '<think>' in response and '</think>' in response:
                    think_start = response.find('<think>') + len('<think>')
                    think_end = response.find('</think>')
                    think_content = response[think_start:think_end].strip()
                    final_answer = response[think_end + len('</think>'):].strip()
                    
                    # 进一步处理，移除<answer>和</answer>标签（如果存在）
                    if '<answer>' in final_answer and '</answer>' in final_answer:
                        answer_start = final_answer.find('<answer>') + len('<answer>')
                        answer_end = final_answer.find('</answer>')
                        final_answer = final_answer[answer_start:answer_end].strip()
                        
                    return final_answer
                else:
                    # 如果没有找到think标签，检查是否有<answer>标签
                    if '<answer>' in response and '</answer>' in response:
                        answer_start = response.find('<answer>') + len('<answer>')
                        answer_end = response.find('</answer>')
                        return response[answer_start:answer_end].strip()
                    else:
                        # 如果没有找到任何标签，返回原始响应
                        return response.strip()
            else:
                # 非思考模式下，直接检查<answer>标签
                if '<answer>' in response and '</answer>' in response:
                    answer_start = response.find('<answer>') + len('<answer>')
                    answer_end = response.find('</answer>')
                    return response[answer_start:answer_end].strip()
                else:
                    # 如果没有找到<answer>标签，返回原始响应
                    return response.strip()
                
        except Exception as e:
            print(f"模型推理过程中发生错误: {str(e)}")
            raise e
    
    def process_video(self, video_path: str, prompt: str):
        """
        调用处理函数。
        """
        return self.process_video_with_kwaikeye(video_path, self.meta_prompt, prompt)
    
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
    parser = argparse.ArgumentParser(description="Kwai-Keye model type selection script")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specify the Kwai-Keye model to use for processing."
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
    test_model = TestModel_KwaiKeye(
        model=args.model, 
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        thinking=args.thinking
    )
    
    # Get the response from the model
    test_model.get_response()

if __name__ == "__main__":
    main()
    
    #思考模式    conda activate shihao && python get_response_Kwai-Keye.py --model Keye-VL-1_5-8B -t   
    #不思考模式    conda activate shihao && python get_response_Kwai-Keye.py --model Keye-VL-1_5-8B