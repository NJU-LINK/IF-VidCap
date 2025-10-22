import argparse
import os
import json
import math
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from decord import VideoReader, cpu

try:
    from scipy.spatial import cKDTree
except ImportError:
    print("警告: 缺少 scipy 库，请使用 'pip install scipy' 安装")
    raise

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
        
        # 生成配置 - 根据思考模式调整参数
        if self.thinking:
            self.temperature = 0.6
            self.max_new_tokens = 8192  # 思考模式需要更多token
        else:
            self.temperature = 0.1
            self.max_new_tokens = 2048
        
        # 视频处理配置 - 基于MiniCPM-V4.5的特性
        self.MAX_NUM_FRAMES = 180  # 最大帧数
        self.MAX_NUM_PACKING = 3   # 3D-Resampler的最大打包数
        self.TIME_SCALE = 0.1
        
        # 根据60s视频和能力设置采样率
        # MiniCPM-V4.5支持高FPS视频理解，可以处理更多帧
        # 根据要求：模型能处理180+帧，按2fps采样（60s*2fps=120帧 < 180帧）
        self.fps = 2.0
        
        # 分辨率配置 - 使用模型默认的448x448
        self.resolution = 448
        
        # 加载MiniCPM-V模型
        model_path = f"./models/MiniCPM/{model}"
        
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径未找到: {model_path}")
        
        print(f"正在加载模型: {model_path}")
        print(f"启用思考模式: {self.thinking}")
        
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation='flash_attention_2',  # 开启 flash-attn 以免 OOM
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True
        )
        
        print(f"成功加载模型: {model}")
        print(f"模型配置:")
        print(f"- 思考模式: {self.thinking}")
        print(f"- 温度参数: {self.temperature}")
        print(f"- 最大token数: {self.max_new_tokens}")
        print(f"- 视频采样率: {self.fps} FPS")
        print(f"- 分辨率: {self.resolution}x{self.resolution}")
        print(f"- 最大视频帧数: {self.MAX_NUM_FRAMES}")
        print(f"- 3D-Resampler最大打包数: {self.MAX_NUM_PACKING}")
        print(f"检测到 {torch.cuda.device_count()} 个GPU")
        
        # 显示GPU内存信息
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
    def map_to_nearest_scale(self, values, scale):
        """映射到最近的时间尺度"""
        tree = cKDTree(np.asarray(scale)[:, None])
        _, indices = tree.query(np.asarray(values)[:, None])
        return np.asarray(scale)[indices]

    def group_array(self, arr, size):
        """将数组分组"""
        return [arr[i:i+size] for i in range(0, len(arr), size)]

    def uniform_sample(self, l, n):
        """均匀采样"""
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    def encode_video(self, video_path, choose_fps=None, force_packing=None):
        """
        编码视频，使用MiniCPM-V4.5的3D-Resampler技术
        """
        if choose_fps is None:
            choose_fps = self.fps
            
        vr = VideoReader(video_path, ctx=cpu(0))
        fps = vr.get_avg_fps()
        video_duration = len(vr) / fps
        
        # 根据视频时长和采样率决定打包策略
        if choose_fps * int(video_duration) <= self.MAX_NUM_FRAMES:
            packing_nums = 1
            choose_frames = round(min(choose_fps, round(fps)) * min(self.MAX_NUM_FRAMES, video_duration))
        else:
            packing_nums = math.ceil(video_duration * choose_fps / self.MAX_NUM_FRAMES)
            if packing_nums <= self.MAX_NUM_PACKING:
                choose_frames = round(video_duration * choose_fps)
            else:
                choose_frames = round(self.MAX_NUM_FRAMES * self.MAX_NUM_PACKING)
                packing_nums = self.MAX_NUM_PACKING

        frame_idx = [i for i in range(0, len(vr))]      
        frame_idx = np.array(self.uniform_sample(frame_idx, choose_frames))

        if force_packing:
            packing_nums = min(force_packing, self.MAX_NUM_PACKING)
        
        print(f'视频信息: 时长={video_duration:.2f}s, 原始FPS={fps:.2f}, 采样FPS={choose_fps:.2f}')
        print(f'采样结果: 帧数={len(frame_idx)}, 3D-Resampler打包数={packing_nums}')
        
        frames = vr.get_batch(frame_idx).asnumpy()

        frame_idx_ts = frame_idx / fps
        scale = np.arange(0, video_duration, self.TIME_SCALE)

        frame_ts_id = self.map_to_nearest_scale(frame_idx_ts, scale) / self.TIME_SCALE
        frame_ts_id = frame_ts_id.astype(np.int32)

        assert len(frames) == len(frame_ts_id)

        frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
        
        # 根据打包数对时间戳ID进行分组
        if packing_nums > 1:
            frames_per_group = len(frame_ts_id) // packing_nums
            if frames_per_group > 0:
                frame_ts_id_group = self.group_array(frame_ts_id, frames_per_group)
            else:
                frame_ts_id_group = [frame_ts_id.tolist()]
        else:
            frame_ts_id_group = [frame_ts_id.tolist()]
        
        return frames, frame_ts_id_group
    
    def process_video_with_minicpm(self, video_path: str, meta_prompt: str, prompt: str):
        """
        使用 MiniCPM-V 4.5 处理视频。
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件未找到: {video_path}")
        
        # 编码视频
        frames, frame_ts_id_group = self.encode_video(video_path, self.fps)
        
        # 构建消息 - 不在提示词中添加思考指令
        full_prompt = f"{meta_prompt}\n{prompt}"
        
        msgs = [
            {'role': 'user', 'content': frames + [full_prompt]}
        ]
        
        try:
            # 使用模型进行推理 - 通过enable_thinking参数控制思考模式
            response = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                use_image_id=False,  # 视频推理时确保设为False
                max_slice_nums=1,    # 每帧的最大切片数
                temporal_ids=frame_ts_id_group,  # 时序ID用于3D-Resampler
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=0.001,
                repetition_penalty=1.05,
                enable_thinking=self.thinking  # 使用enable_thinking参数控制思考功能
            )
            
            # 直接返回模型响应，无需手动处理思考标签
            return response.strip()
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"GPU内存不足: {str(e)}")
            print("建议: 1) 减少视频采样率 2) 使用更少的GPU 3) 清理GPU缓存")
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise e
        except Exception as e:
            print(f"模型推理过程中发生错误: {str(e)}")
            print(f"错误类型: {type(e).__name__}")
            raise e
    
    def process_video(self, video_path: str, prompt: str):
        """
        调用处理函数。
        """
        return self.process_video_with_minicpm(video_path, self.meta_prompt, prompt)
    
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
    parser = argparse.ArgumentParser(description="MiniCPM-V model evaluation script")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specify the MiniCPM-V model to use for processing."
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
        help="Enable thinking mode for complex reasoning tasks."
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
    # 基本使用
    python get_response_minicpm.py --model MiniCPM-V-4_5
    
    # 使用思考模式
    python get_response_minicpm.py --model MiniCPM-V-4_5 --thinking
    
    # 使用自定义输入输出目录
    python get_response_minicpm.py --model MiniCPM-V-4_5 -i ./annotation/test -o ./response_test
    
    # 完整示例命令
    python get_response_minicpm.py --model MiniCPM-V-4_5 -t -o ./response
    '''
