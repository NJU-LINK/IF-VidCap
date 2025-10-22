import argparse
import os
import json
import torch
from vllm import LLM, SamplingParams

os.environ["VLLM_ENABLE_FLASH_ATTN"] = "1"  # 开启 Flash Attention

class PixtralEvaluator:
    def __init__(self,
                 model_name: str = "Pixtral-12B",
                 input_dir: str = './annotation/normal',
                 output_dir: str = './response',
                 fps: float = 2.0,
                 resolution: str = "448P",
                 thinking: bool = False):
        """
        初始化 Pixtral 测评器
        """
        self.input_dir = input_dir
        self.video_meta_info_path = './annotation/video_meta_info.json'
        self.prompt_input_path = os.path.join(input_dir, 'prompt.json')
        self.thinking = thinking

        self.model_name = model_name
        self.model_path = os.path.join("./models/Pixtral", model_name)

        self.output_dir = output_dir
        suffix = "_thinking" if thinking else ""
        self.response_output_path = os.path.join(output_dir, f'{self.model_name}{suffix}_response.json')

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.meta_prompt_file = "meta_prompt/test_vlm_meta_prompt.txt"
        self.tensor_parallel_size = torch.cuda.device_count()

        print(f"检测到 {torch.cuda.device_count()} 个GPU")
        print(f"使用模型: {self.model_name}, thinking模式: {self.thinking}")

        # 加载模型
        self.model = LLM(
            model=self.model_path,
            load_format="mistral",  # 强制指定加载格式
            config_format="mistral",
            tokenizer_mode="mistral",
            tensor_parallel_size=self.tensor_parallel_size,
            limit_mm_per_prompt={"video": 120},  # 限制多模态输入
        )

        # 配置视频处理参数
        self.fps = fps  # 动态帧率
        self.resolution = resolution  # 视频分辨率
        self.max_frames = 120 if fps == 2.0 else 60  # 最大帧数

        print(f"视频处理配置: fps={self.fps}, max_frames={self.max_frames}, resolution={self.resolution}")

        # 配置生成参数
        self.sampling_params = SamplingParams(
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            stop_token_ids=[],
        )

    def process_video(self, video_path: str, meta_prompt: str, prompt: str):
        """
        处理单个视频并生成响应
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件未找到: {video_path}")

        # 构造对话
        conversation = [
            {"role": "system", "content": meta_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": self.fps, "max_frames": self.max_frames}},
                    {"type": "text", "text": prompt},
                ]
            },
        ]

        # 添加 thinking 提示
        if self.thinking:
            conversation[0]["content"] = f"{meta_prompt}\n\nPlease think step by step before answering. Start your thinking process with 'Thinking:' and then provide the final answer after 'Answer:'"

        # 准备输入
        prompt = conversation[0]["content"]
        multi_modal_data = {
            "video": {
                "video_path": video_path,
                "fps": self.fps,
                "max_frames": self.max_frames
            }
        }

        # 生成输出
        outputs = self.model.generate(
            [{"prompt": prompt, "multi_modal_data": multi_modal_data}],
            self.sampling_params
        )
        response = outputs[0].outputs[0].text.strip()

        return response

    def read_data_file(self):
        """
        读取输入数据文件
        """
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

    def get_response(self):
        """
        遍历视频并生成响应
        """
        # 加载元指令
        try:
            with open(self.meta_prompt_file, 'r', encoding='utf-8') as f:
                self.meta_prompt = f.read()
            print(f"成功从 '{self.meta_prompt_file}' 加载元指令。")
        except FileNotFoundError:
            print(f"错误: 元指令文件 '{self.meta_prompt_file}' 未找到。请确保文件存在于正确路径。")
            return

        # 读取数据文件
        video_meta_info, prompt_dict, response_dict = self.read_data_file()

        # 遍历视频
        for video_id, prompts in prompt_dict.items():
            if video_id in response_dict:
                print(f"跳过已处理的视频: {video_id}")
                continue

            video_path = os.path.normpath(os.path.join('.', video_meta_info[video_id]['processed_path'])).replace('\\', '/')
            if not os.path.exists(video_path):
                print(f"视频文件未找到: {video_path}")
                continue

            print(f"正在处理视频: {video_id}")
            responses = []
            for prompt in prompts:
                try:
                    response = self.process_video(video_path, self.meta_prompt, prompt['generated_prompt'])
                    responses.append({
                        "field": prompt['field'],
                        "prompt_id": prompt['prompt_id'],
                        "response": response
                    })
                except Exception as e:
                    print(f"处理视频 {video_id} 时出错: {e}")
                    continue

            # 保存响应
            response_dict[video_id] = responses
            with open(self.response_output_path, 'w', encoding='utf-8') as f:
                json.dump(response_dict, f, ensure_ascii=False, indent=4)

        print("测评完成，结果已保存")

def main():
    parser = argparse.ArgumentParser(description="Pixtral 视频理解推理脚本")
    parser.add_argument("--model_name", type=str, default="Pixtral-12B", help="模型名称")
    parser.add_argument("-i", "--input_dir", type=str, default="./annotation/normal", help="输入目录")
    parser.add_argument("-o", "--output_dir", type=str, default="./response", help="输出目录")
    parser.add_argument("--fps", type=float, default=2.0, help="视频帧率")
    parser.add_argument("--resolution", type=str, default="448P", help="视频分辨率")
    parser.add_argument("--thinking", action="store_true", help="启用思考模式")

    args = parser.parse_args()

    evaluator = PixtralEvaluator(
        model_name=args.model_name,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        resolution=args.resolution,
        thinking=args.thinking
    )
    evaluator.get_response()


if __name__ == "__main__":
    main()