import argparse
import os
import json
import torch
from videollava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from videollava.conversation import conv_templates, SeparatorStyle
from videollava.model.builder import load_pretrained_model
from videollava.utils import disable_torch_init
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

os.environ["CUDA_VISIBLE_DEVICES"] = "0"                  #
torch.cuda.set_device(0)         


def main():
    parser = argparse.ArgumentParser(description="InternVL model type selection script")
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
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    
    # Get the response from the model
    test_model.get_response()

class TestModel:
    def __init__(self,
                 input_dir: str = './annotation/normal',
                 output_dir: str = './response',
                 ):
        disable_torch_init()
        self.input_dir = input_dir
        self.video_meta_info_path = './annotation/video_meta_info.json'
        self.prompt_input_path = os.path.join(input_dir, 'prompt.json')
        
        self.model_name = "Video-LLaVA"
        self.output_dir = output_dir
        

        self.response_output_path = os.path.join(output_dir, f'{self.model_name}_response.json')

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.meta_prompt_file = "meta_prompt/test_vlm_meta_prompt.txt"
        
        # Load meta prompt
        with open(self.meta_prompt_file, 'r', encoding='utf-8') as f:
            self.meta_prompt = f.read()
        
        model_path = '/cpfs01/user/liujiaheng/workspace/shihao/IFEval-Caption/models/Llava/Video-LLaVA'
        cache_dir = ''
        device = 'cuda:0'
        load_4bit, load_8bit = False, False
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device, cache_dir=cache_dir)
        self.model = self.model.to(device)                  #
        self.video_processor = self.processor['video']
        self.conv_mode = "llava_v1"
        self.roles = conv_templates[self.conv_mode].copy().roles

        if next(self.model.parameters()).device.type == 'cuda':
            print("在GPU上运行")
        else:
            print("不好 是CPU")
        # 因为LLaVA使用不同的对话模板
        
        print(f"成功加载模型: {model_path}")
        print(f"检测到 {torch.cuda.device_count()} 个GPU")

        

    def process_video(self, video_path: str, prompt: str):
        """
        Process video using the model API
        """
        # Build full prompt with meta instructions
        conv = conv_templates[self.conv_mode].copy()
        full_prompt = f"{self.meta_prompt}\n{prompt}"
        
        # Call the model API
        ###########################################################################TODO

        video_tensor = self.video_processor(video_path, return_tensors='pt')['pixel_values']
        if type(video_tensor) is list:
            tensor = [video_path.to(self.model.device, dtype=torch.float16) for video_path in video_tensor]
        else:
            tensor = video_tensor.to(self.model.device, dtype=torch.float16)

        inp = full_prompt
        inp = ' '.join([DEFAULT_IMAGE_TOKEN] * self.model.get_video_tower().config.num_frames) + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        real_prompt = conv.get_prompt()

        #input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        #修改如下
        input_ids = tokenizer_image_token(real_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)         #
        input_ids = input_ids.to(self.model.device)           #


        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=tensor,
                do_sample=True,
                temperature=0.1,
                max_new_tokens=1024,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        response = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        
        # Process response if in thinking mode
        
        return response

    def read_data_file(self):
        """Read input data files"""
        # Check input files
        if not os.path.exists(self.video_meta_info_path):
            raise FileNotFoundError(f"Video meta info file not found: {self.video_meta_info_path}")
        if not os.path.exists(self.prompt_input_path):
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_input_path}")
        
        # Read files
        with open(self.video_meta_info_path, 'r', encoding='utf-8') as f:
            video_meta_info = json.load(f)
        
        with open(self.prompt_input_path, 'r', encoding='utf-8') as f:
            prompt_dict = json.load(f)
        
        # Check existing output file
        if os.path.exists(self.response_output_path):
            try:
                with open(self.response_output_path, 'r', encoding='utf-8') as f:
                    response_dict = json.load(f)
            except:
                response_dict = {}
        else:
            response_dict = {}
            
        return video_meta_info, prompt_dict, response_dict

    def _save_sorted_dict(self, data_dict, file_path):
        """Save sorted dictionary to file"""
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

if __name__ == "__main__":
    main()