# check_only_pipeline.py
import os
import json
import glob
import traceback
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime
import argparse
import copy
from tqdm import tqdm
from utils import openai_client, clean_json_response, combined_retry
from utils import AutoRuleChecker


class ProgressManager:
    """统一的进度管理器 - 简化多线程进度跟踪"""
    
    def __init__(self, total_tasks: int, desc: str = "处理进度"):
        self.total_tasks = total_tasks
        self.desc = desc
        self.lock = Lock()
        self.stats = {'completed': 0, 'failed': 0, 'skipped': 0}
        self.progress_bar = None
        self._setup_progress_bar()
    
    def _setup_progress_bar(self):
        """设置进度条"""
        self.progress_bar = tqdm(
            total=self.total_tasks, 
            desc=self.desc,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        # 初始化显示信息
        self.progress_bar.set_postfix({
            '成功': 0,
            '失败': 0,
            '跳过': 0
        })
    
    def update(self, status: str, item_id: str = ""):
        """
        更新进度
        
        Args:
            status: 'completed', 'failed', 'skipped'
            item_id: 当前处理项目的ID（可选）
        """
        with self.lock:
            if status in self.stats:
                self.stats[status] += 1
            
            # 更新进度条显示信息
            self.progress_bar.set_postfix({
                '成功': self.stats['completed'],
                '失败': self.stats['failed'],
                '跳过': self.stats['skipped']
            })
            self.progress_bar.update(1)
    
    def get_stats(self) -> Dict[str, int]:
        """获取当前统计信息"""
        with self.lock:
            return self.stats.copy()
    
    def close(self):
        """关闭进度条"""
        if self.progress_bar:
            self.progress_bar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VideoLogger:
    """用于捕获和存储单个视频处理过程的日志"""
    def __init__(self, video_id: str, log_dir: str):
        self.video_id = video_id
        self.log_dir = log_dir
        self.start_time = datetime.now()
        
        # 创建单独的日志文件
        self.log_file_path = os.path.join(log_dir, f"{video_id}.log")
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8')
        
        # 写入开始信息
        self.write(f"开始处理视频: {video_id}")
        self.write(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.write("-" * 60)
        
    def write(self, message: str):
        """写入日志消息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}\n"
        self.log_file.write(log_line)
        self.log_file.flush()  # 实时写入
        
    def close(self):
        """关闭日志文件"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        self.write("-" * 60)
        self.write(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.write(f"总耗时: {duration:.2f} 秒")
        self.log_file.close()
        
    def get_log_path(self) -> str:
        """返回日志文件路径"""
        return self.log_file_path


class LogManager:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.master_log_path = os.path.join(log_dir, "master.log")
        self.lock = Lock()
        self.completed_logs = {}  # 存储所有已完成的日志路径
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化主日志文件
        self._init_master_log()
    
    def _init_master_log(self):
        """初始化主日志文件"""
        with open(self.master_log_path, 'w', encoding='utf-8') as f:
            f.write(f"Check-Only Pipeline 主日志文件\n")
            f.write(f"创建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def add_completed_log(self, video_id: str, log_file_path: str):
        """添加完成的日志并重新生成主日志"""
        with self.lock:
            # 添加到已完成列表
            self.completed_logs[video_id] = log_file_path
            
            # 重新生成主日志
            self._regenerate_master_log()
    
    def _regenerate_master_log(self):
        """重新生成有序的主日志文件"""
        # 定义排序函数
        def get_video_sort_key(video_id):
            parts = video_id.split('_')
            video_part = parts[0]
            video_num = int(parts[-1]) if parts[-1].isdigit() else 0
            part_order = {'clip': 0, 'short': 1, 'long': 2}
            part_idx = part_order.get(video_part, 999)
            return (part_idx, video_num)
        
        # 排序video_ids
        sorted_video_ids = sorted(self.completed_logs.keys(), key=get_video_sort_key)
        
        # 重新写入主日志
        self._init_master_log()  # 重新初始化
        
        with open(self.master_log_path, 'a', encoding='utf-8') as master_f:
            for video_id in sorted_video_ids:
                log_file_path = self.completed_logs[video_id]
                try:
                    with open(log_file_path, 'r', encoding='utf-8') as f:
                        video_log_content = f.read()
                    
                    master_f.write(f"\n{'='*80}\n")
                    master_f.write(f"视频 {video_id} 的处理日志:\n")
                    master_f.write(f"{'='*80}\n")
                    master_f.write(video_log_content)
                    master_f.write(f"\n{'='*80}\n\n")
                    
                except Exception as e:
                    master_f.write(f"\n错误: 无法读取视频 {video_id} 的日志文件: {str(e)}\n\n")


class VideoProcessor:
    """处理单个视频的类 - 只进行check"""
    def __init__(self, video_id: str, existing_data: Dict, pipeline: 'CheckOnlyPipeline', logger: VideoLogger):
        self.video_id = video_id
        self.existing_data = existing_data
        self.pipeline = pipeline
        self.logger = logger
        self.result = {
            'judge': []
        }
        
    def log(self, message: str):
        """记录日志"""
        self.logger.write(message)
        
    def process(self) -> Tuple[str, Dict]:
        """处理视频并返回结果"""
        try:
            # 获取已有数据
            prompts = self.existing_data['prompts']
            responses = self.existing_data['responses']
            checklists = self.existing_data['checklists']
            
            if len(prompts) != len(responses) or len(prompts) != len(checklists):
                self.log(f"错误: 数据长度不匹配 - prompts: {len(prompts)}, responses: {len(responses)}, checklists: {len(checklists)}")
                return self.video_id, None
            
            self.log(f"开始处理视频 {self.video_id}，共有 {len(prompts)} 个测试用例")
            
            # 处理每个测试用例
            for idx, (prompt_data, response_data, checklist_data) in enumerate(zip(prompts, responses, checklists)):
                prompt = prompt_data['generated_prompt']
                field = prompt_data['field']
                prompt_id = prompt_data['prompt_id']
                self.log(f"处理第 {idx+1}/{len(prompts)} 个测试用例 - field: {field}, prompt_id: {prompt_id}")
                
                # 获取必要的数据
                response = response_data['response']
                checklist = checklist_data['checklist']
                
                # 生成check result
                check_result = self._generate_check_result(
                    prompt, response, checklist, field
                )
                
                # 保存结果
                self.result['judge'].append({
                    "field": field,
                    "prompt_id": prompt_id,
                    "check_result": check_result
                })
                
                self.log(f"✅ 完成第 {idx+1} 个测试用例的检查")
            
            self.log(f"✅ 视频 {self.video_id} 处理完成，共完成 {len(self.result['judge'])} 个检查")
            return self.video_id, self.result
            
        except Exception as e:
            self.log(f"处理视频 {self.video_id} 时发生错误: {str(e)}")
            import traceback
            self.log(f"错误详情: {traceback.format_exc()}")
            return self.video_id, None
        
    def _generate_check_result(self, prompt: str, response: str,  
                              checklist: Dict, field: str) -> Dict:
        """生成检查结果"""
        
        check_result = copy.deepcopy(checklist)
        
        max_inline_retry = 5
        inline_retry = 0
        if 'ruled_based_check' in checklist:
            for idx, checkitem in enumerate(checklist['ruled_based_check']):
                # 使用 judge_llm 生成 rule_based check result
                retry_response=None
                while inline_retry < max_inline_retry:
                    rule_content = self.pipeline.get_rule_based_checkresult_with_llm(
                        response[:self.pipeline.max_token], checkitem, retry_response
                    )
                    if checkitem['constraint_id'] != 'count':
                        if all(item in response for item in rule_content['content']):
                            check_result['ruled_based_check'][idx]['parameters']['content'] = rule_content['content']
                            break
                        else:
                            retry_response = rule_content['content']
                            inline_retry += 1
                            check_result['ruled_based_check'][idx]['parameters']['content'] = ['<error content holder>'*100]
                            self.log(f"❌ field {field} 的rule based check生成错误, content不在response中, 进行重试")
                    else:
                        break
                check_result['ruled_based_check'][idx]['parameters']['content'] = rule_content['content']
                
        inline_retry = 0
        if 'open_ended_check' in checklist:
            for idx, check_content in enumerate(checklist['open_ended_check']):
                for checkitem_idx, checkitem in enumerate(check_content['check_items']):
                    question = checkitem['question']
                    options = checkitem['options']
                    # 使用 judge_llm 生成 open_ended check result
                    while inline_retry < max_inline_retry:
                        answer_response = self.pipeline.get_open_ended_checkresult_with_llm(
                            prompt, response[:self.pipeline.max_token], question, options
                        )
                        current_check_item = check_result['open_ended_check'][idx]['check_items'][checkitem_idx]
                        try:
                            # 填写检查结果
                            current_check_item['answer'] = answer_response['answer'][0] if answer_response['answer'][0] in ['A', 'B', 'C', 'D'] else answer_response['answer']
                            break
                        except Exception as e:
                            self.log(f"❌ field {field} 的open ended check生成错误: {e}")
                            inline_retry += 1
                            continue

                    current_check_item['result_explanation'] = answer_response['result_explanation']
                    current_check_item['result_confidence'] = answer_response['result_confidence']
                    inline_retry = 0
                
        self.log(f"✅ field {field} 的check生成正确")

        # 进行规则检查
        check_result, status = self.pipeline.auto_checker.check_all_rules(check_result)
        if not status:
            self.log(f"❌ field {field} 的check结果不符合规则")
            raise ValueError(f"field {field} 的check结果不符合规则: {check_result}")
        # 进行 open 检查
        if 'open_ended_check' in check_result:
            for check_group in check_result['open_ended_check']: 
                for check_item in check_group['check_items']:
                    # 判断模型答案与正确答案是否一致，填写result属性
                    check_item['result'] = check_item['answer'] == check_item['correct_answer']
            
        self.log(f"✅ 完成field {field} 的check结果生成")
        
        return check_result
    
class CheckOnlyPipeline:
    def __init__(self, meta_input_dir: str = './annotation/normal',
                 response_input_dir: str = './response',
                 model_name: str = 'Qwen2.5-VL-7B-Instruct', 
                 output_dir: str = './check_result',
                 max_workers: int = 10):
        
        self.max_workers = max_workers  # 指定的线程数
        
        # 输入文件路径
        self.meta_input_dir = meta_input_dir
        self.prompt_input_path = os.path.join(meta_input_dir, 'prompt.json')
        self.checklist_input_path = os.path.join(meta_input_dir, 'checklist.json')
        self.response_input_path = os.path.join(response_input_dir, f"{model_name}_response.json")
        
        # 输出文件路径
        self.output_dir = output_dir
        self.judge_output_path = os.path.join(output_dir, f"{model_name}_check_result.json")

        # 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 日志目录
        self.log_dir = os.path.join('./logs/check/', model_name)
        self.log_manager = LogManager(self.log_dir)

        # 配置模型名称
        self.judge_llm = 'gpt-5-mini'

        # 配置文件
        rule_based_judge_llm_meta_prompt_path = "meta_prompt/rule_based_judge_llm_meta_prompt.txt"
        open_ended_judge_llm_meta_prompt_path = "meta_prompt/open_ended_judge_llm_meta_prompt.txt"
        
        print("开始加载元指令...")
        
        try:
            with open(rule_based_judge_llm_meta_prompt_path, 'r', encoding='utf-8') as f:
                rule_based_judge_llm_meta_prompt = f.read()
            print(f"成功从 '{rule_based_judge_llm_meta_prompt_path}' 加载judge元指令。")
            with open(open_ended_judge_llm_meta_prompt_path, 'r', encoding='utf-8') as f:
                open_ended_judge_llm_meta_prompt = f.read()
        except FileNotFoundError as e:
            print(f"错误: 无法加载元指令文件 {e.filename}。请检查路径是否正确。")
            raise e
        

        self.meta_prompt = {
            "rule_based_judge": rule_based_judge_llm_meta_prompt,
            "open_ended_judge": open_ended_judge_llm_meta_prompt
        }
        
        # 初始化模型客户端
        print("开始初始化模型客户端...")
        self.client = {
            'judge_llm': openai_client()
        }
        
        self.auto_checker = AutoRuleChecker()
        
        # 多线程相关
        self.lock = Lock()
        
        # 显示配置信息
        print(f"线程配置: {self.max_workers} 个并发线程")
        
        self.max_token = 2048

    @combined_retry(timeout_seconds=600, 
                    timeout_retries=2, 
                    error_retries=3, 
                    exceptions=(ValueError, ConnectionError),
                    delay=1.0,
                    backoff=2.0)
    def get_rule_based_checkresult_with_llm(self,
                                            response: str, 
                                            checkitem: Dict,
                                            retry_response=None) -> Dict:
        """使用 judge_llm 生成 rule_based check result"""
        json_prompt = json.dumps({
            "response": response,
            "checkitem": checkitem
        }, ensure_ascii=False)
        if retry_response == None:
            api_response = self.client['judge_llm'].chat.completions.create(
                model=self.judge_llm,
                messages=[
                    {"role": "system", "content": self.meta_prompt['rule_based_judge']},
                    {"role": "user", "content": json_prompt}
                ],
                response_format={"type": "json_object"},
                stream=False
            )
        else:
            retry_prompt = """
            The content you extracted has been detected as not being a pure extraction from the response. The "content in response" check failed. Please re-extract, noting that you cannot make any modifications - it must be an original text excerpt from the response without adding any of your own understanding or changes.
            """
            retry_response = json.dumps(retry_response, ensure_ascii=False)
            api_response = self.client['judge_llm'].chat.completions.create(
                model=self.judge_llm,
                messages=[
                    {"role": "system", "content": self.meta_prompt['rule_based_judge']},
                    {"role": "user", "content": json_prompt},
                    {"role": "assistant", "content": retry_response},
                    {"role": "user", "content": retry_prompt}
                ],
                response_format={"type": "json_object"},
                stream=False
            )
        try:
            return(json.loads(clean_json_response(api_response.choices[0].message.content)))
        except json.JSONDecodeError as e:
            print(f"清理后仍然解析失败: {api_response.choices[0].message.content}")
            raise ValueError(f"无法解析LLM响应为有效JSON格式: {e}") from e

    @combined_retry(timeout_seconds=600, 
                    timeout_retries=2, 
                    error_retries=3, 
                    exceptions=(ValueError, ConnectionError),
                    delay=1.0,
                    backoff=2.0)
    def get_open_ended_checkresult_with_llm(self, 
                                            prompt: str,
                                            response: str, 
                                            question: Dict,
                                            options: List[str]
                                            ) -> Dict:
        """使用 judge_llm 生成 open_ended check result"""
        json_prompt = json.dumps({
            "prompt": prompt,
            "response": response,
            "question": question,
            "options": options
        }, ensure_ascii=False)
        api_response = self.client['judge_llm'].chat.completions.create(
            model=self.judge_llm,
            messages=[
                {"role": "system", "content": self.meta_prompt['open_ended_judge']},
                {"role": "user", "content": json_prompt}
            ],
            response_format={"type": "json_object"},
            stream=False
        )
        
        try:
            return(json.loads(clean_json_response(api_response.choices[0].message.content)))
        except json.JSONDecodeError as e:
            print(f"清理后仍然解析失败: {api_response.choices[0].message.content}")
            raise ValueError(f"无法解析LLM响应为有效JSON格式: {e}") from e
    
    def read_data_file(self):
        """读取输入数据文件"""
        # 检查输入文件
        if not os.path.exists(self.prompt_input_path):
            raise FileNotFoundError(f"Prompt文件未找到: {self.prompt_input_path}")
        if not os.path.exists(self.checklist_input_path):
            raise FileNotFoundError(f"Checklist文件未找到: {self.checklist_input_path}")
        if not os.path.exists(self.response_input_path):
            raise FileNotFoundError(f"Response文件未找到: {self.response_input_path}")
        
        # 读取文件
        with open(self.prompt_input_path, 'r', encoding='utf-8') as f:
            prompt_dict = json.load(f)
        print(f"成功从 '{self.prompt_input_path}' 加载prompt数据")
        
        with open(self.checklist_input_path, 'r', encoding='utf-8') as f:
            checklist_dict = json.load(f)
        print(f"成功从 '{self.checklist_input_path}' 加载checklist数据")
        
        with open(self.response_input_path, 'r', encoding='utf-8') as f:
            response_dict = json.load(f)
        print(f"成功从 '{self.response_input_path}' 加载response数据")
        
        # 检查已有的输出文件
        if os.path.exists(self.judge_output_path):
            try:
                with open(self.judge_output_path, 'r', encoding='utf-8') as f:
                    judge_dict = json.load(f)
                print(f"找到已有check result文件 '{self.judge_output_path}'，将从断点继续处理。")
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"无法读取结果文件 '{self.judge_output_path}'，将重新开始处理。")
                judge_dict = {}
        else:
            print("未找到已有结果文件，将重新开始处理。")
            judge_dict = {}
            
        return prompt_dict, checklist_dict, response_dict, judge_dict
    
    def save_data_file(self, judge_dict):
        """保存数据文件，按video_id排序"""
        # 提取video_id的后三位数字进行排序
        def get_video_sort_key(video_id):
            parts = video_id.split('_')
            video_part = parts[0]
            video_num = int(parts[-1]) if parts[-1].isdigit() else 0
            
            # 定义part的顺序
            part_order = {'clip': 0, 'short': 1, 'long': 2}
            part_idx = part_order.get(video_part, 999)  # 未知part放最后
            
            # 返回元组作为排序键：(part顺序, 数字)
            return (part_idx, video_num)
        
        # 对字典按video_id排序
        sorted_judge_dict = dict(sorted(judge_dict.items(), key=lambda x: get_video_sort_key(x[0])))
        
        with self.lock:
            with open(self.judge_output_path, 'w', encoding='utf-8') as f:
                json.dump(sorted_judge_dict, f, ensure_ascii=False, indent=4)

    def process_video_wrapper(self, video_id: str, existing_data: Dict, 
                            judge_dict: Dict) -> Tuple[str, Dict, str]:
        """处理单个视频的包装函数"""
        # 简化的跳过逻辑：检查是否已处理且有完整结果
        if video_id in judge_dict and judge_dict[video_id] and len(judge_dict[video_id]) > 0:
            # 已处理，创建或获取日志文件
            existing_log_path = os.path.join(self.log_dir, f"{video_id}.log")
            if not os.path.exists(existing_log_path):
                # 创建简单的跳过日志
                with open(existing_log_path, 'w', encoding='utf-8') as f:
                    f.write(f"[跳过] 该视频已处理完成\n")
                    f.write(f"视频ID: {video_id}\n")
                    f.write(f"跳过时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"已有结果数量: {len(judge_dict[video_id])}\n")
            return video_id, None, existing_log_path
            
        # 创建日志记录器
        logger = VideoLogger(video_id, self.log_dir)
        
        try:
            # 创建处理器并处理
            processor = VideoProcessor(video_id, existing_data, self, logger)
            video_id, result = processor.process()
            
            # 关闭日志
            logger.close()
            log_path = logger.get_log_path()
            
            return video_id, result, log_path
            
        except Exception as e:
            logger.write(f"处理过程中发生未捕获的错误: {str(e)}")
            logger.close()
            log_path = logger.get_log_path()
            return video_id, None, log_path

    def process_single_video_independently(self, video_id: str, existing_data: Dict, 
                                         judge_dict: Dict, progress_manager: ProgressManager = None):
        """
        独立处理单个视频的完整流程：处理->保存->更新进度
        
        Args:
            video_id: 视频ID
            existing_data: 视频数据
            judge_dict: 判断结果字典
            progress_manager: 进度管理器
            
        Returns:
            str: 处理状态 ('completed', 'skipped', 'failed')
        """
        try:
            # 处理视频
            video_id, result, log_path = self.process_video_wrapper(video_id, existing_data, judge_dict)
            
            if result is None:
                # 检查是否是因为已处理而跳过
                if video_id in judge_dict and len(judge_dict.get(video_id, [])) > 0:
                    if progress_manager:
                        progress_manager.update('skipped', video_id)
                    # 将日志添加到管理器
                    self.log_manager.add_completed_log(video_id, log_path)
                    return 'skipped'
                else:
                    if progress_manager:
                        progress_manager.update('failed', video_id)
                    return 'failed'
            else:
                # 保存结果
                self._save_video_result(video_id, result['judge'])
                
                if progress_manager:
                    progress_manager.update('completed', video_id)
                
                # 将日志添加到管理器
                self.log_manager.add_completed_log(video_id, log_path)
                return 'completed'
                
        except Exception as e:
            print(f"❌ 视频 {video_id} 处理失败: {str(e)}")
            if progress_manager:
                progress_manager.update('failed', video_id)
            return 'failed'
    
    def _save_video_result(self, video_id: str, result_data: List[Dict]):
        """
        线程安全地保存单个视频的结果数据
        
        Args:
            video_id: 视频ID
            result_data: 结果数据列表
        """
        try:
            # 读取当前结果
            if os.path.exists(self.judge_output_path):
                with self.lock:
                    with open(self.judge_output_path, 'r', encoding='utf-8') as f:
                        current_data = json.load(f)
            else:
                current_data = {}
            
            # 更新数据
            current_data[video_id] = result_data
            
            # 保存排序后的数据
            self.save_data_file(current_data)
            
        except Exception as e:
            print(f"❌ 保存视频 {video_id} 结果时出错: {str(e)}")
            raise

    def run(self):
        """执行检查结果生成管道 - 改进的线程管理和进度跟踪"""
        start_time = datetime.now()
        
        # 1. 初始化和数据准备
        self._print_pipeline_header(start_time)
        prompt_dict, checklist_dict, response_dict, judge_dict = self.read_data_file()
        videos_to_process, skipped_count = self._prepare_video_tasks(
            prompt_dict, checklist_dict, response_dict, judge_dict
        )
        
        total_videos = len(videos_to_process)
        total_all_videos = len(set(prompt_dict.keys()) & set(checklist_dict.keys()) & set(response_dict.keys()))
        
        # 2. 显示配置信息
        self._print_configuration_info(total_all_videos, total_videos, skipped_count)
        
        if total_videos == 0:
            print("✅ 所有视频都已处理完成，无需进一步处理。")
            return
        
        # 3. 执行多线程处理
        final_stats = self._execute_multithreaded_processing(videos_to_process)
        
        # 4. 显示最终统计
        self._print_final_statistics(start_time, total_all_videos, final_stats, skipped_count)
    
    def _print_pipeline_header(self, start_time: datetime):
        """打印管道开始信息"""
        print(f"\n{'='*80}")
        print(f"Check-Only Pipeline 开始执行")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"输入目录: {self.meta_input_dir}")
        print(f"输出目录: {self.output_dir}")
        print(f"日志目录: {self.log_dir}")
        print(f"{'='*80}\n")
    
    def _prepare_video_tasks(self, prompt_dict: Dict, checklist_dict: Dict, 
                           response_dict: Dict, judge_dict: Dict) -> Tuple[List[Tuple], int]:
        """
        准备需要处理的视频任务列表
        
        Returns:
            Tuple[List[Tuple], int]: (待处理任务列表, 跳过数量)
        """
        videos_to_process = []
        skipped_initial = 0
        
        # 验证数据一致性
        video_ids = set(prompt_dict.keys()) & set(checklist_dict.keys()) & set(response_dict.keys())
        print(f"找到 {len(video_ids)} 个可处理的视频")
        
        # 恢复已处理视频的日志
        self._recover_completed_logs(judge_dict)
        
        # 准备待处理的视频任务
        for video_id in sorted(video_ids):
            existing_data = {
                'prompts': prompt_dict[video_id],
                'checklists': checklist_dict[video_id],
                'responses': response_dict[video_id]
            }
            
            # 检查是否已经完成处理
            if self._is_video_completed(video_id, judge_dict):
                skipped_initial += 1
                continue
                
            videos_to_process.append((video_id, existing_data))
        
        return videos_to_process, skipped_initial
    
    def _recover_completed_logs(self, judge_dict: Dict):
        """恢复已处理视频的日志"""
        print("恢复已处理视频的日志...")
        recovered_count = 0
        
        for video_id in judge_dict:
            if len(judge_dict.get(video_id, [])) > 0:  # 确认视频已处理
                log_path = os.path.join(self.log_dir, f"{video_id}.log")
                if os.path.exists(log_path):
                    # 日志文件存在且视频已处理，恢复日志
                    self.log_manager.completed_logs[video_id] = log_path
                    recovered_count += 1
                else:
                    # 视频已处理但日志不存在，创建占位日志
                    with open(log_path, 'w', encoding='utf-8') as f:
                        f.write(f"[恢复信息] 该视频已处理但原始日志丢失\n")
                        f.write(f"视频ID: {video_id}\n")
                        f.write(f"恢复时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    self.log_manager.completed_logs[video_id] = log_path
                    recovered_count += 1
        
        if recovered_count > 0:
            print(f"成功恢复 {recovered_count} 个已处理视频的日志")
            # 重新生成主日志
            self.log_manager._regenerate_master_log()
        
        # 清理无效的日志文件
        self._cleanup_invalid_logs(judge_dict)
    
    def _cleanup_invalid_logs(self, judge_dict: Dict):
        """清理无效的日志文件"""
        all_log_files = glob.glob(os.path.join(self.log_dir, "*.log"))
        invalid_logs = 0
        for log_path in all_log_files:
            if os.path.basename(log_path) == "master.log":
                continue
            video_id = os.path.basename(log_path).replace(".log", "")
            # 如果日志对应的视频未处理完成，删除该日志
            if video_id not in judge_dict or len(judge_dict.get(video_id, [])) == 0:
                os.remove(log_path)
                invalid_logs += 1
        
        if invalid_logs > 0:
            print(f"清理了 {invalid_logs} 个无效的日志文件")
    
    def _is_video_completed(self, video_id: str, judge_dict: Dict) -> bool:
        """检查视频是否已完成处理"""
        return video_id in judge_dict and len(judge_dict.get(video_id, [])) > 0
    
    def _print_configuration_info(self, total_all: int, to_process: int, skipped: int):
        """打印配置信息"""
        print(f"总视频数: {total_all}")
        print(f"需要处理: {to_process}")
        print(f"已完成(跳过): {skipped}")
        print(f"线程配置: {self.max_workers} 个并发线程")
        print(f"{'='*80}\n")
    
    def _execute_multithreaded_processing(self, videos_to_process: List[Tuple]) -> Dict[str, int]:
        """执行多线程处理"""
        total_videos = len(videos_to_process)
        
        # 读取当前判断结果
        if os.path.exists(self.judge_output_path):
            with open(self.judge_output_path, 'r', encoding='utf-8') as f:
                judge_dict = json.load(f)
        else:
            judge_dict = {}
        
        with ProgressManager(total_videos, "检查进度") as progress_manager:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_video = {}
                for video_id, existing_data in videos_to_process:
                    future = executor.submit(
                        self.process_single_video_independently,
                        video_id,
                        existing_data,
                        judge_dict,
                        progress_manager
                    )
                    future_to_video[future] = video_id
                
                # 等待所有任务完成
                for future in as_completed(future_to_video):
                    try:
                        future.result()  # 获取结果，捕获异常
                    except Exception as e:
                        video_id = future_to_video[future]
                        print(f"❌ 处理视频 {video_id} 时发生错误: {str(e)}")
        
        return progress_manager.get_stats()
    
    def _print_final_statistics(self, start_time: datetime, total_all: int, 
                              stats: Dict[str, int], skipped_initial: int):
        """打印最终统计信息"""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print(f"Check-Only Pipeline 执行完成")
        print(f"总耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)")
        print(f"处理统计:")
        print(f"  - 总视频数: {total_all}")
        print(f"  - 初始跳过: {skipped_initial}")
        print(f"  - 本次成功: {stats['completed']}")
        print(f"  - 本次失败: {stats['failed']}")
        print(f"  - 本次跳过: {stats['skipped']}")
        
        # 计算处理速度
        total_processed = stats['completed'] + stats['failed']
        if total_processed > 0:
            avg_time_per_video = duration / total_processed
            print(f"  - 平均处理时间: {avg_time_per_video:.2f} 秒/视频")
        
        print(f"结果保存: {self.judge_output_path}")
        print(f"日志目录: {self.log_dir}")
        print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check-Only Pipeline")
    parser.add_argument('-m', '--model', type=str, default='Qwen2.5-VL-7B-Instruct', 
                        help='Model name for test VLM (default: Qwen2.5-VL-7B-Instruct)')
    parser.add_argument('-i', '--input_dir', type=str, default='./annotation/normal', 
                       help='Input directory containing prompt.json and checklist.json')
    parser.add_argument('-r', '--response_dir', type=str, default='./response', 
                       help='Input directory containing model response files')
    parser.add_argument('-o', '--output_dir', type=str, default='./check_result', 
                       help='Output directory for check results')
    parser.add_argument('-w', '--max_workers', type=int, default=10, 
                       help='Maximum number of concurrent threads (default: 10)')
    
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    # 创建并运行管道
    pipeline = CheckOnlyPipeline(
        meta_input_dir=args.input_dir,
        response_input_dir=args.response_dir,
        model_name=args.model,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    try:
        pipeline.run()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断程序执行")
    except Exception as e:
        print(f"\n❌ 程序执行出错: {str(e)}")
        traceback.print_exc()
    
    '''
    使用示例:
    
    # 基础用法
    python generate_check_result.py -w 10 -m Qwen2.5-VL-7B-Instruct
    python generate_check_result.py -w 10 -m human
    
    # 指定输入输出目录
    python generate_check_result.py -w 15 -m gemini-2.0-flash -i ./annotation/test
    
    python generate_check_result.py -w 15 -m 7B-VLM-600 -i ./annotation/test
    python generate_check_result.py -w 15 -m 7B-LLM-845 -i ./annotation/test
    python generate_check_result.py -w 15 -m 7B-LLM-812_25K ./annotation/test
    
    # 多线程处理
    python generate_check_result.py -w 10 -m Qwen2.5-VL-7B-SFT -i ./annotation/test
    '''
