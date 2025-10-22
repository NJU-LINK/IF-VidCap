import json
import os
import argparse
from typing import Dict, Any, Optional
from utils import RuledCheckModule

class AutoRuleChecker:
    """
    自动化规则检查器类，用于执行基于规则的检查。
    """
    
    def __init__(self, utils_path: Optional[str] = None):
        """
        初始化规则检查器。
        
        Args:
            utils_path: utils目录的路径，如果为None则使用默认路径
        """
        self.utils_path = utils_path or os.path.join(os.path.dirname(__file__), 'utils')
        self.rule_functions = {}
        self._load_rule_functions()
    
    def _load_rule_functions(self):
        """动态加载规则检查模块"""
        try:
            ruled_check_module = RuledCheckModule()
            
            # 获取所有检查函数
            self.rule_functions = {
                'plain_text': ruled_check_module.plain_text, 
                'json_object': ruled_check_module.json_object,
                'json_array': ruled_check_module.json_array,
                'unordered_list': ruled_check_module.unordered_list,
                'ordered_list': ruled_check_module.ordered_list,
                'table': ruled_check_module.table,
                'keyword': ruled_check_module.keyword,
                'markdown': ruled_check_module.markdown,
                'prefix_suffix': ruled_check_module.prefix_suffix,
                'delimiter': ruled_check_module.delimiter,
                'length': ruled_check_module.length,
                'count': ruled_check_module.count,
                'case': ruled_check_module.case,
                'language': ruled_check_module.language
            }
        except Exception as e:
            print(f"加载规则检查模块失败: {e}")
            print("请确保 utils/ruled_check.py 文件存在且可访问")
            # 不抛出异常，允许类正常初始化

    def load_check_data(self, check_json_path: str) -> Dict[str, Any]:
        """
        加载检查数据文件。
        
        Args:
            check_json_path: check_result.json文件的路径
            
        Returns:
            检查数据字典
            
        Raises:
            FileNotFoundError: 文件未找到
            json.JSONDecodeError: JSON解析错误
        """
        try:
            with open(check_json_path, 'r', encoding='utf-8') as f:
                check_data = json.load(f)
            print(f"成功加载检查数据文件: {check_json_path}")
            return check_data
        except FileNotFoundError:
            raise FileNotFoundError(f"检查数据文件 '{check_json_path}' 未找到")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"无法解析JSON文件 '{check_json_path}': {e}")
    
    def execute_rule_check(self, api_call: str, parameters: Dict[str, Any]) -> tuple:
        """
        执行单个规则检查。
        
        Args:
            api_call: API调用名称
            parameters: 参数字典
            
        Returns:
            (success: bool, result: Any, error: str)
        """
        if not api_call:
            return False, None, "Missing api_call field"
        
        if api_call not in self.rule_functions:
            return False, None, f"Unknown API call: {api_call}"
        
        try:
            func = self.rule_functions[api_call]
            content_list = parameters.get('content')
            if isinstance(content_list, list):
                shared_params = {k: v for k, v in parameters.items() if k != 'content'}
                results = []
                for content in content_list:
                    call_params = dict(shared_params)
                    call_params['content'] = content
                    result = func(**call_params)
                    results.append(result)
                final_result = all(results)
                return True, final_result, None
            else:
                result = func(**parameters)
                return True, result, None
        except Exception as e:
            return False, None, str(e)
        
    def check_all_rules(self, check_data: dict) -> tuple:
        """
        执行所有规则检查的主要方法。
        
        Args:
            check_data: 要检查的一个 case
            output_json_path: 输出文件路径，如果为None则覆盖原文件
            
        Returns:
            包含统计信息的字典
        """
            
        if not isinstance(check_data, dict) or 'ruled_based_check' not in check_data:
            print(f"  提示: 本检查项没有rule_based_checks字段，跳过检查")
            return check_data, True
        rule_based_checks = check_data['ruled_based_check']
        if not isinstance(rule_based_checks, list):
            print(f"  错误: 本检查项的rule_based_checks不是列表")
            return check_data, False
        
        # 4. 遍历每个规则检查
        for rule_idx, rule_check in enumerate(rule_based_checks):
            
            api_call = rule_check.get('constraint_id')
            parameters = rule_check.get('parameters', {})
            
            if not isinstance(parameters, dict):
                print(f"    错误: 规则检查 {rule_idx + 1} 的parameters字段不是字典")
                return check_data, False
            
            # 5. 执行规则检查
            success, result, error = self.execute_rule_check(api_call, parameters)
            
            if success:
                rule_check['result'] = result
            else:
                print(f"    ✗ 规则检查 {rule_idx + 1} ({api_call}) 执行失败: {error}")
                return check_data, False
        return check_data, True

    def check_all_rules_form_file(self, check_json_path: str, output_json_path: Optional[str] = None) -> Dict[str, int]:
        """
        执行所有规则检查的主要方法。
        
        Args:
            check_json_path: check.json文件的路径
            output_json_path: 输出文件路径，如果为None则覆盖原文件
            
        Returns:
            包含统计信息的字典
        """
        # 1. 读取check.json文件
        try:
            check_data = self.load_check_data(check_json_path)
        except Exception as e:
            print(f"错误: {e}")
            return {}
        
        # 统计信息
        stats = {
            'total_videos': 0,
            'total_rule_checks': 0,
            'successful_checks': 0
        }
        
        print("开始执行自动化规则检查...")
        
        # 2. 遍历每个video_id
        for video_id, check_cases in check_data.items():
            if not isinstance(check_cases, dict) or 'check_case' not in check_cases:
                continue
                
            stats['total_videos'] += 1
            print(f"\n--- 正在处理视频: {video_id} ---")
            
            check_list = check_cases['check_case']
            
            # 3. 遍历每个check项
            for check_idx, check_item in enumerate(check_list):
                if not isinstance(check_item, dict) or 'check_result' not in check_item:
                    print(f"  错误: 第 {check_idx + 1} 个检查项格式不正确")
                    return
                    
                check_result = check_item['check_result']
                if not isinstance(check_result, dict) or 'ruled_based_check' not in check_result:
                    print(f"  提示: 第 {check_idx + 1} 个检查项没有rule_based_checks字段，跳过检查")
                    continue
                    
                rule_based_checks = check_result['ruled_based_check']
                if not isinstance(rule_based_checks, list):
                    print(f"  错误: 第 {check_idx + 1} 个检查项的rule_based_checks不是列表")
                    return
                
                print(f"  处理第 {check_idx + 1} 个检查项，包含 {len(rule_based_checks)} 个规则检查")
                
                # 4. 遍历每个规则检查
                for rule_idx, rule_check in enumerate(rule_based_checks):
                    if not isinstance(rule_check, dict):
                        continue
                        
                    stats['total_rule_checks'] += 1
                    api_call = rule_check.get('constraint_id')
                    parameters = rule_check.get('parameters', {})
                    
                    if not isinstance(parameters, dict):
                        print(f"    错误: 规则检查 {rule_idx + 1} 的parameters字段不是字典")
                        return
                    
                    # 5. 执行规则检查
                    success, result, error = self.execute_rule_check(api_call, parameters)
                    
                    if success:
                        rule_check['result'] = result
                        stats['successful_checks'] += 1
                        print(f"    ✓ 规则检查 {rule_idx + 1} ({api_call}): {result}")
                    else:
                        print(f"    ✗ 规则检查 {rule_idx + 1} ({api_call}) 执行失败: {error}")
                        return
        
        # 6. 保存结果
        output_path = output_json_path if output_json_path else check_json_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(check_data, f, ensure_ascii=False, indent=2)
            print(f"\n--- 自动化规则检查完成 ---")
            print(f"结果已保存到: {output_path}")
        except Exception as e:
            print(f"保存文件时发生错误: {str(e)}")
            return stats
        
        # 7. 输出统计信息
        self._print_statistics(stats)
        return stats
    
    def _print_statistics(self, stats: Dict[str, int]):
        """打印统计信息"""
        print(f"\n--- 检查统计 ---")
        print(f"  处理视频数: {stats['total_videos']}")
        print(f"  总规则检查数: {stats['total_rule_checks']}")
        print(f"  成功检查数: {stats['successful_checks']}")
        if stats['total_rule_checks'] > 0:
            success_rate = (stats['successful_checks'] / stats['total_rule_checks']) * 100
            print(f"  检查成功率: {success_rate:.1f}%")

    def validate_results(self, check_json_path: str) -> Dict[str, int]:
        """
        验证规则检查结果的方法，用于查看检查结果的统计信息。
        
        Args:
            check_json_path: check.json文件的路径
            
        Returns:
            包含验证统计信息的字典
        """
        try:
            check_data = self.load_check_data(check_json_path)
        except Exception as e:
            print(f"读取文件失败: {e}")
            return {}
        
        stats = {
            'total_rules': 0,
            'passed_rules': 0,
            'failed_rules': 0,
            'rules_with_score': 0
        }
        
        for video_id, check_cases in check_data.items():
            if not isinstance(check_cases, dict) or 'check' not in check_cases:
                continue
                
            for check_item in check_cases['check']:
                if not isinstance(check_item, dict) or 'check_result' not in check_item:
                    continue
                    
                rule_based_checks = check_item.get('check_result', {}).get('rule_based_checks', [])
                
                for rule_check in rule_based_checks:
                    if isinstance(rule_check, dict):
                        stats['total_rules'] += 1
                        
                        if 'score' in rule_check:
                            stats['rules_with_score'] += 1
                            score = rule_check['score']
                            
                            if score is True:
                                stats['passed_rules'] += 1
                            elif score is False:
                                stats['failed_rules'] += 1
        
        self._print_validation_statistics(stats)
        return stats
    
    def _print_validation_statistics(self, stats: Dict[str, int]):
        """打印验证统计信息"""
        print(f"--- 规则检查结果统计 ---")
        print(f"  总规则数: {stats['total_rules']}")
        print(f"  已评分规则数: {stats['rules_with_score']}")
        print(f"  通过的规则数: {stats['passed_rules']}")
        print(f"  未通过的规则数: {stats['failed_rules']}")
        if stats['total_rules'] > 0:
            completion_rate = (stats['rules_with_score'] / stats['total_rules']) * 100
            print(f"  评分完成率: {completion_rate:.1f}%")
        if stats['rules_with_score'] > 0:
            pass_rate = (stats['passed_rules'] / stats['rules_with_score']) * 100
            print(f"  规则通过率: {pass_rate:.1f}%")


# 保持向后兼容的函数接口
def auto_rule_checker(check_json_path: str, output_json_path: str = None):
    """
    自动化检查脚本函数，读取check.json文件并执行基于规则的检查。
    保持向后兼容性的包装函数。
    
    Args:
        check_json_path: check.json文件的路径
        output_json_path: 输出文件路径，如果为None则覆盖原文件
    """
    checker = AutoRuleChecker()
    return checker.check_all_rules(check_json_path, output_json_path)


def validate_rule_check_result(check_json_path: str):
    """
    验证规则检查结果的辅助函数，用于查看检查结果的统计信息。
    保持向后兼容性的包装函数。
    
    Args:
        check_json_path: check.json文件的路径
    """
    checker = AutoRuleChecker()
    return checker.validate_results(check_json_path)

# --- 程序入口 ---
def main():

    OUTPUT_FILE = "check_result.json"
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FOLDER), exist_ok=True)


    # 执行自动化检查脚本
    checker = AutoRuleChecker()
    checker.check_all_rules(OUTPUT_FILE, OUTPUT_FILE.replace("checkresult", "checkresult_ruled"))

if __name__ == "__main__":
    main()
    '''
    python auto_rule_checker.py --model gemini-2.0-flash
    python auto_rule_checker.py --model Qwen2.5-VL-7B-Instruct
    python auto_rule_checker.py --model Qwen2.5-VL-32B-Instruct
    python auto_rule_checker.py --model Qwen2.5-VL-72B-Instruct
    '''