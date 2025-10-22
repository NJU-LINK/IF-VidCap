import json
import argparse
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import pandas as pd
import os
import numpy as np

@dataclass
class ScoreResults:
    """存储各项评分结果"""
    # 基础指标 - ISR在前，CSR在后
    isr: float = 0.0  # 指令满足率
    csr: float = 0.0  # 约束满足率
    
    # Rule-based 指标 - ISR在前，CSR在后
    rule_based_isr: float = 0.0  # 基于规则的指令满足率
    rule_based_csr: float = 0.0  # 基于规则的约束满足率
    
    # Open-ended 指标 - ISR在前，CSR在后
    open_ended_isr: float = 0.0  # 开放式指令满足率
    open_ended_csr: float = 0.0  # 开放式约束满足率
    
    # Fact-Free 指标 (忽略correctness类型检查) - ISR在前，CSR在后
    ff_isr: float = 0.0  # Fact-Free指令满足率
    ff_csr: float = 0.0  # Fact-Free约束满足率
    
    # Fact-Only 指标 (只关注correctness类型检查) - ISR在前，CSR在后
    fo_isr: float = 0.0  # Fact-Only指令满足率
    fo_csr: float = 0.0  # Fact-Only约束满足率
    
    # 专项能力指标
    constraint_dimension_scores: Dict[str, float] = field(default_factory=dict)
    
    # 详细统计数据
    stats: Dict[str, Any] = field(default_factory=dict)
    
    # 视频级别的详细得分
    video_scores: Dict[str, Dict[str, float]] = field(default_factory=dict)

class ScoreCalculator:
    """分数计算器"""
    
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.results = ScoreResults()
        self.stats = defaultdict(lambda: defaultdict(int))
        self.video_scores = defaultdict(lambda: {
            'ISR': 0.0,
            'CSR': 0.0,
            'Rule-based ISR': 0.0,
            'Rule-based CSR': 0.0,
            'Open-ended ISR': 0.0,
            'Open-ended CSR': 0.0,
            'FF-ISR': 0.0,
            'FF-CSR': 0.0,
            'FO-ISR': 0.0,
            'FO-CSR': 0.0
        })
        
    def calculate_all_scores(self) -> ScoreResults:
        """计算所有分数"""
        self._calculate_isr()
        self._calculate_csr()
        self._calculate_rule_based_isr_csr()
        self._calculate_open_ended_isr_csr()
        self._calculate_ff_isr_csr()
        self._calculate_fo_isr_csr()
        self._calculate_constraint_dimension_scores()
        
        self.results.stats = dict(self.stats)
        self.results.video_scores = dict(self.video_scores)
        return self.results
    
    def _calculate_csr(self):
        """计算约束满足率 (CSR)"""
        video_csr_scores = defaultdict(float)
        
        # 全局统计
        global_constraints_passed = 0
        global_constraints_total = 0
        
        for video_id, case in self.data.items():
            for check_item in case:
                check_result = check_item.get('check_result', {})
                
                # 约束数量统计
                ruled_based_constraints = len(check_result.get('ruled_based_check', []))
                open_ended_constraints = len(check_result.get('open_ended_check', []))
                total_constraints = ruled_based_constraints + open_ended_constraints
                
                if total_constraints == 0:
                    continue
                
                global_constraints_total += total_constraints
                
                # 统计通过的约束数量
                passed_constraints = 0
                
                # 规则检查 - 每个rule_check都是一个约束
                for rule_check in check_result.get('ruled_based_check', []):
                    if rule_check.get('result', False):
                        passed_constraints += 1
                
                # 开放式检查 - 每个open_check元素是一个约束
                for open_check in check_result.get('open_ended_check', []):
                    check_items = open_check.get('check_items', [])
                    if not check_items:
                        continue
                    
                    # 判断该open_check约束是否满足（所有check_items都通过）
                    constraint_satisfied = True
                    for item in check_items:
                        if not item.get('result', False):
                            constraint_satisfied = False
                            break
                    
                    if constraint_satisfied:
                        passed_constraints += 1
                
                # 计算当前prompt的CSR
                csr = passed_constraints / total_constraints if total_constraints > 0 else 0
                
                # 累积到视频级别统计（用于计算视频平均值）
                video_csr_scores[video_id] += csr
                
                # 累积到全局统计
                global_constraints_passed += passed_constraints
        
        # 计算全局CSR
        self.results.csr = global_constraints_passed / global_constraints_total if global_constraints_total > 0 else 0
        
        # 保存统计信息
        self.stats['csr']['global_constraints_total'] = global_constraints_total
        self.stats['csr']['global_constraints_passed'] = global_constraints_passed
        
        # 计算每个视频的平均CSR（基于该视频的所有prompt）
        for video_id, total_score in video_csr_scores.items():
            # 计算该视频有多少个prompt
            prompt_count = len([item for item in self.data.get(video_id, []) if item.get('check_result')])
            if prompt_count > 0:
                self.video_scores[video_id]['CSR'] = total_score / prompt_count

    
    def _calculate_isr(self):
        """计算指令满足率 (ISR)"""
        # 存储每个prompt的约束满足情况
        prompt_constraint_status = defaultdict(lambda: {
            'total_constraints': 0,
            'passed_constraints': 0
        })
        
        for video_id, case in self.data.items():
            for check_item in case:
                prompt_id = check_item.get('prompt_id', '')
                prompt_key = (video_id, prompt_id)
                
                check_result = check_item.get('check_result', {})
                
                # 约束数量统计
                ruled_based_constraints = len(check_result.get('ruled_based_check', []))
                open_ended_constraints = len(check_result.get('open_ended_check', []))
                total_constraints = ruled_based_constraints + open_ended_constraints
                
                if total_constraints == 0:
                    continue
                
                prompt_constraint_status[prompt_key]['total_constraints'] = total_constraints
                
                # 统计通过的约束数量
                passed_constraints = 0
                
                # 规则检查 - 每个rule_check都是一个约束
                for rule_check in check_result.get('ruled_based_check', []):
                    if rule_check.get('result', False):
                        passed_constraints += 1
                
                # 开放式检查 - 每个open_check元素是一个约束
                for open_check in check_result.get('open_ended_check', []):
                    check_items = open_check.get('check_items', [])
                    if not check_items:
                        continue
                    
                    # 判断该open_check约束是否满足（所有check_items都通过）
                    constraint_satisfied = True
                    for item in check_items:
                        if not item.get('result', False):
                            constraint_satisfied = False
                            break
                    
                    if constraint_satisfied:
                        passed_constraints += 1
                
                prompt_constraint_status[prompt_key]['passed_constraints'] = passed_constraints
        
        # 获取所有唯一的prompt
        all_prompt_keys = list(prompt_constraint_status.keys())
        total_prompts = len(all_prompt_keys)
        
        # 计算ISR (prompt级别的完全满足率)
        # ISR: 该prompt所有约束都满足
        prompts_fully_satisfied = 0
        for prompt_key in all_prompt_keys:
            status = prompt_constraint_status[prompt_key]
            if status['passed_constraints'] == status['total_constraints']:
                prompts_fully_satisfied += 1
        
        self.results.isr = prompts_fully_satisfied / total_prompts if total_prompts > 0 else 0
        
        # 计算每个视频的ISR
        video_prompt_constraint_status = defaultdict(lambda: {})
        
        # 按video_id重新组织数据
        for prompt_key, status in prompt_constraint_status.items():
            video_id, prompt_id = prompt_key
            video_prompt_constraint_status[video_id][prompt_id] = status
        
        # 计算每个视频的ISR
        for video_id, prompts in video_prompt_constraint_status.items():
            total_prompts_in_video = len(prompts)
            
            if total_prompts_in_video == 0:
                continue
            
            # ISR
            prompts_fully_satisfied_in_video = 0
            for prompt_id, status in prompts.items():
                if status['passed_constraints'] == status['total_constraints']:
                    prompts_fully_satisfied_in_video += 1
            
            self.video_scores[video_id]['ISR'] = prompts_fully_satisfied_in_video / total_prompts_in_video

    def _calculate_rule_based_isr_csr(self):
        """计算Rule-based ISR和CSR"""
        # Rule-based约束统计
        rule_based_total = 0
        rule_based_passed = 0
        rule_based_video_constraints = defaultdict(lambda: {'total': 0, 'passed': 0})
        
        # Rule-based ISR统计（按prompt统计）
        prompt_constraint_status = defaultdict(lambda: {
            'total_constraints': 0,
            'passed_constraints': 0
        })
        
        for video_id, case in self.data.items():
            for check_item in case:
                prompt_id = check_item.get('prompt_id', '')
                prompt_key = (video_id, prompt_id)
                
                check_result = check_item.get('check_result', {})
                
                # Rule-based约束 - 每个rule_check就是一个约束项
                ruled_based_checks = check_result.get('ruled_based_check', [])
                rule_based_total += len(ruled_based_checks)
                rule_based_video_constraints[video_id]['total'] += len(ruled_based_checks)
                prompt_constraint_status[prompt_key]['total_constraints'] += len(ruled_based_checks)
                
                for rule_check in ruled_based_checks:
                    if rule_check.get('result', False):
                        rule_based_passed += 1
                        rule_based_video_constraints[video_id]['passed'] += 1
                        prompt_constraint_status[prompt_key]['passed_constraints'] += 1
        
        # 计算Rule-based CSR
        self.results.rule_based_csr = rule_based_passed / rule_based_total if rule_based_total > 0 else 0
        
        # 计算Rule-based ISR
        all_prompt_keys = [k for k in prompt_constraint_status.keys() if prompt_constraint_status[k]['total_constraints'] > 0]
        total_prompts = len(all_prompt_keys)
        prompts_fully_satisfied = sum(1 for prompt_key in all_prompt_keys 
                                    if prompt_constraint_status[prompt_key]['passed_constraints'] == 
                                       prompt_constraint_status[prompt_key]['total_constraints'])
        
        self.results.rule_based_isr = prompts_fully_satisfied / total_prompts if total_prompts > 0 else 0
        
        # 保存统计信息
        self.stats['rule_based_csr'] = {
            'total_constraints': rule_based_total,
            'passed_constraints': rule_based_passed
        }
        self.stats['rule_based_isr'] = {
            'total_prompts': total_prompts,
            'fully_satisfied_prompts': prompts_fully_satisfied
        }
        
        # 计算每个视频的Rule-based CSR
        for video_id, constraints in rule_based_video_constraints.items():
            if constraints['total'] > 0:
                self.video_scores[video_id]['Rule-based CSR'] = constraints['passed'] / constraints['total']
        
        # 计算每个视频的Rule-based ISR
        video_prompt_constraint_status = defaultdict(lambda: {})
        for prompt_key, status in prompt_constraint_status.items():
            video_id, prompt_id = prompt_key
            if status['total_constraints'] > 0:  # 只考虑有rule-based约束的prompt
                video_prompt_constraint_status[video_id][prompt_id] = status
        
        for video_id, prompts in video_prompt_constraint_status.items():
            total_prompts_in_video = len(prompts)
            if total_prompts_in_video == 0:
                continue
            
            prompts_fully_satisfied_in_video = sum(1 for prompt_id, status in prompts.items() 
                                                 if status['passed_constraints'] == status['total_constraints'])
            
            self.video_scores[video_id]['Rule-based ISR'] = prompts_fully_satisfied_in_video / total_prompts_in_video

    def _calculate_open_ended_isr_csr(self):
        """计算Open-ended ISR和CSR"""
        # Open-ended约束统计
        open_ended_total = 0
        open_ended_passed = 0
        open_ended_video_constraints = defaultdict(lambda: {'total': 0, 'passed': 0})
        
        # Open-ended ISR统计（按prompt统计）
        prompt_constraint_status = defaultdict(lambda: {
            'total_constraints': 0,
            'passed_constraints': 0
        })
        
        for video_id, case in self.data.items():
            for check_item in case:
                prompt_id = check_item.get('prompt_id', '')
                prompt_key = (video_id, prompt_id)
                
                check_result = check_item.get('check_result', {})
                
                # Open-ended约束 - 每个open_check元素是一个约束
                open_ended_checks = check_result.get('open_ended_check', [])
                open_ended_total += len(open_ended_checks)
                open_ended_video_constraints[video_id]['total'] += len(open_ended_checks)
                prompt_constraint_status[prompt_key]['total_constraints'] += len(open_ended_checks)
                
                for open_check in open_ended_checks:
                    check_items = open_check.get('check_items', [])
                    if not check_items:
                        continue
                    
                    # 判断该open_check约束是否满足（所有check_items都通过）
                    constraint_satisfied = True
                    for item in check_items:
                        if not item.get('result', False):
                            constraint_satisfied = False
                            break
                    
                    if constraint_satisfied:
                        open_ended_passed += 1
                        open_ended_video_constraints[video_id]['passed'] += 1
                        prompt_constraint_status[prompt_key]['passed_constraints'] += 1
        
        # 计算Open-ended CSR
        self.results.open_ended_csr = open_ended_passed / open_ended_total if open_ended_total > 0 else 0
        
        # 计算Open-ended ISR
        all_prompt_keys = [k for k in prompt_constraint_status.keys() if prompt_constraint_status[k]['total_constraints'] > 0]
        total_prompts = len(all_prompt_keys)
        prompts_fully_satisfied = sum(1 for prompt_key in all_prompt_keys 
                                    if prompt_constraint_status[prompt_key]['passed_constraints'] == 
                                       prompt_constraint_status[prompt_key]['total_constraints'])
        
        self.results.open_ended_isr = prompts_fully_satisfied / total_prompts if total_prompts > 0 else 0
        
        # 保存统计信息
        self.stats['open_ended_csr'] = {
            'total_constraints': open_ended_total,
            'passed_constraints': open_ended_passed
        }
        self.stats['open_ended_isr'] = {
            'total_prompts': total_prompts,
            'fully_satisfied_prompts': prompts_fully_satisfied
        }
        
        # 计算每个视频的Open-ended CSR
        for video_id, constraints in open_ended_video_constraints.items():
            if constraints['total'] > 0:
                self.video_scores[video_id]['Open-ended CSR'] = constraints['passed'] / constraints['total']
        
        # 计算每个视频的Open-ended ISR
        video_prompt_constraint_status = defaultdict(lambda: {})
        for prompt_key, status in prompt_constraint_status.items():
            video_id, prompt_id = prompt_key
            if status['total_constraints'] > 0:  # 只考虑有open-ended约束的prompt
                video_prompt_constraint_status[video_id][prompt_id] = status
        
        for video_id, prompts in video_prompt_constraint_status.items():
            total_prompts_in_video = len(prompts)
            if total_prompts_in_video == 0:
                continue
            
            prompts_fully_satisfied_in_video = sum(1 for prompt_id, status in prompts.items() 
                                                 if status['passed_constraints'] == status['total_constraints'])
            
            self.video_scores[video_id]['Open-ended ISR'] = prompts_fully_satisfied_in_video / total_prompts_in_video

    def _calculate_ff_isr_csr(self):
        """计算Fact-Free ISR和CSR（忽略correctness类型的检查）"""
        ff_total = 0
        ff_passed = 0
        ff_video_constraints = defaultdict(lambda: {'total': 0, 'passed': 0})
        
        # FF ISR统计（按prompt统计）
        prompt_constraint_status = defaultdict(lambda: {
            'total_constraints': 0,
            'passed_constraints': 0
        })
        
        for video_id, case in self.data.items():
            for check_item in case:
                prompt_id = check_item.get('prompt_id', '')
                prompt_key = (video_id, prompt_id)
                
                check_result = check_item.get('check_result', {})
                
                # Rule-based约束 - 每个rule_check就是一个约束项
                ruled_based_checks = check_result.get('ruled_based_check', [])
                ff_total += len(ruled_based_checks)
                ff_video_constraints[video_id]['total'] += len(ruled_based_checks)
                prompt_constraint_status[prompt_key]['total_constraints'] += len(ruled_based_checks)
                
                for rule_check in ruled_based_checks:
                    if rule_check.get('result', False):
                        ff_passed += 1
                        ff_video_constraints[video_id]['passed'] += 1
                        prompt_constraint_status[prompt_key]['passed_constraints'] += 1
                
                # 开放式检查约束 - 只考虑attempt类型的检查项
                open_ended_checks = check_result.get('open_ended_check', [])
                for open_check in open_ended_checks:
                    check_items = open_check.get('check_items', [])
                    if not check_items:
                        continue
                    
                    # 过滤掉correctness类型的检查项
                    attempt_items = [item for item in check_items if item.get('check_type') != 'correctness']
                    
                    if not attempt_items:
                        continue
                    
                    ff_total += 1
                    ff_video_constraints[video_id]['total'] += 1
                    prompt_constraint_status[prompt_key]['total_constraints'] += 1
                    
                    # 该约束项满足需要所有attempt检查项都通过
                    all_attempt_items_passed = all(item.get('result', False) for item in attempt_items)
                    if all_attempt_items_passed:
                        ff_passed += 1
                        ff_video_constraints[video_id]['passed'] += 1
                        prompt_constraint_status[prompt_key]['passed_constraints'] += 1
        
        # 计算FF CSR
        self.results.ff_csr = ff_passed / ff_total if ff_total > 0 else 0
        
        # 计算FF ISR
        all_prompt_keys = list(prompt_constraint_status.keys())
        total_prompts = len(all_prompt_keys)
        prompts_fully_satisfied = sum(1 for prompt_key in all_prompt_keys 
                                    if prompt_constraint_status[prompt_key]['passed_constraints'] == 
                                       prompt_constraint_status[prompt_key]['total_constraints'])
        
        self.results.ff_isr = prompts_fully_satisfied / total_prompts if total_prompts > 0 else 0
        
        # 保存统计信息
        self.stats['ff_csr'] = {
            'total_constraints': ff_total,
            'passed_constraints': ff_passed
        }
        self.stats['ff_isr'] = {
            'total_prompts': total_prompts,
            'fully_satisfied_prompts': prompts_fully_satisfied
        }
        
        # 计算每个视频的FF CSR
        for video_id, constraints in ff_video_constraints.items():
            if constraints['total'] > 0:
                self.video_scores[video_id]['FF-CSR'] = constraints['passed'] / constraints['total']
        
        # 计算每个视频的FF ISR
        video_prompt_constraint_status = defaultdict(lambda: {})
        for prompt_key, status in prompt_constraint_status.items():
            video_id, prompt_id = prompt_key
            video_prompt_constraint_status[video_id][prompt_id] = status
        
        for video_id, prompts in video_prompt_constraint_status.items():
            total_prompts_in_video = len(prompts)
            if total_prompts_in_video == 0:
                continue
            
            prompts_fully_satisfied_in_video = sum(1 for prompt_id, status in prompts.items() 
                                                 if status['passed_constraints'] == status['total_constraints'])
            
            self.video_scores[video_id]['FF-ISR'] = prompts_fully_satisfied_in_video / total_prompts_in_video

    def _calculate_fo_isr_csr(self):
        """计算Fact-Only ISR和CSR（只关注correctness类型的检查）"""
        fo_total = 0
        fo_passed = 0
        fo_video_constraints = defaultdict(lambda: {'total': 0, 'passed': 0})
        
        # FO ISR统计（按prompt统计）
        prompt_constraint_status = defaultdict(lambda: {
            'total_constraints': 0,
            'passed_constraints': 0
        })
        
        for video_id, case in self.data.items():
            for check_item in case:
                prompt_id = check_item.get('prompt_id', '')
                prompt_key = (video_id, prompt_id)
                
                check_result = check_item.get('check_result', {})
                
                # 开放式检查约束 - 只考虑correctness类型的检查项
                open_ended_checks = check_result.get('open_ended_check', [])
                for open_check in open_ended_checks:
                    check_items = open_check.get('check_items', [])
                    if not check_items:
                        continue
                    
                    # 只考虑correctness类型的检查项
                    correctness_items = [item for item in check_items if item.get('check_type') == 'correctness']
                    
                    if not correctness_items:
                        continue
                    
                    fo_total += 1
                    fo_video_constraints[video_id]['total'] += 1
                    prompt_constraint_status[prompt_key]['total_constraints'] += 1
                    
                    # 该约束项满足需要所有correctness检查项都通过
                    all_correctness_items_passed = all(item.get('result', False) for item in correctness_items)
                    if all_correctness_items_passed:
                        fo_passed += 1
                        fo_video_constraints[video_id]['passed'] += 1
                        prompt_constraint_status[prompt_key]['passed_constraints'] += 1
        
        # 计算FO CSR
        self.results.fo_csr = fo_passed / fo_total if fo_total > 0 else 0
        
        # 计算FO ISR
        all_prompt_keys = [k for k in prompt_constraint_status.keys() if prompt_constraint_status[k]['total_constraints'] > 0]
        total_prompts = len(all_prompt_keys)
        prompts_fully_satisfied = sum(1 for prompt_key in all_prompt_keys 
                                    if prompt_constraint_status[prompt_key]['passed_constraints'] == 
                                       prompt_constraint_status[prompt_key]['total_constraints'])
        
        self.results.fo_isr = prompts_fully_satisfied / total_prompts if total_prompts > 0 else 0
        
        # 保存统计信息
        self.stats['fo_csr'] = {
            'total_constraints': fo_total,
            'passed_constraints': fo_passed
        }
        self.stats['fo_isr'] = {
            'total_prompts': total_prompts,
            'fully_satisfied_prompts': prompts_fully_satisfied
        }
        
        # 计算每个视频的FO CSR
        for video_id, constraints in fo_video_constraints.items():
            if constraints['total'] > 0:
                self.video_scores[video_id]['FO-CSR'] = constraints['passed'] / constraints['total']
        
        # 计算每个视频的FO ISR
        video_prompt_constraint_status = defaultdict(lambda: {})
        for prompt_key, status in prompt_constraint_status.items():
            video_id, prompt_id = prompt_key
            if status['total_constraints'] > 0:  # 只考虑有FO约束的prompt
                video_prompt_constraint_status[video_id][prompt_id] = status
        
        for video_id, prompts in video_prompt_constraint_status.items():
            total_prompts_in_video = len(prompts)
            if total_prompts_in_video == 0:
                continue
            
            prompts_fully_satisfied_in_video = sum(1 for prompt_id, status in prompts.items() 
                                                 if status['passed_constraints'] == status['total_constraints'])
            
            self.video_scores[video_id]['FO-ISR'] = prompts_fully_satisfied_in_video / total_prompts_in_video

    def _calculate_constraint_dimension_scores(self):
        """计算约束维度得分"""
        constraint_categories = {
            'format': ['format'],
            'content': ['content'],
            'relation': ['logical', 'conditional']
        }
        
        category_scores = defaultdict(lambda: {'total': 0, 'passed': 0})
        
        for video_id, case in self.data.items():
            for check_item in case:
                check_result = check_item.get('check_result', {})
                
                for rule_check in check_result.get('ruled_based_check', []):
                    constraint_id = rule_check.get('constraint_id', '')
                    result = rule_check.get('result', False)
                    
                    # 根据constraint_id判断类别
                    for category, keywords in constraint_categories.items():
                        if any(keyword in constraint_id.lower() for keyword in keywords):
                            category_scores[category]['total'] += 1
                            if result:
                                category_scores[category]['passed'] += 1
                            break
        
        # 计算各维度得分
        for category, scores in category_scores.items():
            if scores['total'] > 0:
                score = scores['passed'] / scores['total']
                self.results.constraint_dimension_scores[f'{category}_score'] = score
                self.stats['constraint_dimensions'][category] = scores


def calculate_prompt_scores(check_item: Dict[str, Any]) -> Dict[str, float]:
    """计算单个prompt的各项指标"""
    check_result = check_item.get('check_result', {})
    
    # 约束数量统计
    ruled_based_constraints = len(check_result.get('ruled_based_check', []))
    open_ended_constraints = len(check_result.get('open_ended_check', []))
    total_constraints = ruled_based_constraints + open_ended_constraints
    
    if total_constraints == 0:
        return {
            'isr': 0.0,
            'csr': 0.0,
            'rule_based_isr': 0.0,
            'rule_based_csr': 0.0,
            'open_ended_isr': 0.0,
            'open_ended_csr': 0.0,
            'ff_isr': 0.0,
            'ff_csr': 0.0,
            'fo_isr': 0.0,
            'fo_csr': 0.0
        }
    
    # 统计通过的约束数量
    passed_constraints = 0
    
    # 规则检查统计
    ruled_based_passed = 0
    for rule_check in check_result.get('ruled_based_check', []):
        if rule_check.get('result', False):
            passed_constraints += 1
            ruled_based_passed += 1
    
    # 开放式检查统计
    open_ended_passed = 0
    
    # FF (Fact-Free) 统计
    ff_total_constraints = ruled_based_constraints  # Rule-based约束
    ff_passed_constraints = ruled_based_passed     # Rule-based通过数
    
    # FO (Fact-Only) 统计
    fo_total_constraints = 0
    fo_passed_constraints = 0
    
    for open_check in check_result.get('open_ended_check', []):
        check_items = open_check.get('check_items', [])
        if not check_items:
            continue
        
        # 判断该open_check约束是否满足（所有check_items都通过）
        constraint_satisfied = True
        for item in check_items:
            if not item.get('result', False):
                constraint_satisfied = False
                break
        
        if constraint_satisfied:
            passed_constraints += 1
            open_ended_passed += 1
        
        # 统计attempt和correctness检查项
        attempt_items = []
        correctness_items = []
        
        for item in check_items:
            check_type = item.get('check_type')
            if check_type == 'attempt':
                attempt_items.append(item)
            elif check_type == 'correctness':
                correctness_items.append(item)
        
        # FF统计：只考虑attempt类型的检查项
        if attempt_items:
            ff_total_constraints += 1
            all_attempt_items_passed = all(item.get('result', False) for item in attempt_items)
            if all_attempt_items_passed:
                ff_passed_constraints += 1
        
        # FO统计：只考虑correctness类型的检查项
        if correctness_items:
            fo_total_constraints += 1
            all_correctness_items_passed = all(item.get('result', False) for item in correctness_items)
            if all_correctness_items_passed:
                fo_passed_constraints += 1
    
    # 计算各项指标
    csr = passed_constraints / total_constraints if total_constraints > 0 else 0
    
    # ISR: 该prompt所有约束都满足
    isr = 1.0 if passed_constraints == total_constraints else 0.0
    
    # Rule-based指标
    rule_based_csr = ruled_based_passed / ruled_based_constraints if ruled_based_constraints > 0 else 0
    rule_based_isr = 1.0 if ruled_based_passed == ruled_based_constraints else 0.0
    
    # Open-ended指标
    open_ended_csr = open_ended_passed / open_ended_constraints if open_ended_constraints > 0 else 0
    open_ended_isr = 1.0 if open_ended_passed == open_ended_constraints else 0.0
    
    # FF指标
    ff_csr = ff_passed_constraints / ff_total_constraints if ff_total_constraints > 0 else 0
    ff_isr = 1.0 if ff_passed_constraints == ff_total_constraints else 0.0
    
    # FO指标
    fo_csr = fo_passed_constraints / fo_total_constraints if fo_total_constraints > 0 else 0
    fo_isr = 1.0 if fo_passed_constraints == fo_total_constraints else 0.0
    
    return {
        'isr': isr,
        'csr': csr,
        'rule_based_isr': rule_based_isr,
        'rule_based_csr': rule_based_csr,
        'open_ended_isr': open_ended_isr,
        'open_ended_csr': open_ended_csr,
        'ff_isr': ff_isr,
        'ff_csr': ff_csr,
        'fo_isr': fo_isr,
        'fo_csr': fo_csr
    }

def process_multiple_models(model_names: List[str], input_folder: str, output_folder: str):
    """处理多个模型并生成Excel文件"""
    
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 存储所有模型的结果
    all_model_results = []
    
    # 存储prompt级别的结果（按prompt为粒度）
    prompt_scores_by_model = defaultdict(list)
    
    for model_name in model_names:
        print(f"Processing model: {model_name}")
        
        if model_name == 'baseline':
            input_file = os.path.join(input_folder, "check_result.json")
        else:
            input_file = os.path.join(input_folder, f"{model_name}_check_result.json")

        # 检查文件是否存在
        if not os.path.exists(input_file):
            print(f"Warning: File not found - {input_file}")
            continue
        
        # 读取数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 计算分数
        calculator = ScoreCalculator(data)
        results = calculator.calculate_all_scores()
        
        # 转换为百分比并保留两位小数 - ISR在前，CSR在后
        results.isr = round(results.isr * 100, 2)
        results.csr = round(results.csr * 100, 2)
        results.rule_based_isr = round(results.rule_based_isr * 100, 2)
        results.rule_based_csr = round(results.rule_based_csr * 100, 2)
        results.open_ended_isr = round(results.open_ended_isr * 100, 2)
        results.open_ended_csr = round(results.open_ended_csr * 100, 2)
        results.ff_isr = round(results.ff_isr * 100, 2)
        results.ff_csr = round(results.ff_csr * 100, 2)
        results.fo_isr = round(results.fo_isr * 100, 2)
        results.fo_csr = round(results.fo_csr * 100, 2)
        
        for k in results.constraint_dimension_scores:
            results.constraint_dimension_scores[k] = round(results.constraint_dimension_scores[k] * 100, 2)
        
        # 视频级别的分数也转换为百分比
        for video_id, scores in results.video_scores.items():
            scores['ISR'] = round(scores['ISR'] * 100, 2)
            scores['CSR'] = round(scores['CSR'] * 100, 2)
            scores['Rule-based ISR'] = round(scores['Rule-based ISR'] * 100, 2)
            scores['Rule-based CSR'] = round(scores['Rule-based CSR'] * 100, 2)
            scores['Open-ended ISR'] = round(scores['Open-ended ISR'] * 100, 2)
            scores['Open-ended CSR'] = round(scores['Open-ended CSR'] * 100, 2)
            scores['FF-ISR'] = round(scores['FF-ISR'] * 100, 2)
            scores['FF-CSR'] = round(scores['FF-CSR'] * 100, 2)
            scores['FO-ISR'] = round(scores['FO-ISR'] * 100, 2)
            scores['FO-CSR'] = round(scores['FO-CSR'] * 100, 2)

        # 收集模型级别的结果 - ISR在前，CSR在后
        model_result = {
            'Model': model_name,
            'ISR': results.isr,
            'CSR': results.csr,
            'Rule-based ISR': results.rule_based_isr,
            'Rule-based CSR': results.rule_based_csr,
            'Open-ended ISR': results.open_ended_isr,
            'Open-ended CSR': results.open_ended_csr,
            'FF-ISR': results.ff_isr,
            'FF-CSR': results.ff_csr,
            'FO-ISR': results.fo_isr,
            'FO-CSR': results.fo_csr
        }
        
        # 添加约束维度得分
        model_result.update(results.constraint_dimension_scores)
        
        all_model_results.append(model_result)
        
        # 收集prompt级别的结果
        prompt_scores = []
        for video_id, case in data.items():
            for check_item in case:
                prompt_id = check_item.get('prompt_id', '')
                if prompt_id:
                    # 计算单个prompt的各项指标
                    prompt_score = calculate_prompt_scores(check_item)
                    prompt_score_entry = {
                        'Model': model_name,
                        'video_id': video_id,
                        'prompt_id': prompt_id,
                        'ISR': round(prompt_score['isr'] * 100, 2),
                        'CSR': round(prompt_score['csr'] * 100, 2),
                        'Rule-based ISR': round(prompt_score['rule_based_isr'] * 100, 2),
                        'Rule-based CSR': round(prompt_score['rule_based_csr'] * 100, 2),
                        'Open-ended ISR': round(prompt_score['open_ended_isr'] * 100, 2),
                        'Open-ended CSR': round(prompt_score['open_ended_csr'] * 100, 2),
                        'FF-ISR': round(prompt_score['ff_isr'] * 100, 2),
                        'FF-CSR': round(prompt_score['ff_csr'] * 100, 2),
                        'FO-ISR': round(prompt_score['fo_isr'] * 100, 2),
                        'FO-CSR': round(prompt_score['fo_csr'] * 100, 2)
                    }
                    prompt_scores.append(prompt_score_entry)
        
        prompt_scores_by_model[model_name].extend(prompt_scores)

    # 生成汇总的Excel文件
    if all_model_results:
        # 模型级别的指标汇总
        df_models = pd.DataFrame(all_model_results)
        
        # 按ISR降序排序
        df_models = df_models.sort_values('ISR', ascending=False)
        
        # 合并所有模型的prompt级别数据
        all_prompt_scores = []
        for model_name, scores in prompt_scores_by_model.items():
            all_prompt_scores.extend(scores)
        
        # 保存为Excel - 分成两个sheet
        metrics_excel_path = os.path.join(output_folder, "metrics.xlsx")
        with pd.ExcelWriter(metrics_excel_path, engine='openpyxl') as writer:
            # Sheet 1: 主要指标
            main_columns = ['Model', 'ISR', 'CSR', 'Rule-based ISR', 'Rule-based CSR', 
                          'Open-ended ISR', 'Open-ended CSR']
            available_main = [col for col in main_columns if col in df_models.columns]
            df_main = df_models[available_main]
            df_main.to_excel(writer, sheet_name='Main Metrics', index=False)
            
            # Sheet 2: Fact-Free和Fact-Only指标
            detailed_columns = ['Model', 'FF-ISR', 'FF-CSR', 'FO-ISR', 'FO-CSR']
            detailed_columns.extend([col for col in df_models.columns 
                                    if col not in main_columns and col not in detailed_columns])
            available_detailed = [col for col in detailed_columns if col in df_models.columns]
            df_detailed = df_models[available_detailed]
            df_detailed.to_excel(writer, sheet_name='Detailed Metrics', index=False)
            
            # Sheet 3: 所有指标（原来的完整表格）
            df_models.to_excel(writer, sheet_name='All Metrics', index=False)
            
            # Sheet 4: prompt级别的详细表格
            if all_prompt_scores:
                df_prompt_scores = pd.DataFrame(all_prompt_scores)
                df_prompt_scores = df_prompt_scores.sort_values(['Model', 'video_id', 'prompt_id'])
                df_prompt_scores.to_excel(writer, sheet_name='Prompt Detailed Scores', index=False)
            
            # 自动调整列宽
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"Model metrics Excel saved to: {metrics_excel_path}")
        
        # 生成LaTeX表格并保存到txt文件
        latex_table_path = os.path.join(output_folder, "metrics_latex_table.txt")
        generate_latex_table(df_models, latex_table_path)
        print(f"LaTeX tables saved to:")
        print(f"  - {latex_table_path}")
        print(f"  - {latex_table_path.replace('.txt', '_main.txt')}")
        print(f"  - {latex_table_path.replace('.txt', '_detailed.txt')}")
    else:
        df_models = None
    
    # 合并所有模型的prompt级别数据用于返回
    all_prompt_scores = []
    for model_name, scores in prompt_scores_by_model.items():
        all_prompt_scores.extend(scores)
    
    df_prompt_detailed = None
    if all_prompt_scores:
        df_prompt_detailed = pd.DataFrame(all_prompt_scores)
        df_prompt_detailed = df_prompt_detailed.sort_values(['Model', 'video_id', 'prompt_id'])
    
    return df_models, df_prompt_detailed

def generate_latex_table(df_models, output_file):
    """生成LaTeX表格格式并保存到txt文件（只包含&和\\）"""
    if df_models is None or df_models.empty:
        return
    
    latex_lines_main = []  # 主要指标表格
    latex_lines_detailed = []  # 详细指标表格
    
    # 表格1：主要指标 - ISR, CSR, Rule-based ISR, Rule-based CSR, Open-ended ISR, Open-ended CSR
    main_columns = [
        'Model', 'ISR', 'CSR', 
        'Rule-based ISR', 'Rule-based CSR',
        'Open-ended ISR', 'Open-ended CSR'
    ]
    
    # 表格2：Fact-Free和Fact-Only指标
    detailed_columns = [
        'Model',
        'FF-ISR', 'FF-CSR', 
        'FO-ISR', 'FO-CSR'
    ]
    
    # 生成主要指标表格
    available_main_columns = [col for col in main_columns if col in df_models.columns]
    df_main = df_models[available_main_columns]
    
    for _, row in df_main.iterrows():
        row_values = []
        for col in available_main_columns:
            value = row[col]
            if col == 'Model':
                row_values.append(str(value))
            else:
                row_values.append(f"{value:.2f}\\%")
        latex_line = " & ".join(row_values) + " \\\\"
        latex_lines_main.append(latex_line)
    
    # 生成详细指标表格
    available_detailed_columns = [col for col in detailed_columns if col in df_models.columns]
    df_detailed = df_models[available_detailed_columns]
    
    for _, row in df_detailed.iterrows():
        row_values = []
        for col in available_detailed_columns:
            value = row[col]
            if col == 'Model':
                row_values.append(str(value))
            else:
                row_values.append(f"{value:.2f}\\%")
        latex_line = " & ".join(row_values) + " \\\\"
        latex_lines_detailed.append(latex_line)
    
    # 保存到文件，两个表格用分隔符分开
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% Table 1: Main Metrics\n")
        f.write('\n'.join(latex_lines_main))
        f.write("\n\n% Table 2: Fact-Free and Fact-Only Metrics\n")
        f.write('\n'.join(latex_lines_detailed))
    
    # 同时保存为两个单独的文件
    main_table_file = output_file.replace('.txt', '_main.txt')
    detailed_table_file = output_file.replace('.txt', '_detailed.txt')
    
    with open(main_table_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines_main))
    
    with open(detailed_table_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(latex_lines_detailed))

def generate_report(results: ScoreResults, output_file: str = None):
    """生成评分报告"""
    report = []
    report.append("=" * 50)
    report.append("分数统计报告")
    report.append("=" * 50)
    
    # 宏观表现指标 - ISR在前，CSR在后
    report.append("\n## 宏观表现指标")
    report.append(f"指令满足率 (ISR): {results.isr:.2%}")
    report.append(f"约束满足率 (CSR): {results.csr:.2%}")
    
    # 规则检查指标
    report.append("\n## 规则检查指标")
    report.append(f"Rule-based 指令满足率 (Rule-based ISR): {results.rule_based_isr:.2%}")
    report.append(f"Rule-based 约束满足率 (Rule-based CSR): {results.rule_based_csr:.2%}")
    
    # 开放式检查指标
    report.append("\n## 开放式检查指标")
    report.append(f"Open-ended 指令满足率 (Open-ended ISR): {results.open_ended_isr:.2%}")
    report.append(f"Open-ended 约束满足率 (Open-ended CSR): {results.open_ended_csr:.2%}")
    
    # Fact-Free指标
    report.append("\n## Fact-Free指标（忽略correctness检查）")
    report.append(f"FF 指令满足率 (FF-ISR): {results.ff_isr:.2%}")
    report.append(f"FF 约束满足率 (FF-CSR): {results.ff_csr:.2%}")
    
    # Fact-Only指标
    report.append("\n## Fact-Only指标（仅关注correctness检查）")
    report.append(f"FO 指令满足率 (FO-ISR): {results.fo_isr:.2%}")
    report.append(f"FO 约束满足率 (FO-CSR): {results.fo_csr:.2%}")
    
    # 专项能力指标
    report.append("\n## 专项能力指标")
    report.append("约束维度得分:")
    for dimension, score in results.constraint_dimension_scores.items():
        report.append(f"  - {dimension}: {score:.2%}")
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
    else:
        print(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description='计算评测分数')
    parser.add_argument('--models', type=str, nargs='+', required=True, 
                        help='模型名称列表，空格分隔。使用"all"时自动识别所有模型')
    parser.add_argument('--part', type=str, required=False, default=None,
                        help='数据集部分: easy, hard, 或 breeze')
    
    args = parser.parse_args()
    
    # 固定的根目录配置
    if args.models == ['baseline']:
        input_root = 'annotation'  # 输入根目录
    else:
        input_root = 'check_result'
        
    output_root = 'metrics'    # 输出根目录
    
    # 根据part配置输入和输出文件夹
    if args.part is not None:
        input_folder = os.path.join(input_root, f'{args.part}')
        output_folder = os.path.join(output_root, f'{args.part}')
    else:
        input_folder = input_root
        output_folder = output_root

    # 自动识别模型名称逻辑
    if args.models == ['all']:
        print(f"自动识别模型中，扫描目录: {input_folder}")
        
        # 确保输入文件夹存在
        if not os.path.exists(input_folder):
            print(f"Error: Input folder not found - {input_folder}")
            return
        
        # 扫描所有_check_result.json文件
        model_names = []
        
        # 添加baseline模型（如果存在check_result.json）
        baseline_file = os.path.join(input_folder, "check_result.json")
        if os.path.exists(baseline_file):
            model_names.append('baseline')
        
        # 扫描所有*_check_result.json文件
        for filename in os.listdir(input_folder):
            if filename.endswith('_check_result.json') and filename != 'check_result.json':
                # 提取模型名称（去掉_check_result.json后缀）
                model_name = filename[:-len('_check_result.json')]
                model_names.append(model_name)
        
        if not model_names:
            print(f"Warning: No check result files found in {input_folder}")
            return
        
        # 按字母顺序排序模型名称
        model_names.sort()
        print(f"发现 {len(model_names)} 个模型: {', '.join(model_names)}")
        
        # 更新args.models
        args.models = model_names

    print(f"Processing part: {args.part}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Models to process: {args.models}")
    
    # 处理多个模型
    df_models, df_prompt_detailed = process_multiple_models(
        args.models, 
        input_folder, 
        output_folder
    )
    
    # 显示汇总结果
    if df_models is not None:
        print("\n模型级别指标汇总:")
        print(df_models.to_string(index=False))
        
        print("\nPrompt级别详细指标样例 (前10行):")
        if df_prompt_detailed is not None and not df_prompt_detailed.empty:
            # 显示前10行prompt级别的详细数据
            print(df_prompt_detailed.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
    
    '''
    使用示例：
    
    # 处理baseline模型
    python metrics.py --models baseline --part normal
    python metrics.py --models baseline --part hard
    python metrics.py --models baseline --part human_fix_v1
    
    python metrics.py --models humanfix_baseline

    # 处理单个模型
    python metrics.py --models gemini-2.0-flash --part normal
    
    # 处理多个模型
    python metrics.py --models Qwen2.5-VL-7B-Instruct Qwen2.5-VL-32B-Instruct Qwen2.5-VL-72B-Instruct --part normal
    
    python metrics.py --models gemini-2.5-pro gemini-2.5-flash gemini-2.0-flash Qwen2.5-VL-72B-Instruct Qwen2.5-VL-32B-Instruct Qwen2.5-VL-7B-Instruct
    
    # 自动识别所有模型
    python metrics.py --models all --part normal
    python metrics.py --models all
    '''

