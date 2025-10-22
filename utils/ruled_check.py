import re
import json
import jsonschema
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum


class CheckerType(Enum):
    """检查器类型枚举"""
    PLAIN_TEXT = "plain_text"
    JSON_OBJECT = "json_object"
    JSON_ARRAY = "json_array"
    UNORDERED_LIST = "unordered_list"
    ORDERED_LIST = "ordered_list"
    TABLE = "table"
    KEYWORD = "keyword"
    MARKDOWN = "markdown"
    PREFIX_SUFFIX = "prefix_suffix"
    DELIMITER = "delimiter"
    LENGTH = "length"
    COUNT = "count"
    CASE = "case"
    LANGUAGE = "language"


class BaseChecker(ABC):
    """基础检查器抽象类"""
    
    @abstractmethod
    def check(self, content: str, **kwargs) -> bool:
        """执行检查的抽象方法"""
        pass


class PlainTextChecker(BaseChecker):
    """纯文本检查器"""
    
    def check(self, content: str, **kwargs) -> bool:
        """检查是否为纯文本（不包含特殊结构）"""
        # 检查是否包含JSON结构
        try:
            json.loads(content)
            return False
        except json.JSONDecodeError:
            pass
        
        # 检查是否包含列表结构
        list_patterns = [
            r'^\s*[-*+•]\s+',  # 无序列表
            r'^\s*\d+[\.\)]\s+',  # 数字有序列表
            r'^\s*[a-zA-Z][\.\)]\s+',  # 字母有序列表
            r'^\s*[一二三四五六七八九十]+[\.\、]\s+',  # 中文数字列表
            r'^\s*[IVXLCDM]+[\.\)]\s+'  # 罗马数字列表
        ]
        
        lines = content.split('\n')
        for line in lines:
            for pattern in list_patterns:
                if re.match(pattern, line):
                    return False
        
        # 检查是否包含Markdown表格
        if '|' in content and re.search(r'\|[-:\s]+\|', content):
            return False
        
        return True


class JSONChecker(BaseChecker):
    """JSON检查器基类"""
    
    def _validate_json(self, content: str, schema: Dict[str, Any], expected_type: str) -> bool:
        """验证JSON是否符合schema"""
        try:
            data = json.loads(content)
            
            # 检查JSON类型
            if expected_type == "object" and not isinstance(data, dict):
                return False
            elif expected_type == "array" and not isinstance(data, list):
                return False
            
            # 使用jsonschema验证
            jsonschema.validate(instance=data, schema=schema)
            return True
        except (json.JSONDecodeError, jsonschema.exceptions.ValidationError):
            return False


class JSONObjectChecker(JSONChecker):
    """JSON对象检查器"""
    
    def check(self, content: str, schema: Dict[str, Any], **kwargs) -> bool:
        first_brace = content.find('{')
        last_brace = content.rfind('}')
        if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
            return False
        content = content[first_brace:last_brace + 1]
        return self._validate_json(content, schema, "object")


class JSONArrayChecker(JSONChecker):
    """JSON数组检查器"""
    
    def check(self, content: str, schema: Dict[str, Any], **kwargs) -> bool:
        first_bracket = content.find('[')
        last_bracket = content.rfind(']')
        if first_bracket == -1 or last_bracket == -1 or last_bracket < first_bracket:
            return False
        content = content[first_bracket:last_bracket + 1]
        return self._validate_json(content, schema, "array")


class ListChecker(BaseChecker):
    """列表检查器基类"""
    
    def _check_list_format(self, content: str, patterns: List[str], symbol: Optional[str] = None) -> bool:
        """检查列表格式"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return False
        
        if symbol:
            # 如果指定了symbol，构建特定的pattern
            escaped_symbol = re.escape(symbol)
            specific_patterns = [f'^{escaped_symbol}\\s+']
        else:
            specific_patterns = patterns
        
        # 检查每一行是否符合列表格式
        for line in lines:
            matched = False
            for pattern in specific_patterns:
                if re.match(pattern, line):
                    matched = True
                    break
            if not matched:
                return False
        
        return True


class UnorderedListChecker(ListChecker):
    """无序列表检查器"""
    
    def check(self, content: str, symbol: Optional[str] = None, **kwargs) -> bool:
        default_patterns = [r'^[-*+•]\s+']
        return self._check_list_format(content, default_patterns, symbol)


class OrderedListChecker(ListChecker):
    """有序列表检查器"""
    
    def check(self, content: str, symbol: Optional[str] = None, **kwargs) -> bool:
        if symbol:
            # 如果指定了symbol，检查是否以该符号开始且序号有序
            return self._check_ordered_with_symbol(content, symbol)
        else:
            # 对于没有指定symbol的情况，需要检查所有行是否使用同一种编号系统且有序
            return self._check_consistent_numbering(content)
    
    def _check_ordered_with_symbol(self, content: str, symbol: str) -> bool:
        """检查是否以指定symbol开始且序号有序"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return False
        
        # 分析symbol的结构
        sequence_type, separator = self._analyze_symbol(symbol)
        if not sequence_type or not separator:
            return False
        
        # 提取第一行的序号，应该与symbol匹配
        first_number = self._extract_sequence_number(lines[0], sequence_type, separator)
        symbol_number = self._extract_sequence_number(symbol, sequence_type, separator, require_space=False)
        
        if first_number != symbol_number:
            return False
        
        # 检查所有行的序号是否连续有序
        return self._check_sequence_order(lines, sequence_type, separator, first_number)
    
    def _check_consistent_numbering(self, content: str) -> bool:
        """检查是否使用一致的编号系统且序号有序"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return False
        
        # 定义所有支持的编号模式
        pattern_groups = [
            ([r'^\d+[\.\)\:]\s+'], 'arabic'),                           # 阿拉伯数字
            ([r'^[A-Z][\.\)]\s+'], 'upper_alpha'),                      # 大写字母
            ([r'^[a-z][\.\)]\s+'], 'lower_alpha'),                      # 小写字母
            ([r'^[一二三四五六七八九十]+[\.\、]\s+'], 'chinese'),           # 中文数字
            ([r'^[IVXLCDM]+[\.\)]\s+'], 'upper_roman'),                 # 大写罗马数字
            ([r'^[ivxlcdm]+[\.\)]\s+'], 'lower_roman'),                 # 小写罗马数字
        ]
        
        # 尝试每一种编号系统
        for patterns, pattern_type in pattern_groups:
            if self._check_format_and_order(content, patterns, pattern_type):
                return True
        
        return False
    
    def _check_format_and_order(self, content: str, patterns: List[str], pattern_type: str) -> bool:
        """检查格式并验证序号顺序"""
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return False
        
        # 首先检查格式
        if not self._check_list_format(content, patterns, None):
            return False
        
        # 然后检查序号顺序
        # 从第一行推断分隔符
        separator = self._infer_separator_from_line(lines[0])
        if not separator:
            return False
        
        # 提取第一个序号
        first_number = self._extract_sequence_number(lines[0], pattern_type, separator)
        if first_number is None:
            return False
        
        return self._check_sequence_order(lines, pattern_type, separator, first_number)
    
    def _get_patterns_from_symbol(self, symbol: str) -> List[str]:
        """从symbol中提取匹配模式"""
        symbol_stripped = symbol.strip()
        if not symbol_stripped:
            return []
        
        # 分析symbol的结构
        sequence_type, separator = self._analyze_symbol(symbol_stripped)
        if not sequence_type or not separator:
            return []
        
        # 生成对应的正则模式
        return self._generate_patterns_for_type(sequence_type, separator)
    
    def _analyze_symbol(self, symbol: str) -> Tuple[Optional[str], Optional[str]]:
        """分析symbol，返回(序号类型, 分隔符)"""
        # 支持的分隔符
        separators = {'.', ')', '、', ':', '．'}  # 包含全角点号
        
        # 查找分隔符
        separator = None
        for i in range(len(symbol) - 1, -1, -1):
            if symbol[i] in separators:
                separator = symbol[i]
                sequence_part = symbol[:i].strip()
                break
        
        if not separator:
            return None, None
        
        # 识别序号类型
        sequence_type = self._identify_sequence_type(sequence_part)
        return sequence_type, separator
    
    def _identify_sequence_type(self, text: str) -> Optional[str]:
        """识别序号类型"""
        if not text:
            return None
        
        # 1. 阿拉伯数字
        if text.isdigit():
            return 'arabic'
        
        # 2. 中文数字
        if re.fullmatch(r'[一二三四五六七八九十]+', text):
            return 'chinese'
        
        # 3. 检查是否为有效的罗马数字
        if self._is_valid_roman_numeral(text):
            if text.isupper():
                return 'upper_roman'
            else:
                return 'lower_roman'
        
        # 4. 大写字母 A-Z (单个字母)
        if len(text) == 1 and text.isupper() and text.isalpha():
            return 'upper_alpha'
        
        # 5. 小写字母 a-z (单个字母)
        if len(text) == 1 and text.islower() and text.isalpha():
            return 'lower_alpha'
        
        return None
    
    def _is_valid_roman_numeral(self, text: str) -> bool:
        """检查是否为有效的罗马数字"""
        if not text or not re.fullmatch(r'[IVXLCDMivxlcdm]+', text):
            return False
        
        # 定义有效的罗马数字序列（1-20）
        valid_upper_romans = [
            'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
            'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX'
        ]
        valid_lower_romans = [r.lower() for r in valid_upper_romans]
        
        return text in valid_upper_romans or text in valid_lower_romans
    
    def _generate_patterns_for_type(self, sequence_type: str, separator: str) -> List[str]:
        """为特定类型生成正则表达式模式"""
        escaped_separator = re.escape(separator)
        
        patterns_map = {
            # 1. 阿拉伯数字：1. 2. 3. ...
            'arabic': [r'^\d+' + escaped_separator + r'\s+'],
            
            # A. B. C. ... Z.
            'upper_alpha': [r'^[A-Z]' + escaped_separator + r'\s+'],
            
            # a. b. c. ... z.
            'lower_alpha': [r'^[a-z]' + escaped_separator + r'\s+'],
            
            # I. II. III. IV. ...
            'upper_roman': [r'^[IVXLCDM]+' + escaped_separator + r'\s+'],
            
            # i. ii. iii. iv. ...
            'lower_roman': [r'^[ivxlcdm]+' + escaped_separator + r'\s+'],
            
            # 一、二、三、...
            'chinese': [r'^[一二三四五六七八九十]+' + escaped_separator + r'\s+'],
        }
        
        return patterns_map.get(sequence_type, [])
    
    def _infer_separator_from_line(self, line: str) -> Optional[str]:
        """从行中推断分隔符"""
        separators = {'.', ')', '、', ':', '．'}
        for char in line:
            if char in separators:
                return char
        return None
    
    def _extract_sequence_number(self, text: str, sequence_type: str, separator: str, require_space: bool = True) -> Optional[Union[int, str]]:
        """从文本中提取序号，同时验证格式"""
        # 查找分隔符位置
        sep_pos = text.find(separator)
        if sep_pos == -1:
            return None
        
        sequence_part = text[:sep_pos].strip()
        if not sequence_part:
            return None
        
        # 检查分隔符后面是否有空格（只在require_space为True时检查）
        if require_space:
            after_separator = text[sep_pos + len(separator):]
            if not after_separator or not after_separator[0].isspace():
                return None  # 分隔符后面必须有空格
        
        if sequence_type == 'arabic':
            try:
                return int(sequence_part)
            except ValueError:
                return None
        elif sequence_type in ['upper_alpha', 'lower_alpha']:
            if len(sequence_part) == 1 and sequence_part.isalpha():
                return sequence_part
            return None
        elif sequence_type in ['upper_roman', 'lower_roman']:
            return sequence_part  # 罗马数字作为字符串返回
        elif sequence_type == 'chinese':
            return sequence_part  # 中文数字作为字符串返回
        return None
    
    def _check_sequence_order(self, lines: List[str], sequence_type: str, separator: str, start_number: Union[int, str]) -> bool:
        """检查序号是否按顺序递增"""
        if sequence_type == 'arabic':
            return self._check_arabic_order(lines, separator, int(start_number))
        elif sequence_type == 'upper_alpha':
            return self._check_alpha_order(lines, separator, start_number, True)
        elif sequence_type == 'lower_alpha':
            return self._check_alpha_order(lines, separator, start_number, False)
        elif sequence_type == 'upper_roman':
            return self._check_roman_order(lines, separator, start_number, True)
        elif sequence_type == 'lower_roman':
            return self._check_roman_order(lines, separator, start_number, False)
        elif sequence_type == 'chinese':
            return self._check_chinese_order(lines, separator, start_number)
        return False
    
    def _check_arabic_order(self, lines: List[str], separator: str, start_num: int) -> bool:
        """检查阿拉伯数字序号顺序"""
        expected = start_num
        for line in lines:
            current = self._extract_sequence_number(line, 'arabic', separator)
            if current != expected:
                return False
            expected += 1
        return True
    
    def _check_alpha_order(self, lines: List[str], separator: str, start_char: str, is_upper: bool) -> bool:
        """检查字母序号顺序"""
        if is_upper:
            start_ord = ord(start_char.upper())
            sequence_type = 'upper_alpha'
        else:
            start_ord = ord(start_char.lower())
            sequence_type = 'lower_alpha'
        
        expected_ord = start_ord
        for line in lines:
            current = self._extract_sequence_number(line, sequence_type, separator)
            if not current or ord(current) != expected_ord:
                return False
            expected_ord += 1
            if expected_ord > ord('Z') and is_upper:
                return False  # 超出字母范围
            if expected_ord > ord('z') and not is_upper:
                return False  # 超出字母范围
        return True
    
    def _check_roman_order(self, lines: List[str], separator: str, start_roman: str, is_upper: bool) -> bool:
        """检查罗马数字序号顺序"""
        # 简化的罗马数字顺序检查
        roman_sequence = self._get_roman_sequence(is_upper)
        try:
            start_index = roman_sequence.index(start_roman)
        except ValueError:
            return False
        
        sequence_type = 'upper_roman' if is_upper else 'lower_roman'
        expected_index = start_index
        for line in lines:
            current = self._extract_sequence_number(line, sequence_type, separator)
            if expected_index >= len(roman_sequence) or current != roman_sequence[expected_index]:
                return False
            expected_index += 1
        return True
    
    def _check_chinese_order(self, lines: List[str], separator: str, start_chinese: str) -> bool:
        """检查中文数字序号顺序"""
        chinese_sequence = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
        try:
            start_index = chinese_sequence.index(start_chinese)
        except ValueError:
            return False
        
        expected_index = start_index
        for line in lines:
            current = self._extract_sequence_number(line, 'chinese', separator)
            if expected_index >= len(chinese_sequence) or current != chinese_sequence[expected_index]:
                return False
            expected_index += 1
        return True
    
    def _get_roman_sequence(self, is_upper: bool) -> List[str]:
        """获取罗马数字序列"""
        if is_upper:
            return ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
                    'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX']
        else:
            return ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
                    'xi', 'xii', 'xiii', 'xiv', 'xv', 'xvi', 'xvii', 'xviii', 'xix', 'xx']


class TableChecker(BaseChecker):
    """Markdown表格检查器"""
    
    def _clean_markdown_formatting(self, text: str) -> str:
        """清理Markdown修饰符，保留纯文本内容"""
        # 移除加粗：**text** 或 __text__
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        
        # 移除斜体：*text* 或 _text_（但要避免与加粗冲突）
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'\1', text)
        text = re.sub(r'(?<!_)_([^_]+?)_(?!_)', r'\1', text)
        
        # 移除高亮：==text==
        text = re.sub(r'==(.*?)==', r'\1', text)
        
        # 移除行内代码：`text`
        text = re.sub(r'`(.*?)`', r'\1', text)
        
        # 移除删除线：~~text~~
        text = re.sub(r'~~(.*?)~~', r'\1', text)
        
        return text.strip()
    
    def check(self, content: str, col_name: List[str], **kwargs) -> bool:
        lines = content.strip().split('\n')
        if len(lines) < 2:  # 至少需要表头和分隔行
            return False
        
        # 检查是否为Markdown表格格式
        if not all('|' in line for line in lines[:2]):
            return False
        
        # 检查分隔行
        if not re.match(r'^[\s\|:\-]+$', lines[1]):
            return False
        
        # 提取表头并清理Markdown修饰符
        header_cells = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
        cleaned_header_cells = [self._clean_markdown_formatting(cell) for cell in header_cells]

        col_name = [self._clean_markdown_formatting(cell) for cell in col_name]

        # 检查列名是否匹配（比较清理后的版本）
        return cleaned_header_cells == col_name

class KeywordChecker(BaseChecker):
    """关键词检查器"""
    
    def check(self, content: str, keyword: str, keyword_type: str, **kwargs) -> bool:
        """检查内容是否包含所有关键词"""
        if not keyword:
            return True  # 如果没有关键词，直接返回True

        content = content.lower()
        keyword = keyword.lower()
        
        if keyword_type == "include":
            return keyword in content
        elif keyword_type == "exclude":
            return keyword not in content
        return False


class MarkdownChecker(BaseChecker):
    """Markdown样式检查器"""
    
    MARKDOWN_PATTERNS = {
        'title': [r'^#{1,6}\s+.+$'],
        'bold': [r'\*\*.+\*\*', r'__.+__'],
        'highlight': [r'==.+==', r'`.+`'],
        'italic': [r'\*.+\*', r'_.+_'],
        'code': [r'```[\s\S]*```', r'`.+`']
    }
    
    def check(self, content: str, md_type: str, **kwargs) -> bool:
        if md_type not in self.MARKDOWN_PATTERNS:
            return False

        patterns = self.MARKDOWN_PATTERNS[md_type]
        for pattern in patterns:
            if re.search(pattern, content, re.MULTILINE):
                return True
        return False


class PrefixSuffixChecker(BaseChecker):
    """前后缀检查器"""
    
    def check(self, content: str, prefix: Optional[str] = None, suffix: Optional[str] = None, **kwargs) -> bool:
        if prefix and not content.startswith(prefix):
            return False
        if suffix:
            # 允许后缀后面跟标点符号
            import string
            # 定义常见的标点符号（包括中英文标点）
            punctuation = string.punctuation + '，。！？；：、''""（）【】《》〈〉·'
            
            # 检查是否以后缀结尾
            if content.endswith(suffix):
                return True
            
            # 检查是否以后缀+标点符号结尾
            for i in range(len(content) - 1, -1, -1):
                if content[i] in punctuation:
                    continue
                else:
                    # 找到第一个非标点符号的位置
                    potential_suffix_end = i + 1
                    if content[:potential_suffix_end].endswith(suffix):
                        return True
                    break
            return False
        return True


class DelimiterChecker(BaseChecker):
    """分隔符检查器"""
    
    def check(self, content: str, symbol: str, **kwargs) -> bool:
        """
        检查是否使用了指定的分隔符进行分隔：
        1. 检查内容中是否包含指定的分隔符
        2. 确保分隔符前后都有内容（真正起到分隔作用）
        """
        # 检查内容中是否包含指定的分隔符
        if symbol not in content:
            return False
        
        # 验证分隔符是否真正起到分隔作用
        parts = content.split(symbol)
        # 检查分割后是否至少有2个非空部分
        non_empty_parts = [part.strip() for part in parts if part.strip()]
        return len(non_empty_parts) >= 2


class LengthChecker(BaseChecker):
    """长度检查器"""
    
    def _remove_list_prefixes(self, content: str) -> str:
        """去除有序/无序列表的前缀"""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 定义列表前缀的正则模式
            list_patterns = [
                r'^\s*[-*+•]\s+',  # 无序列表：- * + •
                r'^\s*\d+[\.\)]\s+',  # 数字有序列表：1. 2) 等
                r'^\s*[a-zA-Z][\.\)]\s+',  # 字母有序列表：A. b) 等
                r'^\s*[一二三四五六七八九十]+[\.\、]\s+',  # 中文数字列表：一、二、等
                r'^\s*[IVXLCDM]+[\.\)]\s+',  # 大写罗马数字：I. II) 等
                r'^\s*[ivxlcdm]+[\.\)]\s+'  # 小写罗马数字：i. ii) 等
            ]
            
            # 检查是否匹配任何列表模式，如果匹配就去除前缀
            cleaned_line = line
            for pattern in list_patterns:
                match = re.match(pattern, line)
                if match:
                    # 去除匹配的前缀，保留剩余内容
                    cleaned_line = line[match.end():]
                    break
            
            cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def check(self, content: str, unit: str, min_len: int = 0, max_len: int = -1, **kwargs) -> bool:
        # 在检查长度之前，先去除列表前缀
        cleaned_content = self._remove_list_prefixes(content)
        
        if unit == "word":
            # 中英文混合的词数统计
            chinese_words = len(re.findall(r'[\u4e00-\u9fa5]', cleaned_content))
            english_words = len(re.findall(r'\b[a-zA-Z]+(?:-[a-zA-Z]+)*\b', cleaned_content))
            count = chinese_words + english_words
        elif unit == "sentence":
            # 句子数统计
            count = len(re.findall(r'[.!?。！？]+', cleaned_content))
        elif unit == "paragraph":
            # 段落数统计
            paragraphs = [p.strip() for p in cleaned_content.split('\n\n') if p.strip()]
            count = len(paragraphs)
        elif unit == "character":
            # 字符数统计
            count = len(cleaned_content.replace(" ", ""))
        else:
            return False

        if count < min_len:
            return False
        if max_len > 0 and count > max_len:
            return False
        return True

class CountChecker(BaseChecker):
    """计数检查器"""
    
    def check(self, content: str, min_count: int = 0, max_count: int = -1, **kwargs) -> bool:
        # 统计括号对的个数
        # 使用正则表达式匹配形如 (xxx) 的模式
        import re
        parentheses_pattern = r'\([^)]*\)'
        matches = re.findall(parentheses_pattern, content)
        count = len(matches)
        
        if count < min_count:
            return False
        if max_count > 0 and count > max_count:
            return False
        return True

class CaseChecker(BaseChecker):
    """大小写检查器"""
    
    def check(self, content: str, case_type: str, **kwargs) -> bool:
        # 只检查英文字符
        english_chars = re.findall(r'\b[a-zA-Z]+\b', content)
        if not english_chars:
            return True
        
        english_text = ' '.join(english_chars)
        
        if case_type == "upper":
            return english_text.isupper()
        elif case_type == "lower":
            return english_text.islower()
        elif case_type == "title":
            upper = 0
            # 允许缩写（全大写单词）通过
            for word in english_chars:
                if word.isupper():
                    upper += 1
                    continue
                if len(word) > 1 and word[0].isupper() and word[1:].islower():
                    continue
                if len(word) == 1 and word.isupper():
                    continue
                return False
            if upper == len(english_chars):
                return False
            return True
        return False


class LanguageChecker(BaseChecker):
    """语言检查器"""
    
    def check(self, content: str, lang_type: str, **kwargs) -> bool:
        # 只提取英文字母和中文字符，忽略数字和标点符号
        english_chars = len(re.findall(r'[a-zA-Z]', content))
        chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', content))
        
        # 如果没有任何语言字符，返回False
        if english_chars == 0 and chinese_chars == 0:
            return False
        
        if lang_type == "en":
            # 检查是否全部为英文（没有中文字符）
            return chinese_chars == 0 and english_chars > 0
        elif lang_type == "zh":
            # 检查是否全部为中文（没有英文字符）
            return english_chars == 0 and chinese_chars > 0
        return False


class RuledCheckModule:
    """规则检查模块主类"""
    
    def __init__(self):
        self._checkers = {
            CheckerType.PLAIN_TEXT: PlainTextChecker(),
            CheckerType.JSON_OBJECT: JSONObjectChecker(),
            CheckerType.JSON_ARRAY: JSONArrayChecker(),
            CheckerType.UNORDERED_LIST: UnorderedListChecker(),
            CheckerType.ORDERED_LIST: OrderedListChecker(),
            CheckerType.TABLE: TableChecker(),
            CheckerType.KEYWORD: KeywordChecker(),
            CheckerType.MARKDOWN: MarkdownChecker(),
            CheckerType.PREFIX_SUFFIX: PrefixSuffixChecker(),
            CheckerType.DELIMITER: DelimiterChecker(),
            CheckerType.LENGTH: LengthChecker(),
            CheckerType.COUNT: CountChecker(),
            CheckerType.CASE: CaseChecker(),
            CheckerType.LANGUAGE: LanguageChecker()
        }
    
    def plain_text(self, content: str) -> bool:
        return self._checkers[CheckerType.PLAIN_TEXT].check(content)
    
    def json_object(self, content: str, schema: Dict[str, Any]) -> bool:
        return self._checkers[CheckerType.JSON_OBJECT].check(content, schema=schema)
    
    def json_array(self, content: str, schema: Dict[str, Any]) -> bool:
        return self._checkers[CheckerType.JSON_ARRAY].check(content, schema=schema)
    
    def unordered_list(self, content: str, symbol: Optional[str] = None) -> bool:
        return self._checkers[CheckerType.UNORDERED_LIST].check(content, symbol=symbol)
    
    def ordered_list(self, content: str, symbol: Optional[str] = None) -> bool:
        return self._checkers[CheckerType.ORDERED_LIST].check(content, symbol=symbol)
    
    def table(self, content: str, col_name: List[str]) -> bool:
        return self._checkers[CheckerType.TABLE].check(content, col_name=col_name)
    
    def keyword(self, content: str, keyword: str, keyword_type: str) -> bool:
        return self._checkers[CheckerType.KEYWORD].check(content, keyword=keyword, keyword_type=keyword_type)
    
    def markdown(self, content: str, md_type: str) -> bool:
        return self._checkers[CheckerType.MARKDOWN].check(content, md_type=md_type)
    
    def prefix_suffix(self, content: str, prefix: Optional[str] = None, suffix: Optional[str] = None) -> bool:
        return self._checkers[CheckerType.PREFIX_SUFFIX].check(content, prefix=prefix, suffix=suffix)
    
    def delimiter(self, content: str, symbol: str) -> bool:
        return self._checkers[CheckerType.DELIMITER].check(content, symbol=symbol)
    
    def length(self, content: str, unit: str, min_len: int = 0, max_len: int = -1) -> bool:
        return self._checkers[CheckerType.LENGTH].check(content, unit=unit, min_len=min_len, max_len=max_len)

    def count(self, content: str, min_count: int = 0, max_count: int = -1) -> bool:
        return self._checkers[CheckerType.COUNT].check(content, min_count=min_count, max_count=max_count)

    def case(self, content: str, case_type: str) -> bool:
        return self._checkers[CheckerType.CASE].check(content, case_type=case_type)
    
    def language(self, content: str, lang_type: str) -> bool:
        return self._checkers[CheckerType.LANGUAGE].check(content, lang_type=lang_type)