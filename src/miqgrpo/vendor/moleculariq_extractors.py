"""VENDORED FILE -- DO NOT EDIT BY HAND.

Source : https://github.com/ml-jku/moleculariq-eval
Path   : lm_eval/tasks/moleculariq/extractors.py
Commit : 425ecaaa8faf65aa43aa60ec0f584b7b7f060063
Vendored: 2026-09-22

Why vendored
------------
The official MolecularIQ benchmark extracts a model's answer from raw generated
text with exactly this code before handing it to moleculariq_core.evaluate_answer.
The GRPO reward must use the *same* extraction so that training reward and the
held-out benchmark metric measure the same thing.

The upstream module depends only on the Python standard library, so it is copied
verbatim instead of taking a dependency on the whole lm-evaluation-harness fork
(which is not installed in the training environment).

Re-sync with: python -m miqgrpo.vendor.sync --check
"""

"""
Model-specific extraction functions for MolecularIQ chemistry tasks.
"""
import ast
import json
import re
from typing import Optional, Union

# Ether0 model specific tokens
ETHER0_THINK_START = "<|think_start|>"
ETHER0_THINK_END = "<|think_end|>"
ETHER0_ANSWER_START = "<|answer_start|>"
ETHER0_ANSWER_END = "<|answer_end|>"

# Ether0 strict regex patterns for XML-style answer extraction
ETHER0_STRICT_XML_PATTERNS = {
    True: re.compile(
        rf"^\s?{re.escape(ETHER0_THINK_START)}\s*([\s\S]*?)\s*{re.escape(ETHER0_THINK_END)}([\s\S]*?){re.escape(ETHER0_ANSWER_START)}\s*([\s\S]*?)\s*{re.escape(ETHER0_ANSWER_END)}$"
    ),
    False: re.compile(
        rf"^\s?{re.escape(ETHER0_ANSWER_START)}\s*(\S[\s\S]*?)\s*{re.escape(ETHER0_ANSWER_END)}$"
    ),
}


def extract_ether0_answer(response):
    """
    Extract answer specifically for ether0 models using their answer tags.
    Priority order:
    1. <|answer_start|>...<|answer_end|> tags (ether0 specific)
    2. <answer>...</answer> tags (general)
    3. Fallback to general extraction
    """
    # Priority 1: Check for ether0-specific answer tags ANYWHERE in response
    if ETHER0_ANSWER_START in response and ETHER0_ANSWER_END in response:
        answer = response.split(ETHER0_ANSWER_START)[-1].split(ETHER0_ANSWER_END)[0].strip()
        return answer

    # Priority 2: Check for general answer tags
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer

    # Priority 3: Fallback to general extraction
    return extract_general_answer(response)


def extract_general_answer(response):
    """
    Simplified answer extraction for chemistry CoT traces and cleanup.
    """
    # Handle Qwen3 thinking mode
    if "</think>" in response:
        content = response.split("</think>")[-1].strip()
    elif "<|think_end|>" in response:
        content = response.split("<|think_end|>")[-1].strip()
    else:
        content = response.strip()

    answer = None

    if "<answer>" in content and "</answer>" in content:
        answer = content.split("<answer>")[-1].split("</answer>")[0].strip()
    elif "<|answer_start|>" in content and "<|answer_end|>" in content:
        answer = content.split("<|answer_start|>")[-1].split("<|answer_end|>")[0].strip()
    else:
        # Check for boxed or bold answers
        answer = extract_last_formatted_answer(content)

    if answer is not None:
        cleaned = clean_extracted_answer(answer)
        if cleaned is not None:
            return convert_to_appropriate_type(cleaned)

    # Fallback: try other patterns
    return extract_fallback_patterns(content)


def extract_last_formatted_answer(content):
    """
    Extract the last formatted answer from the content.
    Looks for **content**, \\boxed{content}, and (content) patterns.
    """
    all_matches = []

    # Find positions of bold matches **content**
    for match in re.finditer(r'\*\*(.+?)\*\*', content):
        all_matches.append((match.start(), match.group(1)))

    # Find positions of boxed matches \\boxed{content}
    pattern = r'\\boxed\{'
    for match in re.finditer(pattern, content):
        start = match.end()
        brace_count = 1
        end = start

        while end < len(content) and brace_count > 0:
            if content[end] == '{':
                brace_count += 1
            elif content[end] == '}':
                brace_count -= 1
            end += 1

        if brace_count == 0:
            extracted = content[start:end-1]
            all_matches.append((match.start(), extracted))

    if all_matches:
        all_matches.sort(key=lambda x: x[0])
        last_answer = all_matches[-1][1].strip()
        return last_answer

    return None


def extract_fallback_patterns(content):
    """Fallback extraction methods."""
    # Content in quotes
    quoted = re.findall(r'"([^"]+)"', content)
    if quoted:
        return clean_extracted_answer(quoted[-1])

    single_quoted = re.findall(r"'([^']+)'", content)
    if single_quoted:
        return clean_extracted_answer(single_quoted[-1])

    # Numbers with units
    numbers_with_units = re.findall(r'([0-9.]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)', content)
    if numbers_with_units:
        return f"{numbers_with_units[-1][0]} {numbers_with_units[-1][1]}"

    # Just numbers
    numbers = re.findall(r'([0-9.]+(?:\.[0-9]+)?)', content)
    if numbers:
        return numbers[-1]

    return None


def remove_latex_commands(text):
    """Remove LaTeX commands like \\text{}, \\mathrm{}, \\mathbf{}."""
    if not text:
        return text

    latex_commands = ['text', 'mathrm', 'mathbf']
    result = text

    for cmd in latex_commands:
        pattern = rf'\\{cmd}\{{'

        while True:
            match = re.search(pattern, result)
            if not match:
                break

            start = match.end()
            brace_count = 1
            end = start

            while end < len(result) and brace_count > 0:
                if result[end] == '{':
                    brace_count += 1
                elif result[end] == '}':
                    brace_count -= 1
                end += 1

            if brace_count == 0:
                before = result[:match.start()]
                content = result[start:end-1]
                after = result[end:]
                result = before + content + after
            else:
                break

    return result


def clean_extracted_answer(answer):
    """Clean extracted answers."""
    if not answer:
        return None

    answer = answer.strip()

    # Remove common artifacts
    answer = re.sub(r'^\$+|\$+$', '', answer)
    answer = re.sub(r'^\[|\]$', '', answer)
    answer = re.sub(r'[.,;:]+$', '', answer)

    if not (answer.startswith('\\text{') or answer.startswith('\\mathrm{') or answer.startswith('\\mathbf{')):
        answer = re.sub(r'^\{|\}$', '', answer)

    answer = remove_latex_commands(answer)
    answer = normalize_sub_super_scripts(answer)
    answer = remove_chemistry_units(answer)

    return answer.strip()


def remove_chemistry_units(answer):
    """Remove common chemistry units from numerical answers."""
    if not answer:
        return answer

    unit_pattern = r'^([+-]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][+-]?\d+)?)\s*(?:' + \
        r'Å²|Å\^2|A²|A\^2|' + \
        r'g/mol|Da|kDa|amu|' + \
        r'logP|cLogP|' + \
        r'mol/L|mmol/L|' + \
        r'kJ/mol|kcal/mol|' + \
        r'°C|°F|' + \
        r'mmHg|' + \
        r'μM|μL|μg|μm|μs|μA|' + \
        r'uM|uL|ug|um|us|uA|' + \
        r'mM|mL|mg|mm|ms|mV|mA|' + \
        r'nM|nm|ns|' + \
        r'pM|Pa|' + \
        r'kPa|kDa|kJ|kg|km|kHz|' + \
        r'MPa|MHz|' + \
        r'GHz|THz|' + \
        r'eV|' + \
        r'atm|bar|torr|' + \
        r'cal|kcal|' + \
        r'min|hr|hours|' + \
        r'percent|units|' + \
        r'M|L|K|J|V|A|C|F|g|m|s|h|%' + \
        r')(?:\s*[.,;:]?)*$'

    match = re.search(unit_pattern, answer, re.IGNORECASE | re.UNICODE)
    if match:
        return match.group(1)

    return answer


def convert_to_appropriate_type(answer):
    """Convert answer to appropriate type (int, float, or string)."""
    if not answer:
        return answer

    answer_str = str(answer).strip()

    # Try integer first
    try:
        if '.' not in answer_str:
            return int(answer_str)
    except ValueError:
        pass

    # Try float
    try:
        return float(answer_str)
    except ValueError:
        pass

    try:
        return ast.literal_eval(answer_str)
    except (ValueError, SyntaxError, TypeError):
        pass

    return answer_str


def extract_ether0_thought_answer_strict(text: str, reasoning: bool = True) -> tuple:
    """Extract thought and answer from ether0 model output using strict XML pattern."""
    matches = ETHER0_STRICT_XML_PATTERNS[reasoning].split(text, maxsplit=1)
    try:
        _, *inner, suffix = matches
    except (IndexError, ValueError):
        return None, None

    if reasoning:
        thought, inter, answer = inner
    else:
        thought, inter = None, None
        (answer,) = inner

    if (
        ETHER0_THINK_START not in (thought or "")
        and ETHER0_THINK_START not in (inter or "")
        and ETHER0_ANSWER_START not in answer
        and not suffix
    ):
        return thought, answer or None
    return thought, None


def extract_ether0(response):
    """Ether0-specific extraction function using strict XML-style patterns."""
    _, answer = extract_ether0_thought_answer_strict(response, reasoning=True)

    if answer is None:
        _, answer = extract_ether0_thought_answer_strict(response, reasoning=False)

    if answer is not None:
        cleaned = clean_extracted_answer(answer)
        if cleaned is not None:
            return convert_to_appropriate_type(cleaned)

    return extract_general_answer(response)


def extract_qwen3(response):
    """Qwen3-specific extraction function."""
    return extract_general_answer(response)


def extract_general_with_gsm8k(response):
    """Enhanced general extraction that includes GSM8K patterns."""
    if not response or response.strip() == "":
        return None

    if "</think>" in response:
        content = response.split("</think>")[-1].strip()
    elif "<|think_end|>" in response:
        content = response.split("<|think_end|>")[-1].strip()
    else:
        content = response.strip()

    all_matches = []

    patterns = [
        (r'####\s*([^\n]+)', 'GSM8K'),
        (r'<answer>(.*?)</answer>', 'ANSWER_TAG'),
        (r'<\|answer_start\|>(.*?)<\|answer_end\|>', 'ANSWER_START'),
        (r'\\boxed\{([^}]+)\}', 'BOXED'),
        (r'\*\*(.+?)\*\*', 'BOLD'),
        (r'the final answer is\s*[:]*\s*([^\n.!?]+)', 'FINAL_ANSWER'),
        (r'therefore,?\s*(?:the answer is)?\s*[:]*\s*([^\n.!?]+)', 'THEREFORE'),
        (r'so,?\s*(?:the answer is)?\s*[:]*\s*([^\n.!?]+)', 'SO'),
        (r'answer\s*[:]\s*([^\n.!?]+)', 'ANSWER_COLON'),
    ]

    for pattern, pattern_name in patterns:
        for match in re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL):
            extracted = match.group(1).strip()
            all_matches.append((match.start(), pattern_name, extracted))

    if all_matches:
        all_matches.sort(key=lambda x: x[0])
        _, pattern_name, extracted = all_matches[-1]
        extracted = re.sub(r'[.,;:!?]+$', '', extracted).strip()

        cleaned = clean_extracted_answer(extracted)
        if cleaned:
            return convert_to_appropriate_type(cleaned)
        return extracted

    return extract_fallback_patterns(content)


def extract_llasmol(response):
    """LlaSMol-specific extraction function for chemistry-tagged outputs."""
    if not response or response.strip() == "":
        return None

    content = response.strip()

    # Priority 1: Check for <answer> wrapper tags
    if "<answer>" in content and "</answer>" in content:
        answer_content = content.split("<answer>")[-1].split("</answer>")[0].strip()

        llasmol_tags = [
            ('SMILES', r'<SMILES>(.*?)</SMILES>'),
            ('IUPAC', r'<IUPAC>(.*?)</IUPAC>'),
            ('MOLFORMULA', r'<MOLFORMULA>(.*?)</MOLFORMULA>'),
            ('NUMBER', r'<NUMBER>(.*?)</NUMBER>'),
            ('BOOLEAN', r'<BOOLEAN>(.*?)</BOOLEAN>'),
        ]

        for tag_name, pattern in llasmol_tags:
            match = re.search(pattern, answer_content, re.IGNORECASE | re.DOTALL)
            if match:
                extracted = match.group(1).strip()

                if tag_name == 'NUMBER':
                    try:
                        if '.' not in extracted:
                            return int(extracted)
                        else:
                            return float(extracted)
                    except (ValueError, TypeError):
                        pass

                elif tag_name == 'BOOLEAN':
                    lower_val = extracted.lower().strip()
                    if lower_val in ('true', 't', '1', 'yes', 'y'):
                        return True
                    elif lower_val in ('false', 'f', '0', 'no', 'n'):
                        return False

                cleaned = clean_extracted_answer(extracted)
                if cleaned:
                    return convert_to_appropriate_type(cleaned)
                return extracted

        cleaned = clean_extracted_answer(answer_content)
        if cleaned:
            return convert_to_appropriate_type(cleaned)
        return answer_content

    # Priority 2: Check for chemistry tags without answer wrapper
    llasmol_tags = [
        ('SMILES', r'<SMILES>(.*?)</SMILES>'),
        ('IUPAC', r'<IUPAC>(.*?)</IUPAC>'),
        ('MOLFORMULA', r'<MOLFORMULA>(.*?)</MOLFORMULA>'),
        ('NUMBER', r'<NUMBER>(.*?)</NUMBER>'),
        ('BOOLEAN', r'<BOOLEAN>(.*?)</BOOLEAN>'),
    ]

    for tag_name, pattern in llasmol_tags:
        matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
        if matches:
            extracted = matches[-1].strip()

            if tag_name == 'NUMBER':
                try:
                    if '.' not in extracted:
                        return int(extracted)
                    else:
                        return float(extracted)
                except (ValueError, TypeError):
                    pass

            elif tag_name == 'BOOLEAN':
                lower_val = extracted.lower().strip()
                if lower_val in ('true', 't', '1', 'yes', 'y'):
                    return True
                elif lower_val in ('false', 'f', '0', 'no', 'n'):
                    return False

            cleaned = clean_extracted_answer(extracted)
            if cleaned:
                return convert_to_appropriate_type(cleaned)
            return extracted

    return extract_general_with_gsm8k(response)


def normalize_sub_super_scripts(text):
    """Normalize Unicode subscript and superscript digits to standard digits."""
    subscripts = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋", "0123456789+-")
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
    return text.translate(subscripts).translate(superscripts)


def extract_txgemma(response):
    """TXGemma-specific extraction function optimized for chemistry tasks."""
    if not response or response.strip() == "":
        return None

    content = response.strip()

    # Priority 1: <answer></answer> tags
    if "<answer>" in content and "</answer>" in content:
        answer = content.split("<answer>")[-1].split("</answer>")[0].strip()
        cleaned = clean_extracted_answer(answer)
        if cleaned:
            return convert_to_appropriate_type(cleaned)
        return answer

    # Priority 2: **Final Answer:** pattern
    final_answer_pattern = r'\*\*(?:Final\s+)?Answer\*\*\s*[:]*\s*([^\n.!?]+)'
    match = re.search(final_answer_pattern, content, re.IGNORECASE)
    if match:
        extracted = match.group(1).strip()
        cleaned = clean_extracted_answer(extracted)
        if cleaned:
            return convert_to_appropriate_type(cleaned)
        return extracted

    # Priority 3: \\boxed{...} LaTeX formatting
    pattern = r'\\boxed\{'
    for match in re.finditer(pattern, content):
        start = match.end()
        brace_count = 1
        end = start

        while end < len(content) and brace_count > 0:
            if content[end] == '{':
                brace_count += 1
            elif content[end] == '}':
                brace_count -= 1
            end += 1

        if brace_count == 0:
            extracted = content[start:end-1].strip()
            cleaned = clean_extracted_answer(extracted)
            if cleaned:
                return convert_to_appropriate_type(cleaned)
            return extracted

    # Priority 4: Bold **answer** patterns
    bold_patterns = [
        r'\*\*(.+?)\*\*(?:\s*$|\s*[.!?])',
        r'\*\*(\d+(?:\.\d+)?)\*\*',
    ]

    for pattern in bold_patterns:
        matches = re.findall(pattern, content)
        if matches:
            extracted = matches[-1].strip()
            cleaned = clean_extracted_answer(extracted)
            if cleaned:
                return convert_to_appropriate_type(cleaned)
            return extracted

    # Priority 5: "Therefore" and conclusion patterns
    conclusion_patterns = [
        r'therefore,?\s*(?:the\s+(?:final\s+)?answer\s+is)?\s*[:]*\s*([^\n.!?]+)',
        r'thus,?\s*(?:the\s+(?:final\s+)?answer\s+is)?\s*[:]*\s*([^\n.!?]+)',
        r'so,?\s*(?:the\s+(?:final\s+)?answer\s+is)?\s*[:]*\s*([^\n.!?]+)',
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*[:]*\s*([^\n.!?]+)',
        r'(?:the\s+)?(?:final\s+)?result\s+is\s*[:]*\s*([^\n.!?]+)',
    ]

    for pattern in conclusion_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r'[.,;:!?]+$', '', extracted).strip()
            cleaned = clean_extracted_answer(extracted)
            if cleaned:
                return convert_to_appropriate_type(cleaned)
            return extracted

    return extract_general_with_gsm8k(response)


def extract_answer_only(response):
    """
    Enhanced extraction function that extracts content in the following priority:
    1. Content between <answer></answer> tags (highest priority)
    2. Valid JSON objects or arrays
    3. Python dictionary/list formats
    """
    if not response or not isinstance(response, str):
        return None

    # Priority 1: Check for answer tags
    if "<answer>" in response and "</answer>" in response:
        answer = response.split("<answer>")[-1].split("</answer>")[0]
        return answer

    # Priority 2 & 3: Find and validate JSON/dict/list/tuple structures
    candidates = []

    def find_balanced_structures(text):
        """Find all balanced bracket/brace structures in text."""
        structures = []
        stack = {'[': [], '{': [], '(': []}
        reverse_matching = {']': '[', '}': '{', ')': '('}

        i = 0
        while i < len(text):
            char = text[i]

            if char == '"':
                i += 1
                while i < len(text):
                    if text[i] == '"' and (i == 0 or text[i-1] != '\\'):
                        break
                    i += 1
                i += 1
                continue
            elif char == "'":
                if i > 0 and text[i-1] in ['{', '[', ':', ',', ' ', '(']:
                    j = i + 1
                    found_close = False
                    while j < len(text) and j < i + 1000:
                        if text[j] == "'" and (j == 0 or text[j-1] != '\\'):
                            if j + 1 < len(text) and text[j+1] in ['}', ']', ':', ',', ' ', ')']:
                                found_close = True
                                break
                        j += 1

                    if found_close:
                        i = j + 1
                        continue

            if char in stack:
                stack[char].append(i)
            elif char in reverse_matching:
                opener = reverse_matching[char]
                if stack[opener]:
                    start = stack[opener].pop()
                    complete_structure = text[start:i+1]
                    structures.append((start, complete_structure))

            i += 1

        return structures

    structures = find_balanced_structures(response)

    # Filter out nested structures
    filtered_structures = []
    for i, (start1, struct1) in enumerate(structures):
        is_nested = False
        for j, (start2, struct2) in enumerate(structures):
            if i != j:
                end1 = start1 + len(struct1)
                end2 = start2 + len(struct2)
                if start2 <= start1 and end2 >= end1 and len(struct2) > len(struct1):
                    is_nested = True
                    break
        if not is_nested:
            filtered_structures.append((start1, struct1))

    for start_pos, candidate in filtered_structures:
        if candidate.startswith('{'):
            structure_type = 'dict'
        elif candidate.startswith('['):
            structure_type = 'list'
        elif candidate.startswith('('):
            structure_type = 'tuple'
        else:
            continue

        if structure_type in ['dict', 'list']:
            try:
                json.loads(candidate)
                candidates.append((start_pos, candidate, 'json'))
                continue
            except (json.JSONDecodeError, ValueError):
                pass

        try:
            ast.literal_eval(candidate)
            candidates.append((start_pos, candidate, 'python'))
        except (ValueError, SyntaxError, TypeError):
            if structure_type == 'dict' and ':' in candidate:
                if '"' not in candidate and "'" not in candidate:
                    candidates.append((start_pos, candidate, 'simple_dict'))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]

    return None


def extract_moleculariq_answer(response: str) -> Optional[Union[str, int, float, dict, list]]:
    """
    Unified extraction function for MolecularIQ.

    Handles multiple common formats in priority order:
    1. Strip thinking blocks (</think>, <|think_end|>)
    2. <answer>...</answer> tags (standard format)
    3. <|answer_start|>...<|answer_end|> tags (ether0-style)
    4. JSON objects/arrays via extract_answer_only logic
    5. Fallback to general extraction patterns

    Args:
        response: Raw model response string

    Returns:
        Extracted answer (string, number, dict, or list), or None if extraction failed
    """
    if not response or not isinstance(response, str):
        return None

    # Step 1: Strip thinking blocks to get the answer portion
    content = response.strip()
    if "</think>" in content:
        content = content.split("</think>")[-1].strip()
    elif "<|think_end|>" in content:
        content = content.split("<|think_end|>")[-1].strip()

    # Step 2: Check for <answer></answer> tags (standard format)
    if "<answer>" in content and "</answer>" in content:
        answer = content.split("<answer>")[-1].split("</answer>")[0].strip()
        return answer

    # Step 3: Check for <|answer_start|><|answer_end|> tags (ether0 format)
    if "<|answer_start|>" in content and "<|answer_end|>" in content:
        answer = content.split("<|answer_start|>")[-1].split("<|answer_end|>")[0].strip()
        return answer

    # Step 4: Fall back to JSON/dict detection via extract_answer_only
    json_answer = extract_answer_only(content)
    if json_answer is not None:
        return json_answer

    # Step 5: Fall back to general extraction patterns (boxed, bold, etc.)
    return extract_general_answer(response)
