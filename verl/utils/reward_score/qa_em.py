# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 0:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score
        
def kg_em_check(prediction, golden_answers):
    """Order-insensitive exact matching for knowledge graph search results.
    
    Args:
        prediction: A string containing comma-separated values from model output
        golden_answers: A list of reference answers or a single string
        
    Returns:
        # score: 1 if sets match exactly, 0 otherwise
        F1 score
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
        
    # Split the prediction by commas and normalize each item
    # First handle the case where prediction might be empty
    if not prediction or prediction.strip() == "":
        return 0 if len(golden_answers) != 0 else 1  # Only match if golden_answers is also empty
        
    # Split prediction by commas and normalize each element
    pred_items = [normalize_answer(item.strip()) for item in prediction.split(',')]
    pred_items = [item for item in pred_items if item]  # Remove empty items
    
    
    # Create a set of normalized golden answers
    golden_set = set()
    for answer in golden_answers:
        # Handle different formats of golden_answers
        if isinstance(answer, list) or hasattr(answer, '__iter__') and not isinstance(answer, str):
            # If answer is already a list/array, add each normalized item
            golden_set.update([normalize_answer(str(item)) for item in answer])
        else:
            # Otherwise add the single normalized answer
            golden_set.add(normalize_answer(str(answer)))
            
    # Remove empty items from golden set
    golden_set = {item for item in golden_set if item}
    golden_zl = len(golden_set) == 0
    pred_zl = len(pred_items) == 0
    if golden_zl and pred_zl:
        return 1
    elif golden_zl or pred_zl:
        return 0
    
    pred_set = set(pred_items)
    
    intersection = pred_set & golden_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(golden_set)
    if precision + recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    # Check if sets match exactly (order-insensitive)
    # return 1 if set(pred_items) == golden_set else 0
    return f1

def kg_format_check(solution_str):
    if solution_str.strip().startswith('<search>') or not solution_str.strip().endswith('</answer>'):
        return 0.1

    '''
    thought_pattern = f'</information>(.*?)<answer>'
    info_pos = solution_str.rfind('</information>')
    if info_pos != -1:
        match = re.finditer(thought_pattern, solution_str[info_pos:], re.DOTALL)
        match = list(match)
        if len(match) == 0 or match[-1].group(1).strip() == '':
            return 0.2
    else:
        return 0.1
    '''    
    
    return 1

def lightrag_format_check(solution_str):
    """
    Check the format of LightRAG solution string and assign score based on format compliance.
    
    Scoring rule:
    - 0.1: If starts with <think> tag and ends with </answer> tag
    - 0.4: If adjacent different tags have content between them (no empty transitions)
    - 0.6: If same tags have content (e.g., <think>content...</think>)
    - 1.0: If uses at least two different tools (minimum two <search> calls with different tools)
    
    Note: Each higher score requires satisfying all previous conditions.
    
    Args:
        solution_str: The complete solution string from model output
        
    Returns:
        A format score between 0.0 and 1.0
    """
    # Base score
    score = 0.0
    
    # Strip leading/trailing whitespace
    solution_str = solution_str.strip()
    
    # Check if starts with <think> and ends with </answer>
    if solution_str.startswith('<think>') and solution_str.endswith('</answer>'):
        score = 0.1
    else:
        return score  # Format completely incorrect, return 0
    
    # Look for empty transitions between different tags
    # Common patterns: </think><search>, </search><information>, </information><think>, </think><answer>
    empty_transitions = [
        '</think><search>', 
        '</search><information>', 
        '</information><think>', 
        '</think><answer>'
    ]
    
    has_empty_transition = False
    for transition in empty_transitions:
        if transition in solution_str:
            has_empty_transition = True
            break
    
    if not has_empty_transition:
        score = 0.4
    else:
        return score  # Has empty transitions, return 0.1
    
    # Check for content within same tags
    # Need to check <think>...</think>, <search>...</search>, <answer>...</answer>
    tag_pairs = [
        ('<think>', '</think>'),
        ('<search>', '</search>'),
        ('<answer>', '</answer>')
    ]
    
    all_tags_have_content = True
    for open_tag, close_tag in tag_pairs:
        # Find all occurrences of each tag pair
        pattern = f'{re.escape(open_tag)}(.*?){re.escape(close_tag)}'
        matches = list(re.finditer(pattern, solution_str, re.DOTALL))
        
        # Check if all occurrences have content
        for match in matches:
            if not match.group(1).strip():
                all_tags_have_content = False
                break
        
        if not all_tags_have_content:
            break
    
    if all_tags_have_content:
        score = 0.6
    else:
        return score  # Not all tags have content, return 0.4
    
    # Check for at least two different tool calls in <search> tags
    search_pattern = r'<search>(.*?)</search>'
    search_matches = list(re.finditer(search_pattern, solution_str, re.DOTALL))
    
    if len(search_matches) < 2:
        return score  # Less than two search calls, return 0.6
    
    # Extract tool names from search calls
    tools_used = set()
    for match in search_matches:
        search_content = match.group(1).strip()
        # Extract tool name (everything before the first parenthesis)
        if '(' in search_content:
            tool_name = search_content.split('(')[0].strip()
            tools_used.add(tool_name)
    
    # At least two different tools were used
    if len(tools_used) >= 2:
        score = 1.0
    
    return score

def compute_score_em_kgqa(solution_str, ground_truth, format_score=0., score=1., stage='train'):
    """Scoring function for knowledge graph search with order-insensitive exact matching.
    
    Args:
        solution_str: The complete solution string from model output
        ground_truth: Dictionary containing 'target' field with golden answers
        format_score: Score to give for correct format but wrong answer
        score: Score to give for correct answer
        
    Returns:
        A reward score (0 to 1 by default)
    """
    # Extract answer from the solution string
    answer = extract_solution(solution_str)
    
    # Default score if no answer is found
    rw_score = 0
    if answer is not None:
        format_score = kg_format_check(solution_str)
        # Check for exact match with order insensitivity
        rw_score = kg_em_check(answer, ground_truth['target'])
        # if there is no think at the begin
        rw_score = format_score*rw_score
    
    # Occasionally print examples for debugging (1/64 chance)
    do_print = random.randint(1, 64) == 1
    if stage != 'train':
        do_print = do_print or rw_score > 1
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Score: {rw_score}")
        print(f"Format Score: {format_score}")
        print(f"Solution string: {solution_str}")
    
    return rw_score

def compute_score_em_rag(solution_str, ground_truth, question_raw='', path='', stage='train'):
    """Scoring function for knowledge graph search with exact matching.
    
    Args:
        solution_str: The complete solution string from model output
        ground_truth: Dictionary containing 'target' field with golden answers
    """
    # Extract answer from the solution string
    answer = extract_solution(solution_str)
    
    # Default score if no answer is found
    format_score = lightrag_format_check(solution_str)
    
    rw_score = 0
    if answer is not None:
        # Check for exact match - direct comparison without order insensitivity
        normalized_answer = normalize_answer(answer)
        for target in ground_truth['target']:
            if normalized_answer == normalize_answer(target):
                rw_score = 1
                break
        # Apply format check to the score
        rw_score = format_score * (0.1 + rw_score * 0.9)
    
    # Occasionally print examples for debugging (1/64 chance)
    do_print = random.randint(1, 64) == 1
    if stage != 'train':
        do_print = do_print or rw_score > 1
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Score: {rw_score}")
        print(f"Format Score: {format_score}")
        print(f"Question: {question_raw}")
        print(f"Path: {path}")
        print(f"Solution string: {solution_str}")
    
    return rw_score


def lightrag_format_check2(solution_str):
    """
    Check the format of LightRAG solution string and assign score based on format compliance.
    
    Scoring rule:
    - 0.1: If starts with <think> tag and ends with </answer> tag
    - 0.4: If adjacent different tags have content between them (no empty transitions)
    - 0.6: If same tags have content (e.g., <think>content...</think>)
    - 1.0: If uses at least two different tools (minimum two <search> calls with different tools)
    
    Note: Each higher score requires satisfying all previous conditions.
    
    Args:
        solution_str: The complete solution string from model output
        
    Returns:
        A format score between 0.0 and 1.0
    """
    # Strip leading/trailing whitespace
    solution_str = solution_str.strip()
    
    # Check if starts with <think> and ends with </answer>
    if not solution_str.startswith('<think>') or not solution_str.endswith('</answer>'):
        return 0.1

    # Check for at least two different tool calls in <search> tags
    search_pattern = r'<search>(.*?)</search>'
    search_matches = list(re.finditer(search_pattern, solution_str, re.DOTALL))
    empty_str_cnt = 0
    if len(search_matches) >= 2:
        for search_item in search_matches:
            if search_item.group(1).strip() == '':
                empty_str_cnt += 1
    if empty_str_cnt != len(search_matches):
        return 0.1
    
    return 1.0


def compute_score_lightrag_em2(solution_str, ground_truth, format_score=0., score=1., stage='train'):
    """Scoring function for knowledge graph search with exact matching.
    
    Args:
        solution_str: The complete solution string from model output
        ground_truth: Dictionary containing 'target' field with golden answers
    """
    # Extract answer from the solution string
    answer = extract_solution(solution_str)
    
    # Default score if no answer is found
    rw_score = 0
    if answer is not None:
        # Use the new lightrag_format_check instead of kg_format_check
        format_score = lightrag_format_check2(solution_str)
        # Check for exact match - direct comparison without order insensitivity
        normalized_answer = normalize_answer(answer)
        for target in ground_truth['target']:
            if normalized_answer == normalize_answer(target):
                rw_score = 1
                break
        # Apply format check to the score
        rw_score = format_score * (0.1 + rw_score)
    
    # Occasionally print examples for debugging (1/64 chance)
    do_print = random.randint(1, 64) == 1
    if stage != 'train':
        do_print = do_print or rw_score > 1
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Score: {rw_score}")
        print(f"Format Score: {format_score}")
        print(f"Solution string: {solution_str}")
    
    return rw_score
