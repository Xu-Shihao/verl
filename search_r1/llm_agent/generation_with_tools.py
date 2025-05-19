from concurrent.futures import ThreadPoolExecutor, as_completed
from tensordict import TensorDict
import torch
import re
import os
import ast
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
import subprocess

# Import tools
from .tools import (
    Knowledge_Graph_Entity_Matcher_Tool,
    Knowledge_Graph_Get_Node_Info_Tool, 
    Knowledge_Graph_Node_Edge_To_Node_Tool
)

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    n_repeat: int=1
    do_sample: bool=False


class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        endpoint: str,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        # self.logger = logger
        self.is_validation = is_validation
        
        # Initialize tools
        self._init_tools(endpoint)

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

    def _init_tools(self, endpoint):
        """Initialize the knowledge graph tools."""
        
        # Setup entity matcher tool with no specific graph_type
        # The graph_type will be provided at runtime when executing the tool
        self.entity_matcher = Knowledge_Graph_Entity_Matcher_Tool(
            api_endpoint=endpoint
        )
        
        # Setup node info tool with no specific graph_type
        self.node_info = Knowledge_Graph_Get_Node_Info_Tool(
            api_endpoint=endpoint
        )
        
        # Setup edge to node tool with no specific graph_type
        self.edge_to_node = Knowledge_Graph_Node_Edge_To_Node_Tool(
            api_endpoint=endpoint
        )

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            #print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]
            max_len = 0
            for obs in next_obs:
                cur_len = len(obs)
                if cur_len > max_len:
                    max_len = cur_len
                    longest_obs = obs
            #print(f"Longest observation: {longest_obs}")

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        padding_start = time.time()
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            print(f"INFO: Generating sequences without padding (single GPU) at {padding_start}")
            generate_start = time.time()
            result = self.actor_rollout_wg.generate_sequences(active_batch)
            generate_end = time.time()
            print(f"INFO: Single GPU sequence generation completed in {generate_end - generate_start:.2f}s")
            return result
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus

        if remainder == 0:
            print(f"INFO: Generating sequences without padding (batch size divisible by GPUs) at {padding_start}")
            generate_start = time.time()
            result = self.actor_rollout_wg.generate_sequences(active_batch)
            generate_end = time.time()
            print(f"INFO: Multi-GPU sequence generation (no padding) completed in {generate_end - generate_start:.2f}s")
            return result
        
        print(f"INFO: Preparing padding for GPU batch (remainder={remainder}) at {padding_start}")
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        padding_end = time.time()
        print(f"INFO: Padding preparation completed in {padding_end - padding_start:.2f}s")

        # Generate with padded batch
        generate_start = time.time()
        print(f"INFO: Generating sequences with padded batch at {generate_start}")
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        generate_end = time.time()
        print(f"INFO: Multi-GPU sequence generation (with padding) completed in {generate_end - generate_start:.2f}s")
        
        # Remove padding from output
        unpad_start = time.time()
        print(f"INFO: Starting padding removal at {unpad_start}")
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        unpad_end = time.time()
        print(f"INFO: Padding removal completed in {unpad_end - unpad_start:.2f}s")
        
        return padded_output

    def run_llm_loop(self, gen_batch: DataProto, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        print(f"INFO: Starting run_llm_loop at {time.time()}")
        loop_start_time = time.time()
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []], 'responses_with_info_mask': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        turns_stats = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_action_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        valid_search_stats = torch.zeros(gen_batch.batch['input_ids'].shape[0], dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        
        print(f'n_repeat: {self.config.n_repeat}, do_sample: {self.config.do_sample}')

        # Main generation loop
        for step in range(self.config.max_turns):
            step_start_time = time.time()
            print(f"INFO: Starting step {step} at {step_start_time}")
            
            if not active_mask.sum():
                break
            print(f'--- Main generation loop: step {step} ---')
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            
            #print(f'rollings_active.size: {len(rollings_active)}')
            
            gen_start_time = time.time()
            print(f"INFO: Starting model generation at {gen_start_time}")
            gen_output = self._generate_with_gpu_padding(rollings_active)
            gen_end_time = time.time()
            print(f"INFO: Model generation completed in {gen_end_time - gen_start_time:.2f}s")
            
            #print(f'gen_output.size: {len(gen_output)}')
            
            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            execute_start_time = time.time()
            print(f"INFO: Starting execute_predictions at {execute_start_time}")
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            execute_end_time = time.time()
            print(f"INFO: execute_predictions completed in {execute_end_time - execute_start_time:.2f}s")
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
            
            step_end_time = time.time()
            print(f"INFO: Step {step} completed in {step_end_time - step_start_time:.2f}s")
            
        # final LLM rollout
        if active_mask.sum():
            final_rollout_start = time.time()
            print(f"INFO: Starting final LLM rollout at {final_rollout_start}")
            
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            
            final_gen_start = time.time()
            print(f"INFO: Starting final model generation at {final_gen_start}")
            gen_output = self._generate_with_gpu_padding(rollings_active)
            final_gen_end = time.time()
            print(f"INFO: Final model generation completed in {final_gen_end - final_gen_start:.2f}s")

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            final_execute_start = time.time()
            print(f"INFO: Starting final execute_predictions at {final_execute_start}")
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )
            final_execute_end = time.time()
            print(f"INFO: Final execute_predictions completed in {final_execute_end - final_execute_start:.2f}s")

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            
            meta_info['turns_stats'] = turns_stats.tolist()
            meta_info['active_mask'] = active_mask.tolist()
            meta_info['valid_action_stats'] = valid_action_stats.tolist()
            meta_info['valid_search_stats'] = valid_search_stats.tolist()

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
            )
            
            final_rollout_end = time.time()
            print(f"INFO: Final LLM rollout completed in {final_rollout_end - final_rollout_start:.2f}s")
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        compose_start = time.time()
        print(f"INFO: Starting final output composition at {compose_start}")
        final_output = self._compose_final_output(original_left_side, original_right_side, meta_info)
        compose_end = time.time()
        print(f"INFO: Final output composition completed in {compose_end - compose_start:.2f}s")
        
        loop_end_time = time.time()
        print(f"INFO: Total run_llm_loop completed in {loop_end_time - loop_start_time:.2f}s")
        
        return final_output

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        process_start = time.time()
        print(f"INFO: Starting postprocess_predictions at {process_start}")
        cur_actions, tool_calls = self.postprocess_predictions(predictions)
        process_end = time.time()
        print(f"INFO: postprocess_predictions completed in {process_end - process_start:.2f}s")
        
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [tool_call for action, tool_call in zip(cur_actions, tool_calls) if action == 'search']
        if do_search:
            search_start = time.time()
            search_count = len(search_queries)
            print(f"INFO: Starting batch_execute_tool_calls for {search_count} search queries at {search_start}")
            search_results = self.batch_execute_tool_calls(search_queries)
            search_end = time.time()
            print(f"INFO: batch_execute_tool_calls completed in {search_end - search_start:.2f}s, avg {(search_end-search_start)/max(1,search_count):.2f}s per query")
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        process_results_start = time.time()
        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(0)
                    is_search.append(0)
                elif action == 'search':
                    #print(f'Search Results: {search_results}')
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to call a tool, I should put the call between <search> and </search> using the format tool_name(param="value"). \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
        
        process_results_end = time.time()
        print(f"INFO: Processing results completed in {process_results_end - process_results_start:.2f}s")
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str]]:
        """
        Process predictions from LLM into actions and tool calls.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, tool calls list)
        """
        actions = []
        tool_calls = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                # Check for search actions (tool calls)
                search_pattern = r'<search>(.*?)</search>'
                search_match = re.search(search_pattern, prediction, re.DOTALL)
                
                # Check for answer actions
                answer_pattern = r'<answer>(.*?)</answer>'
                answer_match = re.search(answer_pattern, prediction, re.DOTALL)
                
                if search_match:
                    tool_call = search_match.group(1).strip()
                    actions.append('search')
                    tool_calls.append(tool_call)
                elif answer_match:
                    answer = answer_match.group(1).strip()
                    actions.append('answer')
                    tool_calls.append(answer)
                else:
                    actions.append(None)
                    tool_calls.append('')
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
        return actions, tool_calls

    def batch_execute_tool_calls(self, tool_calls: List[str]) -> List[str]:
        """
        Execute a batch of tool calls and return the results.
        
        Args:
            tool_calls: List of tool call strings in the format "tool_name(param1='value1', param2='value2')"
            
        Returns:
            List of tool execution results as strings
        """
        # 如果没有工具调用，返回空列表
        if not tool_calls:
            return []

        # 创建一个线程池，最大工作线程数可以根据需要调整
        max_workers = min(len(tool_calls), 32)  # 限制最大线程数为10或调用数量
        print(f"INFO: Setting up ThreadPoolExecutor with {max_workers} workers for {len(tool_calls)} tool calls")

        # 定义工作函数，包含错误处理
        def process_tool_call(idx, tool_call):
            try:
                call_start = time.time()
                print(f"INFO: Executing tool call {idx} at {call_start}")
                # 执行工具调用
                result = self.execute_tool_call(tool_call)
                # 格式化结果
                formatted_result = self._format_tool_result(tool_call, result)
                call_end = time.time()
                print(f"INFO: Tool call {idx} completed in {call_end - call_start:.2f}s")
                return idx, formatted_result
            except Exception as e:
                # 处理错误情况
                error_message = f"Error executing tool call '{tool_call}': {str(e)}"
                return idx, error_message

        # 初始化结果列表，保持与输入相同的长度
        results = [None] * len(tool_calls)

        # 使用线程池并行执行工具调用
        pool_start = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务到线程池
            future_to_idx = {
                executor.submit(process_tool_call, i, call): i
                for i, call in enumerate(tool_calls)
            }

            # 获取结果，保持原始顺序
            for future in as_completed(future_to_idx):
                idx, result = future.result()
                results[idx] = result

        pool_end = time.time()
        print(f"INFO: ThreadPoolExecutor completed all tool calls in {pool_end - pool_start:.2f}s")

        # 确保所有结果都已填充
        assert None not in results, "Some tool calls did not complete"
                
        return results
    
    def execute_tool_call(self, tool_call: str) -> str:
        tool_name, params = parse(tool_call)
        # Execute the appropriate tool
        if tool_name == "entity_matcher":
            return self.entity_matcher.execute(**params)
        elif tool_name == "node_info":
            return self.node_info.execute(**params)
        elif tool_name == "edge_to_node":
            return self.edge_to_node.execute(**params)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _format_tool_result(self, tool_call: str, result: str) -> str:
        """
        Format the result of a tool execution as a string for the LLM.
        
        Args:
            tool_call: The original tool call string
            result: The result of the tool execution (pre-formatted string)
            
        Returns:
            Formatted result string
        """
        formatted_result = ""
        
        # Add the pre-formatted result directly
        formatted_result += result
        
        return formatted_result
    
def parse(tool_call: str):
    """
        Parse and execute a single tool call.
        
        Args:
            tool_call: Tool call string in the format "tool_name(param1='value1', param2='value2')"
            
        Returns:
            The result of the tool execution as a formatted string
        """
        # Check for multiple tool calls in the same string
    #if tool_call.count('(') > 1 and tool_call.count(')') > 1:
    #    raise ValueError(f"Multiple tool calls detected in a single request. Only one tool can be called at a time: {tool_call}")
            
        # Extract tool name and parameters
    match = re.match(r'(\w+)\((.*)\)', tool_call.strip())
        
    if not match:
        raise ValueError(f"Invalid tool call format: {tool_call}")
            
    tool_name = match.group(1)
    params_str = match.group(2)
        
        # Parse parameters
        # Add outer braces to make it a dictionary literal
    if params_str:
            # Fix boolean values for proper parsing
        params_str = params_str.replace('=True', '=true').replace('=False', '=false')
            # Replace single quotes with double quotes for JSON parsing
        params_str = params_str.replace("'", '"')
            
            # Try to parse parameters directly first
        try:
            # Use a regex to extract parameter names and values
            params = {}
            param_pattern = r'(\w+)=([^,]+)(?:,|$)'
            param_matches = re.finditer(param_pattern, params_str)
                
            for param_match in param_matches:
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                    
                # Convert string representations to Python types
                if param_value.lower() == 'true':
                    params[param_name] = True
                elif param_value.lower() == 'false':
                    params[param_name] = False
                elif param_value.startswith('"') and param_value.endswith('"'):
                    # String value
                    params[param_name] = param_value[1:-1]  # Remove quotes
                elif param_value.isdigit():
                    # Integer value
                    params[param_name] = int(param_value)
                elif param_value.replace('.', '', 1).isdigit():
                    # Float value
                    params[param_name] = float(param_value)
                else:
                    # Default to string
                    params[param_name] = param_value
            
        except Exception as e:
            raise ValueError(f"Failed to parse parameters '{params_str}': {str(e)}")
    else:
        params = {}
        
    return tool_name, params
