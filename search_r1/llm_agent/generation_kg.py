import torch
import re
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
import subprocess

class Knowledge_Graph_Searcher:
    """
    A simplified version of the Knowledge_Graph_Searcher_Tool for use in the search_r1 project.
    """
    
    def __init__(self, database_path):
        """
        Initialize the Knowledge Graph Searcher.
        """
        self.database_path = database_path
        self._verify_database()
        
    def _verify_database(self):
        """
        Verify that the database file exists and is accessible.
        """
        if not os.path.exists(self.database_path):
            print(f"Warning: Knowledge graph database file not found at {self.database_path}")
            exit()
    
    def search_knowledge_graph(self, entity: str, relation: str, reverse_search: bool = False) -> List[str]:
        """
        Search the knowledge graph for entities related to the input entity through the specified relationship.
        
        Args:
            entity (str): The entity to search for.
            relationship (str): The relationship to filter by.
            reverse_search (bool): If True, search for entity as the object of the relationship instead of the subject.
            
        Returns:
            List[str]: A list of entities that have the specified relationship with the input entity.
        """
        '''
        if not os.path.exists(self.database_path):
            return [f"Error: Database file not found at {self.database_path}"]
        '''
        
        results = []
        
        try:
            # Use grep for efficient searching of large files
            
            if not reverse_search:
                # Forward search: entity is subject (entity|relationship|?)
                search_pattern = f"^{entity}\\|{relation}\\|"
                position = 0  # Subject position
                result_position = 2  # From 0-based indexing, the third field
            else:
                # Reverse search: entity is object (?|relationship|entity)
                search_pattern = f"\\|{relation}\\|{entity}$"
                position = 2  # Object position
                result_position = 0  # From 0-based indexing, the first field
                
            # Execute grep command
            grep_cmd = ["grep", "-E", search_pattern, self.database_path]
            grep_result = subprocess.run(grep_cmd, capture_output=True, text=True)
            #print(f'grep result: {grep_result}')
            
            # Process results
            if grep_result.returncode == 0 and grep_result.stdout:
                for line in grep_result.stdout.strip().split('\n'):
                    # Split line and get result element
                    parts = line.split('|')
                    if len(parts) >= 3 and parts[position] == entity:
                        results.append(parts[result_position])
            
            return results
            
        except Exception as e:
            return [f"Error searching knowledge graph: {str(e)}"]

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    # logging: dict
    num_gpus: int
    no_think_rl: bool=False
    kg_path: str=None  #= "/mnt/afs/theta/fengji/works/omne-kg/search_r1/tools/knowledge_graph_searcher/data/kb.txt"
    n_repeat: int=1
    do_sample: bool=False

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        # logger: Tracking,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        # self.logger = logger
        self.is_validation = is_validation
        self.kg_searcher = Knowledge_Graph_Searcher(config.kg_path)

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))

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
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus

        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
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
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
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
            
            gen_output = self._generate_with_gpu_padding(rollings_active)
            
            #print(f'gen_output.size: {len(gen_output)}')
            
            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask
            )
            
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
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # # Execute in environment and process observations
            _, dones, valid_action, is_search = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, active_mask, do_search=False
            )

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
        
        print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        return self._compose_final_output(original_left_side, original_right_side, meta_info)

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
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []
        
        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

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
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
            
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

    def _parse_kg_query(self, query):
        """
        Parse a knowledge graph query string into components.
        Expected format: "entity|relationship|reverse:true/false"
        Default for reverse is False if not specified.
        
        Args:
            query (str): The query string to parse
            
        Returns:
            tuple: (entity, relationship, reverse_search)
        """
        parts = query.split('|')
        
        # Handle basic case with just entity
        if len(parts) == 1:
            return parts[0].strip(), "", False
            
        # Handle entity and relationship
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip(), False
            
        # Handle complete case with reverse parameter
        if len(parts) >= 3:
            entity = parts[0].strip()
            relationship = parts[1].strip()
            reverse_param = parts[2].strip()
            
            # Check if reverse parameter is specified
            if reverse_param.lower().startswith('reverse:'):
                reverse_value = reverse_param.split(':')[1].strip().lower()
                reverse_search = (reverse_value == 'true')
            else:
                # Third part is not a reverse parameter, treat as regular text
                reverse_search = False
                
            return entity, relationship, reverse_search
            
        return "", "", False

    def batch_search(self, queries: List[str] = None) -> List[str]:
        """
        Perform batch knowledge graph searches.
        
        Args:
            queries: List of knowledge graph queries in the format "entity|relationship|reverse:true/false"
                    where reverse is optional (defaults to false if not provided)
        
        Returns:
            List of search results as formatted strings
        """
        results = []
        
        for query in queries:
            # Process multiple KG queries in a single search request if present
            if '&&' in query:
                sub_queries = query.split('&&')
                combined_results = []
                
                for sub_query in sub_queries:
                    entity, relationship, reverse_search = self._parse_kg_query(sub_query.strip())
                    sub_result = self.kg_searcher.search_knowledge_graph(entity, relationship, reverse_search)
                    
                    if sub_result:
                        query_type = "Reverse" if reverse_search else "Forward"
                        result_text = f"KG {query_type} Search: {entity} | {relationship}\n"
                        
                        for i, item in enumerate(sub_result):
                            result_text += f"- {item}\n"
                            
                        combined_results.append(result_text)
                
                results.append("\n".join(combined_results))
            else:
                # Process a single KG query
                entity, relationship, reverse_search = self._parse_kg_query(query)
                kg_results = self.kg_searcher.search_knowledge_graph(entity, relationship, reverse_search)
                
                query_type = "Reverse" if reverse_search else "Forward"
                result_text = f"KG {query_type} Search: {entity} | {relationship}\n"
                
                if kg_results:
                    for i, item in enumerate(kg_results):
                        result_text += f"- {item}\n"
                else:
                    result_text += "No results found.\n"
                    
                results.append(result_text)
                
        return results

    def _batch_search(self, queries):
        """
        Knowledge graph search implementation.
        
        Args:
            queries: List of knowledge graph queries
            
        Returns:
            Dict with results formatted to match the original API structure
        """
        search_results = []
        
        for query in queries:
            result_documents = []
            
            # Process multiple KG queries if present
            if '&&' in query:
                sub_queries = query.split('&&')
                combined_content = ""
                
                for sub_query in sub_queries:
                    entity, relationship, reverse_search = self._parse_kg_query(sub_query.strip())
                    sub_result = self.kg_searcher.search_knowledge_graph(entity, relationship, reverse_search)
                    
                    if sub_result:
                        query_type = "Reverse" if reverse_search else "Forward"
                        combined_content += f"KG {query_type} Search: {entity} | {relationship}\n"
                        
                        for item in sub_result:
                            combined_content += f"- {item}\n"
                            
                        combined_content += "\n"
                
                # Create a document for the combined results
                if combined_content:
                    result_documents.append({
                        "document": {
                            "contents": f"Knowledge Graph Results\n{combined_content.strip()}"
                        },
                        "score": 1.0
                    })
            else:
                # Process a single KG query
                entity, relationship, reverse_search = self._parse_kg_query(query)
                kg_results = self.kg_searcher.search_knowledge_graph(entity, relationship, reverse_search)
                
                query_type = "Reverse" if reverse_search else "Forward"
                content = f"Knowledge Graph Results\nKG {query_type} Search: {entity} | {relationship}\n"
                
                if kg_results:
                    for item in kg_results:
                        content += f"- {item}\n"
                else:
                    content += "No results found.\n"
                
                result_documents.append({
                    "document": {
                        "contents": content
                    },
                    "score": 1.0
                })
            
            search_results.append(result_documents)
        
        return {"result": search_results}

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            format_reference += f"{content}\n"
        return format_reference 
