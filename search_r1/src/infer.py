import sys
sys.path.append('/mnt/tanka/fengji/works/Omne-RAG')

import transformers
import torch
import random
from datasets import load_dataset
import requests
import re

import warnings

from search_r1.llm_agent.tools.entity_matcher import Knowledge_Graph_Entity_Matcher_Tool
from search_r1.llm_agent.tools.node_info import Knowledge_Graph_Get_Node_Info_Tool
from verl.utils.tokenizer import hf_tokenizer
warnings.filterwarnings("ignore", category=UserWarning)

from search_r1.llm_agent.generation_with_tools import parse

class ToolEnv:
    def __init__(self, endpoint):
        self.entity_matcher = Knowledge_Graph_Entity_Matcher_Tool(
            api_endpoint=endpoint
        )
        self.node_info = Knowledge_Graph_Get_Node_Info_Tool(
            api_endpoint=endpoint
        )
         
    def execute_action(self, tool_call: str):
        method, params = parse(tool_call)
        if method == "entity_matcher":
                return self.entity_matcher.execute(**params)
        elif method == "node_info":
                return self.node_info.execute(**params)
        elif method == "edge_to_node":
                return self.edge_to_node.execute(**params)
        else:
            raise ValueError(f"Unknown tool: {method}")

# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i]:], target):
                return True

        return False

def get_tool_call(text):
    
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None

def search(query: str):
    payload = {
        "queries": [query],
        "topk": 3,
        "return_scores": True
    }
    results = requests.post("http://10.119.16.45:9000/retrieve", json=payload).json()['result']
                
    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
                        
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0]) 


def _generate_sequences(input_ids, attention_mask, max_turn=4, use_cache=False):
        cnt = 0
        # Encode the chat-formatted prompt and move it to the correct device
        while True:
            print(f'rolling_ids: {input_ids.shape}')
            
            # Generate text with the stopping criteria
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1024,
                stopping_criteria=stopping_criteria,
                #pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True,
                output_logits=True,
                use_cache=use_cache
            )
            
            outputs = outputs.sequences
            
            print(outputs[0][-1].item())
            if outputs[0][-1].item() in curr_eos:
                generated_tokens = outputs[0][input_ids.shape[1]:]
                output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                print(f'rolling_ids: {rolling_ids.shape}, output_ids: {outputs.shape}')
                
                return output_text

            generated_tokens = outputs[0][input_ids.shape[1]:]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            tmp_query = get_tool_call(tokenizer.decode(outputs[0], skip_special_tokens=True))
            if tmp_query:
                # print(f'searching "{tmp_query}"...')
                search_results = search(tmp_query)
            else:
                search_results = ''
                
            print(search_results)

            search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
            prompt += search_text
            cnt += 1
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            attention_mask = torch.ones_like(input_ids)
            

def _generate_sequences2(input_ids, attention_mask, max_turn=4, use_cache=False):
    cnt = 0
    
    rolling_ids = input_ids
    while True:
        print(f'rolling_ids: {rolling_ids.shape}')
        # Generate text with the stopping criteria
        #print(f'model: {model}')
        rolling_ids = model.generate(
            rolling_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            #pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            return_dict_in_generate=True,
            output_logits=True,
            use_cache=use_cache
        )

        logits = rolling_ids.logits
        rolling_ids = rolling_ids.sequences

        if rolling_ids[0][-1].item() in curr_eos or cnt > max_turn:
            generated_tokens = rolling_ids[0][input_ids.shape[1]:]
            final_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return final_text

        #generated_tokens = rolling_ids[0][input_ids.shape[1]:]
        #output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        tmp_query = get_tool_call(tokenizer.decode(rolling_ids[0], skip_special_tokens=True))
        if tmp_query:
            print(f'tmp_query: {tmp_query}')
            search_results = tool_env.execute_action(tmp_query)
        else:
            search_results = ''

        #print(f'search results: {search_results}')
        cnt += 1
        
        info_res = curr_template.format(search_results=search_results)
        info_ids = tokenizer.encode(info_res, return_tensors='pt').to(device)
        rolling_ids = torch.cat((rolling_ids, info_ids), dim=1)
        attention_mask = torch.ones_like(rolling_ids)
        
# Model ID and device setup
model_id = "/mnt/tanka/models/Qwen3-30B-A3B"
print(model_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = hf_tokenizer(model_id, trust_remote_code=True)
tokenizer.padding_size = 'left' ## attention, padding_size is required by flash-attention
#print(tokenizer)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation='flash_attention_2',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device,
)
curr_eos = [tokenizer.eos_token_id, tokenizer.pad_token_id] #\[tokenizer.eos_token_id] # for Qwen2.5 series models #tokenizer.pad_token_id

tool_env = ToolEnv(endpoint='10.119.16.48:9000')

#question = "Mike Barnett negotiated many contracts including which player that went on to become general manager of CSKA Moscow of the Kontinental Hockey League?"
question = 'What country of origin does House of Cosbys and Bill Cosby have in common?'
question = question.strip()
if question[-1] != '?':
    question += '?'
curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
curr_template = '\n<information>{search_results}</information>\n\n'

# Initialize the stopping criteria
target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

# Prepare the message
prompt = f"""Answer the given question. \
You must first conduct reasoning inside <think> and </think> every time. \
After reasoning, if you find you lack some knowledge and need to get new information, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without any detailed illustrations. 

Question: {question}\n"""

data = load_dataset('parquet', data_files='./data/lightrag_0422/test_stage1.parquet', split='train')
print(data)

prompt = data[0]['prompt']


if tokenizer.chat_template:
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt[0]['content']}], add_generation_prompt=True, tokenize=False)

input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
attention_mask = torch.ones_like(input_ids)

prompt_start = input_ids.shape[1]

print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
print('Question:', data[0]['extra_info']['question'])
print('Path:', data[0]['extra_info']['path'])
print(f'prompt: \n{prompt}\nresponse:')
# Encode the chat-formatted prompt and move it to the correct device
rolling_ids = input_ids
attention_mask = torch.ones_like(rolling_ids)
output_text = _generate_sequences2(input_ids, attention_mask)
print(output_text)
