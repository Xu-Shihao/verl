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
"""
Preprocess the Knowledge Graph extraction dataset to parquet format
"""

import argparse
import os
import json

import pandas as pd
from datasets import Dataset

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/kg_extraction")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--input_train", default="/mnt/afs/tanka/shihao/project/verl/data/kg_extraction/kg_extraction_train.jsonl")
    parser.add_argument("--input_val", default="/mnt/afs/tanka/shihao/project/verl/data/kg_extraction/kg_extraction_val.jsonl")

    args = parser.parse_args()

    data_source = "kg_extraction"
    
    # Load the train and validation data
    train_data = []
    with open(args.input_train, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    val_data = []
    with open(args.input_val, 'r') as f:
        for line in f:
            val_data.append(json.loads(line))
    
    # Convert to datasets
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))
    
    # Process each dataset
    def process_dataset(dataset, split):
        def process_fn(example, idx):
            # Most fields are already in the correct format in the source data
            # Just ensure we have the proper structure for consistency
            
            data = {
                "data_source": data_source,
                "prompt": example["prompt"],
                "ability": "knowledge_extraction",
                "reward_model": example["reward_model"],
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "original_input": example.get("extra_info", {}).get("original_input", ""),
                }
            }
            
            # Add response if it exists and is not empty
            if "response" in example and example["response"]:
                data["response"] = example["response"]
                
            return data

        return dataset.map(function=lambda example, idx: process_fn(example, idx), with_indices=True)

    train_dataset = process_dataset(train_dataset, "train")
    val_dataset = process_dataset(val_dataset, "val")

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir) 