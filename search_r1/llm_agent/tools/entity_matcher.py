import httpx
import os
from typing import Dict, Any

# Import the base tool
from .base import BaseTool


class Knowledge_Graph_Entity_Matcher_Tool(BaseTool):
    """
    A tool that uses vector similarity to match approximate entity names to precise entities 
    that actually exist in the knowledge graph.
    """
    
    def __init__(self, api_endpoint: str = None, graph_type: str = None):
        """
        Initialize the Knowledge Graph Entity Matcher Tool.
        
        Args:
            api_endpoint (str, optional): Custom API endpoint URL. If not provided, it will be constructed at runtime.
            graph_type (str, optional): Default type of knowledge graph to use. If not provided, must be specified during execution.
        """
        super().__init__(
            tool_name="Knowledge_Graph_Entity_Matcher_Tool",
            tool_description="This tool uses vector similarity to find precise entity names in the knowledge graph that match an approximate query. It helps avoid guessing exact entity names by returning similar entities that actually exist in the graph, along with their descriptions.",
            tool_version="1.0.0",
            input_types={
                "node_name": "str - An approximate name or description of the entity you're looking for.",
                "graph_type": "str - Type of knowledge graph to query. Options: 'agriculture', 'mix', 'legal', 'cs','health',. Required if not set during initialization."
            },
            output_type="list - A list of matching entities with their names and descriptions, ordered by similarity.",
            demo_commands=[
                {
                    "command": 'result = tool.execute(node_name="Electric Company", graph_type="agriculture")',
                    "description": "Find entities in the agriculture knowledge graph similar to 'Electric Company'."
                },
                {
                    "command": 'result = tool.execute(node_name="sugar production", graph_type="mix")',
                    "description": "Find entities related to sugar production in the mixed domain knowledge graph."
                },
                {
                    "command": 'result = tool.execute(node_name="law enforcement", graph_type="legal")',
                    "description": "Find entities related to law enforcement in the legal knowledge graph."
                },
                {
                    "command": 'result = tool.execute(node_name="algorithm", graph_type="cs")',
                    "description": "Find entities related to algorithms in the computer science knowledge graph."
                }
            ],
            user_metadata={
                "api_endpoint": api_endpoint,
                "description": "A vector similarity service that matches approximate entity names to precise entities in the knowledge graph.",
                "supported_graph_types": ["agriculture", "mix", "legal", "cs"]
            }
        )
        self.api_endpoint = api_endpoint if api_endpoint is not None else '127.0.0.1:9000'
        self.graph_type = graph_type
    
    def execute(self, node_name: str, graph_type: str) -> str:
        """
        Query the knowledge graph to find entities similar to the provided name.
        
        Args:
            node_name (str): An approximate name or description of the entity you're looking for.
            graph_type (str): Type of knowledge graph to query. Options: 'agriculture', 'mix', 'legal', 'cs'.
            
        Returns:
            str: A formatted string containing the matching entities information.
        """
        if not graph_type:
            return f"Error: graph_type must be specified ('agriculture', 'mix', 'legal', or 'cs'). Query: {node_name}"
            
        #print(f"Finding entities similar to '{node_name}' in the {graph_type} knowledge graph...")
        
        try:
            # Construct the API endpoint based on the specified graph_type
            endpoint = f"http://{self.api_endpoint}/{graph_type}/get_precise_entity_names"
            
            # Prepare the request payload
            payload = {
                "node_name": node_name
            }
            
            # Make the API request using httpx
            with httpx.Client() as client:
                response = client.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
            
            # print(f"Response: {response}")
            # print(f"Response content: {response.text}")
            
            # Check if the request was successful
            if response.status_code == 200:
                # Extract the rendered_content field from the JSON response
                response_json = response.json()
                if "rendered_content" in response_json:
                    return response_json["rendered_content"]
                else:
                    # Fallback to returning the full JSON if rendered_content is not found
                    return response.text
            else:
                return f"Error finding similar entities: {response.status_code}, {response.text}"
                
        except Exception as e:
            return f"Error executing entity matching: {str(e)}"


# Test script for the Knowledge Graph Entity Matcher Tool
if __name__ == "__main__":
    import json
    
    # Create an instance of the tool
    entity_matcher = Knowledge_Graph_Entity_Matcher_Tool()
    
    # Test cases
    test_cases = [
        {"node_name": "King Solomon's Mines", "graph_type": "mix"},
        {"node_name": "Prairie Prince", "graph_type": "mix"},
        {"node_name": "Transgenic Modification", "graph_type": "agriculture"},
        {"node_name": "algorithm", "graph_type": "cs"}
    ]
    
    # Run tests
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['node_name']} ---")
        try:
            result = entity_matcher.execute(**test_case)
            
            # Print result (now it's a string)
            print(f"Result (first 200 chars):")
            print(result[:200] + "..." if len(result) > 200 else result)
            
            # Check for errors
            if result.startswith("Error"):
                print(f"Error detected in response")
                
        except Exception as e:
            print(f"Exception during test: {str(e)}")
    
    print("\nAll tests completed.")