import httpx
import os
from typing import Dict, Any

# Import the base tool
from .base import BaseTool

class Knowledge_Graph_Get_Node_Info_Tool(BaseTool):
    """
    A tool that queries a knowledge graph for information about a specific node.
    """
    
    def __init__(self, api_endpoint: str = None, graph_type: str = None):
        """
        Initialize the Knowledge Graph Node Tool.
        
        Args:
            api_endpoint (str, optional): Custom API endpoint URL. If not provided, it will be constructed at runtime.
            graph_type (str, optional): Default type of knowledge graph to use. If not provided, must be specified during execution.
        """
        super().__init__(
            tool_name="Knowledge_Graph_Get_Node_Info_Tool",
            tool_description="A tool that queries a knowledge graph for information about a specific node, returning the node and its relationships. When using this tool, make sure you have obtained the precise entity name through the Knowledge_Graph_Entity_Matcher_Tool.",
            tool_version="1.0.0",
            input_types={
                "node_name": "str - The name of the node to search for in the knowledge graph. (The entity name must be the precise entity name obtained through the Knowledge_Graph_Entity_Matcher_Tool)",
                "graph_type": "str - Type of knowledge graph to query. Options: 'agriculture', 'mix', 'legal', 'cs'. Required if not set during initialization."
            },
            output_type="dict - A dictionary containing information about the node and its relationships.",
            demo_commands=[
                {
                    "command": 'result = tool.execute(node_name="Tampa Electric Company", graph_type="agriculture")',
                    "description": "Get information about the 'Tampa Electric Company' node in the agriculture knowledge graph."
                },
                {
                    "command": 'result = tool.execute(node_name="Supreme Court", graph_type="legal")',
                    "description": "Get information about the 'Supreme Court' node in the legal knowledge graph."
                },
                {
                    "command": 'result = tool.execute(node_name="Python", graph_type="cs")',
                    "description": "Get information about the 'Python' programming language in the computer science knowledge graph."
                }
            ],
            user_metadata={
                "api_endpoint": api_endpoint,
                "description": "A knowledge graph API that provides information about various entities and their relationships.",
                "supported_graph_types": ["agriculture", "mix", "legal", "cs"]
            }
        )
        self.api_endpoint = api_endpoint if api_endpoint is not None else '127.0.0.1:9000'
        self.graph_type = graph_type
    
    def execute(self, node_name: str, graph_type: str) -> str:
        """
        Query the knowledge graph for information about a specific node.
        
        Args:
            node_name (str): The name of the node to search for.
            graph_type (str): Type of knowledge graph to query. Options: 'agriculture', 'mix', 'legal', 'cs'.
            
        Returns:
            str: A formatted string containing information about the node and its relationships.
        """
        if not graph_type:
            return "Error: graph_type must be specified ('agriculture', 'mix', 'legal', or 'cs')"
            
        #print(f"Querying information for node '{node_name}' in the {graph_type} knowledge graph...")
        
        try:
            # Construct the API endpoint based on the specified graph_type
            endpoint = f"http://{self.api_endpoint}/{graph_type}/get_node"
            
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
                return f"Error querying node: {response.status_code}, {response.text}"
                
        except Exception as e:
            return f"Error executing node query: {str(e)}"


# Test script for the Knowledge Graph Get Node Info Tool
if __name__ == "__main__":
    import json
    
    # Create an instance of the tool
    node_info_tool = Knowledge_Graph_Get_Node_Info_Tool()
    
    # Test cases
    test_cases = [
        # {"node_name": "King Solomon's Mines", "graph_type": "mix"},
        {"node_name": "Tampa Electric Company", "graph_type": "legal"},
        # {"node_name": "Supreme Court", "graph_type": "legal"}
    ]
    
    # Run tests
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['node_name']} ---")
        try:
            result = node_info_tool.execute(**test_case)
            
            # Check for errors
            if result.startswith("Error"):
                print(f"Error detected in response: {result}")
            else:
                print(f"Successfully retrieved information for '{test_case['node_name']}'")
                print(f"Graph type: {test_case['graph_type']}")
                
                # Print first 300 characters of the result
                print(f"Result preview (first 300 chars):")
                print(result[:300] + "..." if len(result) > 300 else result)
                
        except Exception as e:
            print(f"Exception during test: {str(e)}")
    
    print("\nAll tests completed.")