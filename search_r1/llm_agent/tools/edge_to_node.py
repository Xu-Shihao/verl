import httpx
import os
from typing import Dict, Any

# Import the base tool
from .base import BaseTool

class Knowledge_Graph_Node_Edge_To_Node_Tool(BaseTool):
    """
    A tool that finds connected nodes in a knowledge graph based on a node and relationship.
    """
    
    def __init__(self, api_endpoint: str = None, graph_type: str = None):
        """
        Initialize the Knowledge Graph Edge Tool.
        
        Args:
            api_endpoint (str, optional): Custom API endpoint URL. If not provided, it will be constructed at runtime.
            graph_type (str, optional): Default type of knowledge graph to use. If not provided, must be specified during execution.
        """
        super().__init__(
            tool_name="Knowledge_Graph_Node_Edge_To_Node_Tool",
            tool_description="A tool that finds connected nodes in a knowledge graph based on a starting node and a specified relationship. When using this tool, make sure you have obtained the precise entity and relationship name through the Knowledge_Graph_Entity_Matcher_Tool and Knowledge_Graph_Get_Node_Info_Tool.",
            tool_version="1.0.0",
            input_types={
                "node_name": "str - The name of the starting node. (The entity name must be the precise entity name obtained through the Knowledge_Graph_Entity_Matcher_Tool)",
                "edge_name": "str - The name of the relationship/edge to follow. (The relationship name must be the precise relationship name obtained through the Knowledge_Graph_Get_Node_Info_Tool)",
                "reverse": "bool - Whether to follow the relationship in reverse direction. Default is False.",
                "graph_type": "str - Type of knowledge graph to query. Options: 'agriculture', 'mix', 'legal', 'cs'. Required if not set during initialization."
            },
            output_type="dict - A dictionary containing information about the connected nodes.",
            demo_commands=[
                {
                    "command": 'result = tool.execute(node_name="Tampa Electric Company", edge_name="borrowing, financial assistance", reverse=False, graph_type="agriculture")',
                    "description": "Find nodes connected to 'Tampa Electric Company' through the 'borrowing, financial assistance' relationship in the agriculture knowledge graph."
                },
                {
                    "command": 'result = tool.execute(node_name="Tampa Electric Company", edge_name="borrowing, financial assistance", graph_type="legal")',
                    "description": "Find nodes connected to 'Tampa Electric Company' through the 'borrowing, financial assistance' relationship in the legal knowledge graph."
                },
                {
                    "command": 'result = tool.execute(node_name="Algorithm", edge_name="used in", reverse=True, graph_type="cs")',
                    "description": "Find nodes that use 'Algorithm' in the computer science knowledge graph (reverse direction)."
                }
            ],
            user_metadata={
                "api_endpoint": api_endpoint,
                "description": "A knowledge graph API that finds connected nodes based on a specified relationship.",
                "supported_graph_types": ["agriculture", "mix", "legal", "cs"]
            }
        )
        self.api_endpoint = api_endpoint if api_endpoint is not None else '127.0.0.1:9000'
        self.graph_type = graph_type
    
    def execute(self, node_name: str, edge_name: str, graph_type: str, reverse: bool = False) -> str:
        """
        Find nodes connected to the starting node through the specified relationship.
        
        Args:
            node_name (str): The name of the starting node.
            edge_name (str): The name of the relationship/edge to follow.
            graph_type (str): Type of knowledge graph to query. Options: 'agriculture', 'mix', 'legal', 'cs'.
            reverse (bool): Whether to follow the relationship in reverse direction. Default is False.
            
        Returns:
            str: A formatted string containing information about the connected nodes.
        """
        if not graph_type:
            return "Error: graph_type must be specified ('agriculture', 'mix', 'legal', or 'cs')"
            
        direction = "reverse" if reverse else "forward"
        #print(f"Finding nodes connected to '{node_name}' through '{edge_name}' relationship ({direction} direction) in the {graph_type} knowledge graph...")
        
        try:
            # Construct the API endpoint based on the specified graph_type
            endpoint = f"http://{self.api_endpoint}/{graph_type}/get_precise_entity_names"
            
            # Prepare the request payload
            payload = {
                "node_name": node_name,
                "edge_name": edge_name,
                "reverse": reverse
            }
            
            # Make the API request using httpx
            with httpx.Client() as client:
                response = client.post(
                    endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
            
            #print(f"Response: {response}")
            #print(f"Response content: {response.text}")
            
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
                return f"Error finding connected nodes: {response.status_code}, {response.text}"
                
        except Exception as e:
            return f"Error executing edge query: {str(e)}"


# Test script for the Knowledge Graph Node Edge To Node Tool
if __name__ == "__main__":
    import json
    
    # Create an instance of the tool
    edge_to_node_tool = Knowledge_Graph_Node_Edge_To_Node_Tool()
    
    # Test cases
    test_cases = [
        {"node_name": "Tampa Electric Company", "edge_name": "credit management", "graph_type": "legal", "reverse": False},
        # {"node_name": "Algorithm", "edge_name": "used in", "graph_type": "cs", "reverse": True},
        # {"node_name": "Python", "edge_name": "is a", "graph_type": "cs", "reverse": False}
    ]
    
    # Run tests
    for i, test_case in enumerate(test_cases):
        print(f"\n--- Test Case {i+1}: {test_case['node_name']} with edge '{test_case['edge_name']}' ---")
        try:
            result = edge_to_node_tool.execute(**test_case)
            
            # Check for errors
            if result.startswith("Error"):
                print(f"Error detected in response: {result}")
            else:
                print(f"Successfully found connected nodes for '{test_case['node_name']}' through '{test_case['edge_name']}'")
                print(f"Graph type: {test_case['graph_type']}")
                print(f"Direction: {'reverse' if test_case['reverse'] else 'forward'}")
                
                # Print result preview
                print(f"Result preview (first 300 chars):")
                print(result[:300] + "..." if len(result) > 300 else result)
                
        except Exception as e:
            print(f"Exception during test: {str(e)}")
    
    print("\nAll tests completed.")