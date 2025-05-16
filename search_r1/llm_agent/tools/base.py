from typing import Dict, Any

class BaseTool:
    """
    Base class for all knowledge graph tools.
    """
    
    def __init__(
        self,
        tool_name: str,
        tool_description: str,
        tool_version: str,
        input_types: Dict[str, str],
        output_type: str,
        demo_commands: list,
        user_metadata: Dict[str, Any]
    ):
        """
        Initialize the base tool.
        
        Args:
            tool_name: Name of the tool
            tool_description: Description of what the tool does
            tool_version: Version of the tool
            input_types: Dictionary of input parameter names and their types/descriptions
            output_type: Description of the tool's output
            demo_commands: List of example commands for the tool
            user_metadata: Additional metadata for the tool
        """
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.tool_version = tool_version
        self.input_types = input_types
        self.output_type = output_type
        self.demo_commands = demo_commands
        self.user_metadata = user_metadata
    
    def execute(self, **kwargs):
        """
        Execute the tool's functionality. This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the execute method.") 