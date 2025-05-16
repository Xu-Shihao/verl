import os
import sys
from typing import List, Dict, Any, Optional

#sys.path.append('/mnt/afs/fengji/works/Search-R1')

from search_r1.tools.base import BaseTool

class Knowledge_Graph_Searcher_Tool(BaseTool):
    """
    A tool that searches a knowledge graph database for entities related to a given entity and relationship.
    """
    
    def __init__(self):
        """
        Initialize the Knowledge Graph Searcher Tool.
        """
        super().__init__(
            tool_name="Knowledge_Graph_Searcher_Tool",
            tool_description="A tool that searches a knowledge graph database for entities related to a given entity and relationship. The database contains triples in the format 'entity|relationship|entity'.",
            tool_version="1.0.0",
            input_types={
                "entity": "str - The entity to search for in the knowledge graph.",
                "relationship": "str - The relationship to filter by. You can only choose from the following relationships: directed_by, has_genre, has_imdb_rating, has_imdb_votes, has_tags, in_language, release_year, starred_actors, written_by",
                "reverse_search": "bool - Whether to search for the entity as the subject (False) or object (True) of the relationship. Default is False."
            },
            output_type="list - A list of entities that have the specified relationship with the input entity.",
            demo_commands=[
                {
                    "command": 'result = tool.execute(entity="Kismet", relationship="directed_by")',
                    "description": "Search for who directed the movie 'Kismet'."
                },
                {
                    "command": 'result = tool.execute(entity="Kismet", relationship="starred_actors")',
                    "description": "Search for all actors who starred in the movie 'Kismet'."
                },
                {
                    "command": 'result = tool.execute(entity="Keanu Reeves", relationship="starred_actors", reverse_search=True)',
                    "description": "Search for all movies that starred Keanu Reeves."
                },
            ],
            user_metadata={
                "database_path": "kgdb/kb.txt",
                "format": "entity|relationship|entity",
                "description": "A knowledge graph database containing information about various entities and their relationships.",
                "available_relationships": [
                    "directed_by", "has_genre", "has_imdb_rating", "has_imdb_votes", 
                    "has_tags", "in_language", "release_year", "starred_actors", "written_by"
                ]
            }
        )
        self.database_path = os.path.dirname(os.path.abspath(__file__))
        self.database_path = os.path.join(self.database_path, 'data/kb.txt')
        self._verify_database()
        
    def _verify_database(self):
        """
        Verify that the database file exists and is accessible.
        """
        if not os.path.exists(self.database_path):
            print(f"Warning: Knowledge graph database file not found at {self.database_path}")
    
    def get_available_relationships(self) -> List[str]:
        """
        Get all available relationship types in the knowledge graph database.
        
        Returns:
            List[str]: A list of all unique relationship types found in the database.
        """
        if not os.path.exists(self.database_path):
            return [f"Error: Database file not found at {self.database_path}"]
        
        try:
            import subprocess
            
            # 使用shell命令提取所有独特的关系类型
            cmd = ["bash", "-c", f"cut -d'|' -f2 {self.database_path} | sort | uniq"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                relationships = result.stdout.strip().split('\n')
                return relationships
            else:
                return ["Error: Failed to extract relationships from database"]
            
        except Exception as e:
            return [f"Error extracting relationships: {str(e)}"]
    
    def get_relationship_counts(self) -> Dict[str, int]:
        """
        Get counts of all relationship types in the knowledge graph database.
        
        Returns:
            Dict[str, int]: A dictionary mapping relationship types to their counts.
        """
        if not os.path.exists(self.database_path):
            return {"error": f"Database file not found at {self.database_path}"}
        
        try:
            relationships = self.get_available_relationships()
            if relationships and isinstance(relationships[0], str) and relationships[0].startswith("Error"):
                return {"error": relationships[0]}
            
            counts = {}
            import subprocess
            
            for rel in relationships:
                cmd = ["bash", "-c", f"grep -E '\\|{rel}\\|' {self.database_path} | wc -l"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout:
                    counts[rel] = int(result.stdout.strip())
            
            return counts
            
        except Exception as e:
            return {"error": f"Error counting relationships: {str(e)}"}
    
    def search_knowledge_graph(self, entity: str, relationship: str, reverse_search: bool = False) -> List[str]:
        """
        Search the knowledge graph for entities related to the input entity through the specified relationship.
        
        Args:
            entity (str): The entity to search for.
            relationship (str): The relationship to filter by.
            reverse_search (bool): If True, search for entity as the object of the relationship instead of the subject.
            
        Returns:
            List[str]: A list of entities that have the specified relationship with the input entity.
        """
        if not os.path.exists(self.database_path):
            return [f"Error: Database file not found at {self.database_path}"]
        
        results = []
        
        try:
            # 使用grep高效搜索大文件
            import subprocess
            
            if not reverse_search:
                # 正向搜索: entity是主语 (entity|relationship|?)
                search_pattern = f"^{entity}\\|{relationship}\\|"
                position = 0  # 主语位置
                result_position = 2  # 从0开始计数，第三个字段
            else:
                # 反向搜索: entity是宾语 (?|relationship|entity)
                search_pattern = f"\\|{relationship}\\|{entity}$"
                position = 2  # 宾语位置
                result_position = 0  # 从0开始计数，第一个字段
                
            # 执行grep命令
            grep_cmd = ["grep", "-E", search_pattern, self.database_path]
            grep_result = subprocess.run(grep_cmd, capture_output=True, text=True)
            
            # 处理结果
            if grep_result.returncode == 0 and grep_result.stdout:
                for line in grep_result.stdout.strip().split('\n'):
                    # 分割行并获取结果元素
                    parts = line.split('|')
                    if len(parts) >= 3 and parts[position] == entity:
                        results.append(parts[result_position])
            
            return results
            
        except Exception as e:
            return [f"Error searching knowledge graph: {str(e)}"]
    
    def execute(self, entity: str, relationship: str, reverse_search: bool = False) -> List[str]:
        """
        Execute the knowledge graph search.
        
        Args:
            entity (str): The entity to search for.
            relationship (str): The relationship to filter by.
            reverse_search (bool): If True, search for entity as the object of the relationship instead of the subject.
            
        Returns:
            List[str]: A list of entities that have the specified relationship with the input entity.
        """
        search_type = "reverse" if reverse_search else "forward"
        print(f"Performing {search_type} search for '{entity}' with relationship '{relationship}'...")
        
        results = self.search_knowledge_graph(entity, relationship, reverse_search)
        
        if not results:
            print(f"No results found for '{entity}' with relationship '{relationship}' ({search_type} search)")
        else:
            print(f"Found {len(results)} results")
        
        return results


if __name__ == "__main__":
    # Test the tool
    tool = Knowledge_Graph_Searcher_Tool()
    
    # Print tool metadata
    print("\nTool Metadata:")
    metadata = tool.get_metadata()
    for key, value in metadata.items():
        if key != "demo_commands" and key != "input_types":
            print(f"{key}: {value}")
    
    # Print available relationships
    print("\nAvailable Relationships:")
    relationships = tool.get_available_relationships()
    for rel in relationships:
        print(f"- {rel}")
    
    # Print relationship counts
    print("\nRelationship Counts:")
    counts = tool.get_relationship_counts()
    for rel, count in counts.items():
        print(f"- {rel}: {count} instances")
    
    # Test with examples
    test_examples = [
        # Forward searches (entity is subject)
        ("Kismet", "directed_by", False),
        ("Kismet", "starred_actors", False),
        ("Flags of Our Fathers", "has_tags", False),
        
        # Reverse searches (entity is object)
        ("Keanu Reeves", "starred_actors", True),
        ("Clint Eastwood", "directed_by", True)
    ]
    
    for entity, relationship, reverse in test_examples:
        search_type = "reverse" if reverse else "forward"
        print(f"\nTest: entity='{entity}', relationship='{relationship}', {search_type} search")
        result = tool.execute(entity=entity, relationship=relationship, reverse_search=reverse)
        print(f"Results ({len(result)}):")
        for item in result:
            print(f"- {item}") 