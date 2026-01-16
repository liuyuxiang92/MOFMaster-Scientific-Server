"""
Load tests for MOF Tools.

Ensures consistency between tool definitions (YAML) and implementation (code).
"""

import pytest
import tools
from tool_registry import get_registry


class TestToolConsistency:
    """Test consistency between configuration and implementation."""
    
    def test_all_yaml_tools_exist_in_code(self):
        """Verify every tool in tool_definitions.yaml exists in tools.py."""
        from main import load_tool_definitions
        
        definitions = load_tool_definitions()
        
        for tool_def in definitions:
            function_name = tool_def['function_name']
            
            # Check if function exists in tools module
            assert hasattr(tools, function_name), f"Function {function_name} defined in YAML not found in tools.py"
            
            # Check if it is callable
            func = getattr(tools, function_name)
            assert callable(func), f"Object {function_name} in tools.py is not callable"
            
    def test_tool_registry_consistency(self):
        """Verify that registered tools match the YAML definitions."""
        from main import load_tool_definitions, register_tools_in_registry
        
        # Reload and register
        registry = get_registry()
        registry.clear()
        register_tools_in_registry()
        
        definitions = load_tool_definitions()
        
        assert len(registry) == len(definitions)
        
        for tool_def in definitions:
            tool_name = tool_def['name']
            metadata = registry.get(tool_name)
            
            assert metadata is not None, f"Tool {tool_name} from YAML not found in registry"
            assert metadata.description == tool_def['description']
            assert metadata.category.name == tool_def['category']


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
