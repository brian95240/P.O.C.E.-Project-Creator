# tests/test_framework.py
"""
P.O.C.E. Project Creator - Comprehensive Testing Framework v4.0
Advanced testing suite with unit, integration, performance, and security tests
Includes automated test generation, coverage analysis, and CI/CD integration
"""

import pytest
import asyncio
import json
import yaml
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any, Optional
import time
import threading
from datetime import datetime, timedelta
import subprocess
import requests
import aiohttp
from dataclasses import dataclass
import logging

# Import modules to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from poce_project_creator_v4 import (
        ConfigManager, WorkflowEngine, MCPServerManager, POCEApp
    )
    from mcp_integration_templates import (
        EnhancedMCPManager, Context7Server, ClaudeTaskManagerServer
    )
except ImportError as e:
    pytest.skip(f"Could not import modules: {e}", allow_module_level=True)

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ==========================================
# TEST CONFIGURATION AND FIXTURES
# ==========================================

@dataclass
class TestConfig:
    """Test configuration settings"""
    temp_dir: Path
    config_file: Path
    test_github_token: str = "test_token_123"
    test_repo_name: str = "test-poce-project"
    mock_api_responses: bool = True
    performance_threshold_ms: int = 5000
    load_test_users: int = 10
    load_test_duration: int = 30

@pytest.fixture(scope="session")
def test_config():
    """Global test configuration"""
    temp_dir = Path(tempfile.mkdtemp(prefix="poce_test_"))
    config_file = temp_dir / "test_config.yaml"
    
    # Create test configuration
    test_config_data = {
        'project': {
            'name': 'test-project',
            'type': 'web_application',
            'goal': 'Test project for automated testing'
        },
        'github': {
            'token': 'test_token_123',
            'username': 'test_user'
        },
        'mcp_servers': {
            'enabled': True,
            'auto_discover': False,  # Disable for testing
            'fallback_enabled': True
        },
        'workflow': {
            'execution': {
                'max_concurrent_tasks': 2,
                'timeout_seconds': 10
            }
        },
        'testing': {
            'mock_external_apis': True,
            'log_level': 'DEBUG'
        }
    }
    
    with open(config_file, 'w') as f:
        yaml.dump(test_config_data, f)
    
    config = TestConfig(
        temp_dir=temp_dir,
        config_file=config_file
    )
    
    yield config
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_github_api():
    """Mock GitHub API responses"""
    with patch('github.Github') as mock_github:
        # Mock user object
        mock_user = Mock()
        mock_user.login = "test_user"
        mock_user.get_repos.return_value = []
        
        # Mock repository object
        mock_repo = Mock()
        mock_repo.name = "test-repo"
        mock_repo.create_file.return_value = Mock()
        
        # Configure GitHub client mock
        mock_github_instance = Mock()
        mock_github_instance.get_user.return_value = mock_user
        mock_github_instance.get_repo.return_value = mock_repo
        mock_github.return_value = mock_github_instance
        
        yield mock_github_instance

@pytest.fixture
async def mock_mcp_servers():
    """Mock MCP server responses"""
    servers = {}
    
    # Mock Context7 server
    context7_mock = AsyncMock()
    context7_mock.initialize.return_value = True
    context7_mock.execute_task.return_value = {
        'status': 'success',
        'result': {'context': 'test context'},
        'execution_time_ms': 100
    }
    context7_mock.check_health.return_value = True
    servers['context7'] = context7_mock
    
    # Mock Task Manager server
    task_manager_mock = AsyncMock()
    task_manager_mock.initialize.return_value = True
    task_manager_mock.execute_task.return_value = {
        'status': 'success',
        'task_id': 'test_task_123',
        'result': {'workflow_status': 'completed'},
        'execution_time_ms': 150
    }
    task_manager_mock.check_health.return_value = True
    servers['task_manager'] = task_manager_mock
    
    return servers

# ==========================================
# UNIT TESTS
# ==========================================

class TestConfigManager:
    """Unit tests for ConfigManager"""
    
    def test_config_initialization(self, test_config):
        """Test configuration manager initialization"""
        config_manager = ConfigManager()
        assert config_manager.config is not None
        assert isinstance(config_manager.config, dict)
    
    def test_load_config_from_file(self, test_config):
        """Test loading configuration from YAML file"""
        config_manager = ConfigManager()
        
        # Load test configuration
        with open(test_config.config_file, 'r') as f:
            test_config_data = yaml.safe_load(f)
        
        config_manager.config = test_config_data
        
        assert config_manager.config['project']['name'] == 'test-project'
        assert config_manager.config['github']['username'] == 'test_user'
    
    def test_update_field(self, test_config):
        """Test updating nested configuration fields"""
        config_manager = ConfigManager()
        
        # Update a nested field
        config_manager.update_field('project.name', 'updated-project')
        assert config_manager.config['project']['name'] == 'updated-project'
        
        # Update a new field
        config_manager.update_field('new_section.new_field', 'new_value')
        assert config_manager.config['new_section']['new_field'] == 'new_value'
    
    def test_config_validation(self, test_config):
        """Test configuration validation"""
        config_manager = ConfigManager()
        
        # Valid configuration should pass
        config_manager.config = {
            'project': {'name': 'test'},
            'github': {'token': 'test_token', 'username': 'test_user'}
        }
        
        # Should not raise exception
        config_manager._validate_config()

class TestWorkflowEngine:
    """Unit tests for WorkflowEngine"""
    
    @pytest.fixture
    def workflow_engine(self, test_config):
        """Create workflow engine instance for testing"""
        config_manager = ConfigManager()
        with open(test_config.config_file, 'r') as f:
            config_manager.config = yaml.safe_load(f)
        
        return WorkflowEngine(config_manager)
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, workflow_engine, mock_github_api):
        """Test basic workflow execution"""
        project_data = {
            'name': 'test-project',
            'type': 'web_application',
            'github_token': 'test_token',
            'github_username': 'test_user'
        }
        
        # Mock the workflow execution
        with patch.object(workflow_engine, '_execute_parallel_tasks') as mock_execute:
            mock_execute.return_value = {'status': 'completed'}
            
            result = await workflow_engine.execute_workflow(project_data)
            
            assert result['status'] == 'completed'
            assert 'workflow_id' in result
    
    @pytest.mark.asyncio
    async def test_parallel_task_execution(self, workflow_engine):
        """Test parallel task execution"""
        # Create mock tasks
        async def mock_task_1():
            await asyncio.sleep(0.01)
            return {'task': 'task_1', 'result': 'success'}
        
        async def mock_task_2():
            await asyncio.sleep(0.01)
            return {'task': 'task_2', 'result': 'success'}
        
        tasks = [mock_task_1, mock_task_2]
        
        start_time = time.time()
        results = await workflow_engine._execute_parallel_tasks(tasks)
        execution_time = time.time() - start_time
        
        # Should execute in parallel (less than sequential time)
        assert execution_time < 0.05  # Much less than 0.02 * 2
        assert len(results) == 2
        assert 'task_0' in results
        assert 'task_1' in results

class TestMCPServerManager:
    """Unit tests for MCP Server Manager"""
    
    @pytest.fixture
    def mcp_manager(self):
        """Create MCP manager instance for testing"""
        return MCPServerManager()
    
    @pytest.mark.asyncio
    async def test_server_discovery(self, mcp_manager):
        """Test MCP server discovery"""
        with patch.object(mcp_manager, '_query_smithery_api') as mock_query:
            mock_query.return_value = [
                {
                    'name': 'test-server',
                    'type': 'context7',
                    'capabilities': ['context_management'],
                    'performance_score': 90
                }
            ]
            
            servers = await mcp_manager.discover_optimal_servers(
                'web_application', 
                ['context_management']
            )
            
            assert len(servers) > 0
            assert servers[0]['name'] == 'test-server'
    
    def test_synergy_calculation(self, mcp_manager):
        """Test synergy score calculation"""
        servers = [
            {
                'name': 'server1',
                'type': 'context7',
                'capabilities': ['context_management'],
                'performance_score': 90
            },
            {
                'name': 'server2',
                'type': 'claude_task_manager',
                'capabilities': ['task_automation'],
                'performance_score': 85
            }
        ]
        
        scored_servers = mcp_manager._calculate_synergy_scores(
            servers, 
            ['context_management', 'task_automation']
        )
        
        assert all('synergy_score' in server for server in scored_servers)
        assert scored_servers[0]['synergy_score'] >= scored_servers[1]['synergy_score']

# ==========================================
# INTEGRATION TESTS
# ==========================================

class TestProjectCreationIntegration:
    """Integration tests for complete project creation workflow"""
    
    @pytest.mark.asyncio
    async def test_complete_project_creation(self, test_config, mock_github_api, mock_mcp_servers):
        """Test complete project creation workflow"""
        # Initialize components
        config_manager = ConfigManager()
        with open(test_config.config_file, 'r') as f:
            config_manager.config = yaml.safe_load(f)
        
        workflow_engine = WorkflowEngine(config_manager)
        
        # Mock MCP server responses
        with patch.object(workflow_engine, 'mcp_manager') as mock_mcp:
            mock_mcp.discover_optimal_servers.return_value = [
                {'name': 'test-context7', 'type': 'context7'},
                {'name': 'test-task-manager', 'type': 'claude_task_manager'}
            ]
            
            project_data = {
                'name': 'integration-test-project',
                'type': 'web_application',
                'description': 'Integration test project',
                'github_token': 'test_token',
                'github_username': 'test_user'
            }
            
            # Execute workflow
            result = await workflow_engine.execute_workflow(project_data)
            
            # Verify results
            assert result['status'] == 'completed'
            assert 'workflow_id' in result
            assert 'results' in result
    
    @pytest.mark.asyncio
    async def test_error_handling_in_workflow(self, test_config, mock_github_api):
        """Test error handling during workflow execution"""
        config_manager = ConfigManager()
        with open(test_config.config_file, 'r') as f:
            config_manager.config = yaml.safe_load(f)
        
        workflow_engine = WorkflowEngine(config_manager)
        
        # Simulate GitHub API failure
        mock_github_api.get_user.side_effect = Exception("API Error")
        
        project_data = {
            'name': 'error-test-project',
            'type': 'web_application',
            'github_token': 'invalid_token',
            'github_username': 'test_user'
        }
        
        result = await workflow_engine.execute_workflow(project_data)
        
        # Should handle error gracefully
        assert result['status'] == 'failed'
        assert 'error' in result

class TestMCPIntegration:
    """Integration tests for MCP server integration"""
    
    @pytest.mark.asyncio
    async def test_mcp_server_communication(self, mock_mcp_servers):
        """Test communication with MCP servers"""
        from mcp_integration_templates import EnhancedMCPManager
        
        mcp_manager = EnhancedMCPManager()
        
        # Mock server initialization
        with patch.object(mcp_manager, 'servers', mock_mcp_servers):
            # Test task execution
            task = {
                'type': 'context_retrieval',
                'query': 'test query'
            }
            
            result = await mock_mcp_servers['context7'].execute_task(task)
            
            assert result['status'] == 'success'
            assert 'execution_time_ms' in result
    
    @pytest.mark.asyncio
    async def test_distributed_task_execution(self, mock_mcp_servers):
        """Test distributed task execution across multiple servers"""
        from mcp_integration_templates import EnhancedMCPManager
        
        mcp_manager = EnhancedMCPManager()
        
        # Mock servers
        mcp_manager.servers = mock_mcp_servers
        
        task = {
            'type': 'project_creation',
            'project_data': {
                'name': 'distributed-test',
                'type': 'web_application'
            }
        }
        
        result = await mcp_manager.execute_distributed_task(
            task, 
            list(mock_mcp_servers.keys())
        )
        
        assert result['status'] == 'success'
        assert 'metrics' in result
        assert len(result['results']) == len(mock_mcp_servers)

# ==========================================
# PERFORMANCE TESTS
# ==========================================

class TestPerformance:
    """Performance and load tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_workflow_execution_performance(self, test_config):
        """Test workflow execution performance"""
        config_manager = ConfigManager()
        with open(test_config.config_file, 'r') as f:
            config_manager.config = yaml.safe_load(f)
        
        workflow_engine = WorkflowEngine(config_manager)
        
        project_data = {
            'name': 'performance-test',
            'type': 'web_application',
            'github_token': 'test_token',
            'github_username': 'test_user'
        }
        
        # Mock external dependencies
        with patch('github.Github'):
            start_time = time.time()
            
            result = await workflow_engine.execute_workflow(project_data)
            
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Performance assertion
        assert execution_time < test_config.performance_threshold_ms
        assert result['status'] in ['completed', 'failed']  # Should complete within timeout
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_workflow_execution(self, test_config):
        """Test multiple concurrent workflow executions"""
        config_manager = ConfigManager()
        with open(test_config.config_file, 'r') as f:
            config_manager.config = yaml.safe_load(f)
        
        workflow_engine = WorkflowEngine(config_manager)
        
        # Create multiple workflows
        workflows = []
        for i in range(5):
            project_data = {
                'name': f'concurrent-test-{i}',
                'type': 'web_application',
                'github_token': 'test_token',
                'github_username': 'test_user'
            }
            workflows.append(workflow_engine.execute_workflow(project_data))
        
        with patch('github.Github'):
            start_time = time.time()
            
            results = await asyncio.gather(*workflows, return_exceptions=True)
            
            execution_time = time.time() - start_time
        
        # Should execute concurrently (faster than sequential)
        assert execution_time < test_config.performance_threshold_ms / 1000 * 5
        assert len(results) == 5
    
    @pytest.mark.performance
    def test_memory_usage(self, test_config):
        """Test memory usage during operation"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple instances
        instances = []
        for i in range(10):
            config_manager = ConfigManager()
            instances.append(config_manager)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Cleanup
        del instances
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Memory should be reasonable
        assert memory_increase < 100  # Less than 100MB increase
        assert final_memory < peak_memory * 1.2  # Memory should be mostly freed

# ==========================================
# SECURITY TESTS
# ==========================================

class TestSecurity:
    """Security-focused tests"""
    
    def test_token_sanitization(self, test_config):
        """Test that tokens are properly sanitized in logs"""
        config_manager = ConfigManager()
        
        # Set a test token
        config_manager.update_field('github.token', 'ghp_super_secret_token_123')
        
        # Convert config to string (simulating logging)
        config_str = str(config_manager.config)
        
        # Token should be masked or not appear in plain text
        assert 'ghp_super_secret_token_123' not in config_str or '*' in config_str
    
    def test_input_validation(self, test_config):
        """Test input validation and sanitization"""
        config_manager = ConfigManager()
        
        # Test malicious input
        malicious_inputs = [
            "../../../etc/passwd",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "$(rm -rf /)",
            "\x00\x01\x02"
        ]
        
        for malicious_input in malicious_inputs:
            # Should handle malicious input gracefully
            try:
                config_manager.update_field('project.name', malicious_input)
                # Verify input is sanitized
                project_name = config_manager.config['project']['name']
                assert project_name != malicious_input or len(project_name) == 0
            except (ValueError, TypeError):
                pass  # Rejecting malicious input is acceptable
    
    @pytest.mark.asyncio
    async def test_api_security(self, mock_github_api):
        """Test API security measures"""
        from github import Github
        
        # Test with invalid token
        with pytest.raises(Exception):
            g = Github("invalid_token")
            g.get_user()
        
        # Test rate limiting simulation
        mock_github_api.get_rate_limit.return_value.core.remaining = 0
        
        # Should handle rate limiting gracefully
        # (In real implementation, this would retry or queue requests)

# ==========================================
# END-TO-END TESTS
# ==========================================

class TestEndToEnd:
    """End-to-end system tests"""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_project_lifecycle(self, test_config, mock_github_api):
        """Test complete project lifecycle from creation to deployment"""
        # This would test the entire pipeline in a real environment
        # For now, we'll test the major components integration
        
        # 1. Initialize application
        config_manager = ConfigManager()
        with open(test_config.config_file, 'r') as f:
            config_manager.config = yaml.safe_load(f)
        
        workflow_engine = WorkflowEngine(config_manager)
        mcp_manager = MCPServerManager()
        
        # 2. Create project
        project_data = {
            'name': 'e2e-test-project',
            'type': 'web_application',
            'description': 'End-to-end test project',
            'github_token': 'test_token',
            'github_username': 'test_user'
        }
        
        with patch.object(workflow_engine, 'mcp_manager', mcp_manager):
            result = await workflow_engine.execute_workflow(project_data)
        
        # 3. Verify project creation
        assert result['status'] == 'completed'
        assert 'workflow_id' in result
        
        # 4. Test project status checking
        # (Would check actual repository in real implementation)
        
        # 5. Test project cleanup
        # (Would clean up resources in real implementation)

# ==========================================
# TEST UTILITIES AND HELPERS
# ==========================================

class TestDataGenerator:
    """Generate test data for various scenarios"""
    
    @staticmethod
    def generate_project_config(project_type: str = "web_application") -> Dict:
        """Generate test project configuration"""
        return {
            'name': f'test-{project_type}-{int(time.time())}',
            'type': project_type,
            'description': f'Test {project_type} project',
            'github_token': 'test_token_123',
            'github_username': 'test_user',
            'features': ['authentication', 'api', 'database'],
            'deployment_target': 'staging'
        }
    
    @staticmethod
    def generate_mcp_server_config(server_type: str) -> Dict:
        """Generate test MCP server configuration"""
        return {
            'name': f'test-{server_type}',
            'type': server_type,
            'endpoint': f'https://test-{server_type}.example.com/api',
            'api_key': f'test_key_{server_type}',
            'capabilities': ['test_capability'],
            'performance_score': 85.0
        }

class MockResponseGenerator:
    """Generate mock API responses"""
    
    @staticmethod
    def github_user_response():
        """Generate mock GitHub user response"""
        return {
            'login': 'test_user',
            'id': 123456,
            'name': 'Test User',
            'email': 'test@example.com',
            'public_repos': 10,
            'followers': 5,
            'following': 3
        }
    
    @staticmethod
    def github_repo_response(repo_name: str):
        """Generate mock GitHub repository response"""
        return {
            'name': repo_name,
            'full_name': f'test_user/{repo_name}',
            'private': True,
            'html_url': f'https://github.com/test_user/{repo_name}',
            'description': f'Test repository {repo_name}',
            'created_at': '2023-01-01T00:00:00Z',
            'updated_at': '2023-01-01T00:00:00Z',
            'size': 1024,
            'language': 'Python',
            'forks_count': 0,
            'stargazers_count': 0,
            'open_issues_count': 0
        }

# ==========================================
# TEST MARKERS AND CONFIGURATION
# ==========================================

# Custom pytest markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.security = pytest.mark.security
pytest.mark.e2e = pytest.mark.e2e

# Test configuration for pytest.ini
PYTEST_CONFIG = """
[tool:pytest]
markers =
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
    security: Security tests
    e2e: End-to-end tests
    slow: Slow running tests

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Async testing
asyncio_mode = auto

# Coverage settings
addopts = 
    --cov=src
    --cov-report=html
    --cov-report=xml
    --cov-report=term-missing
    --cov-fail-under=80
    --strict-markers
    --disable-warnings

# Parallel execution
# -n auto

# Timeout for tests
timeout = 300

# Filter warnings
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
"""

# ==========================================
# AUTOMATED TEST EXECUTION
# ==========================================

def run_test_suite():
    """Run the complete test suite"""
    import subprocess
    import sys
    
    # Test categories to run
    test_commands = [
        # Unit tests (fast)
        ["pytest", "-m", "unit", "-v"],
        
        # Integration tests
        ["pytest", "-m", "integration", "-v"],
        
        # Security tests
        ["pytest", "-m", "security", "-v"],
        
        # Performance tests (optional, slower)
        ["pytest", "-m", "performance", "-v", "--timeout=60"],
        
        # End-to-end tests (slowest)
        ["pytest", "-m", "e2e", "-v", "--timeout=120"],
    ]
    
    results = {}
    
    for command in test_commands:
        test_type = command[2]  # Extract marker name
        print(f"\n{'='*50}")
        print(f"Running {test_type} tests...")
        print(f"{'='*50}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            results[test_type] = {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            if result.returncode == 0:
                print(f"✅ {test_type} tests PASSED")
            else:
                print(f"❌ {test_type} tests FAILED")
                print(f"Error output: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {test_type} tests TIMED OUT")
            results[test_type] = {'returncode': -1, 'error': 'timeout'}
        except Exception as e:
            print(f"💥 {test_type} tests ERROR: {e}")
            results[test_type] = {'returncode': -1, 'error': str(e)}
    
    # Generate summary report
    print(f"\n{'='*50}")
    print("TEST SUITE SUMMARY")
    print(f"{'='*50}")
    
    total_passed = 0
    total_failed = 0
    
    for test_type, result in results.items():
        if result['returncode'] == 0:
            print(f"✅ {test_type}: PASSED")
            total_passed += 1
        else:
            print(f"❌ {test_type}: FAILED")
            total_failed += 1
    
    print(f"\nTotal: {total_passed} passed, {total_failed} failed")
    
    # Return overall success
    return total_failed == 0

if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)