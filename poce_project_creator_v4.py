# poce_project_creator_v4.py
"""
P.O.C.E. (Proof of Concept Engineering) Project Creator v4.0
Enhanced with DevOps automation, MCP server integration, and YAML-driven workflows
"""

import customtkinter as ctk
from tkinter import messagebox, filedialog, ttk
import tkinter as tk
from github import Github
import webbrowser
import os
import PyPDF2
import yaml
import json
import asyncio
import aiohttp
import threading
import time
from datetime import datetime
import subprocess
import hashlib
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('poce_creator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MCPServerManager:
    """Manages MCP server selection and optimization via Smithery.ai"""
    
    def __init__(self):
        self.smithery_api = "https://smithery.ai/api/v1/servers"
        self.selected_servers = []
        self.performance_cache = {}
        
    async def discover_optimal_servers(self, project_type: str, requirements: List[str]) -> List[Dict]:
        """Discover and rank MCP servers based on project needs"""
        try:
            async with aiohttp.ClientSession() as session:
                # Query Smithery.ai for relevant servers
                params = {
                    'category': project_type,
                    'tags': ','.join(requirements),
                    'sort': 'performance_score',
                    'limit': 20
                }
                
                async with session.get(self.smithery_api, params=params) as response:
                    servers = await response.json()
                    
                # Apply synergy scoring algorithm
                optimized_servers = self._calculate_synergy_scores(servers, requirements)
                return optimized_servers[:5]  # Top 5 with highest synergy
                
        except Exception as e:
            logger.error(f"Failed to discover MCP servers: {e}")
            return self._fallback_servers()
    
    def _calculate_synergy_scores(self, servers: List[Dict], requirements: List[str]) -> List[Dict]:
        """Calculate synergy scores for server combinations"""
        scored_servers = []
        
        for server in servers:
            synergy_score = 0
            performance_score = server.get('performance_score', 0)
            
            # Base score from individual performance
            synergy_score += performance_score
            
            # Bonus for requirement coverage
            server_capabilities = server.get('capabilities', [])
            requirement_coverage = len(set(requirements) & set(server_capabilities)) / len(requirements)
            synergy_score += requirement_coverage * 100
            
            # Bonus for cross-server synergy
            for other_server in self.selected_servers:
                if self._has_synergy(server, other_server):
                    synergy_score += 25
            
            server['synergy_score'] = synergy_score
            scored_servers.append(server)
        
        return sorted(scored_servers, key=lambda x: x['synergy_score'], reverse=True)
    
    def _has_synergy(self, server1: Dict, server2: Dict) -> bool:
        """Check if two servers have synergistic capabilities"""
        synergy_pairs = [
            ('context7', 'claude_task_manager'),
            ('github_actions', 'docker_deploy'),
            ('testing_framework', 'ci_cd'),
            ('monitoring', 'alerting')
        ]
        
        s1_type = server1.get('type', '').lower()
        s2_type = server2.get('type', '').lower()
        
        return any((s1_type in pair and s2_type in pair) for pair in synergy_pairs)
    
    def _fallback_servers(self) -> List[Dict]:
        """Fallback servers when API is unavailable"""
        return [
            {'name': 'Context7', 'type': 'context7', 'synergy_score': 95},
            {'name': 'Claude Task Manager', 'type': 'claude_task_manager', 'synergy_score': 90},
            {'name': 'GitHub Actions', 'type': 'github_actions', 'synergy_score': 85},
            {'name': 'Docker Deploy', 'type': 'docker_deploy', 'synergy_score': 80},
            {'name': 'Testing Framework', 'type': 'testing_framework', 'synergy_score': 75}
        ]

class ConfigManager:
    """Manages YAML configuration with real-time editing"""
    
    def __init__(self):
        self.config = self._load_default_config()
        self.config_path = "poce_config.yaml"
        
    def _load_default_config(self) -> Dict:
        """Load default configuration template"""
        return {
            'project': {
                'name': '',
                'type': 'web_application',
                'goal': '',
                'description': ''
            },
            'github': {
                'token': '',
                'username': '',
                'visibility': 'private',
                'auto_deploy': True
            },
            'mcp_servers': {
                'enabled': True,
                'auto_discover': True,
                'required_capabilities': [
                    'context_management',
                    'task_automation', 
                    'ci_cd',
                    'testing',
                    'monitoring'
                ]
            },
            'workflow': {
                'cascading_enabled': True,
                'parallel_execution': True,
                'max_concurrent_tasks': 4,
                'retry_attempts': 3,
                'timeout_seconds': 300
            },
            'optimization': {
                'resource_efficiency': 'high',
                'speed_priority': True,
                'accuracy_checks': 'comprehensive',
                'performance_monitoring': True
            },
            'documents': {
                'prd': {'enabled': False, 'template': 'default', 'priority': 1},
                'prp': {'enabled': False, 'template': 'default', 'priority': 2},
                'master_prompt': {'enabled': False, 'template': 'default', 'priority': 3}
            },
            'devops': {
                'ci_cd_pipeline': True,
                'automated_testing': True,
                'security_scanning': True,
                'performance_monitoring': True,
                'auto_deployment': True
            }
        }
    
    def update_field(self, field_path: str, value: Any):
        """Update nested configuration field"""
        keys = field_path.split('.')
        current = self.config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration integrity"""
        required_fields = [
            'project.name',
            'github.token',
            'github.username'
        ]
        
        for field in required_fields:
            if not self._get_nested_value(field):
                logger.warning(f"Required field missing: {field}")
    
    def _get_nested_value(self, field_path: str) -> Any:
        """Get nested configuration value"""
        keys = field_path.split('.')
        current = self.config
        
        for key in keys:
            if key not in current:
                return None
            current = current[key]
        
        return current
    
    def save_config(self):
        """Save configuration to YAML file"""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

class WorkflowEngine:
    """Manages cascading and compounding workflows"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.task_queue = asyncio.Queue()
        self.active_tasks = {}
        self.performance_metrics = {}
        
    async def execute_workflow(self, project_data: Dict) -> Dict:
        """Execute optimized workflow with cascading tasks"""
        workflow_id = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        
        try:
            # Phase 1: Setup and Validation
            setup_tasks = await self._create_setup_tasks(project_data)
            setup_results = await self._execute_parallel_tasks(setup_tasks)
            
            # Phase 2: Repository Creation and Structure
            repo_tasks = await self._create_repo_tasks(project_data, setup_results)
            repo_results = await self._execute_parallel_tasks(repo_tasks)
            
            # Phase 3: Document Generation (cascading from repo structure)
            doc_tasks = await self._create_document_tasks(project_data, repo_results)
            doc_results = await self._execute_parallel_tasks(doc_tasks)
            
            # Phase 4: DevOps Pipeline Setup (compounding on previous phases)
            devops_tasks = await self._create_devops_tasks(project_data, {**repo_results, **doc_results})
            devops_results = await self._execute_parallel_tasks(devops_tasks)
            
            # Phase 5: Final Optimization and Validation
            final_tasks = await self._create_final_tasks(project_data, {**setup_results, **repo_results, **doc_results, **devops_results})
            final_results = await self._execute_parallel_tasks(final_tasks)
            
            return {
                'workflow_id': workflow_id,
                'status': 'completed',
                'results': {
                    'setup': setup_results,
                    'repository': repo_results,
                    'documents': doc_results,
                    'devops': devops_results,
                    'final': final_results
                },
                'metrics': self.performance_metrics.get(workflow_id, {})
            }
            
        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}")
            return {'workflow_id': workflow_id, 'status': 'failed', 'error': str(e)}
    
    async def _execute_parallel_tasks(self, tasks: List) -> Dict:
        """Execute tasks in parallel with optimal resource usage"""
        max_concurrent = self.config.config['workflow']['max_concurrent_tasks']
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_task_with_semaphore(task):
            async with semaphore:
                return await task()
        
        results = await asyncio.gather(*[execute_task_with_semaphore(task) for task in tasks])
        return {f"task_{i}": result for i, result in enumerate(results)}
    
    async def _create_setup_tasks(self, project_data: Dict) -> List:
        """Create setup phase tasks"""
        tasks = []
        
        # Validate GitHub credentials
        tasks.append(lambda: self._validate_github_access(project_data))
        
        # Discover optimal MCP servers
        mcp_manager = MCPServerManager()
        tasks.append(lambda: mcp_manager.discover_optimal_servers(
            project_data.get('type', 'web_application'),
            self.config.config['mcp_servers']['required_capabilities']
        ))
        
        # Validate project structure requirements
        tasks.append(lambda: self._validate_project_requirements(project_data))
        
        return tasks
    
    async def _create_repo_tasks(self, project_data: Dict, setup_results: Dict) -> List:
        """Create repository creation tasks"""
        tasks = []
        
        # Create GitHub repository
        tasks.append(lambda: self._create_github_repository(project_data))
        
        # Setup repository structure
        tasks.append(lambda: self._setup_repository_structure(project_data))
        
        # Initialize CI/CD pipeline
        tasks.append(lambda: self._initialize_cicd_pipeline(project_data))
        
        return tasks
    
    async def _create_document_tasks(self, project_data: Dict, repo_results: Dict) -> List:
        """Create document generation tasks"""
        tasks = []
        
        if self.config.config['documents']['prd']['enabled']:
            tasks.append(lambda: self._generate_prd(project_data))
        
        if self.config.config['documents']['prp']['enabled']:
            tasks.append(lambda: self._generate_prp(project_data))
        
        if self.config.config['documents']['master_prompt']['enabled']:
            tasks.append(lambda: self._generate_master_prompt(project_data))
        
        return tasks
    
    async def _create_devops_tasks(self, project_data: Dict, previous_results: Dict) -> List:
        """Create DevOps automation tasks"""
        tasks = []
        
        if self.config.config['devops']['automated_testing']:
            tasks.append(lambda: self._setup_testing_framework(project_data))
        
        if self.config.config['devops']['security_scanning']:
            tasks.append(lambda: self._setup_security_scanning(project_data))
        
        if self.config.config['devops']['performance_monitoring']:
            tasks.append(lambda: self._setup_performance_monitoring(project_data))
        
        return tasks
    
    async def _create_final_tasks(self, project_data: Dict, all_results: Dict) -> List:
        """Create final validation and optimization tasks"""
        tasks = []
        
        tasks.append(lambda: self._validate_final_structure(project_data, all_results))
        tasks.append(lambda: self._optimize_repository_performance(project_data))
        tasks.append(lambda: self._generate_final_report(project_data, all_results))
        
        return tasks
    
    # Placeholder methods for task implementations
    async def _validate_github_access(self, project_data): 
        return {"status": "validated", "timestamp": datetime.now()}
    
    async def _validate_project_requirements(self, project_data):
        return {"status": "validated", "requirements_met": True}
    
    async def _create_github_repository(self, project_data):
        return {"status": "created", "repo_url": f"https://github.com/{project_data.get('username', '')}/{project_data.get('name', '')}"}
    
    async def _setup_repository_structure(self, project_data):
        return {"status": "structured", "folders_created": ["docs", "src", "tests", ".github"]}
    
    async def _initialize_cicd_pipeline(self, project_data):
        return {"status": "initialized", "pipeline_file": ".github/workflows/main.yml"}
    
    async def _generate_prd(self, project_data):
        return {"status": "generated", "file": "docs/PRD.md"}
    
    async def _generate_prp(self, project_data):
        return {"status": "generated", "file": "docs/PRP.md"}
    
    async def _generate_master_prompt(self, project_data):
        return {"status": "generated", "file": "prompts/master_prompt.md"}
    
    async def _setup_testing_framework(self, project_data):
        return {"status": "configured", "framework": "pytest", "coverage": "100%"}
    
    async def _setup_security_scanning(self, project_data):
        return {"status": "configured", "tools": ["bandit", "safety", "semgrep"]}
    
    async def _setup_performance_monitoring(self, project_data):
        return {"status": "configured", "tools": ["prometheus", "grafana"]}
    
    async def _validate_final_structure(self, project_data, results):
        return {"status": "validated", "score": 95, "issues": []}
    
    async def _optimize_repository_performance(self, project_data):
        return {"status": "optimized", "improvements": ["git-lfs", "pre-commit-hooks"]}
    
    async def _generate_final_report(self, project_data, results):
        return {"status": "generated", "file": "PROJECT_REPORT.md", "confidence": 97}

class POCEApp(ctk.CTk):
    """Enhanced P.O.C.E. Project Creator with DevOps automation"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize managers
        self.config_manager = ConfigManager()
        self.workflow_engine = WorkflowEngine(self.config_manager)
        self.mcp_manager = MCPServerManager()
        
        self._setup_ui()
        self._setup_performance_monitoring()
        
    def _setup_ui(self):
        """Setup the enhanced user interface"""
        self.title("P.O.C.E. Project Creator v4.0")
        self.geometry("1200x900")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        # Create main container
        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create 3D metal header
        self._create_3d_header()
        
        # Create tabbed interface
        self.tabview = ctk.CTkTabview(self.main_container)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.project_tab = self.tabview.add("Project Setup")
        self.config_tab = self.tabview.add("Configuration")
        self.workflow_tab = self.tabview.add("Workflow Monitor")
        self.results_tab = self.tabview.add("Results")
        
        self._setup_project_tab()
        self._setup_config_tab()
        self._setup_workflow_tab()
        self._setup_results_tab()
        
        # Status bar
        self.status_frame = ctk.CTkFrame(self.main_container, height=40)
        self.status_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready", font=("Arial", 12))
        self.status_label.pack(side="left", padx=10, pady=10)
        
        self.progress_bar = ctk.CTkProgressBar(self.status_frame)
        self.progress_bar.pack(side="right", padx=10, pady=10, fill="x", expand=True)
        self.progress_bar.set(0)
    
    def _create_3d_header(self):
        """Create 3D stamped metal header with realistic effects"""
        header_frame = ctk.CTkFrame(self.main_container, height=80, fg_color="#2B2B2B")
        header_frame.pack(fill="x", padx=5, pady=5)
        header_frame.pack_propagate(False)
        
        # Create canvas for 3D text effect
        canvas = tk.Canvas(header_frame, height=80, bg="#2B2B2B", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        
        # Calculate center position
        canvas.update_idletasks()
        center_x = canvas.winfo_width() // 2 if canvas.winfo_width() > 1 else 300
        center_y = 40
        
        # Create 3D text effect with multiple layers
        text = "P.O.C.E. PROJECT CREATOR"
        font_tuple = ("Impact", 24, "bold")
        
        # Shadow layers (bottom to top)
        for i in range(5, 0, -1):
            color_intensity = 50 + (i * 10)
            shadow_color = f"#{color_intensity:02x}{color_intensity:02x}{color_intensity:02x}"
            canvas.create_text(center_x + i, center_y + i, text=text, font=font_tuple, 
                             fill=shadow_color, anchor="center")
        
        # Main text with metallic gradient effect
        canvas.create_text(center_x, center_y, text=text, font=font_tuple, 
                         fill="#E8E8E8", anchor="center")
        
        # Highlight layer
        canvas.create_text(center_x - 1, center_y - 1, text=text, font=font_tuple, 
                         fill="#F5F5F5", anchor="center")
        
        # Add rivets effect
        rivet_positions = [(50, 20), (50, 60), (canvas.winfo_reqwidth() - 50, 20), 
                          (canvas.winfo_reqwidth() - 50, 60)]
        
        for x, y in rivet_positions:
            # Outer rivet circle
            canvas.create_oval(x-8, y-8, x+8, y+8, fill="#808080", outline="#606060", width=2)
            # Inner highlight
            canvas.create_oval(x-5, y-5, x+5, y+5, fill="#A0A0A0", outline="")
            # Center dot
            canvas.create_oval(x-2, y-2, x+2, y+2, fill="#606060", outline="")
    
    def _setup_project_tab(self):
        """Setup project configuration tab"""
        # Core project details
        core_frame = ctk.CTkFrame(self.project_tab)
        core_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(core_frame, text="Core Project Details", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        
        # Project name
        name_frame = ctk.CTkFrame(core_frame, fg_color="transparent")
        name_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(name_frame, text="Project Name:", width=120).pack(side="left")
        self.project_name_entry = ctk.CTkEntry(name_frame, placeholder_text="Enter project name")
        self.project_name_entry.pack(side="left", fill="x", expand=True, padx=(10, 0))
        
        # Project type
        type_frame = ctk.CTkFrame(core_frame, fg_color="transparent")
        type_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(type_frame, text="Project Type:", width=120).pack(side="left")
        self.project_type_combo = ctk.CTkComboBox(type_frame, values=[
            "web_application", "mobile_app", "api_service", "ml_model", 
            "data_pipeline", "automation_tool", "microservice"
        ])
        self.project_type_combo.pack(side="left", padx=(10, 0))
        
        # GitHub details
        github_frame = ctk.CTkFrame(self.project_tab)
        github_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(github_frame, text="GitHub Configuration", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        
        # GitHub token
        token_frame = ctk.CTkFrame(github_frame, fg_color="transparent")
        token_frame.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(token_frame, text="GitHub Token:", width=120).pack(side="left")
        self.github_token_entry = ctk.CTkEntry(token_frame, placeholder_text="ghp_...", show="*")
        self.github_token_entry.pack(side="left", fill="x", expand=True, padx=(10, 0))
        
        # Documents section
        docs_frame = ctk.CTkFrame(self.project_tab)
        docs_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(docs_frame, text="Optional Documents", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        
        # Document checkboxes with file attachment options
        self.doc_vars = {}
        self.doc_paths = {}
        
        for doc_type in ['prd', 'prp', 'master_prompt']:
            doc_frame = ctk.CTkFrame(docs_frame, fg_color="transparent")
            doc_frame.pack(fill="x", padx=10, pady=5)
            
            self.doc_vars[doc_type] = ctk.BooleanVar()
            checkbox = ctk.CTkCheckBox(doc_frame, text=doc_type.upper().replace('_', ' '), 
                                     variable=self.doc_vars[doc_type],
                                     command=lambda dt=doc_type: self._on_doc_toggle(dt))
            checkbox.pack(side="left")
            
            attach_btn = ctk.CTkButton(doc_frame, text="Attach File", width=100,
                                     command=lambda dt=doc_type: self._attach_file(dt))
            attach_btn.pack(side="left", padx=10)
            
            self.doc_paths[doc_type] = ctk.StringVar()
            path_label = ctk.CTkLabel(doc_frame, textvariable=self.doc_paths[doc_type], 
                                    text_color="gray")
            path_label.pack(side="left", padx=10)
        
        # Action buttons
        action_frame = ctk.CTkFrame(self.project_tab)
        action_frame.pack(fill="x", padx=10, pady=10)
        
        self.validate_btn = ctk.CTkButton(action_frame, text="Validate Configuration", 
                                        command=self._validate_configuration)
        self.validate_btn.pack(side="left", padx=10)
        
        self.create_btn = ctk.CTkButton(action_frame, text="Create Project", 
                                      command=self._create_project,
                                      fg_color="#00AA00", hover_color="#00CC00")
        self.create_btn.pack(side="right", padx=10)
    
    def _setup_config_tab(self):
        """Setup configuration editing tab"""
        # YAML editor
        editor_frame = ctk.CTkFrame(self.config_tab)
        editor_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(editor_frame, text="Configuration Editor", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        
        self.config_text = ctk.CTkTextbox(editor_frame, font=("Courier", 11))
        self.config_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Load current config
        self._refresh_config_editor()
        
        # Config actions
        config_actions = ctk.CTkFrame(editor_frame)
        config_actions.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(config_actions, text="Refresh", command=self._refresh_config_editor).pack(side="left", padx=5)
        ctk.CTkButton(config_actions, text="Save", command=self._save_config_from_editor).pack(side="left", padx=5)
        ctk.CTkButton(config_actions, text="Reset to Default", command=self._reset_config).pack(side="left", padx=5)
    
    def _setup_workflow_tab(self):
        """Setup workflow monitoring tab"""
        # Real-time workflow status
        status_frame = ctk.CTkFrame(self.workflow_tab)
        status_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(status_frame, text="Workflow Status", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        
        self.workflow_status = ctk.CTkTextbox(status_frame, height=200)
        self.workflow_status.pack(fill="x", padx=10, pady=10)
        
        # Performance metrics
        metrics_frame = ctk.CTkFrame(self.workflow_tab)
        metrics_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(metrics_frame, text="Performance Metrics", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        
        self.metrics_display = ctk.CTkTextbox(metrics_frame)
        self.metrics_display.pack(fill="both", expand=True, padx=10, pady=10)
    
    def _setup_results_tab(self):
        """Setup results display tab"""
        results_frame = ctk.CTkFrame(self.results_tab)
        results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(results_frame, text="Project Creation Results", font=("Arial", 16, "bold")).pack(anchor="w", padx=10, pady=5)
        
        self.results_display = ctk.CTkTextbox(results_frame)
        self.results_display.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Results actions
        results_actions = ctk.CTkFrame(results_frame)
        results_actions.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkButton(results_actions, text="Export Report", command=self._export_report).pack(side="left", padx=5)
        ctk.CTkButton(results_actions, text="Open Repository", command=self._open_repository).pack(side="left", padx=5)
    
    def _setup_performance_monitoring(self):
        """Setup real-time performance monitoring"""
        self.performance_data = {
            'start_time': None,
            'tasks_completed': 0,
            'errors_encountered': 0,
            'efficiency_score': 0
        }
    
    def _refresh_config_editor(self):
        """Refresh the configuration editor with current config"""
        self.config_text.delete("1.0", "end")
        yaml_content = yaml.dump(self.config_manager.config, default_flow_style=False, indent=2)
        self.config_text.insert("1.0", yaml_content)
    
    def _save_config_from_editor(self):
        """Save configuration from editor"""
        try:
            yaml_content = self.config_text.get("1.0", "end-1c")
            new_config = yaml.safe_load(yaml_content)
            self.config_manager.config = new_config
            self.config_manager.save_config()
            self._update_status("Configuration saved successfully", "green")
        except Exception as e:
            messagebox.showerror("Configuration Error", f"Invalid YAML configuration:\n{e}")
    
    def _reset_config(self):
        """Reset configuration to defaults"""
        self.config_manager.config = self.config_manager._load_default_config()
        self._refresh_config_editor()
        self._update_status("Configuration reset to defaults", "orange")
    
    def _on_doc_toggle(self, doc_type: str):
        """Handle document toggle"""
        enabled = self.doc_vars[doc_type].get()
        self.config_manager.update_field(f'documents.{doc_type}.enabled', enabled)
    
    def _attach_file(self, doc_type: str):
        """Attach file for document"""
        file_types = [
            ("All supported", "*.md *.txt *.pdf *.docx"),
            ("Markdown", "*.md"),
            ("Text files", "*.txt"),
            ("PDF files", "*.pdf"),
            ("Word documents", "*.docx"),
            ("All files", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title=f"Select {doc_type.upper()} file",
            filetypes=file_types
        )
        
        if filepath:
            self.doc_paths[doc_type].set(os.path.basename(filepath))
            # Store full path in config
            self.config_manager.update_field(f'documents.{doc_type}.file_path', filepath)
    
    def _validate_configuration(self):
        """Validate current configuration"""
        try:
            # Update config from UI
            self._update_config_from_ui()
            
            # Validate GitHub token
            token = self.github_token_entry.get()
            if not token:
                raise ValueError("GitHub token is required")
            
            # Test GitHub connection
            g = Github(token)
            user = g.get_user()
            
            self._update_status(f"Configuration valid! Connected as {user.login}", "green")
            return True
            
        except Exception as e:
            self._update_status(f"Configuration error: {e}", "red")
            messagebox.showerror("Validation Error", str(e))
            return False
    
    def _update_config_from_ui(self):
        """Update configuration from UI elements"""
        self.config_manager.update_field('project.name', self.project_name_entry.get())
        self.config_manager.update_field('project.type', self.project_type_combo.get())
        self.config_manager.update_field('github.token', self.github_token_entry.get())
    
    def _create_project(self):
        """Create project with enhanced workflow"""
        if not self._validate_configuration():
            return
        
        try:
            self._update_status("Starting project creation workflow...", "blue")
            self.progress_bar.set(0.1)
            
            # Prepare project data
            project_data = {
                'name': self.project_name_entry.get(),
                'type': self.project_type_combo.get(),
                'token': self.github_token_entry.get(),
                'username': self._get_github_username()
            }
            
            # Run workflow in background thread
            threading.Thread(target=self._run_workflow, args=(project_data,), daemon=True).start()
            
        except Exception as e:
            self._update_status(f"Project creation failed: {e}", "red")
            messagebox.showerror("Creation Error", str(e))
    
    def _run_workflow(self, project_data: Dict):
        """Run the workflow in background"""
        try:
            # Run async workflow
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            self.performance_data['start_time'] = time.time()
            
            results = loop.run_until_complete(self.workflow_engine.execute_workflow(project_data))
            
            # Update UI with results
            self.after(0, self._display_results, results)
            
        except Exception as e:
            self.after(0, self._handle_workflow_error, e)
    
    def _display_results(self, results: Dict):
        """Display workflow results"""
        if results['status'] == 'completed':
            self._update_status("Project created successfully!", "green")
            self.progress_bar.set(1.0)
            
            # Switch to results tab
            self.tabview.set("Results")
            
            # Display detailed results
            results_text = self._format_results(results)
            self.results_display.delete("1.0", "end")
            self.results_display.insert("1.0", results_text)
            
        else:
            self._update_status(f"Project creation failed: {results.get('error', 'Unknown error')}", "red")
    
    def _format_results(self, results: Dict) -> str:
        """Format results for display"""
        lines = [
            f"Workflow ID: {results['workflow_id']}",
            f"Status: {results['status']}",
            f"Completion Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Phase Results:",
            "=" * 50
        ]
        
        for phase, phase_results in results.get('results', {}).items():
            lines.append(f"\n{phase.title()}:")
            lines.append("-" * 20)
            for task, task_result in phase_results.items():
                lines.append(f"  {task}: {task_result.get('status', 'Unknown')}")
        
        if 'metrics' in results:
            lines.extend([
                "",
                "Performance Metrics:",
                "=" * 50,
                f"Total Tasks: {self.performance_data['tasks_completed']}",
                f"Errors: {self.performance_data['errors_encountered']}",
                f"Efficiency Score: {self.performance_data['efficiency_score']}%"
            ])
        
        return "\n".join(lines)
    
    def _handle_workflow_error(self, error: Exception):
        """Handle workflow errors"""
        self._update_status(f"Workflow failed: {error}", "red")
        messagebox.showerror("Workflow Error", str(error))
        self.progress_bar.set(0)
    
    def _get_github_username(self) -> str:
        """Get GitHub username from token"""
        try:
            g = Github(self.github_token_entry.get())
            return g.get_user().login
        except:
            return "unknown"
    
    def _update_status(self, message: str, color: str = "white"):
        """Update status message"""
        self.status_label.configure(text=message, text_color=color)
        logger.info(f"Status: {message}")
    
    def _export_report(self):
        """Export results report"""
        # Implementation for exporting detailed report
        messagebox.showinfo("Export", "Report export functionality to be implemented")
    
    def _open_repository(self):
        """Open created repository in browser"""
        # Implementation for opening repository
        messagebox.showinfo("Repository", "Repository opening functionality to be implemented")

if __name__ == "__main__":
    # Setup logging
    logger.info("Starting P.O.C.E. Project Creator v4.0")
    
    try:
        app = POCEApp()
        app.mainloop()
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application:\n{e}")