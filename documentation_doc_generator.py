# documentation/doc_generator.py
"""
P.O.C.E. Project Creator - Automated Documentation Generation System v4.0
Comprehensive documentation generator with API docs, code documentation,
user guides, and multi-format output support
"""

import os
import ast
import inspect
import json
import yaml
import re
import subprocess
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import tempfile
import shutil

# Documentation generation libraries
try:
    import sphinx
    from sphinx.cmd.build import build_main
    SPHINX_AVAILABLE = True
except ImportError:
    SPHINX_AVAILABLE = False

try:
    import mkdocs
    from mkdocs import config
    from mkdocs.commands import build
    MKDOCS_AVAILABLE = True
except ImportError:
    MKDOCS_AVAILABLE = False

try:
    import pydoc
    PYDOC_AVAILABLE = True
except ImportError:
    PYDOC_AVAILABLE = False

# Markdown processing
try:
    import markdown
    from markdown.extensions import toc, codehilite, tables
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False

# API documentation
try:
    from openapi_spec_validator import validate_spec
    OPENAPI_VALIDATOR_AVAILABLE = True
except ImportError:
    OPENAPI_VALIDATOR_AVAILABLE = False

logger = logging.getLogger(__name__)

# ==========================================
# DOCUMENTATION CONFIGURATION
# ==========================================

class DocFormat(Enum):
    """Supported documentation formats"""
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    SPHINX = "sphinx"
    MKDOCS = "mkdocs"
    DOCX = "docx"

class DocType(Enum):
    """Types of documentation"""
    API_REFERENCE = "api_reference"
    USER_GUIDE = "user_guide"
    DEVELOPER_GUIDE = "developer_guide"
    INSTALLATION_GUIDE = "installation_guide"
    CONFIGURATION_GUIDE = "configuration_guide"
    TROUBLESHOOTING = "troubleshooting"
    CHANGELOG = "changelog"
    README = "readme"

@dataclass
class DocSection:
    """Documentation section"""
    title: str
    content: str
    level: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    subsections: List['DocSection'] = field(default_factory=list)

@dataclass
class APIEndpoint:
    """API endpoint documentation"""
    path: str
    method: str
    summary: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

@dataclass
class CodeDocumentation:
    """Code documentation extracted from source"""
    module_name: str
    classes: List[Dict[str, Any]] = field(default_factory=list)
    functions: List[Dict[str, Any]] = field(default_factory=list)
    constants: List[Dict[str, Any]] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    docstring: Optional[str] = None

# ==========================================
# CODE DOCUMENTATION EXTRACTOR
# ==========================================

class CodeDocExtractor:
    """Extracts documentation from Python source code"""
    
    def __init__(self):
        self.extracted_docs: Dict[str, CodeDocumentation] = {}
    
    def extract_from_file(self, file_path: Path) -> CodeDocumentation:
        """Extract documentation from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse the AST
            tree = ast.parse(source_code, filename=str(file_path))
            
            # Extract module-level information
            module_doc = CodeDocumentation(
                module_name=file_path.stem,
                docstring=ast.get_docstring(tree)
            )
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_doc.imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            module_doc.imports.append(f"{node.module}.{alias.name}")
            
            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = self._extract_class_info(node)
                    module_doc.classes.append(class_doc)
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:  # Top-level functions only
                    function_doc = self._extract_function_info(node)
                    module_doc.functions.append(function_doc)
                elif isinstance(node, ast.Assign) and node.col_offset == 0:  # Top-level assignments
                    constants = self._extract_constants(node)
                    module_doc.constants.extend(constants)
            
            self.extracted_docs[str(file_path)] = module_doc
            return module_doc
            
        except Exception as e:
            logger.error(f"Failed to extract documentation from {file_path}: {e}")
            return CodeDocumentation(module_name=file_path.stem)
    
    def _extract_class_info(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Extract class information"""
        class_info = {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'methods': [],
            'attributes': [],
            'bases': [self._get_name(base) for base in node.bases],
            'decorators': [self._get_name(decorator) for decorator in node.decorator_list]
        }
        
        # Extract methods and attributes
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function_info(item, is_method=True)
                class_info['methods'].append(method_info)
            elif isinstance(item, ast.Assign):
                # Class attributes
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_info['attributes'].append({
                            'name': target.id,
                            'type': self._infer_type(item.value),
                            'value': self._get_value_repr(item.value)
                        })
        
        return class_info
    
    def _extract_function_info(self, node: ast.FunctionDef, is_method: bool = False) -> Dict[str, Any]:
        """Extract function/method information"""
        function_info = {
            'name': node.name,
            'docstring': ast.get_docstring(node),
            'parameters': [],
            'return_type': None,
            'decorators': [self._get_name(decorator) for decorator in node.decorator_list],
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'is_method': is_method
        }
        
        # Extract parameters
        for arg in node.args.args:
            param_info = {
                'name': arg.arg,
                'type': self._get_annotation(arg.annotation) if arg.annotation else None,
                'default': None
            }
            function_info['parameters'].append(param_info)
        
        # Extract defaults
        defaults = node.args.defaults
        if defaults:
            num_defaults = len(defaults)
            num_args = len(node.args.args)
            for i, default in enumerate(defaults):
                param_index = num_args - num_defaults + i
                if param_index < len(function_info['parameters']):
                    function_info['parameters'][param_index]['default'] = self._get_value_repr(default)
        
        # Extract return type
        if node.returns:
            function_info['return_type'] = self._get_annotation(node.returns)
        
        return function_info
    
    def _extract_constants(self, node: ast.Assign) -> List[Dict[str, Any]]:
        """Extract module-level constants"""
        constants = []
        
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                constants.append({
                    'name': target.id,
                    'type': self._infer_type(node.value),
                    'value': self._get_value_repr(node.value)
                })
        
        return constants
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(node)
    
    def _get_annotation(self, node: ast.AST) -> str:
        """Get type annotation as string"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Subscript):
            return f"{self._get_name(node.value)}[{self._get_name(node.slice)}]"
        else:
            return str(node)
    
    def _infer_type(self, node: ast.AST) -> str:
        """Infer type from AST node"""
        if isinstance(node, ast.Constant):
            return type(node.value).__name__
        elif isinstance(node, ast.List):
            return "list"
        elif isinstance(node, ast.Dict):
            return "dict"
        elif isinstance(node, ast.Set):
            return "set"
        elif isinstance(node, ast.Tuple):
            return "tuple"
        else:
            return "unknown"
    
    def _get_value_repr(self, node: ast.AST) -> str:
        """Get value representation from AST node"""
        try:
            if isinstance(node, ast.Constant):
                return repr(node.value)
            elif isinstance(node, ast.Name):
                return node.id
            else:
                return "..."
        except:
            return "..."

# ==========================================
# API DOCUMENTATION GENERATOR
# ==========================================

class APIDocGenerator:
    """Generates API documentation in OpenAPI format"""
    
    def __init__(self):
        self.endpoints: List[APIEndpoint] = []
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.info: Dict[str, Any] = {
            'title': 'P.O.C.E. Project Creator API',
            'version': '4.0.0',
            'description': 'Advanced DevOps automation platform API'
        }
    
    def add_endpoint(self, endpoint: APIEndpoint):
        """Add an API endpoint"""
        self.endpoints.append(endpoint)
    
    def add_schema(self, name: str, schema: Dict[str, Any]):
        """Add a data schema"""
        self.schemas[name] = schema
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate OpenAPI 3.0 specification"""
        spec = {
            'openapi': '3.0.0',
            'info': self.info,
            'paths': {},
            'components': {
                'schemas': self.schemas
            }
        }
        
        # Group endpoints by path
        paths = {}
        for endpoint in self.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}
            
            paths[endpoint.path][endpoint.method.lower()] = {
                'summary': endpoint.summary,
                'description': endpoint.description,
                'tags': endpoint.tags,
                'parameters': endpoint.parameters,
                'responses': endpoint.responses
            }
            
            if endpoint.request_body:
                paths[endpoint.path][endpoint.method.lower()]['requestBody'] = endpoint.request_body
        
        spec['paths'] = paths
        
        return spec
    
    def extract_from_flask_app(self, app) -> None:
        """Extract API documentation from Flask application"""
        try:
            for rule in app.url_map.iter_rules():
                if rule.endpoint == 'static':
                    continue
                
                view_function = app.view_functions[rule.endpoint]
                
                # Extract docstring
                docstring = inspect.getdoc(view_function) or ""
                
                # Create endpoint documentation
                endpoint = APIEndpoint(
                    path=rule.rule,
                    method=list(rule.methods)[0] if rule.methods else 'GET',
                    summary=rule.endpoint.replace('_', ' ').title(),
                    description=docstring,
                    tags=[rule.endpoint.split('.')[0] if '.' in rule.endpoint else 'default']
                )
                
                self.add_endpoint(endpoint)
                
        except Exception as e:
            logger.error(f"Failed to extract Flask API documentation: {e}")
    
    def extract_from_fastapi_app(self, app) -> None:
        """Extract API documentation from FastAPI application"""
        try:
            # FastAPI already provides OpenAPI spec
            openapi_spec = app.openapi()
            
            # Extract endpoints
            for path, methods in openapi_spec.get('paths', {}).items():
                for method, details in methods.items():
                    endpoint = APIEndpoint(
                        path=path,
                        method=method.upper(),
                        summary=details.get('summary', ''),
                        description=details.get('description', ''),
                        parameters=details.get('parameters', []),
                        request_body=details.get('requestBody'),
                        responses=details.get('responses', {}),
                        tags=details.get('tags', [])
                    )
                    
                    self.add_endpoint(endpoint)
            
            # Extract schemas
            components = openapi_spec.get('components', {})
            self.schemas.update(components.get('schemas', {}))
            
        except Exception as e:
            logger.error(f"Failed to extract FastAPI documentation: {e}")

# ==========================================
# MARKDOWN DOCUMENT GENERATOR
# ==========================================

class MarkdownGenerator:
    """Generates Markdown documentation"""
    
    def __init__(self):
        self.sections: List[DocSection] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_section(self, section: DocSection):
        """Add a documentation section"""
        self.sections.append(section)
    
    def add_table_of_contents(self) -> str:
        """Generate table of contents"""
        toc_lines = ["## Table of Contents", ""]
        
        def add_toc_item(section: DocSection, indent: int = 0):
            anchor = section.title.lower().replace(' ', '-').replace('(', '').replace(')', '')
            toc_lines.append(f"{'  ' * indent}- [{section.title}](#{anchor})")
            
            for subsection in section.subsections:
                add_toc_item(subsection, indent + 1)
        
        for section in self.sections:
            add_toc_item(section)
        
        return '\n'.join(toc_lines) + '\n\n'
    
    def generate_markdown(self, include_toc: bool = True) -> str:
        """Generate complete Markdown document"""
        content_lines = []
        
        # Add metadata as front matter if available
        if self.metadata:
            content_lines.append("---")
            for key, value in self.metadata.items():
                content_lines.append(f"{key}: {value}")
            content_lines.append("---")
            content_lines.append("")
        
        # Add table of contents
        if include_toc and self.sections:
            content_lines.append(self.add_table_of_contents())
        
        # Add sections
        for section in self.sections:
            content_lines.append(self._format_section(section))
        
        return '\n'.join(content_lines)
    
    def _format_section(self, section: DocSection, base_level: int = 0) -> str:
        """Format a section as Markdown"""
        lines = []
        
        # Section header
        level = section.level + base_level
        header_prefix = "#" * min(level, 6)
        lines.append(f"{header_prefix} {section.title}")
        lines.append("")
        
        # Section content
        if section.content:
            lines.append(section.content)
            lines.append("")
        
        # Subsections
        for subsection in section.subsections:
            lines.append(self._format_section(subsection, level))
        
        return '\n'.join(lines)
    
    def generate_api_documentation(self, api_spec: Dict[str, Any]) -> str:
        """Generate API documentation from OpenAPI spec"""
        lines = [
            f"# {api_spec['info']['title']}",
            "",
            api_spec['info'].get('description', ''),
            "",
            f"**Version:** {api_spec['info']['version']}",
            "",
            "## Endpoints",
            ""
        ]
        
        # Sort paths
        paths = api_spec.get('paths', {})
        for path in sorted(paths.keys()):
            methods = paths[path]
            
            for method in sorted(methods.keys()):
                endpoint = methods[method]
                
                lines.extend([
                    f"### {method.upper()} {path}",
                    "",
                    endpoint.get('summary', ''),
                    "",
                    endpoint.get('description', ''),
                    ""
                ])
                
                # Parameters
                parameters = endpoint.get('parameters', [])
                if parameters:
                    lines.extend([
                        "#### Parameters",
                        "",
                        "| Name | Type | Required | Description |",
                        "|------|------|----------|-------------|"
                    ])
                    
                    for param in parameters:
                        name = param.get('name', '')
                        param_type = param.get('schema', {}).get('type', 'string')
                        required = 'Yes' if param.get('required', False) else 'No'
                        description = param.get('description', '')
                        
                        lines.append(f"| {name} | {param_type} | {required} | {description} |")
                    
                    lines.append("")
                
                # Responses
                responses = endpoint.get('responses', {})
                if responses:
                    lines.extend([
                        "#### Responses",
                        "",
                        "| Code | Description |",
                        "|------|-------------|"
                    ])
                    
                    for code, response in responses.items():
                        description = response.get('description', '')
                        lines.append(f"| {code} | {description} |")
                    
                    lines.append("")
        
        return '\n'.join(lines)

# ==========================================
# PROJECT DOCUMENTATION GENERATOR
# ==========================================

class ProjectDocGenerator:
    """Generates comprehensive project documentation"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.code_extractor = CodeDocExtractor()
        self.api_generator = APIDocGenerator()
        self.markdown_generator = MarkdownGenerator()
        self.extracted_code_docs: List[CodeDocumentation] = []
    
    def scan_project(self):
        """Scan project directory for documentation sources"""
        logger.info(f"Scanning project directory: {self.project_path}")
        
        # Find Python files
        python_files = list(self.project_path.rglob("*.py"))
        
        # Extract documentation from Python files
        for py_file in python_files:
            if not self._should_skip_file(py_file):
                doc = self.code_extractor.extract_from_file(py_file)
                self.extracted_code_docs.append(doc)
        
        logger.info(f"Extracted documentation from {len(self.extracted_code_docs)} Python files")
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = [
            '__pycache__',
            '.git',
            '.pytest_cache',
            'venv',
            'env',
            '.tox',
            'build',
            'dist'
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def generate_readme(self) -> str:
        """Generate README.md file"""
        project_name = self.project_path.name
        
        # Try to find existing config for project info
        config_files = ['pyproject.toml', 'setup.py', 'package.json', 'poce_config.yaml']
        project_info = self._extract_project_info(config_files)
        
        readme_sections = [
            DocSection(
                title=project_name,
                content=project_info.get('description', 'Advanced DevOps automation project created with P.O.C.E.'),
                level=1
            ),
            DocSection(
                title="Features",
                content=self._generate_features_list(),
                level=2
            ),
            DocSection(
                title="Installation",
                content=self._generate_installation_guide(),
                level=2
            ),
            DocSection(
                title="Usage",
                content=self._generate_usage_guide(),
                level=2
            ),
            DocSection(
                title="API Reference",
                content="See [API Documentation](docs/api.md) for detailed API reference.",
                level=2
            ),
            DocSection(
                title="Contributing",
                content=self._generate_contributing_guide(),
                level=2
            ),
            DocSection(
                title="License",
                content=project_info.get('license', 'MIT License'),
                level=2
            )
        ]
        
        self.markdown_generator.sections = readme_sections
        self.markdown_generator.metadata = {
            'title': project_name,
            'description': project_info.get('description', ''),
            'version': project_info.get('version', '1.0.0')
        }
        
        return self.markdown_generator.generate_markdown()
    
    def generate_api_documentation(self) -> str:
        """Generate API documentation"""
        # Look for API definitions in the code
        self._extract_api_endpoints()
        
        # Generate OpenAPI spec
        api_spec = self.api_generator.generate_openapi_spec()
        
        # Convert to Markdown
        return self.markdown_generator.generate_api_documentation(api_spec)
    
    def generate_code_documentation(self) -> str:
        """Generate code documentation"""
        sections = []
        
        # Group by modules
        for code_doc in self.extracted_code_docs:
            if not code_doc.classes and not code_doc.functions:
                continue
            
            module_section = DocSection(
                title=f"Module: {code_doc.module_name}",
                content=code_doc.docstring or "No module description available.",
                level=2
            )
            
            # Add classes
            if code_doc.classes:
                classes_section = DocSection(
                    title="Classes",
                    content="",
                    level=3
                )
                
                for class_info in code_doc.classes:
                    class_content = self._format_class_documentation(class_info)
                    class_section = DocSection(
                        title=class_info['name'],
                        content=class_content,
                        level=4
                    )
                    classes_section.subsections.append(class_section)
                
                module_section.subsections.append(classes_section)
            
            # Add functions
            if code_doc.functions:
                functions_section = DocSection(
                    title="Functions",
                    content="",
                    level=3
                )
                
                for func_info in code_doc.functions:
                    func_content = self._format_function_documentation(func_info)
                    func_section = DocSection(
                        title=func_info['name'],
                        content=func_content,
                        level=4
                    )
                    functions_section.subsections.append(func_section)
                
                module_section.subsections.append(functions_section)
            
            sections.append(module_section)
        
        self.markdown_generator.sections = sections
        return self.markdown_generator.generate_markdown()
    
    def _extract_project_info(self, config_files: List[str]) -> Dict[str, Any]:
        """Extract project information from config files"""
        project_info = {}
        
        for config_file in config_files:
            config_path = self.project_path / config_file
            if config_path.exists():
                try:
                    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                        with open(config_path, 'r') as f:
                            config = yaml.safe_load(f)
                            if 'project' in config:
                                project_info.update(config['project'])
                    elif config_file == 'package.json':
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            project_info.update({
                                'name': config.get('name', ''),
                                'description': config.get('description', ''),
                                'version': config.get('version', ''),
                                'license': config.get('license', '')
                            })
                    # Add more config file parsers as needed
                    
                except Exception as e:
                    logger.error(f"Failed to parse {config_file}: {e}")
        
        return project_info
    
    def _generate_features_list(self) -> str:
        """Generate features list based on detected components"""
        features = []
        
        # Detect features based on code analysis
        for code_doc in self.extracted_code_docs:
            if 'api' in code_doc.module_name.lower():
                features.append("🔌 RESTful API")
            if 'auth' in code_doc.module_name.lower():
                features.append("🔐 Authentication & Authorization")
            if 'database' in code_doc.module_name.lower() or 'db' in code_doc.module_name.lower():
                features.append("💾 Database Integration")
            if 'test' in code_doc.module_name.lower():
                features.append("🧪 Comprehensive Testing")
            if 'monitor' in code_doc.module_name.lower():
                features.append("📊 Performance Monitoring")
            if 'security' in code_doc.module_name.lower():
                features.append("🛡️ Security Features")
        
        # Add default features
        default_features = [
            "⚡ High Performance",
            "🐳 Docker Support",
            "☸️ Kubernetes Ready",
            "📖 Auto-generated Documentation",
            "🔄 CI/CD Pipeline"
        ]
        
        features.extend(default_features)
        
        return '\n'.join(f"- {feature}" for feature in sorted(set(features)))
    
    def _generate_installation_guide(self) -> str:
        """Generate installation guide"""
        return """
## Prerequisites

- Python 3.9 or higher
- Docker (optional)
- Kubernetes cluster (optional)

## Install from PyPI

```bash
pip install {project_name}
```

## Install from Source

```bash
git clone https://github.com/yourusername/{project_name}.git
cd {project_name}
pip install -r requirements.txt
pip install -e .
```

## Docker Installation

```bash
docker pull yourusername/{project_name}:latest
docker run -p 8080:8080 yourusername/{project_name}:latest
```
        """.strip()
    
    def _generate_usage_guide(self) -> str:
        """Generate usage guide"""
        return """
## Quick Start

```python
from {project_name} import ProjectCreator

# Create a new project creator instance
creator = ProjectCreator()

# Configure your project
creator.configure(
    name="my-awesome-project",
    type="web_application"
)

# Create the project
result = creator.create_project()
print(f"Project created: {result}")
```

## Command Line Interface

```bash
# Create a new project
poce create --name my-project --type web_application

# List existing projects
poce list

# Get project status
poce status my-project
```

For more examples, see the [examples](examples/) directory.
        """.strip()
    
    def _generate_contributing_guide(self) -> str:
        """Generate contributing guide"""
        return """
## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/repo.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\\Scripts\\activate` (Windows)
5. Install dependencies: `pip install -r requirements-dev.txt`
6. Install pre-commit hooks: `pre-commit install`

## Running Tests

```bash
pytest tests/
```

## Code Style

We use Black for code formatting and flake8 for linting:

```bash
black .
flake8 .
```

## Submitting Changes

1. Create a new branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Add tests for your changes
4. Run the test suite
5. Commit your changes: `git commit -am 'Add some feature'`
6. Push to the branch: `git push origin feature/my-feature`
7. Submit a pull request
        """.strip()
    
    def _extract_api_endpoints(self):
        """Extract API endpoints from code"""
        # This is a simplified extraction
        # In practice, you would analyze Flask/FastAPI decorators
        
        for code_doc in self.extracted_code_docs:
            for func_info in code_doc.functions:
                # Look for API-like function names
                if any(keyword in func_info['name'].lower() for keyword in ['get_', 'post_', 'put_', 'delete_', 'api_']):
                    endpoint = APIEndpoint(
                        path=f"/{func_info['name'].replace('_', '/')}",
                        method='GET',  # Default method
                        summary=func_info['name'].replace('_', ' ').title(),
                        description=func_info['docstring'] or f"API endpoint for {func_info['name']}"
                    )
                    self.api_generator.add_endpoint(endpoint)
    
    def _format_class_documentation(self, class_info: Dict[str, Any]) -> str:
        """Format class documentation"""
        lines = []
        
        # Class description
        if class_info['docstring']:
            lines.append(class_info['docstring'])
            lines.append("")
        
        # Inheritance
        if class_info['bases']:
            lines.append(f"**Inherits from:** {', '.join(class_info['bases'])}")
            lines.append("")
        
        # Methods
        if class_info['methods']:
            lines.append("**Methods:**")
            lines.append("")
            for method in class_info['methods']:
                lines.append(f"- `{method['name']}()`: {method['docstring'] or 'No description'}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def _format_function_documentation(self, func_info: Dict[str, Any]) -> str:
        """Format function documentation"""
        lines = []
        
        # Function signature
        params = ', '.join([
            f"{p['name']}: {p['type'] or 'Any'}" + (f" = {p['default']}" if p['default'] else "")
            for p in func_info['parameters']
        ])
        
        return_type = f" -> {func_info['return_type']}" if func_info['return_type'] else ""
        
        lines.append(f"```python")
        lines.append(f"def {func_info['name']}({params}){return_type}:")
        lines.append("```")
        lines.append("")
        
        # Function description
        if func_info['docstring']:
            lines.append(func_info['docstring'])
            lines.append("")
        
        # Parameters
        if func_info['parameters']:
            lines.append("**Parameters:**")
            lines.append("")
            for param in func_info['parameters']:
                param_desc = f"- `{param['name']}`"
                if param['type']:
                    param_desc += f" ({param['type']})"
                if param['default']:
                    param_desc += f" - Default: `{param['default']}`"
                lines.append(param_desc)
            lines.append("")
        
        return '\n'.join(lines)
    
    def generate_all_documentation(self, output_dir: Path):
        """Generate all project documentation"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating complete project documentation...")
        
        # Scan project
        self.scan_project()
        
        # Generate README
        readme_content = self.generate_readme()
        with open(output_dir / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Generate API documentation
        api_content = self.generate_api_documentation()
        docs_dir = output_dir / "docs"
        docs_dir.mkdir(exist_ok=True)
        with open(docs_dir / "api.md", 'w', encoding='utf-8') as f:
            f.write(api_content)
        
        # Generate code documentation
        code_content = self.generate_code_documentation()
        with open(docs_dir / "code.md", 'w', encoding='utf-8') as f:
            f.write(code_content)
        
        # Generate OpenAPI spec
        api_spec = self.api_generator.generate_openapi_spec()
        with open(docs_dir / "openapi.json", 'w', encoding='utf-8') as f:
            json.dump(api_spec, f, indent=2)
        
        logger.info(f"Documentation generated in {output_dir}")

# ==========================================
# SPHINX DOCUMENTATION BUILDER
# ==========================================

class SphinxDocBuilder:
    """Builds Sphinx documentation"""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.docs_dir = project_path / "docs"
        self.build_dir = self.docs_dir / "_build"
    
    def setup_sphinx_project(self):
        """Setup Sphinx project structure"""
        if not SPHINX_AVAILABLE:
            logger.error("Sphinx not available")
            return False
        
        self.docs_dir.mkdir(exist_ok=True)
        
        # Create conf.py
        conf_content = self._generate_sphinx_config()
        with open(self.docs_dir / "conf.py", 'w') as f:
            f.write(conf_content)
        
        # Create index.rst
        index_content = self._generate_index_rst()
        with open(self.docs_dir / "index.rst", 'w') as f:
            f.write(index_content)
        
        # Create Makefile
        makefile_content = self._generate_makefile()
        with open(self.docs_dir / "Makefile", 'w') as f:
            f.write(makefile_content)
        
        return True
    
    def build_documentation(self, output_format: str = "html") -> bool:
        """Build Sphinx documentation"""
        if not SPHINX_AVAILABLE:
            logger.error("Sphinx not available")
            return False
        
        try:
            # Build arguments
            source_dir = str(self.docs_dir)
            build_dir = str(self.build_dir / output_format)
            
            # Run Sphinx build
            args = ['-b', output_format, source_dir, build_dir]
            result = build_main(args)
            
            if result == 0:
                logger.info(f"Sphinx documentation built successfully: {build_dir}")
                return True
            else:
                logger.error(f"Sphinx build failed with code: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to build Sphinx documentation: {e}")
            return False
    
    def _generate_sphinx_config(self) -> str:
        """Generate Sphinx configuration"""
        return '''
# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'P.O.C.E. Project Creator'
copyright = '2024, P.O.C.E. Technologies'
author = 'P.O.C.E. Technologies'
version = '4.0.0'
release = '4.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.githubpages',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for autodoc extension -------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# -- Options for napoleon extension ------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
        '''.strip()
    
    def _generate_index_rst(self) -> str:
        """Generate index.rst"""
        return '''
P.O.C.E. Project Creator Documentation
======================================

Welcome to the P.O.C.E. Project Creator documentation!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
        '''.strip()
    
    def _generate_makefile(self) -> str:
        """Generate Makefile for Sphinx"""
        return '''
# Minimal makefile for Sphinx documentation

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help Makefile

help:
\t@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: Makefile
%: Makefile
\t@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
        '''.strip()

# ==========================================
# EXAMPLE USAGE
# ==========================================

def example_documentation_generation():
    """Example of using the documentation generation system"""
    
    # Create project documentation generator
    project_path = Path(".")  # Current directory
    doc_generator = ProjectDocGenerator(project_path)
    
    # Generate all documentation
    output_dir = Path("generated_docs")
    doc_generator.generate_all_documentation(output_dir)
    
    # Build Sphinx documentation if available
    if SPHINX_AVAILABLE:
        sphinx_builder = SphinxDocBuilder(project_path)
        sphinx_builder.setup_sphinx_project()
        sphinx_builder.build_documentation("html")
    
    print(f"Documentation generated in {output_dir}")

if __name__ == "__main__":
    example_documentation_generation()