# poce_cli.py
"""
P.O.C.E. Project Creator CLI Interface v4.0
Command-line interface for headless DevOps automation
Supports batch operations, CI/CD integration, and remote execution
"""

import click
import asyncio
import json
import yaml
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import getpass
import subprocess
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint

# Import core modules
from poce_project_creator_v4 import (
    ConfigManager, 
    WorkflowEngine, 
    MCPServerManager,
    logger
)

# CLI Configuration
console = Console()
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

class CLIContext:
    """Shared context for CLI commands"""
    def __init__(self):
        self.config_manager = None
        self.workflow_engine = None
        self.mcp_manager = None
        self.verbose = False
        self.output_format = 'table'
        self.config_path = 'poce_config.yaml'

pass_context = click.make_pass_decorator(CLIContext, ensure=True)

# Utility functions
def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('poce_cli.log')
        ]
    )

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[red]Error: Configuration file {config_path} not found[/red]")
        sys.exit(1)
    except yaml.YAMLError as e:
        console.print(f"[red]Error parsing YAML configuration: {e}[/red]")
        sys.exit(1)

def save_config(config: Dict, config_path: str):
    """Save configuration to YAML file"""
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        console.print(f"[green]Configuration saved to {config_path}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving configuration: {e}[/red]")
        sys.exit(1)

def format_output(data: Any, format_type: str = 'table') -> None:
    """Format and display output based on specified format"""
    if format_type == 'json':
        console.print_json(json.dumps(data, indent=2, default=str))
    elif format_type == 'yaml':
        console.print(yaml.dump(data, default_flow_style=False))
    elif format_type == 'table' and isinstance(data, list):
        if data and isinstance(data[0], dict):
            table = Table()
            # Add columns from first item
            for key in data[0].keys():
                table.add_column(str(key).title(), style="cyan")
            # Add rows
            for item in data:
                table.add_row(*[str(v) for v in item.values()])
            console.print(table)
        else:
            console.print(data)
    else:
        console.print(data)

# Main CLI Group
@click.group(context_settings=CONTEXT_SETTINGS)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', default='poce_config.yaml', help='Configuration file path')
@click.option('--output', '-o', type=click.Choice(['table', 'json', 'yaml']), 
              default='table', help='Output format')
@pass_context
def cli(ctx: CLIContext, verbose: bool, config: str, output: str):
    """
    P.O.C.E. Project Creator CLI v4.0
    
    Advanced DevOps automation platform with MCP server integration,
    cascading workflows, and comprehensive CI/CD pipeline generation.
    """
    ctx.verbose = verbose
    ctx.output_format = output
    ctx.config_path = config
    
    setup_logging(verbose)
    
    # Initialize managers
    try:
        ctx.config_manager = ConfigManager()
        if os.path.exists(config):
            with open(config, 'r') as f:
                ctx.config_manager.config = yaml.safe_load(f)
        
        ctx.workflow_engine = WorkflowEngine(ctx.config_manager)
        ctx.mcp_manager = MCPServerManager()
        
    except Exception as e:
        console.print(f"[red]Initialization error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)

# Configuration Commands
@cli.group()
def config():
    """Configuration management commands"""
    pass

@config.command()
@pass_context
def show(ctx: CLIContext):
    """Display current configuration"""
    format_output(ctx.config_manager.config, ctx.output_format)

@config.command()
@click.argument('key')
@click.argument('value')
@pass_context
def set(ctx: CLIContext, key: str, value: str):
    """Set configuration value"""
    try:
        # Parse value as JSON if possible, otherwise use as string
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError:
            parsed_value = value
        
        ctx.config_manager.update_field(key, parsed_value)
        ctx.config_manager.save_config()
        console.print(f"[green]Set {key} = {parsed_value}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error setting configuration: {e}[/red]")

@config.command()
@click.argument('key')
@pass_context
def get(ctx: CLIContext, key: str):
    """Get configuration value"""
    try:
        keys = key.split('.')
        current = ctx.config_manager.config
        
        for k in keys:
            current = current[k]
        
        format_output({key: current}, ctx.output_format)
        
    except KeyError:
        console.print(f"[red]Configuration key '{key}' not found[/red]")
    except Exception as e:
        console.print(f"[red]Error getting configuration: {e}[/red]")

@config.command()
@pass_context
def validate(ctx: CLIContext):
    """Validate configuration"""
    try:
        # Validate GitHub token
        github_token = ctx.config_manager.config.get('github', {}).get('token')
        if not github_token:
            console.print("[red]GitHub token not configured[/red]")
            return
        
        from github import Github
        g = Github(github_token)
        user = g.get_user()
        
        console.print(f"[green]✓ GitHub connection successful (user: {user.login})[/green]")
        console.print("[green]✓ Configuration is valid[/green]")
        
    except Exception as e:
        console.print(f"[red]Configuration validation failed: {e}[/red]")

# Project Management Commands
@cli.group()
def project():
    """Project creation and management commands"""
    pass

@project.command()
@click.option('--name', '-n', required=True, help='Project name')
@click.option('--type', '-t', default='web_application', 
              type=click.Choice(['web_application', 'mobile_app', 'api_service', 
                               'ml_model', 'data_pipeline', 'automation_tool']),
              help='Project type')
@click.option('--description', '-d', help='Project description')
@click.option('--private/--public', default=True, help='Repository visibility')
@click.option('--template', help='Project template path')
@click.option('--dry-run', is_flag=True, help='Simulate creation without actual execution')
@pass_context
def create(ctx: CLIContext, name: str, type: str, description: str, 
           private: bool, template: str, dry_run: bool):
    """Create a new project with full DevOps automation"""
    
    if dry_run:
        console.print("[yellow]DRY RUN MODE - No actual changes will be made[/yellow]")
    
    # Prepare project data
    project_data = {
        'name': name,
        'type': type,
        'description': description or f"A {type.replace('_', ' ')} project created with P.O.C.E.",
        'private': private,
        'template': template,
        'github_token': ctx.config_manager.config.get('github', {}).get('token'),
        'github_username': None  # Will be auto-detected
    }
    
    # Validate required data
    if not project_data['github_token']:
        console.print("[red]Error: GitHub token not configured[/red]")
        console.print("Use: poce config set github.token YOUR_TOKEN")
        return
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Creating project...", total=None)
            
            if dry_run:
                console.print(Panel(
                    f"Would create project '{name}' with:\n"
                    f"Type: {type}\n"
                    f"Private: {private}\n"
                    f"Description: {project_data['description']}\n"
                    f"Template: {template or 'Default'}",
                    title="Dry Run Summary",
                    border_style="yellow"
                ))
                return
            
            # Execute workflow
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            progress.update(task, description="Executing workflow...")
            results = loop.run_until_complete(
                ctx.workflow_engine.execute_workflow(project_data)
            )
            
            progress.update(task, description="Complete!")
            
        # Display results
        if results['status'] == 'completed':
            console.print(Panel(
                f"✅ Project '{name}' created successfully!\n"
                f"Repository: https://github.com/{project_data.get('github_username', 'USER')}/{name}\n"
                f"Workflow ID: {results['workflow_id']}",
                title="Success",
                border_style="green"
            ))
            
            if ctx.verbose:
                format_output(results, ctx.output_format)
        else:
            console.print(f"[red]Project creation failed: {results.get('error', 'Unknown error')}[/red]")
            
    except Exception as e:
        console.print(f"[red]Error creating project: {e}[/red]")
        if ctx.verbose:
            console.print_exception()

@project.command()
@click.argument('name')
@pass_context
def status(ctx: CLIContext, name: str):
    """Check project status"""
    try:
        # This would integrate with GitHub API to check repository status
        from github import Github
        
        github_token = ctx.config_manager.config.get('github', {}).get('token')
        if not github_token:
            console.print("[red]GitHub token not configured[/red]")
            return
        
        g = Github(github_token)
        user = g.get_user()
        
        try:
            repo = g.get_repo(f"{user.login}/{name}")
            
            # Get repository information
            info = {
                'name': repo.name,
                'private': repo.private,
                'created': repo.created_at,
                'updated': repo.updated_at,
                'size': repo.size,
                'language': repo.language,
                'forks': repo.forks_count,
                'stars': repo.stargazers_count,
                'issues': repo.open_issues_count,
                'default_branch': repo.default_branch
            }
            
            format_output(info, ctx.output_format)
            
        except Exception:
            console.print(f"[red]Repository '{name}' not found[/red]")
            
    except Exception as e:
        console.print(f"[red]Error checking project status: {e}[/red]")

@project.command()
@pass_context
def list(ctx: CLIContext):
    """List all projects"""
    try:
        from github import Github
        
        github_token = ctx.config_manager.config.get('github', {}).get('token')
        if not github_token:
            console.print("[red]GitHub token not configured[/red]")
            return
        
        g = Github(github_token)
        user = g.get_user()
        
        repos = []
        for repo in user.get_repos():
            repos.append({
                'name': repo.name,
                'private': '🔒' if repo.private else '🌐',
                'language': repo.language or 'N/A',
                'updated': repo.updated_at.strftime('%Y-%m-%d'),
                'stars': repo.stargazers_count,
                'forks': repo.forks_count
            })
        
        if not repos:
            console.print("[yellow]No repositories found[/yellow]")
            return
        
        format_output(repos, ctx.output_format)
        
    except Exception as e:
        console.print(f"[red]Error listing projects: {e}[/red]")

# MCP Server Commands
@cli.group()
def mcp():
    """MCP server management commands"""
    pass

@mcp.command()
@click.option('--refresh', is_flag=True, help='Refresh server cache')
@pass_context
def discover(ctx: CLIContext, refresh: bool):
    """Discover optimal MCP servers"""
    try:
        project_type = ctx.config_manager.config.get('project', {}).get('type', 'web_application')
        requirements = ctx.config_manager.config.get('mcp_servers', {}).get('required_capabilities', [])
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Discovering MCP servers...", total=None)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            servers = loop.run_until_complete(
                ctx.mcp_manager.discover_optimal_servers(project_type, requirements)
            )
            
            progress.update(task, description="Discovery complete!")
        
        if servers:
            console.print(f"[green]Found {len(servers)} optimal MCP servers:[/green]")
            format_output(servers, ctx.output_format)
        else:
            console.print("[yellow]No MCP servers discovered[/yellow]")
            
    except Exception as e:
        console.print(f"[red]Error discovering MCP servers: {e}[/red]")

@mcp.command()
@click.argument('server_name')
@pass_context
def info(ctx: CLIContext, server_name: str):
    """Get information about a specific MCP server"""
    try:
        # This would query the server for detailed information
        console.print(f"[blue]Querying MCP server: {server_name}[/blue]")
        
        # Placeholder for actual server query
        server_info = {
            'name': server_name,
            'status': 'active',
            'capabilities': ['context_management', 'task_automation'],
            'performance_score': 85,
            'synergy_score': 92,
            'last_updated': datetime.now().isoformat()
        }
        
        format_output(server_info, ctx.output_format)
        
    except Exception as e:
        console.print(f"[red]Error getting server info: {e}[/red]")

# Workflow Commands
@cli.group()
def workflow():
    """Workflow management commands"""
    pass

@workflow.command()
@click.option('--project-data', type=click.File('r'), help='JSON file with project data')
@click.option('--async', 'async_execution', is_flag=True, help='Execute workflow asynchronously')
@pass_context
def execute(ctx: CLIContext, project_data, async_execution: bool):
    """Execute a custom workflow"""
    try:
        if project_data:
            data = json.load(project_data)
        else:
            # Use interactive mode to gather project data
            data = {
                'name': click.prompt('Project name'),
                'type': click.prompt('Project type', default='web_application'),
                'github_token': ctx.config_manager.config.get('github', {}).get('token')
            }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Executing workflow...", total=None)
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            results = loop.run_until_complete(
                ctx.workflow_engine.execute_workflow(data)
            )
            
            progress.update(task, description="Workflow complete!")
        
        format_output(results, ctx.output_format)
        
    except Exception as e:
        console.print(f"[red]Error executing workflow: {e}[/red]")

@workflow.command()
@pass_context
def status(ctx: CLIContext):
    """Show workflow engine status"""
    try:
        status_info = {
            'engine_status': 'active',
            'max_concurrent_tasks': ctx.config_manager.config.get('workflow', {}).get('execution', {}).get('max_concurrent_tasks', 6),
            'cascading_enabled': ctx.config_manager.config.get('workflow', {}).get('execution', {}).get('cascading_enabled', True),
            'active_workflows': 0,  # Would be tracked in production
            'completed_workflows': 0,  # Would be tracked in production
            'performance_metrics': {
                'average_execution_time': '0s',
                'success_rate': '100%',
                'resource_efficiency': 'high'
            }
        }
        
        format_output(status_info, ctx.output_format)
        
    except Exception as e:
        console.print(f"[red]Error getting workflow status: {e}[/red]")

# Utility Commands
@cli.group()
def utils():
    """Utility commands"""
    pass

@utils.command()
@click.option('--github-token', help='GitHub token to validate')
@pass_context
def validate_token(ctx: CLIContext, github_token: str):
    """Validate GitHub token"""
    try:
        token = github_token or ctx.config_manager.config.get('github', {}).get('token')
        
        if not token:
            token = getpass.getpass("Enter GitHub token: ")
        
        from github import Github
        g = Github(token)
        user = g.get_user()
        
        console.print(f"[green]✅ Token is valid for user: {user.login}[/green]")
        console.print(f"[green]✅ Remaining rate limit: {g.get_rate_limit().core.remaining}[/green]")
        
        # Test repository access
        repos = list(user.get_repos())
        console.print(f"[green]✅ Can access {len(repos)} repositories[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Token validation failed: {e}[/red]")

@utils.command()
@click.option('--include-dev', is_flag=True, help='Include development dependencies')
@pass_context
def check_deps(ctx: CLIContext, include_dev: bool):
    """Check system dependencies"""
    try:
        console.print("[blue]Checking system dependencies...[/blue]")
        
        deps = {
            'Python': sys.version,
            'Platform': sys.platform,
        }
        
        # Check Python packages
        import pkg_resources
        installed_packages = [d.project_name for d in pkg_resources.working_set]
        
        required_packages = [
            'customtkinter', 'PyGithub', 'PyPDF2', 'pyyaml', 
            'aiohttp', 'requests', 'click', 'rich'
        ]
        
        if include_dev:
            required_packages.extend(['pytest', 'black', 'flake8', 'mypy'])
        
        missing_packages = []
        for package in required_packages:
            if package not in installed_packages:
                missing_packages.append(package)
            else:
                deps[f'✅ {package}'] = 'installed'
        
        if missing_packages:
            for package in missing_packages:
                deps[f'❌ {package}'] = 'missing'
        
        format_output(deps, ctx.output_format)
        
        if missing_packages:
            console.print(f"\n[yellow]Install missing packages with:[/yellow]")
            console.print(f"pip install {' '.join(missing_packages)}")
        
    except Exception as e:
        console.print(f"[red]Error checking dependencies: {e}[/red]")

@utils.command()
@click.option('--output-file', '-o', help='Output file for the report')
@pass_context
def generate_report(ctx: CLIContext, output_file: str):
    """Generate system and configuration report"""
    try:
        report = {
            'generated_at': datetime.now().isoformat(),
            'poce_version': '4.0.0',
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': os.getcwd()
            },
            'configuration': ctx.config_manager.config,
            'mcp_servers': [],  # Would be populated with discovered servers
            'recent_projects': []  # Would be populated with project history
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                if output_file.endswith('.json'):
                    json.dump(report, f, indent=2, default=str)
                else:
                    yaml.dump(report, f, default_flow_style=False)
            console.print(f"[green]Report saved to {output_file}[/green]")
        else:
            format_output(report, ctx.output_format)
        
    except Exception as e:
        console.print(f"[red]Error generating report: {e}[/red]")

# Interactive Commands
@cli.command()
@pass_context
def interactive(ctx: CLIContext):
    """Start interactive mode"""
    console.print(Panel(
        "🚀 P.O.C.E. Project Creator Interactive Mode\n\n"
        "Available commands:\n"
        "• create - Create a new project\n"
        "• config - Manage configuration\n"
        "• mcp - MCP server operations\n"
        "• workflow - Workflow management\n"
        "• exit - Exit interactive mode",
        title="Interactive Mode",
        border_style="blue"
    ))
    
    while True:
        try:
            command = console.input("\n[bold blue]poce>[/bold blue] ")
            
            if command.lower() in ['exit', 'quit', 'q']:
                console.print("[green]Goodbye![/green]")
                break
            
            if command.strip():
                # Parse and execute command
                args = command.split()
                try:
                    cli.main(args, standalone_mode=False)
                except SystemExit:
                    pass  # Ignore SystemExit in interactive mode
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit[/yellow]")
        except EOFError:
            console.print("\n[green]Goodbye![/green]")
            break

# Version Command
@cli.command()
def version():
    """Show version information"""
    console.print(Panel(
        "P.O.C.E. Project Creator CLI v4.0.0\n"
        "Advanced DevOps Automation Platform\n\n"
        "🔧 DevOps automation with CI/CD\n"
        "🤖 Intelligent MCP server integration\n"
        "⚡ Cascading workflow execution\n"
        "🛡️ Security-first approach\n"
        "📊 Performance monitoring\n"
        "🎨 Beautiful GUI and CLI interfaces",
        title="Version Information",
        border_style="green"
    ))

if __name__ == '__main__':
    cli()