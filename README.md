# P.O.C.E. Project Creator v4.0

**Prompt Orchestrator + Context Engineering**

An advanced DevOps automation tool that creates professional GitHub repositories with comprehensive CI/CD pipelines, automated testing, security scanning, and performance monitoring.

## 🚀 Key Features

- **DevOps Automation**: Complete CI/CD pipeline setup with GitHub Actions
- **Intelligent MCP Server Selection**: Optimized server discovery and synergy calculation via Smithery.ai
- **Cascading Workflows**: Parallel task execution with dependency resolution
- **Security-First**: Automated security scanning and compliance checks
- **Performance Monitoring**: Real-time metrics and alerting
- **3D Metal GUI**: Professional interface with realistic metal styling
- **YAML-Driven Configuration**: Granular control over all aspects
- **Resource Optimization**: Maximum efficiency and speed

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.9 or higher (3.11 recommended)
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space
- **Network**: Stable internet connection for GitHub API and MCP server discovery

### Recommended Requirements
- **OS**: Latest stable versions
- **Python**: 3.11 or 3.12
- **RAM**: 16GB for optimal performance
- **CPU**: Multi-core processor for parallel task execution
- **Storage**: SSD with 10GB free space

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/brian95240/P.O.C.E.-Project-Creator.git
cd P.O.C.E.-Project-Creator
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv poce_env

# Activate virtual environment
# On Windows:
poce_env\Scripts\activate
# On macOS/Linux:
source poce_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Usage

```bash
python src/poce_project_creator_v4.py
```

### CLI Usage

```bash
python src/poce_cli.py --help
```

### Configuration

Edit `src/poce_config.yaml` to customize your project settings:

```yaml
# Example configuration
project:
  name: "my-awesome-project"
  description: "An amazing project created with P.O.C.E."
  
github:
  auto_create_repo: true
  enable_actions: true
  
security:
  enable_scanning: true
  compliance_checks: true
```

## 📁 Project Structure

```
P.O.C.E.-Project-Creator/
├── 📄 README.md                      # Project documentation
├── 📄 LICENSE-GPL                    # GPL v3.0 license
├── 📄 LICENSE-COMMERCIAL              # Commercial license
├── 📄 setup.py                       # PyPI packaging configuration
├── 📄 requirements.txt               # Python dependencies
├── 📄 Dockerfile.txt                 # Docker configuration
├── 📁 src/                           # Source code directory
│   ├── poce_project_creator_v4.py    # Main application
│   ├── poce_master_orchestrator.py   # Core orchestration engine
│   ├── poce_cli.py                   # Command-line interface
│   ├── poce_config.yaml              # Configuration file
│   ├── documentation_doc_generator.py # Documentation generator
│   ├── infrastructure_iac_manager.py # Infrastructure as Code manager
│   ├── monitoring_monitoring_system.py # Monitoring system
│   ├── security_security_manager.py  # Security manager
│   ├── performance_optimization_engine.py # Performance optimizer
│   ├── mcp_integration_templates.py  # MCP integration templates
│   ├── tests_test_framework.py       # Testing framework
│   └── k8s_namespace.yaml            # Kubernetes configuration
├── 📁 docs/                          # Documentation directory
│   ├── P.O.C.E. Project Creator.txt
│   └── P.O.C.E. Project Creator v4.0 - Installation & Setup Guide.txt
└── 📁 .github/workflows/             # CI/CD automation
    └── github_workflows_main.yml     # GitHub Actions workflow
```

## 🔧 Core Components

### Master Orchestrator
The central engine that coordinates all project creation activities with intelligent workflow management.

### MCP Integration
Advanced Model Context Protocol integration with Smithery.ai for optimal server selection and synergy calculation.

### Security Manager
Comprehensive security scanning, vulnerability assessment, and compliance checking.

### Performance Engine
Real-time performance monitoring, optimization suggestions, and resource management.

### Documentation Generator
Automated generation of professional documentation, README files, and API docs.

## 🐳 Docker Support

Build and run with Docker:

```bash
# Build the image
docker build -f Dockerfile.txt -t poce-creator .

# Run the container
docker run -it poce-creator
```

## ☸️ Kubernetes Deployment

Deploy to Kubernetes:

```bash
kubectl apply -f k8s_namespace.yaml
```

## 🧪 Testing

Run the test suite:

```bash
python src/tests_test_framework.py
```

## 📊 Monitoring

The built-in monitoring system provides:
- Real-time performance metrics
- Resource utilization tracking
- Error rate monitoring
- Custom alerting

## 🛡️ Security

P.O.C.E. includes comprehensive security features:
- Automated vulnerability scanning
- Dependency security checks
- Code quality analysis
- Compliance validation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is available under dual licensing:

- **GPL v3.0**: For open source projects and personal use (see `LICENSE-GPL`)
- **Commercial License**: For proprietary applications and commercial use (see `LICENSE-COMMERCIAL`)

For commercial licensing inquiries, please contact: brian95240@users.noreply.github.com

## 🆘 Support

For support and documentation, please refer to the `How to set-up/` directory or open an issue on GitHub.

## 🔄 Version History

- **v4.0**: Latest version with MCP integration and advanced orchestration
- **v3.x**: Previous versions with basic automation features

---

**Created with ❤️ by the P.O.C.E. Team**

