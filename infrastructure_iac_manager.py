# infrastructure/iac_manager.py
"""
P.O.C.E. Project Creator - Infrastructure as Code Manager v4.0
Comprehensive IaC templates and deployment automation with Terraform, Ansible,
Kubernetes, and cloud provider integrations
"""

import os
import json
import yaml
import subprocess
import tempfile
import shutil
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import logging
import re
import base64
import hashlib

logger = logging.getLogger(__name__)

# ==========================================
# INFRASTRUCTURE CONFIGURATION
# ==========================================

class CloudProvider(Enum):
    """Supported cloud providers"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    DIGITALOCEAN = "digitalocean"
    LINODE = "linode"
    LOCAL = "local"

class InfrastructureType(Enum):
    """Types of infrastructure"""
    KUBERNETES_CLUSTER = "kubernetes_cluster"
    CONTAINER_REGISTRY = "container_registry"
    DATABASE = "database"
    LOAD_BALANCER = "load_balancer"
    CDN = "cdn"
    MONITORING = "monitoring"
    LOGGING = "logging"
    SECURITY = "security"
    NETWORKING = "networking"
    STORAGE = "storage"

class DeploymentEnvironment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"

@dataclass
class InfrastructureResource:
    """Infrastructure resource definition"""
    name: str
    type: InfrastructureType
    provider: CloudProvider
    configuration: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    environment: DeploymentEnvironment = DeploymentEnvironment.DEVELOPMENT

@dataclass
class DeploymentPlan:
    """Infrastructure deployment plan"""
    plan_id: str
    resources: List[InfrastructureResource]
    execution_order: List[str]
    estimated_cost: Optional[float] = None
    deployment_time_estimate: Optional[int] = None  # minutes
    created_at: datetime = field(default_factory=datetime.utcnow)

# ==========================================
# TERRAFORM TEMPLATE GENERATOR
# ==========================================

class TerraformTemplateGenerator:
    """Generates Terraform templates for infrastructure"""
    
    def __init__(self):
        self.templates: Dict[str, str] = {}
        self.providers: Dict[CloudProvider, Dict[str, Any]] = {}
        self._load_provider_configs()
    
    def _load_provider_configs(self):
        """Load provider configurations"""
        self.providers = {
            CloudProvider.AWS: {
                'provider_block': '''
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.0"
}

provider "aws" {
  region = var.aws_region
}
                '''.strip(),
                'variables': {
                    'aws_region': {'default': 'us-west-2', 'description': 'AWS region'}
                }
            },
            CloudProvider.GCP: {
                'provider_block': '''
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
  required_version = ">= 1.0"
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}
                '''.strip(),
                'variables': {
                    'gcp_project_id': {'description': 'GCP Project ID'},
                    'gcp_region': {'default': 'us-central1', 'description': 'GCP region'}
                }
            },
            CloudProvider.AZURE: {
                'provider_block': '''
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
  required_version = ">= 1.0"
}

provider "azurerm" {
  features {}
}
                '''.strip(),
                'variables': {
                    'azure_location': {'default': 'East US', 'description': 'Azure location'}
                }
            }
        }
    
    def generate_kubernetes_cluster(self, resource: InfrastructureResource) -> str:
        """Generate Kubernetes cluster Terraform template"""
        config = resource.configuration
        
        if resource.provider == CloudProvider.AWS:
            return self._generate_eks_cluster(resource)
        elif resource.provider == CloudProvider.GCP:
            return self._generate_gke_cluster(resource)
        elif resource.provider == CloudProvider.AZURE:
            return self._generate_aks_cluster(resource)
        else:
            raise ValueError(f"Kubernetes cluster not supported for {resource.provider}")
    
    def _generate_eks_cluster(self, resource: InfrastructureResource) -> str:
        """Generate AWS EKS cluster template"""
        config = resource.configuration
        cluster_name = resource.name
        
        template = f'''
# EKS Cluster
resource "aws_eks_cluster" "{cluster_name}" {{
  name     = "{cluster_name}"
  role_arn = aws_iam_role.{cluster_name}_cluster_role.arn
  version  = "{config.get('kubernetes_version', '1.27')}"

  vpc_config {{
    subnet_ids              = [for subnet in aws_subnet.{cluster_name}_subnet : subnet.id]
    endpoint_private_access = {str(config.get('private_endpoint', True)).lower()}
    endpoint_public_access  = {str(config.get('public_endpoint', True)).lower()}
    public_access_cidrs     = {json.dumps(config.get('public_access_cidrs', ['0.0.0.0/0']))}
  }}

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  tags = {{
    Name        = "{cluster_name}"
    Environment = "{resource.environment.value}"
    ManagedBy   = "terraform"
  }}

  depends_on = [
    aws_iam_role_policy_attachment.{cluster_name}_cluster_policy,
    aws_iam_role_policy_attachment.{cluster_name}_service_policy,
  ]
}}

# IAM Role for EKS Cluster
resource "aws_iam_role" "{cluster_name}_cluster_role" {{
  name = "{cluster_name}-cluster-role"

  assume_role_policy = jsonencode({{
    Version = "2012-10-17"
    Statement = [
      {{
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {{
          Service = "eks.amazonaws.com"
        }}
      }}
    ]
  }})
}}

resource "aws_iam_role_policy_attachment" "{cluster_name}_cluster_policy" {{
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.{cluster_name}_cluster_role.name
}}

resource "aws_iam_role_policy_attachment" "{cluster_name}_service_policy" {{
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSServicePolicy"
  role       = aws_iam_role.{cluster_name}_cluster_role.name
}}

# VPC for EKS Cluster
resource "aws_vpc" "{cluster_name}_vpc" {{
  cidr_block           = "{config.get('vpc_cidr', '10.0.0.0/16')}"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {{
    Name = "{cluster_name}-vpc"
    "kubernetes.io/cluster/{cluster_name}" = "shared"
  }}
}}

# Subnets
resource "aws_subnet" "{cluster_name}_subnet" {{
  count             = {config.get('subnet_count', 2)}
  vpc_id            = aws_vpc.{cluster_name}_vpc.id
  cidr_block        = cidrsubnet(aws_vpc.{cluster_name}_vpc.cidr_block, 8, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  map_public_ip_on_launch = true

  tags = {{
    Name = "{cluster_name}-subnet-${{count.index + 1}}"
    "kubernetes.io/cluster/{cluster_name}" = "shared"
    "kubernetes.io/role/elb" = "1"
  }}
}}

# Internet Gateway
resource "aws_internet_gateway" "{cluster_name}_igw" {{
  vpc_id = aws_vpc.{cluster_name}_vpc.id

  tags = {{
    Name = "{cluster_name}-igw"
  }}
}}

# Route Table
resource "aws_route_table" "{cluster_name}_rt" {{
  vpc_id = aws_vpc.{cluster_name}_vpc.id

  route {{
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.{cluster_name}_igw.id
  }}

  tags = {{
    Name = "{cluster_name}-rt"
  }}
}}

resource "aws_route_table_association" "{cluster_name}_rta" {{
  count          = length(aws_subnet.{cluster_name}_subnet)
  subnet_id      = aws_subnet.{cluster_name}_subnet[count.index].id
  route_table_id = aws_route_table.{cluster_name}_rt.id
}}

# EKS Node Group
resource "aws_eks_node_group" "{cluster_name}_nodes" {{
  cluster_name    = aws_eks_cluster.{cluster_name}.name
  node_group_name = "{cluster_name}-nodes"
  node_role_arn   = aws_iam_role.{cluster_name}_node_role.arn
  subnet_ids      = [for subnet in aws_subnet.{cluster_name}_subnet : subnet.id]
  instance_types  = {json.dumps(config.get('node_instance_types', ['t3.medium']))}

  scaling_config {{
    desired_size = {config.get('desired_nodes', 2)}
    max_size     = {config.get('max_nodes', 4)}
    min_size     = {config.get('min_nodes', 1)}
  }}

  update_config {{
    max_unavailable = 1
  }}

  depends_on = [
    aws_iam_role_policy_attachment.{cluster_name}_worker_policy,
    aws_iam_role_policy_attachment.{cluster_name}_cni_policy,
    aws_iam_role_policy_attachment.{cluster_name}_registry_policy,
  ]

  tags = {{
    Environment = "{resource.environment.value}"
  }}
}}

# IAM Role for EKS Node Group
resource "aws_iam_role" "{cluster_name}_node_role" {{
  name = "{cluster_name}-node-role"

  assume_role_policy = jsonencode({{
    Version = "2012-10-17"
    Statement = [
      {{
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {{
          Service = "ec2.amazonaws.com"
        }}
      }}
    ]
  }})
}}

resource "aws_iam_role_policy_attachment" "{cluster_name}_worker_policy" {{
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.{cluster_name}_node_role.name
}}

resource "aws_iam_role_policy_attachment" "{cluster_name}_cni_policy" {{
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.{cluster_name}_node_role.name
}}

resource "aws_iam_role_policy_attachment" "{cluster_name}_registry_policy" {{
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.{cluster_name}_node_role.name
}}

# Data sources
data "aws_availability_zones" "available" {{
  state = "available"
}}

# Outputs
output "{cluster_name}_cluster_endpoint" {{
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.{cluster_name}.endpoint
}}

output "{cluster_name}_cluster_security_group_id" {{
  description = "Security group ids attached to the cluster control plane"
  value       = aws_eks_cluster.{cluster_name}.vpc_config[0].cluster_security_group_id
}}

output "{cluster_name}_cluster_name" {{
  description = "Kubernetes Cluster Name"
  value       = aws_eks_cluster.{cluster_name}.name
}}
        '''.strip()
        
        return template
    
    def _generate_gke_cluster(self, resource: InfrastructureResource) -> str:
        """Generate GCP GKE cluster template"""
        config = resource.configuration
        cluster_name = resource.name
        
        template = f'''
# GKE Cluster
resource "google_container_cluster" "{cluster_name}" {{
  name     = "{cluster_name}"
  location = var.gcp_region
  
  # We can't create a cluster with no node pool defined, but we want to only use
  # separately managed node pools. So we create the smallest possible default
  # node pool and immediately delete it.
  remove_default_node_pool = true
  initial_node_count       = 1
  
  network    = google_compute_network.{cluster_name}_vpc.name
  subnetwork = google_compute_subnetwork.{cluster_name}_subnet.name
  
  # Configure various addons
  addons_config {{
    horizontal_pod_autoscaling {{
      disabled = false
    }}
    
    http_load_balancing {{
      disabled = false
    }}
    
    network_policy_config {{
      disabled = false
    }}
  }}
  
  # Enable network policy
  network_policy {{
    enabled = true
  }}
  
  # Configure cluster networking
  ip_allocation_policy {{
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }}
  
  # Configure master auth
  master_auth {{
    client_certificate_config {{
      issue_client_certificate = false
    }}
  }}
  
  # Configure private cluster
  private_cluster_config {{
    enable_private_nodes    = {str(config.get('private_nodes', True)).lower()}
    enable_private_endpoint = {str(config.get('private_endpoint', False)).lower()}
    master_ipv4_cidr_block  = "{config.get('master_cidr', '10.1.0.0/28')}"
  }}
  
  workload_identity_config {{
    workload_pool = "${{var.gcp_project_id}}.svc.id.goog"
  }}
  
  resource_labels = {{
    environment = "{resource.environment.value}"
    managed_by  = "terraform"
  }}
}}

# Separately Managed Node Pool
resource "google_container_node_pool" "{cluster_name}_nodes" {{
  name       = "{cluster_name}-nodes"
  location   = var.gcp_region
  cluster    = google_container_cluster.{cluster_name}.name
  node_count = {config.get('initial_node_count', 1)}
  
  autoscaling {{
    min_node_count = {config.get('min_nodes', 1)}
    max_node_count = {config.get('max_nodes', 4)}
  }}
  
  management {{
    auto_repair  = true
    auto_upgrade = true
  }}
  
  node_config {{
    preemptible  = {str(config.get('preemptible', False)).lower()}
    machine_type = "{config.get('machine_type', 'e2-medium')}"
    
    # Google recommends custom service accounts that have cloud-platform scope and permissions granted via IAM Roles.
    service_account = google_service_account.{cluster_name}_sa.email
    oauth_scopes    = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
    
    labels = {{
      env = "{resource.environment.value}"
    }}
    
    tags = ["gke-node", "{cluster_name}-node"]
    
    metadata = {{
      disable-legacy-endpoints = "true"
    }}
  }}
}}

# VPC
resource "google_compute_network" "{cluster_name}_vpc" {{
  name                    = "{cluster_name}-vpc"
  auto_create_subnetworks = false
}}

# Subnet
resource "google_compute_subnetwork" "{cluster_name}_subnet" {{
  name          = "{cluster_name}-subnet"
  ip_cidr_range = "{config.get('subnet_cidr', '10.0.0.0/24')}"
  region        = var.gcp_region
  network       = google_compute_network.{cluster_name}_vpc.name
  
  secondary_ip_range {{
    range_name    = "services"
    ip_cidr_range = "{config.get('services_cidr', '10.1.0.0/20')}"
  }}
  
  secondary_ip_range {{
    range_name    = "pods"
    ip_cidr_range = "{config.get('pods_cidr', '10.2.0.0/16')}"
  }}
}}

# Service Account
resource "google_service_account" "{cluster_name}_sa" {{
  account_id   = "{cluster_name}-sa"
  display_name = "GKE Service Account for {cluster_name}"
}}

# Outputs
output "{cluster_name}_cluster_name" {{
  description = "GKE Cluster Name"
  value       = google_container_cluster.{cluster_name}.name
}}

output "{cluster_name}_cluster_endpoint" {{
  description = "GKE Cluster Endpoint"
  value       = google_container_cluster.{cluster_name}.endpoint
  sensitive   = true
}}

output "{cluster_name}_cluster_location" {{
  description = "GKE Cluster Location"
  value       = google_container_cluster.{cluster_name}.location
}}
        '''.strip()
        
        return template
    
    def generate_database(self, resource: InfrastructureResource) -> str:
        """Generate database Terraform template"""
        config = resource.configuration
        db_name = resource.name
        
        if resource.provider == CloudProvider.AWS:
            return self._generate_rds_instance(resource)
        elif resource.provider == CloudProvider.GCP:
            return self._generate_cloud_sql_instance(resource)
        else:
            raise ValueError(f"Database not supported for {resource.provider}")
    
    def _generate_rds_instance(self, resource: InfrastructureResource) -> str:
        """Generate AWS RDS instance template"""
        config = resource.configuration
        db_name = resource.name
        
        template = f'''
# RDS Subnet Group
resource "aws_db_subnet_group" "{db_name}_subnet_group" {{
  name       = "{db_name}-subnet-group"
  subnet_ids = [for subnet in aws_subnet.{db_name}_subnet : subnet.id]

  tags = {{
    Name = "{db_name} DB subnet group"
  }}
}}

# RDS Instance
resource "aws_db_instance" "{db_name}" {{
  allocated_storage    = {config.get('allocated_storage', 20)}
  max_allocated_storage = {config.get('max_allocated_storage', 100)}
  storage_type         = "{config.get('storage_type', 'gp2')}"
  engine              = "{config.get('engine', 'postgres')}"
  engine_version      = "{config.get('engine_version', '14.9')}"
  instance_class      = "{config.get('instance_class', 'db.t3.micro')}"
  identifier          = "{db_name}"
  
  db_name  = "{config.get('database_name', 'pocedb')}"
  username = "{config.get('master_username', 'postgres')}"
  password = var.{db_name}_password
  
  vpc_security_group_ids = [aws_security_group.{db_name}_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.{db_name}_subnet_group.name
  
  backup_retention_period = {config.get('backup_retention_period', 7)}
  backup_window          = "{config.get('backup_window', '07:00-09:00')}"
  maintenance_window     = "{config.get('maintenance_window', 'Sun:09:00-Sun:11:00')}"
  
  storage_encrypted = {str(config.get('storage_encrypted', True)).lower()}
  
  skip_final_snapshot = {str(config.get('skip_final_snapshot', False)).lower()}
  deletion_protection = {str(config.get('deletion_protection', True)).lower()}
  
  # Enable monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.{db_name}_monitoring_role.arn
  
  # Enable logging
  enabled_cloudwatch_logs_exports = {json.dumps(config.get('log_exports', ['postgresql']))}
  
  tags = {{
    Name        = "{db_name}"
    Environment = "{resource.environment.value}"
    ManagedBy   = "terraform"
  }}
}}

# Security Group for RDS
resource "aws_security_group" "{db_name}_sg" {{
  name        = "{db_name}-sg"
  description = "Security group for {db_name} RDS instance"
  vpc_id      = aws_vpc.{db_name}_vpc.id

  ingress {{
    from_port   = {config.get('port', 5432)}
    to_port     = {config.get('port', 5432)}
    protocol    = "tcp"
    cidr_blocks = ["{config.get('allowed_cidr', '10.0.0.0/16')}"]
  }}

  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}

  tags = {{
    Name = "{db_name}-sg"
  }}
}}

# IAM Role for Enhanced Monitoring
resource "aws_iam_role" "{db_name}_monitoring_role" {{
  name = "{db_name}-monitoring-role"

  assume_role_policy = jsonencode({{
    Version = "2012-10-17"
    Statement = [
      {{
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {{
          Service = "monitoring.rds.amazonaws.com"
        }}
      }}
    ]
  }})
}}

resource "aws_iam_role_policy_attachment" "{db_name}_monitoring_policy" {{
  role       = aws_iam_role.{db_name}_monitoring_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole"
}}

# Variable for database password
variable "{db_name}_password" {{
  description = "Password for the RDS instance"
  type        = string
  sensitive   = true
}}

# Outputs
output "{db_name}_endpoint" {{
  description = "RDS instance endpoint"
  value       = aws_db_instance.{db_name}.endpoint
  sensitive   = true
}}

output "{db_name}_port" {{
  description = "RDS instance port"
  value       = aws_db_instance.{db_name}.port
}}
        '''.strip()
        
        return template
    
    def generate_complete_template(self, resources: List[InfrastructureResource]) -> str:
        """Generate complete Terraform template for all resources"""
        template_parts = []
        
        # Determine primary provider
        providers = set(resource.provider for resource in resources)
        if len(providers) > 1:
            logger.warning("Multiple providers detected, template may need manual adjustment")
        
        primary_provider = list(providers)[0] if providers else CloudProvider.AWS
        
        # Add provider configuration
        provider_config = self.providers.get(primary_provider, {})
        if provider_config:
            template_parts.append(provider_config['provider_block'])
            template_parts.append("")
            
            # Add variables
            variables = provider_config.get('variables', {})
            for var_name, var_config in variables.items():
                template_parts.append(f'variable "{var_name}" {{')
                for key, value in var_config.items():
                    if isinstance(value, str):
                        template_parts.append(f'  {key} = "{value}"')
                    else:
                        template_parts.append(f'  {key} = {json.dumps(value)}')
                template_parts.append("}")
                template_parts.append("")
        
        # Generate resource templates
        for resource in resources:
            if resource.type == InfrastructureType.KUBERNETES_CLUSTER:
                template_parts.append(self.generate_kubernetes_cluster(resource))
            elif resource.type == InfrastructureType.DATABASE:
                template_parts.append(self.generate_database(resource))
            # Add more resource types as needed
            
            template_parts.append("")
        
        return "\n".join(template_parts)

# ==========================================
# ANSIBLE PLAYBOOK GENERATOR
# ==========================================

class AnsiblePlaybookGenerator:
    """Generates Ansible playbooks for configuration management"""
    
    def __init__(self):
        self.playbooks: Dict[str, Dict] = {}
    
    def generate_kubernetes_setup_playbook(self, cluster_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate Ansible playbook for Kubernetes setup"""
        playbook = {
            'name': 'Setup Kubernetes cluster and applications',
            'hosts': 'localhost',
            'connection': 'local',
            'gather_facts': False,
            'vars': {
                'cluster_name': cluster_config.get('cluster_name', 'poce-cluster'),
                'namespace': cluster_config.get('namespace', 'poce-system'),
                'app_version': cluster_config.get('app_version', '4.0.0')
            },
            'tasks': [
                {
                    'name': 'Create namespace',
                    'kubernetes.core.k8s': {
                        'name': '{{ namespace }}',
                        'api_version': 'v1',
                        'kind': 'Namespace',
                        'state': 'present'
                    }
                },
                {
                    'name': 'Apply ConfigMap',
                    'kubernetes.core.k8s': {
                        'state': 'present',
                        'definition': {
                            'apiVersion': 'v1',
                            'kind': 'ConfigMap',
                            'metadata': {
                                'name': 'poce-config',
                                'namespace': '{{ namespace }}'
                            },
                            'data': {
                                'config.yaml': '{{ lookup("file", "config/poce_config.yaml") }}'
                            }
                        }
                    }
                },
                {
                    'name': 'Apply Secrets',
                    'kubernetes.core.k8s': {
                        'state': 'present',
                        'definition': {
                            'apiVersion': 'v1',
                            'kind': 'Secret',
                            'metadata': {
                                'name': 'poce-secrets',
                                'namespace': '{{ namespace }}'
                            },
                            'type': 'Opaque',
                            'data': {
                                'github-token': '{{ github_token | b64encode }}',
                                'database-password': '{{ database_password | b64encode }}'
                            }
                        }
                    },
                    'vars': {
                        'github_token': '{{ vault_github_token }}',
                        'database_password': '{{ vault_database_password }}'
                    }
                },
                {
                    'name': 'Deploy application',
                    'kubernetes.core.k8s': {
                        'state': 'present',
                        'src': 'manifests/deployment.yaml'
                    }
                },
                {
                    'name': 'Deploy service',
                    'kubernetes.core.k8s': {
                        'state': 'present',
                        'src': 'manifests/service.yaml'
                    }
                },
                {
                    'name': 'Deploy ingress',
                    'kubernetes.core.k8s': {
                        'state': 'present',
                        'src': 'manifests/ingress.yaml'
                    }
                },
                {
                    'name': 'Wait for deployment to be ready',
                    'kubernetes.core.k8s_info': {
                        'api_version': 'apps/v1',
                        'kind': 'Deployment',
                        'name': 'poce-app',
                        'namespace': '{{ namespace }}',
                        'wait': True,
                        'wait_condition': {
                            'type': 'Progressing',
                            'status': 'True',
                            'reason': 'NewReplicaSetAvailable'
                        },
                        'wait_timeout': 600
                    }
                }
            ]
        }
        
        return playbook
    
    def generate_monitoring_setup_playbook(self) -> Dict[str, Any]:
        """Generate playbook for monitoring setup"""
        playbook = {
            'name': 'Setup monitoring stack',
            'hosts': 'localhost',
            'connection': 'local',
            'gather_facts': False,
            'vars': {
                'monitoring_namespace': 'monitoring',
                'prometheus_version': '2.45.0',
                'grafana_version': '10.0.0'
            },
            'tasks': [
                {
                    'name': 'Create monitoring namespace',
                    'kubernetes.core.k8s': {
                        'name': '{{ monitoring_namespace }}',
                        'api_version': 'v1',
                        'kind': 'Namespace',
                        'state': 'present'
                    }
                },
                {
                    'name': 'Add Prometheus Helm repository',
                    'kubernetes.core.helm_repository': {
                        'name': 'prometheus-community',
                        'repo_url': 'https://prometheus-community.github.io/helm-charts'
                    }
                },
                {
                    'name': 'Install Prometheus',
                    'kubernetes.core.helm': {
                        'name': 'prometheus',
                        'chart_ref': 'prometheus-community/kube-prometheus-stack',
                        'release_namespace': '{{ monitoring_namespace }}',
                        'create_namespace': True,
                        'values': {
                            'prometheus': {
                                'prometheusSpec': {
                                    'retention': '30d',
                                    'storageSpec': {
                                        'volumeClaimTemplate': {
                                            'spec': {
                                                'storageClassName': 'standard',
                                                'accessModes': ['ReadWriteOnce'],
                                                'resources': {
                                                    'requests': {
                                                        'storage': '50Gi'
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            'grafana': {
                                'enabled': True,
                                'adminPassword': '{{ vault_grafana_password }}',
                                'persistence': {
                                    'enabled': True,
                                    'size': '10Gi'
                                }
                            }
                        }
                    }
                },
                {
                    'name': 'Create Grafana dashboards ConfigMap',
                    'kubernetes.core.k8s': {
                        'state': 'present',
                        'definition': {
                            'apiVersion': 'v1',
                            'kind': 'ConfigMap',
                            'metadata': {
                                'name': 'poce-dashboards',
                                'namespace': '{{ monitoring_namespace }}',
                                'labels': {
                                    'grafana_dashboard': '1'
                                }
                            },
                            'data': {
                                'poce-dashboard.json': '{{ lookup("file", "monitoring/grafana-dashboard.json") }}'
                            }
                        }
                    }
                }
            ]
        }
        
        return playbook
    
    def generate_security_hardening_playbook(self) -> Dict[str, Any]:
        """Generate security hardening playbook"""
        playbook = {
            'name': 'Security hardening for Kubernetes cluster',
            'hosts': 'localhost',
            'connection': 'local',
            'gather_facts': False,
            'tasks': [
                {
                    'name': 'Apply Network Policies',
                    'kubernetes.core.k8s': {
                        'state': 'present',
                        'definition': {
                            'apiVersion': 'networking.k8s.io/v1',
                            'kind': 'NetworkPolicy',
                            'metadata': {
                                'name': 'default-deny-all',
                                'namespace': 'poce-system'
                            },
                            'spec': {
                                'podSelector': {},
                                'policyTypes': ['Ingress', 'Egress']
                            }
                        }
                    }
                },
                {
                    'name': 'Apply Pod Security Policy',
                    'kubernetes.core.k8s': {
                        'state': 'present',
                        'definition': {
                            'apiVersion': 'policy/v1beta1',
                            'kind': 'PodSecurityPolicy',
                            'metadata': {
                                'name': 'poce-restricted-psp'
                            },
                            'spec': {
                                'privileged': False,
                                'allowPrivilegeEscalation': False,
                                'requiredDropCapabilities': ['ALL'],
                                'volumes': [
                                    'configMap', 'emptyDir', 'projected',
                                    'secret', 'downwardAPI', 'persistentVolumeClaim'
                                ],
                                'runAsUser': {
                                    'rule': 'MustRunAsNonRoot'
                                },
                                'seLinux': {
                                    'rule': 'RunAsAny'
                                },
                                'fsGroup': {
                                    'rule': 'RunAsAny'
                                }
                            }
                        }
                    }
                },
                {
                    'name': 'Install Falco for runtime security',
                    'kubernetes.core.helm_repository': {
                        'name': 'falcosecurity',
                        'repo_url': 'https://falcosecurity.github.io/charts'
                    }
                },
                {
                    'name': 'Deploy Falco',
                    'kubernetes.core.helm': {
                        'name': 'falco',
                        'chart_ref': 'falcosecurity/falco',
                        'release_namespace': 'falco-system',
                        'create_namespace': True,
                        'values': {
                            'falco': {
                                'grpc': {
                                    'enabled': True
                                },
                                'grpcOutput': {
                                    'enabled': True
                                }
                            }
                        }
                    }
                }
            ]
        }
        
        return playbook

# ==========================================
# DEPLOYMENT ORCHESTRATOR
# ==========================================

class DeploymentOrchestrator:
    """Orchestrates infrastructure deployment using multiple tools"""
    
    def __init__(self):
        self.terraform_generator = TerraformTemplateGenerator()
        self.ansible_generator = AnsiblePlaybookGenerator()
        self.deployment_history: List[DeploymentPlan] = []
    
    def create_deployment_plan(self, resources: List[InfrastructureResource]) -> DeploymentPlan:
        """Create a deployment plan for infrastructure resources"""
        plan_id = hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8]
        
        # Determine execution order based on dependencies
        execution_order = self._calculate_execution_order(resources)
        
        # Estimate costs and time (simplified)
        estimated_cost = self._estimate_cost(resources)
        deployment_time = self._estimate_deployment_time(resources)
        
        plan = DeploymentPlan(
            plan_id=plan_id,
            resources=resources,
            execution_order=execution_order,
            estimated_cost=estimated_cost,
            deployment_time_estimate=deployment_time
        )
        
        self.deployment_history.append(plan)
        return plan
    
    def _calculate_execution_order(self, resources: List[InfrastructureResource]) -> List[str]:
        """Calculate optimal execution order based on dependencies"""
        # Simple topological sort for dependency resolution
        order = []
        remaining = {r.name: r for r in resources}
        
        while remaining:
            # Find resources with no unresolved dependencies
            ready = []
            for name, resource in remaining.items():
                if not any(dep in remaining for dep in resource.dependencies):
                    ready.append(name)
            
            if not ready:
                # Circular dependency or missing dependency
                logger.warning("Circular dependency detected, using arbitrary order")
                ready = [list(remaining.keys())[0]]
            
            # Add ready resources to order and remove from remaining
            for name in ready:
                order.append(name)
                del remaining[name]
        
        return order
    
    def _estimate_cost(self, resources: List[InfrastructureResource]) -> float:
        """Estimate infrastructure costs (simplified)"""
        cost_estimates = {
            InfrastructureType.KUBERNETES_CLUSTER: 150.0,  # Monthly
            InfrastructureType.DATABASE: 50.0,
            InfrastructureType.LOAD_BALANCER: 25.0,
            InfrastructureType.STORAGE: 10.0,
            InfrastructureType.MONITORING: 30.0
        }
        
        total_cost = 0.0
        for resource in resources:
            base_cost = cost_estimates.get(resource.type, 20.0)
            
            # Adjust for environment
            if resource.environment == DeploymentEnvironment.PRODUCTION:
                base_cost *= 2.0
            elif resource.environment == DeploymentEnvironment.STAGING:
                base_cost *= 1.5
            
            total_cost += base_cost
        
        return total_cost
    
    def _estimate_deployment_time(self, resources: List[InfrastructureResource]) -> int:
        """Estimate deployment time in minutes"""
        time_estimates = {
            InfrastructureType.KUBERNETES_CLUSTER: 20,
            InfrastructureType.DATABASE: 10,
            InfrastructureType.LOAD_BALANCER: 5,
            InfrastructureType.STORAGE: 3,
            InfrastructureType.MONITORING: 15
        }
        
        total_time = 0
        for resource in resources:
            base_time = time_estimates.get(resource.type, 5)
            total_time += base_time
        
        # Add buffer for dependencies and coordination
        return int(total_time * 1.3)
    
    def deploy_infrastructure(self, plan: DeploymentPlan, output_dir: Path) -> Dict[str, Any]:
        """Deploy infrastructure according to plan"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting deployment of plan {plan.plan_id}")
        
        # Generate Terraform templates
        terraform_content = self.terraform_generator.generate_complete_template(plan.resources)
        terraform_file = output_dir / "main.tf"
        with open(terraform_file, 'w') as f:
            f.write(terraform_content)
        
        # Generate Ansible playbooks
        ansible_dir = output_dir / "ansible"
        ansible_dir.mkdir(exist_ok=True)
        
        # Kubernetes setup playbook
        k8s_config = self._extract_kubernetes_config(plan.resources)
        if k8s_config:
            k8s_playbook = self.ansible_generator.generate_kubernetes_setup_playbook(k8s_config)
            with open(ansible_dir / "k8s-setup.yml", 'w') as f:
                yaml.dump([k8s_playbook], f, default_flow_style=False)
        
        # Monitoring setup playbook
        monitoring_playbook = self.ansible_generator.generate_monitoring_setup_playbook()
        with open(ansible_dir / "monitoring-setup.yml", 'w') as f:
            yaml.dump([monitoring_playbook], f, default_flow_style=False)
        
        # Security hardening playbook
        security_playbook = self.ansible_generator.generate_security_hardening_playbook()
        with open(ansible_dir / "security-hardening.yml", 'w') as f:
            yaml.dump([security_playbook], f, default_flow_style=False)
        
        # Generate deployment scripts
        self._generate_deployment_scripts(plan, output_dir)
        
        # Execute deployment (if terraform/ansible are available)
        deployment_result = self._execute_deployment(output_dir)
        
        return {
            'plan_id': plan.plan_id,
            'status': 'completed' if deployment_result else 'generated',
            'terraform_file': str(terraform_file),
            'ansible_dir': str(ansible_dir),
            'estimated_cost': plan.estimated_cost,
            'estimated_time': plan.deployment_time_estimate,
            'execution_result': deployment_result
        }
    
    def _extract_kubernetes_config(self, resources: List[InfrastructureResource]) -> Optional[Dict[str, Any]]:
        """Extract Kubernetes configuration from resources"""
        for resource in resources:
            if resource.type == InfrastructureType.KUBERNETES_CLUSTER:
                return {
                    'cluster_name': resource.name,
                    'namespace': resource.configuration.get('namespace', 'poce-system'),
                    'app_version': '4.0.0'
                }
        return None
    
    def _generate_deployment_scripts(self, plan: DeploymentPlan, output_dir: Path):
        """Generate deployment scripts"""
        # Deploy script
        deploy_script = f'''#!/bin/bash
set -e

echo "Starting deployment of plan {plan.plan_id}"
echo "Estimated time: {plan.deployment_time_estimate} minutes"
echo "Estimated cost: ${plan.estimated_cost:.2f}/month"

# Initialize Terraform
echo "Initializing Terraform..."
terraform init

# Plan Terraform deployment
echo "Planning Terraform deployment..."
terraform plan -out=tfplan

# Apply Terraform (uncomment to execute)
# echo "Applying Terraform..."
# terraform apply tfplan

# Run Ansible playbooks (uncomment to execute)
# echo "Running Ansible playbooks..."
# ansible-playbook ansible/k8s-setup.yml
# ansible-playbook ansible/monitoring-setup.yml
# ansible-playbook ansible/security-hardening.yml

echo "Deployment preparation completed!"
echo "Review the generated files and uncomment the execution commands to deploy."
        '''
        
        deploy_script_file = output_dir / "deploy.sh"
        with open(deploy_script_file, 'w') as f:
            f.write(deploy_script)
        deploy_script_file.chmod(0o755)
        
        # Destroy script
        destroy_script = '''#!/bin/bash
set -e

echo "Destroying infrastructure..."
echo "WARNING: This will destroy all infrastructure managed by Terraform!"
read -p "Are you sure? (yes/no): " -r
if [[ $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    terraform destroy -auto-approve
    echo "Infrastructure destroyed."
else
    echo "Operation cancelled."
fi
        '''
        
        destroy_script_file = output_dir / "destroy.sh"
        with open(destroy_script_file, 'w') as f:
            f.write(destroy_script)
        destroy_script_file.chmod(0o755)
    
    def _execute_deployment(self, output_dir: Path) -> Optional[Dict[str, Any]]:
        """Execute deployment (if tools are available)"""
        try:
            # Check if terraform is available
            terraform_check = subprocess.run(['terraform', '--version'], 
                                           capture_output=True, text=True)
            if terraform_check.returncode != 0:
                logger.info("Terraform not available, skipping execution")
                return None
            
            # Initialize terraform
            init_result = subprocess.run(['terraform', 'init'], 
                                       cwd=output_dir, capture_output=True, text=True)
            
            if init_result.returncode == 0:
                # Plan terraform
                plan_result = subprocess.run(['terraform', 'plan', '-out=tfplan'], 
                                           cwd=output_dir, capture_output=True, text=True)
                
                return {
                    'terraform_init': init_result.returncode == 0,
                    'terraform_plan': plan_result.returncode == 0,
                    'plan_output': plan_result.stdout if plan_result.returncode == 0 else plan_result.stderr
                }
            
        except FileNotFoundError:
            logger.info("Terraform not found, deployment files generated only")
        except Exception as e:
            logger.error(f"Deployment execution failed: {e}")
        
        return None

# ==========================================
# INFRASTRUCTURE MANAGER
# ==========================================

class InfrastructureManager:
    """Main infrastructure management system"""
    
    def __init__(self):
        self.orchestrator = DeploymentOrchestrator()
        self.resources: List[InfrastructureResource] = []
        self.current_plan: Optional[DeploymentPlan] = None
    
    def add_kubernetes_cluster(self, name: str, provider: CloudProvider, 
                             environment: DeploymentEnvironment,
                             **config) -> InfrastructureResource:
        """Add Kubernetes cluster resource"""
        resource = InfrastructureResource(
            name=name,
            type=InfrastructureType.KUBERNETES_CLUSTER,
            provider=provider,
            configuration=config,
            environment=environment,
            tags={'component': 'kubernetes', 'managed_by': 'poce'}
        )
        
        self.resources.append(resource)
        return resource
    
    def add_database(self, name: str, provider: CloudProvider, 
                    environment: DeploymentEnvironment,
                    **config) -> InfrastructureResource:
        """Add database resource"""
        resource = InfrastructureResource(
            name=name,
            type=InfrastructureType.DATABASE,
            provider=provider,
            configuration=config,
            environment=environment,
            tags={'component': 'database', 'managed_by': 'poce'}
        )
        
        self.resources.append(resource)
        return resource
    
    def create_deployment_plan(self) -> DeploymentPlan:
        """Create deployment plan for all resources"""
        self.current_plan = self.orchestrator.create_deployment_plan(self.resources)
        return self.current_plan
    
    def deploy(self, output_dir: Path) -> Dict[str, Any]:
        """Deploy infrastructure"""
        if not self.current_plan:
            self.create_deployment_plan()
        
        return self.orchestrator.deploy_infrastructure(self.current_plan, output_dir)
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        return {
            'total_resources': len(self.resources),
            'resource_types': list(set(r.type.value for r in self.resources)),
            'providers': list(set(r.provider.value for r in self.resources)),
            'environments': list(set(r.environment.value for r in self.resources)),
            'current_plan': self.current_plan.plan_id if self.current_plan else None,
            'deployment_history': len(self.orchestrator.deployment_history)
        }

# ==========================================
# EXAMPLE USAGE
# ==========================================

def example_infrastructure_deployment():
    """Example of using the infrastructure management system"""
    
    # Create infrastructure manager
    infra_manager = InfrastructureManager()
    
    # Add Kubernetes cluster
    k8s_cluster = infra_manager.add_kubernetes_cluster(
        name="poce-production-cluster",
        provider=CloudProvider.AWS,
        environment=DeploymentEnvironment.PRODUCTION,
        kubernetes_version="1.27",
        node_instance_types=["t3.medium"],
        desired_nodes=3,
        max_nodes=6,
        min_nodes=2
    )
    
    # Add database
    database = infra_manager.add_database(
        name="poce-production-db",
        provider=CloudProvider.AWS,
        environment=DeploymentEnvironment.PRODUCTION,
        engine="postgres",
        engine_version="14.9",
        instance_class="db.t3.small",
        allocated_storage=100,
        storage_encrypted=True
    )
    
    # Add database dependency to cluster
    k8s_cluster.dependencies.append(database.name)
    
    # Create deployment plan
    plan = infra_manager.create_deployment_plan()
    
    print(f"Deployment Plan: {plan.plan_id}")
    print(f"Resources: {len(plan.resources)}")
    print(f"Execution Order: {plan.execution_order}")
    print(f"Estimated Cost: ${plan.estimated_cost:.2f}/month")
    print(f"Estimated Time: {plan.deployment_time_estimate} minutes")
    
    # Deploy infrastructure
    output_dir = Path("infrastructure_deployment")
    result = infra_manager.deploy(output_dir)
    
    print(f"Deployment Status: {result['status']}")
    print(f"Generated files in: {output_dir}")

if __name__ == "__main__":
    example_infrastructure_deployment()