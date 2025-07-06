#!/usr/bin/env python3
"""
P.O.C.E. Project Creator - Setup Script
Prompt Orchestrator + Context Engineering
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="poce-project-creator",
    version="4.0.0",
    author="brian95240",
    author_email="brian95240@users.noreply.github.com",
    description="Advanced DevOps automation tool with MCP integration for professional GitHub repository creation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brian95240/P.O.C.E.-Project-Creator",
    project_urls={
        "Bug Tracker": "https://github.com/brian95240/P.O.C.E.-Project-Creator/issues",
        "Documentation": "https://github.com/brian95240/P.O.C.E.-Project-Creator/tree/main/docs",
        "Source Code": "https://github.com/brian95240/P.O.C.E.-Project-Creator",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Build Tools",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "poce=poce_cli:main",
            "poce-creator=poce_project_creator_v4:main",
            "poce-orchestrator=poce_master_orchestrator:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.txt", "*.md"],
    },
    keywords=[
        "devops",
        "automation",
        "github",
        "ci-cd",
        "mcp",
        "orchestration",
        "project-creator",
        "infrastructure",
        "security",
        "monitoring",
    ],
    license="GPL-3.0-or-later",
    zip_safe=False,
)

