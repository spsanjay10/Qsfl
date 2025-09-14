#!/usr/bin/env python3
"""
QSFL-CAAD Project Status Dashboard
Real-time project health and workflow monitoring
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.tree import Tree
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.live import Live
from rich.layout import Layout
from rich.align import Align

console = Console()

class ProjectStatusDashboard:
    """Comprehensive project status monitoring."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.status_cache = {}
        self.last_update = None
    
    def get_git_status(self) -> Dict[str, any]:
        """Get Git repository status."""
        try:
            # Current branch
            branch_result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True, text=True, cwd=self.project_root
            )
            current_branch = branch_result.stdout.strip()
            
            # Commit count
            commit_result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                capture_output=True, text=True, cwd=self.project_root
            )
            commit_count = int(commit_result.stdout.strip()) if commit_result.stdout.strip() else 0
            
            # Last commit
            last_commit_result = subprocess.run(
                ["git", "log", "-1", "--format=%h %s (%cr)"],
                capture_output=True, text=True, cwd=self.project_root
            )
            last_commit = last_commit_result.stdout.strip()
            
            # Status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=self.project_root
            )
            modified_files = len(status_result.stdout.strip().split('\n')) if status_result.stdout.strip() else 0
            
            # Remote status
            try:
                subprocess.run(["git", "fetch"], capture_output=True, cwd=self.project_root)
                ahead_result = subprocess.run(
                    ["git", "rev-list", "--count", f"origin/{current_branch}..HEAD"],
                    capture_output=True, text=True, cwd=self.project_root
                )
                behind_result = subprocess.run(
                    ["git", "rev-list", "--count", f"HEAD..origin/{current_branch}"],
                    capture_output=True, text=True, cwd=self.project_root
                )
                commits_ahead = int(ahead_result.stdout.strip()) if ahead_result.stdout.strip() else 0
                commits_behind = int(behind_result.stdout.strip()) if behind_result.stdout.strip() else 0
            except:
                commits_ahead = commits_behind = 0
            
            return {
                "branch": current_branch,
                "commits": commit_count,
                "last_commit": last_commit,
                "modified_files": modified_files,
                "commits_ahead": commits_ahead,
                "commits_behind": commits_behind,
                "status": "clean" if modified_files == 0 else "modified"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_code_metrics(self) -> Dict[str, any]:
        """Get code quality metrics."""
        metrics = {}
        
        # Line count
        try:
            result = subprocess.run(
                ["find", ".", "-name", "*.py", "-not", "-path", "./venv/*", "-not", "-path", "./.venv/*", 
                 "-not", "-path", "./build/*", "-not", "-path", "./dist/*", "-exec", "wc", "-l", "{}", "+"],
                capture_output=True, text=True, cwd=self.project_root
            )
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                total_lines = int(lines[-1].split()[0]) if lines else 0
                metrics["lines_of_code"] = total_lines
        except:
            metrics["lines_of_code"] = 0
        
        # File count
        try:
            py_files = list(self.project_root.rglob("*.py"))
            metrics["python_files"] = len([f for f in py_files if "venv" not in str(f) and ".venv" not in str(f)])
        except:
            metrics["python_files"] = 0
        
        # Test coverage (if available)
        coverage_file = self.project_root / "coverage.xml"
        if coverage_file.exists():
            try:
                with open(coverage_file) as f:
                    content = f.read()
                    import re
                    match = re.search(r'line-rate="([0-9.]+)"', content)
                    if match:
                        metrics["test_coverage"] = float(match.group(1)) * 100
            except:
                pass
        
        return metrics
    
    def get_dependency_status(self) -> Dict[str, any]:
        """Get dependency information."""
        deps = {}
        
        # Count requirements
        req_files = ["requirements.txt", "requirements-dev.txt", "requirements-test.txt", "requirements-ui.txt"]
        total_deps = 0
        
        for req_file in req_files:
            req_path = self.project_root / req_file
            if req_path.exists():
                try:
                    with open(req_path) as f:
                        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                        deps[req_file] = len(lines)
                        total_deps += len(lines)
                except:
                    deps[req_file] = 0
        
        deps["total"] = total_deps
        
        # Check for outdated packages
        try:
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True, text=True
            )
            if result.stdout:
                outdated = json.loads(result.stdout)
                deps["outdated"] = len(outdated)
            else:
                deps["outdated"] = 0
        except:
            deps["outdated"] = 0
        
        return deps
    
    def get_test_status(self) -> Dict[str, any]:
        """Get test suite status."""
        test_dir = self.project_root / "tests"
        if not test_dir.exists():
            return {"error": "No tests directory found"}
        
        # Count test files
        test_files = list(test_dir.rglob("test_*.py"))
        
        # Get last test run results (if available)
        pytest_cache = self.project_root / ".pytest_cache"
        last_run = None
        if pytest_cache.exists():
            try:
                # Look for recent test results
                cache_files = list(pytest_cache.rglob("*"))
                if cache_files:
                    latest_file = max(cache_files, key=lambda f: f.stat().st_mtime)
                    last_run = datetime.fromtimestamp(latest_file.stat().st_mtime)
            except:
                pass
        
        return {
            "test_files": len(test_files),
            "last_run": last_run.strftime("%Y-%m-%d %H:%M:%S") if last_run else "Never",
            "test_types": {
                "unit": len(list((test_dir / "unit").rglob("test_*.py"))) if (test_dir / "unit").exists() else 0,
                "integration": len(list((test_dir / "integration").rglob("test_*.py"))) if (test_dir / "integration").exists() else 0,
                "performance": len(list((test_dir / "performance").rglob("test_*.py"))) if (test_dir / "performance").exists() else 0,
            }
        }
    
    def get_docker_status(self) -> Dict[str, any]:
        """Get Docker and containerization status."""
        status = {}
        
        # Check if Docker files exist
        dockerfile = self.project_root / "Dockerfile"
        docker_compose = self.project_root / "docker-compose.yml"
        
        status["dockerfile_exists"] = dockerfile.exists()
        status["docker_compose_exists"] = docker_compose.exists()
        
        # Check running containers
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "json"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                containers = []
                for line in result.stdout.strip().split('\n'):
                    if line:
                        try:
                            container = json.loads(line)
                            if "qsfl" in container.get("Names", "").lower():
                                containers.append(container)
                        except:
                            pass
                status["running_containers"] = len(containers)
                status["docker_available"] = True
            else:
                status["docker_available"] = False
                status["running_containers"] = 0
        except:
            status["docker_available"] = False
            status["running_containers"] = 0
        
        return status
    
    def get_ci_status(self) -> Dict[str, any]:
        """Get CI/CD pipeline status."""
        ci_files = {
            "github_actions": self.project_root / ".github" / "workflows",
            "pre_commit": self.project_root / ".pre-commit-config.yaml",
            "makefile": self.project_root / "Makefile"
        }
        
        status = {}
        for name, path in ci_files.items():
            status[name] = path.exists()
        
        # Count workflow files
        if ci_files["github_actions"].exists():
            workflow_files = list(ci_files["github_actions"].glob("*.yml"))
            status["workflow_files"] = len(workflow_files)
        else:
            status["workflow_files"] = 0
        
        return status
    
    def create_git_panel(self, git_status: Dict) -> Panel:
        """Create Git status panel."""
        if "error" in git_status:
            content = f"[red]Error: {git_status['error']}[/red]"
        else:
            status_color = "green" if git_status["status"] == "clean" else "yellow"
            sync_status = ""
            if git_status["commits_ahead"] > 0:
                sync_status += f" [yellow]↑{git_status['commits_ahead']}[/yellow]"
            if git_status["commits_behind"] > 0:
                sync_status += f" [red]↓{git_status['commits_behind']}[/red]"
            
            content = f"""[bold]Branch:[/bold] {git_status['branch']}{sync_status}
[bold]Commits:[/bold] {git_status['commits']}
[bold]Status:[/bold] [{status_color}]{git_status['status']}[/{status_color}]
[bold]Modified:[/bold] {git_status['modified_files']} files
[bold]Last:[/bold] {git_status['last_commit']}"""
        
        return Panel(content, title="🔄 Git Status", border_style="blue")
    
    def create_code_panel(self, metrics: Dict) -> Panel:
        """Create code metrics panel."""
        content = f"""[bold]Python Files:[/bold] {metrics.get('python_files', 0)}
[bold]Lines of Code:[/bold] {metrics.get('lines_of_code', 0):,}
[bold]Test Coverage:[/bold] {metrics.get('test_coverage', 'N/A')}{'%' if 'test_coverage' in metrics else ''}"""
        
        return Panel(content, title="📊 Code Metrics", border_style="green")
    
    def create_deps_panel(self, deps: Dict) -> Panel:
        """Create dependencies panel."""
        content = f"""[bold]Total Dependencies:[/bold] {deps.get('total', 0)}
[bold]Production:[/bold] {deps.get('requirements.txt', 0)}
[bold]Development:[/bold] {deps.get('requirements-dev.txt', 0)}
[bold]Testing:[/bold] {deps.get('requirements-test.txt', 0)}
[bold]UI:[/bold] {deps.get('requirements-ui.txt', 0)}
[bold]Outdated:[/bold] [red]{deps.get('outdated', 0)}[/red]"""
        
        return Panel(content, title="📦 Dependencies", border_style="yellow")
    
    def create_test_panel(self, test_status: Dict) -> Panel:
        """Create test status panel."""
        if "error" in test_status:
            content = f"[red]{test_status['error']}[/red]"
        else:
            content = f"""[bold]Test Files:[/bold] {test_status['test_files']}
[bold]Unit Tests:[/bold] {test_status['test_types']['unit']}
[bold]Integration:[/bold] {test_status['test_types']['integration']}
[bold]Performance:[/bold] {test_status['test_types']['performance']}
[bold]Last Run:[/bold] {test_status['last_run']}"""
        
        return Panel(content, title="🧪 Tests", border_style="purple")
    
    def create_docker_panel(self, docker_status: Dict) -> Panel:
        """Create Docker status panel."""
        docker_color = "green" if docker_status["docker_available"] else "red"
        docker_text = "Available" if docker_status["docker_available"] else "Not Available"
        
        content = f"""[bold]Docker:[/bold] [{docker_color}]{docker_text}[/{docker_color}]
[bold]Dockerfile:[/bold] {'✅' if docker_status['dockerfile_exists'] else '❌'}
[bold]Compose:[/bold] {'✅' if docker_status['docker_compose_exists'] else '❌'}
[bold]Running:[/bold] {docker_status['running_containers']} containers"""
        
        return Panel(content, title="🐳 Docker", border_style="cyan")
    
    def create_ci_panel(self, ci_status: Dict) -> Panel:
        """Create CI/CD status panel."""
        content = f"""[bold]GitHub Actions:[/bold] {'✅' if ci_status['github_actions'] else '❌'}
[bold]Workflows:[/bold] {ci_status['workflow_files']}
[bold]Pre-commit:[/bold] {'✅' if ci_status['pre_commit'] else '❌'}
[bold]Makefile:[/bold] {'✅' if ci_status['makefile'] else '❌'}"""
        
        return Panel(content, title="🚀 CI/CD", border_style="magenta")
    
    def create_project_tree(self) -> Tree:
        """Create project structure tree."""
        tree = Tree("📁 QSFL-CAAD Project")
        
        # Core modules
        core = tree.add("🔧 Core Modules")
        core.add("qsfl_caad/ - Main system")
        core.add("anomaly_detection/ - Anomaly detection")
        core.add("auth/ - Authentication")
        core.add("federated_learning/ - FL algorithms")
        core.add("monitoring/ - System monitoring")
        core.add("pq_security/ - Post-quantum crypto")
        
        # UI and interfaces
        ui = tree.add("🎨 User Interfaces")
        ui.add("ui/ - Web dashboard")
        ui.add("scripts/ - CLI tools")
        ui.add("working_dashboard.py - Main dashboard")
        
        # Development
        dev = tree.add("🛠️ Development")
        dev.add("tests/ - Test suite")
        dev.add("docs/ - Documentation")
        dev.add("requirements*.txt - Dependencies")
        dev.add(".github/ - CI/CD workflows")
        
        return tree
    
    def display_dashboard(self):
        """Display the complete project status dashboard."""
        console.clear()
        
        # Header
        header = Panel(
            Align.center(
                f"[bold blue]QSFL-CAAD Project Status Dashboard[/bold blue]\n"
                f"[dim]Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]"
            ),
            style="bold blue"
        )
        console.print(header)
        console.print()
        
        # Collect all status information
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Collecting project status...", total=None)
            
            git_status = self.get_git_status()
            progress.update(task, description="Getting code metrics...")
            code_metrics = self.get_code_metrics()
            progress.update(task, description="Checking dependencies...")
            deps_status = self.get_dependency_status()
            progress.update(task, description="Analyzing tests...")
            test_status = self.get_test_status()
            progress.update(task, description="Checking Docker...")
            docker_status = self.get_docker_status()
            progress.update(task, description="Checking CI/CD...")
            ci_status = self.get_ci_status()
        
        console.clear()
        console.print(header)
        console.print()
        
        # Create panels
        panels = [
            self.create_git_panel(git_status),
            self.create_code_panel(code_metrics),
            self.create_deps_panel(deps_status),
            self.create_test_panel(test_status),
            self.create_docker_panel(docker_status),
            self.create_ci_panel(ci_status)
        ]
        
        # Display panels in columns
        console.print(Columns(panels[:3]))
        console.print()
        console.print(Columns(panels[3:]))
        console.print()
        
        # Project structure
        console.print(self.create_project_tree())
        console.print()
        
        # Quick actions
        actions_panel = Panel(
            "[bold]Quick Actions:[/bold]\n"
            "• [cyan]make install-dev[/cyan] - Setup development environment\n"
            "• [cyan]make test[/cyan] - Run all tests\n"
            "• [cyan]make quality-check[/cyan] - Run quality checks\n"
            "• [cyan]make run-dashboard[/cyan] - Start web dashboard\n"
            "• [cyan]make ci-local[/cyan] - Run full CI pipeline locally",
            title="⚡ Quick Actions",
            border_style="green"
        )
        console.print(actions_panel)
    
    def monitor_live(self, interval: int = 30):
        """Live monitoring with auto-refresh."""
        def generate_display():
            self.display_dashboard()
            return ""
        
        with Live(generate_display(), refresh_per_second=1/interval, console=console) as live:
            try:
                while True:
                    import time
                    time.sleep(interval)
                    live.update(generate_display())
            except KeyboardInterrupt:
                console.print("\n[yellow]Monitoring stopped.[/yellow]")


@click.command()
@click.option('--live', is_flag=True, help='Enable live monitoring')
@click.option('--interval', default=30, help='Refresh interval for live monitoring (seconds)')
def main(live, interval):
    """QSFL-CAAD Project Status Dashboard."""
    dashboard = ProjectStatusDashboard()
    
    if live:
        console.print("[blue]Starting live monitoring... Press Ctrl+C to stop[/blue]")
        dashboard.monitor_live(interval)
    else:
        dashboard.display_dashboard()

if __name__ == "__main__":
    main()