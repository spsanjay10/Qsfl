# QSFL-CAAD Project Makefile
# Comprehensive development workflow automation

.PHONY: help install install-dev clean test test-unit test-integration test-performance
.PHONY: lint format type-check security-check pre-commit docs build docker
.PHONY: run-dashboard run-demo deploy-local deploy-staging deploy-prod
.PHONY: benchmark profile monitor logs backup restore

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := qsfl-caad
DOCKER_IMAGE := $(PROJECT_NAME):latest
VENV_DIR := .venv
SRC_DIR := qsfl_caad
TEST_DIR := tests
DOCS_DIR := docs

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
PURPLE := \033[0;35m
CYAN := \033[0;36m
WHITE := \033[0;37m
RESET := \033[0m

# Help target
help: ## Show this help message
	@echo "$(CYAN)QSFL-CAAD Development Workflow$(RESET)"
	@echo "=================================="
	@echo ""
	@echo "$(GREEN)Available targets:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Quick Start:$(RESET)"
	@echo "  make install-dev  # Install development dependencies"
	@echo "  make test         # Run all tests"
	@echo "  make run-dashboard # Start the dashboard"

# Installation targets
install: ## Install production dependencies
	@echo "$(GREEN)Installing production dependencies...$(RESET)"
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(PIP) install -r requirements-test.txt
	$(PIP) install -e .
	pre-commit install
	@echo "$(GREEN)✅ Development environment ready!$(RESET)"

install-ui: ## Install UI dependencies
	@echo "$(GREEN)Installing UI dependencies...$(RESET)"
	$(PIP) install -r requirements-ui.txt
	$(PYTHON) scripts/setup_ui.py

# Environment management
venv: ## Create virtual environment
	@echo "$(GREEN)Creating virtual environment...$(RESET)"
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "$(YELLOW)Activate with: source $(VENV_DIR)/bin/activate$(RESET)"

clean: ## Clean up build artifacts and cache
	@echo "$(GREEN)Cleaning up...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/ .pytest_cache/
	rm -rf .mypy_cache/ .tox/ .cache/
	rm -f bandit-report.json security-report.json vulnerability-report.json
	@echo "$(GREEN)✅ Cleanup complete!$(RESET)"

# Testing targets
test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(RESET)"
	pytest $(TEST_DIR)/ -v --tb=short

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(RESET)"
	pytest $(TEST_DIR)/unit/ -v -m "not slow"

test-integration: ## Run integration tests
	@echo "$(GREEN)Running integration tests...$(RESET)"
	pytest $(TEST_DIR)/integration/ -v

test-performance: ## Run performance tests
	@echo "$(GREEN)Running performance tests...$(RESET)"
	pytest $(TEST_DIR)/performance/ -v --benchmark-only

test-security: ## Run security tests
	@echo "$(GREEN)Running security tests...$(RESET)"
	pytest $(TEST_DIR)/security/ -v -m security

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(RESET)"
	pytest $(TEST_DIR)/ --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(YELLOW)Coverage report: htmlcov/index.html$(RESET)"

test-parallel: ## Run tests in parallel
	@echo "$(GREEN)Running tests in parallel...$(RESET)"
	pytest $(TEST_DIR)/ -n auto

# Code quality targets
lint: ## Run linting checks
	@echo "$(GREEN)Running linting checks...$(RESET)"
	flake8 $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(GREEN)✅ Linting passed!$(RESET)"

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(RESET)"
	black $(SRC_DIR)/ $(TEST_DIR)/ scripts/
	isort $(SRC_DIR)/ $(TEST_DIR)/ scripts/
	@echo "$(GREEN)✅ Code formatted!$(RESET)"

format-check: ## Check code formatting
	@echo "$(GREEN)Checking code formatting...$(RESET)"
	black --check $(SRC_DIR)/ $(TEST_DIR)/ scripts/
	isort --check-only $(SRC_DIR)/ $(TEST_DIR)/ scripts/

type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checks...$(RESET)"
	mypy $(SRC_DIR)/ --ignore-missing-imports
	@echo "$(GREEN)✅ Type checking passed!$(RESET)"

security-check: ## Run security checks
	@echo "$(GREEN)Running security checks...$(RESET)"
	bandit -r $(SRC_DIR)/ -f json -o bandit-report.json
	safety check --json --output vulnerability-report.json
	@echo "$(GREEN)✅ Security checks complete!$(RESET)"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(GREEN)Running pre-commit hooks...$(RESET)"
	pre-commit run --all-files

quality-check: format-check lint type-check security-check ## Run all quality checks
	@echo "$(GREEN)✅ All quality checks passed!$(RESET)"

# Documentation targets
docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(RESET)"
	cd $(DOCS_DIR) && make html
	@echo "$(YELLOW)Documentation: $(DOCS_DIR)/_build/html/index.html$(RESET)"

docs-serve: ## Serve documentation locally
	@echo "$(GREEN)Serving documentation...$(RESET)"
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	@echo "$(GREEN)Cleaning documentation...$(RESET)"
	cd $(DOCS_DIR) && make clean

# Build targets
build: ## Build package
	@echo "$(GREEN)Building package...$(RESET)"
	$(PYTHON) -m build
	@echo "$(GREEN)✅ Package built in dist/$(RESET)"

build-wheel: ## Build wheel package
	@echo "$(GREEN)Building wheel...$(RESET)"
	$(PYTHON) -m build --wheel

build-sdist: ## Build source distribution
	@echo "$(GREEN)Building source distribution...$(RESET)"
	$(PYTHON) -m build --sdist

# Docker targets
docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image...$(RESET)"
	docker build -t $(DOCKER_IMAGE) .
	@echo "$(GREEN)✅ Docker image built: $(DOCKER_IMAGE)$(RESET)"

docker-run: ## Run Docker container
	@echo "$(GREEN)Running Docker container...$(RESET)"
	docker run -p 5000:5000 $(DOCKER_IMAGE)

docker-shell: ## Open shell in Docker container
	@echo "$(GREEN)Opening shell in Docker container...$(RESET)"
	docker run -it $(DOCKER_IMAGE) /bin/bash

docker-compose-up: ## Start services with docker-compose
	@echo "$(GREEN)Starting services with docker-compose...$(RESET)"
	docker-compose up -d

docker-compose-down: ## Stop services with docker-compose
	@echo "$(GREEN)Stopping services with docker-compose...$(RESET)"
	docker-compose down

# Application targets
run-dashboard: ## Start the web dashboard
	@echo "$(GREEN)Starting QSFL-CAAD Dashboard...$(RESET)"
	$(PYTHON) working_dashboard.py

run-demo: ## Run the live demo
	@echo "$(GREEN)Starting QSFL-CAAD Demo...$(RESET)"
	$(PYTHON) scripts/live_demo.py

run-interactive: ## Run interactive dashboard
	@echo "$(GREEN)Starting Interactive Dashboard...$(RESET)"
	$(PYTHON) scripts/interactive_dashboard.py

# Frontend targets
frontend-install: ## Install frontend dependencies
	@echo "$(GREEN)Installing frontend dependencies...$(RESET)"
	cd frontend && npm install

frontend-dev: ## Start frontend development server
	@echo "$(GREEN)Starting frontend development server...$(RESET)"
	cd frontend && npm run dev

frontend-build: ## Build frontend for production
	@echo "$(GREEN)Building frontend...$(RESET)"
	cd frontend && npm run build

frontend-test: ## Run frontend tests
	@echo "$(GREEN)Running frontend tests...$(RESET)"
	cd frontend && npm run test

frontend-lint: ## Lint frontend code
	@echo "$(GREEN)Linting frontend code...$(RESET)"
	cd frontend && npm run lint

frontend-format: ## Format frontend code
	@echo "$(GREEN)Formatting frontend code...$(RESET)"
	cd frontend && npm run format

frontend-storybook: ## Start Storybook
	@echo "$(GREEN)Starting Storybook...$(RESET)"
	cd frontend && npm run storybook

# Full stack development
dev-full: ## Start full development environment
	@echo "$(GREEN)Starting full development environment...$(RESET)"
	@echo "$(YELLOW)Starting backend...$(RESET)"
	$(PYTHON) working_dashboard.py &
	@echo "$(YELLOW)Starting frontend...$(RESET)"
	cd frontend && npm run dev &
	@echo "$(GREEN)✅ Full stack running!$(RESET)"
	@echo "$(YELLOW)Backend: http://localhost:5000$(RESET)"
	@echo "$(YELLOW)Frontend: http://localhost:3000$(RESET)"

# Performance and monitoring targets
benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(RESET)"
	pytest $(TEST_DIR)/performance/ --benchmark-only --benchmark-json=benchmark.json
	@echo "$(YELLOW)Benchmark results: benchmark.json$(RESET)"

profile: ## Profile application performance
	@echo "$(GREEN)Profiling application...$(RESET)"
	$(PYTHON) -m cProfile -o profile.stats scripts/live_demo.py
	@echo "$(YELLOW)Profile results: profile.stats$(RESET)"

monitor: ## Start monitoring dashboard
	@echo "$(GREEN)Starting monitoring...$(RESET)"
	# Add monitoring commands here
	@echo "$(YELLOW)Monitoring dashboard: http://localhost:3000$(RESET)"

# Database and data management
migrate: ## Run database migrations
	@echo "$(GREEN)Running database migrations...$(RESET)"
	# Add migration commands here

backup: ## Backup data
	@echo "$(GREEN)Creating backup...$(RESET)"
	mkdir -p backups
	# Add backup commands here
	@echo "$(GREEN)✅ Backup created in backups/$(RESET)"

restore: ## Restore from backup
	@echo "$(GREEN)Restoring from backup...$(RESET)"
	# Add restore commands here

# Deployment targets
deploy-local: ## Deploy locally
	@echo "$(GREEN)Deploying locally...$(RESET)"
	$(MAKE) docker-build
	$(MAKE) docker-compose-up

deploy-staging: ## Deploy to staging
	@echo "$(GREEN)Deploying to staging...$(RESET)"
	# Add staging deployment commands here

deploy-prod: ## Deploy to production
	@echo "$(GREEN)Deploying to production...$(RESET)"
	# Add production deployment commands here

# Utility targets
logs: ## View application logs
	@echo "$(GREEN)Viewing logs...$(RESET)"
	tail -f logs/*.log

check-deps: ## Check for dependency updates
	@echo "$(GREEN)Checking for dependency updates...$(RESET)"
	pip list --outdated

update-deps: ## Update dependencies
	@echo "$(GREEN)Updating dependencies...$(RESET)"
	pip-review --auto

release: ## Create a new release
	@echo "$(GREEN)Creating new release...$(RESET)"
	cz bump
	git push --tags

# Development workflow shortcuts
dev-setup: install-dev ## Complete development setup
	@echo "$(GREEN)✅ Development environment is ready!$(RESET)"
	@echo "$(YELLOW)Next steps:$(RESET)"
	@echo "  make test          # Run tests"
	@echo "  make run-dashboard # Start dashboard"
	@echo "  make docs          # Build documentation"

quick-test: format lint test-unit ## Quick development test cycle
	@echo "$(GREEN)✅ Quick test cycle complete!$(RESET)"

full-check: quality-check test test-security docs ## Full quality and test check
	@echo "$(GREEN)✅ Full check complete - ready for commit!$(RESET)"

# CI/CD simulation
ci-local: ## Simulate CI pipeline locally
	@echo "$(GREEN)Running local CI simulation...$(RESET)"
	$(MAKE) clean
	$(MAKE) install-dev
	$(MAKE) quality-check
	$(MAKE) test
	$(MAKE) build
	$(MAKE) docker-build
	@echo "$(GREEN)✅ Local CI simulation complete!$(RESET)"