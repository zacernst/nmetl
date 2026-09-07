# PyCypher + nmetl Makefile
# ------------------------------------------------------------------------------
# Configuration variables
PYTHON_VERSION = 3.14
SUPPORTED_PYTHON_VERSIONS = 3.14
PYTHON_TEST_THREADS = 8
DOCKER = podman

# Cross-platform browser opener (macOS: open, Linux: xdg-open, WSL: wslview)
BROWSER := $(shell command -v xdg-open 2>/dev/null || command -v wslview 2>/dev/null || echo open)


# Project paths
export PROJECT_ROOT := ${PWD}
export PACKAGES_DIR := ${PROJECT_ROOT}/packages
# Package-specific paths
export PYCYPHER_DIR := ${PACKAGES_DIR}/pycypher
export NMETL_DIR := ${PACKAGES_DIR}/nmetl
export SHARED_DIR := ${PACKAGES_DIR}/shared
export LC_ALL := C

# Documentation paths
export DOCS_DIR := ${PROJECT_ROOT}/docs

# Test and coverage paths
export TESTS_DIR := ${PROJECT_ROOT}/tests
export COVERAGE_DIR := ${PROJECT_ROOT}/coverage_report

# Main targets
.PHONY: help pycypher nmetl test tests docs lsp clean veryclean venv uv start format lint lint-changed audit typecheck coverage coverage-check check setup test-file test-find test-k test-mark watch reset lock-check dev-check bench bench-save bench-compare bench-memory metrics-snapshot metrics-prometheus test-telemetry dev-up dev-up-minimal dev-up-full dev-down dev-shell dev-rebuild dev-logs dev-test dev-typecheck dev-format spark-up spark-down spark-logs spark-ui spark-shell spark-scale infra-up infra-down test-spark import-cycles import-cycles-ratchet

# Default target - run the complete build process
all: clean venv format pycypher docs

## Show available targets with descriptions
help:
	@echo "PyCypher Development Targets"
	@echo "============================"
	@echo ""
	@echo "Setup & Build:"
	@echo "  make uv              Install/upgrade uv package manager"
	@echo "  make venv            Create virtual environment"
	@echo "  make build           Build wheel package s"
	@echo "  make pycypher        Build and install pycypher package"
	@echo "  make nmetl           Build and install nmetl package (depends on pycypher)"
	@echo "  make format          Run ruff import sorting + format"
	@echo "  make all             Full rebuild (veryclean + venv + format + pycypher)"
	@echo ""
	@echo "Testing:"
	@echo "  make test            Run all tests (parallel, $(PYTHON_TEST_THREADS) threads)"
	@echo "  make test-fast       Run tests (auto threads, stop on first failure)"
	@echo "  make test-serial     Run tests (single thread)"
	@echo "  make test-quick      Run tests (minimal output, no coverage)"
	@echo "  make test-failed     Re-run only previously failed tests"
	@echo "  make test-changed    Re-run failed tests first, then rest"
	@echo "  make test-verbose    Run tests with verbose output"
	@echo "  make test-unit       Run only unit-marked tests"
	@echo "  make test-no-slow    Run all tests except slow-marked"
	@echo "  make coverage        Run tests with HTML coverage report"
	@echo "  make coverage-check  Run tests with coverage floor (COVERAGE_FLOOR=50)"
	@echo ""
	@echo "Benchmarking:"
	@echo "  make bench                Run performance benchmarks (3 suites)"
	@echo "  make bench-save           Run benchmarks and save baseline"
	@echo "  make bench-compare        Compare benchmarks against baseline"
	@echo "  make bench-memory         Run memory profiling benchmark"
	@echo "  make bench-profile        Profile benchmark with cProfile evidence"
	@echo "  make bench-characterize   Characterize query workload patterns"
	@echo ""
	@echo "Telemetry & Monitoring:"
	@echo "  make metrics-snapshot     Show current metrics (human-readable)"
	@echo "  make metrics-prometheus   Export metrics in Prometheus text format"
	@echo "  make test-telemetry       Run telemetry/exporter tests"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint            Run ruff import check + format check + lint"
	@echo "  make lint-changed    Lint only files changed vs main (CI gate)"
	@echo "  make typecheck       Run ty type checker"
	@echo "  make audit           Scan dependencies for known vulnerabilities (pip-audit)"
	@echo "  make quality         Code quality dashboard (complexity, lint, types)"
	@echo "  make complexity      Show complexity hotspots only"
	@echo "  make quality-changed Quality check on changed files only"
	@echo ""
	@echo "Docker Development:"
	@echo "  make dev-up          Start dev container only (fast; attach VS Code to it)"
	@echo "  make dev-up-full     Start dev container + Spark"
	@echo "  make dev-down        Stop all containers"
	@echo "  make dev-shell       Open shell in dev container"
	@echo "  make dev-rebuild     Rebuild and restart dev container"
	@echo "  make dev-logs        Tail dev container logs"
	@echo "  make dev-test        Run tests inside container"
	@echo "  make dev-typecheck   Run ty type checker inside container"
	@echo "  make dev-format      Run ruff format inside container"
	@echo "  make dev-jupyter     Start with Jupyter Lab (port 8888)"
	@echo "  make dev-vscode      Start with VS Code server (port 8080)"
	@echo ""
	@echo "Infrastructure:"
	@echo "  make infra-up        Start Spark"
	@echo "  make infra-down      Stop Spark"
	@echo "  make spark-up        Start Spark cluster"
	@echo "  make spark-down      Stop Spark cluster"
	@echo "  make spark-ui        Open Spark UI (port 8090)"
	@echo "  make spark-shell     Open PySpark shell"
	@echo "  make spark-scale     Scale workers (WORKERS=N)"
	@echo ""
	@echo "Integration Tests:"
	@echo "  make test-spark      Run Spark tests (requires dev container)"
	@echo "  make test-large-dataset Run large-dataset tests (timeout=120s)"
	@echo "  make test-backends   Run backend equivalence tests (timeout=60s)"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs            Build Sphinx documentation"
	@echo ""
	@echo "Examples:"
	@echo "  uv run python examples/social_network/run_demo.py"
	@echo "  uv run python examples/functions_in_where.py"
	@echo "  uv run python examples/scalar_functions_in_with.py"
	@echo "  uv run python examples/ast_conversion_example.py"
	@echo "  uv run python examples/advanced_grammar_examples.py"
	@echo ""
	@echo "Developer Workflow:"
	@echo "  make setup           One-command onboarding (core deps, no Spark/Dask/Polars)"
	@echo "  make setup-full      Full onboarding (all deps including Spark/Dask/Polars)"
	@echo "  make check           Run lock-check + format + lint + typecheck + test-fast"
	@echo "  make lock-check      Verify uv.lock matches pyproject.toml (CI parity)"
	@echo "  make dev-check       Validate .env before Docker targets"
	@echo "  make test-file FILE=tests/test_foo.py  Run a single test file"
	@echo "  make test-find QUERY=binding           Search test names"
	@echo "  make test-k EXPR=\"binding AND frame\"   Run tests matching keyword expression"
	@echo "  make test-mark MARK=security           Run tests by marker"
	@echo "  make watch                             Re-run tests on file change (TDD)"
	@echo "  make watch WATCH_FILE=tests/test_foo.py  Watch a specific test file"
	@echo ""
	@echo "Getting Started:"
	@echo "  1. make setup            (core deps, or make setup-full for everything)"
	@echo "  2. cp .env.example .env  (set real credentials for Docker)"
	@echo "  3. uv sync               (install dependencies)"
	@echo "  4. make test             (run tests)"
	@echo ""
	@echo "Cleaning:"
	@echo "  make clean           Remove build artifacts"
	@echo "  make reset           Deep clean (clean + remove .venv + uv cache)"
	@echo "  make veryclean       Alias for reset"

uv:
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }

venv: uv
	@echo "Setting up virtual environment..."
	uv venv .venv

start: veryclean install

# ------------------------------------------------------------------------------
# Cleaning targets

# Remove all generated files and virtual environment
reset: clean
	@echo "Deep cleaning project..."
	uv cache clean && rm -rfv ./.venv

# Keep veryclean as alias for backwards compatibility
veryclean: reset

# Clean up build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rfv ./venv
	rm -rfv ${DOCS_DIR}/build/html/*
	rm -rfv ${DOCS_DIR}/build/doctrees/*
	rm -fv ./requirements.txt
	rm -rfv ./dist/*
	rm -rfv ${COVERAGE_DIR}

test:
	uv run pytest -n ${PYTHON_TEST_THREADS}

tests: test

test-fast:
	uv run pytest -n auto -x -m "not slow" --ignore=tests/load_testing --ignore=tests/large_dataset .

test-serial:
	uv run pytest .

test-quick:
	uv run pytest -n auto --tb=line --no-cov -q -m "not slow" --ignore=tests/load_testing --ignore=tests/large_dataset --ignore=tests/property_based .

test-failed:
	uv run pytest --lf -n ${PYTHON_TEST_THREADS} .

test-changed:
	uv run pytest --lf --ff -n ${PYTHON_TEST_THREADS} .

test-verbose:
	uv run pytest -n ${PYTHON_TEST_THREADS} -v .

test-unit:
	uv run pytest -m unit -n ${PYTHON_TEST_THREADS} .

test-no-slow:
	uv run pytest -m "not slow" -n ${PYTHON_TEST_THREADS} .

test-large-dataset:
	@echo "Running large-dataset integration tests..."
	uv run pytest -m integration tests/test_distributed_scaffolding.py tests/test_large_dataset_dependency_compat.py -v --timeout=120

test-backends:
	@echo "Running backend equivalence and compatibility tests..."
	uv run pytest tests/test_large_dataset_dependency_compat.py tests/test_distributed_scaffolding.py -k "not Dask" -v --timeout=60

# ------------------------------------------------------------------------------
# Benchmarking targets (pytest-benchmark)

# All benchmark suite files (must match CI workflow)
BENCH_SUITES := tests/benchmarks/bench_core_operations.py \
                tests/benchmarks/bench_optimizer.py \
                tests/benchmarks/bench_multi_type.py

# Run all benchmarks (excludes slow/100K scale by default)
bench:
	@echo "Running performance benchmarks (3 suites)..."
	uv run pytest $(BENCH_SUITES) -v --benchmark-only -m "not slow" --timeout=120

# Run benchmarks and save as named baseline for regression comparison
BENCH_NAME ?= baseline
bench-save:
	@echo "Running benchmarks and saving as '$(BENCH_NAME)'..."
	uv run pytest $(BENCH_SUITES) -v --benchmark-only -m "not slow" --benchmark-save=$(BENCH_NAME) --timeout=120
	@echo "Saved to .benchmarks/ — compare later with: make bench-compare"

# Run benchmarks and compare against most recent saved baseline
bench-compare:
	@echo "Running benchmarks and comparing against saved baseline..."
	uv run pytest $(BENCH_SUITES) -v --benchmark-only -m "not slow" --benchmark-compare=0001_baseline --benchmark-compare-fail=mean:5.0 --timeout=120

# Run memory profiling benchmark (all scales including slow)
bench-memory:
	@echo "Running memory profiling benchmark..."
	uv run python tests/benchmarks/bench_memory_baseline.py

# Profile a specific benchmark or query and save cProfile evidence
# Usage: make bench-profile PROFILE_TARGET=tests/benchmarks/bench_core_operations.py::TestQueryBenchmarks1K::test_simple_scan
PROFILE_TARGET ?= tests/benchmarks/bench_core_operations.py
PROFILE_OUTPUT ?= .profiles
bench-profile:
	@mkdir -p $(PROFILE_OUTPUT)
	@echo "Profiling $(PROFILE_TARGET) → $(PROFILE_OUTPUT)/"
	uv run python tests/benchmarks/profile_helper.py \
		--target "$(PROFILE_TARGET)" \
		--output-dir "$(PROFILE_OUTPUT)"
	@echo "Profile saved to $(PROFILE_OUTPUT)/ — view with: uv run snakeviz $(PROFILE_OUTPUT)/*.prof"

# Characterize query workloads from benchmark results
bench-characterize:
	@echo "Characterizing query workloads..."
	uv run python tests/benchmarks/workload_characterization.py \
		--benchmark-dir .benchmarks \
		--output workload-report.md

# ------------------------------------------------------------------------------
# Telemetry and monitoring targets

# Show current in-process metrics snapshot (human-readable)
metrics-snapshot:
	@uv run nmetl metrics

# Export current metrics in Prometheus text exposition format
metrics-prometheus:
	@uv run python -c "from shared.metrics import QUERY_METRICS; from shared.exporters import PrometheusExporter; print(PrometheusExporter().render(QUERY_METRICS.snapshot()))"

# Run telemetry and exporter test suites
test-telemetry:
	@echo "Running telemetry integration tests..."
	uv run pytest tests/test_otel_integration.py tests/test_metrics_exporters.py -v

# ------------------------------------------------------------------------------
# Code quality targets (local equivalents of CI checks)

lint:
	@echo "Running linters..."
	uv run ruff check --select I .
	uv run ruff format --check .
	uv run ruff check .

# Lint only files changed vs main (enforces quality on new code)
lint-changed:
	@./scripts/lint_changed.sh

# Scan dependencies for known vulnerabilities (CVEs)
# --skip-editable excludes workspace packages not on PyPI
audit:
	@echo "Auditing dependencies for known vulnerabilities..."
	uv run pip-audit --desc --skip-editable --cache-dir "$$(mktemp -d)"

# Static Application Security Testing (SAST) — scan source code for
# injection risks, hardcoded secrets, and other security anti-patterns.
# Configuration in pyproject.toml [tool.bandit].
sast:
	@echo "Running SAST scan (bandit)..."
	uv run bandit -r packages/pycypher/src/ packages/nmetl/src/ packages/shared/src/ \
		-c pyproject.toml --severity-level medium -f txt

# Combined security scan: dependencies + code
security: audit sast

# Feature completeness tracking (skip/xfail/TODO/NotImplementedError patterns)
feature-completeness:
	@uv run python scripts/check_feature_completeness.py

# Import cycle detection (circular dependency analysis)
import-cycles:
	@uv run python scripts/check_import_cycles.py

# Import cycle ratchet — fail if a new cycle appears or an existing cycle
# gains a member, relative to scripts/import_cycles_baseline.txt
import-cycles-ratchet:
	@uv run python scripts/check_import_cycles.py --ratchet

# Orphan module detection (zero inbound imports)
orphans:
	@uv run python scripts/check_orphan_modules.py

typecheck:
	@echo "Running type checker..."
	uv run ty check

# ------------------------------------------------------------------------------
# Developer workflow targets

# One-command onboarding: copy .env, install core deps, install pre-commit hooks.
# Uses dev-core group (no Spark/Dask/Polars). For full deps: make setup-full
setup:
	@echo "Setting up development environment (core)..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example — edit it to set real credentials for Docker."; \
	else \
		echo ".env already exists, skipping copy."; \
	fi
	uv sync --group dev-core
	uv run pre-commit install
	@echo ""
	@echo "Setup complete! Next steps:"
	@echo "  make test        Run the test suite (use -m 'not spark' for core-only)"
	@echo "  make dev-up      Start Docker dev environment (edit .env first)"
	@echo ""
	@echo "Dependency groups available:"
	@echo "  dev-core  (installed) — testing, linting, docs"
	@echo "  dev                   — adds Spark, Dask, Polars  (make setup-full)"
	@echo "  dev-full              — adds Jupyter, profiling, visualization"

# Full onboarding: all dev dependencies including Spark, Dask, Polars
setup-full:
	@echo "Setting up full development environment..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env from .env.example — edit it to set real credentials for Docker."; \
	else \
		echo ".env already exists, skipping copy."; \
	fi
	uv sync --group dev
	uv run pre-commit install
	@echo ""
	@echo "Full setup complete! All dependency groups installed."
	@echo "  make test        Run all tests"
	@echo "  make dev-up      Start Docker dev environment (edit .env first)"

# Verify uv.lock is in sync with pyproject.toml (matches CI --frozen)
lock-check:
	@echo "Checking lockfile is in sync with pyproject.toml..."
	uv sync --frozen --dry-run 2>&1 || { echo "ERROR: uv.lock is out of date. Run 'uv sync' to update."; exit 1; }
	@echo "Lockfile is in sync."

# Validate .env before docker-compose (prevents cryptic startup errors)
dev-check:
	@echo "Checking Docker environment prerequisites..."
	@if [ ! -f .env ]; then \
		echo "ERROR: .env file not found. Run 'cp .env.example .env' and set real credentials."; \
		exit 1; \
	fi
	@missing=""; \
	for var in SPARK_MASTER_URL SPARK_RPC_SECRET; do \
		val=$$(grep "^$$var=" .env 2>/dev/null | cut -d= -f2-); \
		if [ -z "$$val" ] || echo "$$val" | grep -q '<.*>'; then \
			missing="$$missing $$var"; \
		fi; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "ERROR: The following .env variables are missing or still have placeholder values:"; \
		echo " $$missing"; \
		echo "Edit .env and set real values before running Docker targets."; \
		exit 1; \
	fi
	@echo "Environment OK — all required Docker variables are set."

# Run format + lint + typecheck + fast tests (local CI equivalent)
check: lock-check format lint typecheck import-cycles-ratchet test-fast

## Code quality dashboard — complexity hotspots, lint summary, type coverage
quality:
	uv run python scripts/code_quality.py

## Complexity analysis only
complexity:
	uv run python scripts/code_quality.py --complexity

## Quality check on changed files only
quality-changed:
	uv run python scripts/code_quality.py --changed

# Run a single test file: make test-file FILE=tests/test_foo.py
FILE ?= tests/
test-file:
	uv run pytest -n auto -x $(FILE)

# Search test names by keyword: make test-find QUERY=binding
QUERY ?= ""
test-find:
	@uv run pytest --collect-only -q 2>/dev/null | grep -i "$(QUERY)" || echo "No tests matching '$(QUERY)'"

# Run tests matching a keyword expression: make test-k EXPR="binding AND frame"
EXPR ?= ""
test-k:
	uv run pytest -n auto -x -k "$(EXPR)" .

# Run tests by marker: make test-mark MARK=security
MARK ?= ""
test-mark:
	uv run pytest -n ${PYTHON_TEST_THREADS} -m "$(MARK)" .

# Watch files and re-run tests on change (TDD workflow)
WATCH_FILE ?= tests/
watch:
	uv run ptw -- -x --tb=short $(WATCH_FILE)

# ------------------------------------------------------------------------------
# Docker development targets

# Start only the dev container — fast, no Spark. The dev container still
# joins pycypher-dev-network, so it can reach Spark by service name once
# that is started separately (make spark-up).
dev-up: dev-check
	@echo "Starting pycypher dev container..."
	${DOCKER} compose up -d --no-deps pycypher-dev
	@echo ""
	@echo "pycypher-dev is up on pycypher-dev-network."
	@echo "  Shell:   make dev-shell"
	@echo "  VS Code: Cmd/Ctrl+Shift+P -> \"Dev Containers: Attach to Running Container\" -> pycypher-dev"
	@echo "           (or \"Dev Containers: Reopen in Container\" — see .devcontainer/devcontainer.json)"
	@echo "  Full stack (+ Spark): make dev-up-full"

# Alias retained for anyone with dev-up-minimal in muscle memory — identical
# to dev-up now that dev-up itself is the lean target.
dev-up-minimal: dev-up

# Start the full stack: dev container + Spark.
dev-up-full: dev-check
	@echo "Starting full pycypher stack (dev + Spark)..."
	${DOCKER} compose up -d
	@echo "  pycypher-dev : make dev-shell"
	@echo "  Spark UI     : http://localhost:8090"

# Stop all containers
dev-down:
	@echo "Stopping development container..."
	${DOCKER} compose down

# Access shell in development container
dev-shell:
	@echo "Accessing pycypher development container shell..."
	${DOCKER} compose exec pycypher-dev bash

# Rebuild and start development container
dev-rebuild:
	@echo "Rebuilding pycypher development container..."
	${DOCKER} compose build pycypher-dev
	${DOCKER} compose up -d pycypher-dev

# View logs from development container
dev-logs:
	@echo "Viewing pycypher development container logs..."
	${DOCKER} compose logs -f pycypher-dev

# Start with Jupyter Lab for interactive development
dev-jupyter:
	@echo "Starting development environment with Jupyter Lab..."
	${DOCKER} compose --profile jupyter up -d
	@echo "Jupyter Lab available at http://localhost:8888"

# Start with code-server (VS Code in browser)
dev-vscode:
	@echo "Starting development environment with VS Code server..."
	${DOCKER} compose --profile code-server up -d
	@echo "VS Code available at http://localhost:8080"
	@echo "Password: ${CODE_SERVER_PASSWORD:-pycypher}"

# Run tests inside the container
dev-test:
	@echo "Running tests in development container..."
	${DOCKER} compose exec pycypher-dev bash -c "cd /workspace && uv run pytest -n auto -x tests/"

# Run type checking inside the container
dev-typecheck:
	@echo "Running type checker in development container..."
	${DOCKER} compose exec pycypher-dev bash -c "cd /workspace && uv run ty check packages/pycypher/"

# Format code inside the container
dev-format:
	@echo "Formatting code in development container..."
	${DOCKER} compose exec pycypher-dev bash -c "cd /workspace && uv run ruff format packages/pycypher/"

# ------------------------------------------------------------------------------
# Spark targets

spark-up:
	@echo "Starting Spark cluster..."
	${DOCKER} compose up -d spark-master spark-worker

spark-down:
	${DOCKER} compose stop spark-master spark-worker

spark-logs:
	${DOCKER} compose logs -f spark-master spark-worker

spark-ui:
	$(BROWSER) http://localhost:8090

spark-shell:
	${DOCKER} compose exec pycypher-dev bash -c \
	  "cd /workspace && PYSPARK_DRIVER_PYTHON=python3 \
	   uv run pyspark --master spark://spark-master:7077"

WORKERS ?= 2
spark-scale:
	${DOCKER} compose up -d --scale spark-worker=$(WORKERS) spark-worker

# ------------------------------------------------------------------------------
# Combined infrastructure targets

infra-up: spark-up
	@echo "Spark services running."

infra-down:
	${DOCKER} compose stop spark-master spark-worker

# ------------------------------------------------------------------------------
# Integration test targets

test-spark:
	@echo "Running Spark integration tests..."
	${DOCKER} compose exec pycypher-dev bash -c \
	  "cd /workspace && uv run pytest -m spark -v"

# ------------------------------------------------------------------------------
# Development targets

# Format code
format:
	@echo "Formatting code..."
	uv run ruff check --select I --fix .
	uv run ruff format . --config ./pyproject.toml

# Build packages
build:
	@echo "Building packages..."
	@for v in ${SUPPORTED_PYTHON_VERSIONS}; do \
		echo "Building with Python version ${PYTHON_VERSION}..." && \
		uv run --python $v uv build -t wheel || exit 1; \
	done

# Install packages in development mode
install: build
	@echo "Installing packages in development mode..."
	uv pip install --upgrade -e ${PYCYPHER_DIR}
# ------------------------------------------------------------------------------
# Testing targets

# Run tests with coverage (parallel)
coverage:
	@echo "Running tests with coverage..."
	uv run pytest -n ${PYTHON_TEST_THREADS} --cov-report html:${COVERAGE_DIR} --cov

# Run tests with coverage (detailed, serial)
coverage-detailed:
	@echo "Running tests with detailed coverage..."
	uv run pytest --cov-report html:${COVERAGE_DIR} --cov --cov-report term-missing

# Run tests with coverage floor enforcement (CI gate)
COVERAGE_FLOOR ?= 80
coverage-check:
	@echo "Running tests with coverage floor ($(COVERAGE_FLOOR)%)..."
	uv run pytest -n ${PYTHON_TEST_THREADS} --cov --cov-fail-under=$(COVERAGE_FLOOR) -q

# ------------------------------------------------------------------------------
# Documentation targets

docs:
	@echo "Building documentation..."
	cd ${DOCS_DIR} && uv run make html
	# @echo "Building PDF documentation..."
	# cd ${DOCS_DIR} && uv run make latexpdf

lsp:
	@echo "Starting PyCypher LSP server (stdin/stdout)..."
	uv run python -m pycypher.cypher_lsp

# ------------------------------------------------------------------------------
# Package-specific targets

# Build and install only pycypher
pycypher:
	@echo "Building and installing pycypher package..."
	cd ${PYCYPHER_DIR} && uv build
	uv pip install --upgrade -e ${PYCYPHER_DIR}

# Build and install only nmetl (depends on pycypher)
nmetl: pycypher
	@echo "Building and installing nmetl package..."
	cd ${NMETL_DIR} && uv build
	uv pip install --upgrade -e ${NMETL_DIR}

