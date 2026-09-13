# Contributing to qlib-tw-trader

Thank you for your interest in contributing! This guide will help you get started.

## Prerequisites

- **Python 3.12+**
- **Node.js 20+** and npm
- **Git**
- (Optional) [DVC](https://dvc.org/) for downloading pre-built data and models

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/qlib-tw-trader.git
cd qlib-tw-trader
```

### 2. Set up the backend

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Set up the frontend

```bash
cd frontend
npm install
cd ..
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your FinMind API token (optional for basic development)
```

### 5. Initialize the database

If you have DVC configured with Google Drive Desktop:

```bash
python -m dvc pull
```

Otherwise, start with an empty database -- the system will create `data/data.db` on first run.

### 6. Seed factors

Start the backend, then seed the factor library:

```bash
# Terminal 1: start backend
uvicorn src.interfaces.app:app --reload --port 8000

# Terminal 2: seed factors
curl -X POST http://localhost:8000/api/v1/factors/seed
```

## Development Workflow

### Running locally

```bash
# Backend (auto-reload on changes)
uvicorn src.interfaces.app:app --reload --port 8000

# Frontend (Vite dev server with HMR)
cd frontend && npm run dev
```

- Backend API: http://localhost:8000
- Frontend: http://localhost:5173 (proxies `/api` to backend)
- Swagger docs: http://localhost:8000/docs

### Running with Docker

```bash
docker compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:3000

### Running tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_services.py

# With verbose output
pytest -v
```

## Project Structure

```
qlib-tw-trader/
├── src/
│   ├── adapters/          # External data source clients (TWSE, FinMind, yfinance)
│   ├── flows/             # Orchestration workflows
│   ├── interfaces/        # FastAPI routes, schemas, WebSocket handlers
│   │   ├── routers/       # API endpoint definitions
│   │   └── schemas/       # Pydantic request/response models
│   ├── repositories/      # Database access layer and factor definitions
│   │   └── factors/       # Factor libraries (Alpha158, Taiwan chips, etc.)
│   ├── services/          # Business logic
│   │   ├── factor_selection/  # IC-based factor selection algorithms
│   │   └── stability/        # Model quality monitoring
│   └── shared/            # Shared types, utilities, week helpers
├── frontend/
│   └── src/
│       ├── pages/         # React page components
│       ├── components/    # Reusable UI components
│       └── stores/        # Zustand state management
├── tests/                 # pytest test suite
├── scripts/               # Reproducible analysis and evaluation scripts
├── data/                  # Database, models, qlib exports (gitignored)
└── docker-compose.yml     # Container orchestration
```

## Coding Standards

### Python

- **Naming**: `camelCase` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- **Files**: `kebab-case` for filenames
- **Style**: Single responsibility, prefer pure functions, no unused imports
- **Errors**: Never silently ignore errors
- **Types**: Use type hints for function signatures

### TypeScript / React

- **Components**: `PascalCase` for component names and files
- **State**: Zustand for global state, React state for local UI state
- **Styling**: TailwindCSS utility classes

### Git Commits

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new factor category for volatility regime
fix: correct lookahead bias in backtest date alignment
refactor: extract IC calculation into standalone service
docs: update API endpoint reference
test: add integration tests for factor seeding
chore: update dependencies
```

## Key Concepts

Before contributing, familiarize yourself with these domain-specific patterns:

- **Lookahead bias prevention**: Trade on day T must only use features from T-1 or earlier.
- **Label definition**: 2-day forward return (`Ref($close,-3)/Ref($close,-1)-1`), meaning T+1 close to T+3 close.
- **IC (Information Coefficient)**: Rank correlation between predicted scores and actual returns. Primary model evaluation metric.
- **Walk-Forward backtesting**: Rolling window retraining to simulate realistic out-of-sample performance.
- **Timezone**: All dates use `Asia/Taipei` (UTC+8).

## Submitting Changes

1. Fork the repository and create a feature branch from `main`.
2. Make your changes with clear, focused commits.
3. Ensure all tests pass: `pytest`
4. Lint the frontend: `cd frontend && npm run lint`
5. Open a pull request with a clear description of the change and its motivation.

## Questions?

Open an issue for bugs, feature requests, or questions about the codebase.
