# Episteme 

> **Know what your AI actually knows.**  
> An epistemic workspace that decomposes AI responses, quantifies uncertainty, and manages long-term memory adaptively.

Episteme is a unified interface powered by two core engines:
1. **KAIROS** (Epistemic Uncertainty Engine)
2. **MAKS** (Adaptive Memory System)

---

## The Engines

### KAIROS: Epistemic Uncertainty Engine
LLMs generate fluent, confident text regardless of whether the claim is solid or completely fabricated. KAIROS runs every response through a three-axis epistemic analysis without requiring fine-tuning:
- **Entropy (H)**: Measures token distribution uncertainty.
- **Gradient (G)**: Measures structural importance (how much the answer changes if the claim is negated).
- **Consistency (C)**: Measures self-contradiction across multiple samplings.

**The Core Formula**: `U = H × G × (1 - C)`  
Claims are classified into three epistemic zones:
- ■ **SOLID** (`U ≤ 0.03`): Low uncertainty. Confident, consistent, and structurally safe.
- ■ **GRADIENT** (`0.03 < U < 0.08`): Moderate uncertainty. Worth double-checking.
- ■ **FAULT LINE** (`U ≥ 0.08`): High uncertainty. The model is guessing, contradicting itself, or the claim is dangerously load-bearing.

### MAKS: Adaptive Memory System
Traditional chatbots simply dump old context when they hit token limits. MAKS uses biologically-inspired survival scoring to decide what to keep, what to compress, and what to archive.
- **Survival Scoring**: Memories are scored based on recency, access frequency, and epistemic entropy (uncertain claims are forgotten faster).
- **Fidelity Tiers**: Memories naturally degrade from **FULL** → **PARTIAL** (compressed) → **GHOST** (archived).
- **Reconsolidation**: Ghost memories are stored in SQLite and can be seamlessly "reconsolidated" (brought back to active memory) if a new query is semantically similar to them.

---

## Running Locally

### Prerequisites
- Python 3.11+
- Nvidia NIM API Key (or equivalent OpenAI-compatible endpoint)

### 1. Setup Environment
```bash
# Clone the repository
git clone <your-repo-url>
cd QILA

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Credentials
Create a `.env` file in the root directory and add your API keys and configuration:
```env
NIM_API_KEY="your-nvidia-api-key-here"
NIM_BASE_URL="https://integrate.api.nvidia.com/v1"
NIM_MODEL="meta/llama3-70b-instruct"
EPISTEME_SECRET="generate-a-secure-random-string-for-cookies"
```

### 3. Run the Server
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Visit `http://127.0.0.1:8000` in your browser.

---

## Running with Docker (Recommended)

Episteme comes with a production-ready Dockerfile and Docker Compose configuration.

```bash
# Make sure your .env file is populated, then run:
docker compose up --build
```
The app will be available at `http://localhost:8000`. The SQLite database will be persisted in a Docker volume automatically.

---

## Testing

Episteme includes a comprehensive suite of 43 tests covering pure math logic, memory transitions, and API integration.

```bash
# Run the full test suite
python -m pytest tests/ -v
```

---

## Project Structure

```
├── app.py                  # FastAPI production server & routing
├── KAIROS/                 # Epistemic Uncertainty Engine
│   ├── KAIROS.py           # Core orchestrator
│   ├── sampler.py          # Parallel N-sampling (Consistency)
│   ├── gradient_engine.py  # Counterfactual perturbations (Gradient)
│   └── ...
├── MAKS/                   # Adaptive Memory System
│   ├── memory_unit.py      # Core data structures
│   ├── ghost_store.py      # Vector archival logic
│   └── ...
├── qila/                   # Integration & Production Services
│   ├── qila_session.py     # Glue between KAIROS, MAKS, and FastAPI
│   ├── persistence.py      # SQLite database adapter
│   └── logger.py           # Centralized structured logging
├── tests/                  # Pytest suite
├── landing.html            # Landing page UI
├── index.html              # Main Workspace UI
└── requirements.txt        # Python dependencies
```

---

## Logs & Observability
All server interactions, memory lifecycle events, and epistemic calculations are logged to `episteme.log` in the project root, as well as the console. Structured logging is managed by `qila/logger.py`.
