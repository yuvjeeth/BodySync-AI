# BodySync-AI

## Quickstart (Windows)

### Prerequisites

- Install **Python 3.10+** (and ensure `python` works in PowerShell)
- (Recommended) Install **Git** so you can version and submit the project

### Setup

1. Create a virtual environment and install deps:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure your Groq API key:

- Copy `.env.example` to `.env`
- Set `GROQ_API_KEY=...`

### Run

```powershell
python .\main.py
```

