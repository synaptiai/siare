# SIARE Documentation

Welcome to the SIARE (Self-Improving Agentic RAG Engine) documentation.

> **Open-Core Architecture**: SIARE follows an open-core model:
> - **siare** (this package): MIT-licensed core with evolution engine, execution, evaluation
> - **siare-cloud** (enterprise): Proprietary features including auth, billing, audit, approval workflows

## Quick Start

| Goal | Guide |
|------|-------|
| Understand what makes SIARE unique | [Why SIARE?](WHY_SIARE.md) |
| Run your first pipeline | [Quick Start](QUICKSTART.md) |
| Configure SIARE | [Configuration](CONFIGURATION.md) |
| Fix common issues | [Troubleshooting](TROUBLESHOOTING.md) |

## Guides

| Topic | Guide |
|-------|-------|
| Build your first custom pipeline | [First Custom Pipeline](guides/first-custom-pipeline.md) |
| Add custom metrics, tools, constraints | [Custom Extensions](guides/custom-extensions.md) |
| Write effective multi-agent prompts | [Prompt Engineering](guides/prompt-engineering.md) |
| Implement domain use cases | [Use Cases](guides/USE_CASES.md) |

## Architecture

| Document | Description |
|----------|-------------|
| [System Architecture](architecture/SYSTEM_ARCHITECTURE.md) | Complete system design |
| [Data Models](architecture/DATA_MODELS.md) | Core Pydantic model definitions |

## Reference

| Document | Description |
|----------|-------------|
| [Glossary](GLOSSARY.md) | SIARE terminology |
| [PRD](PRD.md) | Product vision and requirements |
| [Contributing](CONTRIBUTING.md) | Developer guide |

## Documentation Map

```
docs/
├── Getting Started
│   ├── WHY_SIARE.md               # What makes SIARE unique
│   ├── QUICKSTART.md              # 10-minute first run
│   ├── CONFIGURATION.md           # All settings explained
│   └── TROUBLESHOOTING.md         # Common issues
│
├── Guides
│   ├── first-custom-pipeline.md   # Customize for your domain
│   ├── custom-extensions.md       # Add metrics, tools, constraints
│   ├── prompt-engineering.md      # Effective multi-agent prompts
│   └── USE_CASES.md               # Domain implementation patterns
│
├── Architecture
│   ├── SYSTEM_ARCHITECTURE.md     # System design
│   └── DATA_MODELS.md             # Pydantic models
│
├── Reference
│   ├── GLOSSARY.md                # Terminology
│   ├── PRD.md                     # Product requirements
│   └── CONTRIBUTING.md            # Developer guide
│
└── README.md                      # This file
```

## API Server 🔒

> **Enterprise Feature**: The REST API server is available in siare-cloud.

The open-source core provides:
- **CLI interface**: `siare init`, `siare evolve`, `siare run`
- **Python library**: Direct import and use of all services

```bash
# CLI usage (open-source)
siare init
siare evolve --generations 10
siare run "Your query here"

# Python library usage (open-source)
from siare.services import DirectorService, GenePool, ExecutionEngine
```

## Quick Links

| Resource | Description |
|----------|-------------|
| [Main README](../README.md) | Project overview and installation |
| [CLAUDE.md](../CLAUDE.md) | Project conventions for AI assistants |
| [GitHub Issues](https://github.com/synaptiai/siare/issues) | Report bugs and request features |

---

*Last Updated: 2025-12-17*
