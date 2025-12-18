---
layout: default
title: Architecture
nav_order: 5
has_children: true
---

# Architecture

Technical documentation for SIARE's internal architecture and data models.

## Overview

SIARE follows an **open-core architecture**:
- **siare** (MIT): Core evolution engine, execution, evaluation, gene pool
- **siare-cloud** (Enterprise): API server, auth, billing, audit logging

## Sections

- [System Architecture]({% link architecture/SYSTEM_ARCHITECTURE.md %}) - Complete system design and component interactions
- [Data Models]({% link architecture/DATA_MODELS.md %}) - All Pydantic models and their relationships
