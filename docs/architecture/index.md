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

- [System Architecture](SYSTEM_ARCHITECTURE.html) - Complete system design and component interactions
- [Data Models](DATA_MODELS.html) - All Pydantic models and their relationships
