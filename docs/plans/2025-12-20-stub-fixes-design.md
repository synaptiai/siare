# Stub Fixes Design

**Date:** 2025-12-20
**Status:** Approved

## Overview

This design addresses 4 stubs/placeholders identified in the SIARE codebase:

1. Allowed tools validation (security-relevant)
2. Bing date filter (feature gap)
3. >2D QD grid visualization (NotImplementedError)
4. Publication benchmark evolution history placeholder

## 1. Allowed Tools Validation

**Location:** `siare/services/director.py`

**Problem:** The `allowed_tools` constraint check at lines 698-703 is an empty `pass` - mutations can add roles with unauthorized tools.

**Solution:** Validate tools in `_apply_mutation` after parsing the new role from LLM response.

```python
# In _apply_mutation, after: new_role = self._parse_new_role_from_llm(...)

if constraints and constraints.get("allowed_tools") is not None:
    allowed_tools = set(constraints["allowed_tools"])
    new_role_tools = new_role["role"].tools or []
    unauthorized_tools = set(new_role_tools) - allowed_tools
    if unauthorized_tools:
        raise ValueError(
            f"Constraint violation: Role '{new_role['role'].id}' uses unauthorized tools: "
            f"{unauthorized_tools}. Allowed tools: {allowed_tools}"
        )
```

**Also:** Replace empty `pass` at lines 698-703 with explanatory comment.

**Behavior:**
- `allowed_tools=["vector_search"]` - only vector_search permitted
- `allowed_tools=[]` - no tools allowed
- `allowed_tools=None` (or not set) - no restriction

## 2. Bing Date Filter

**Location:** `siare/adapters/web_search.py` (lines 270-273)

**Problem:** Date filter parameter is ignored for Bing adapter.

**Solution:** Use Bing's query operators (`after:`, `before:`).

```python
if filter_date:
    if isinstance(filter_date, str):
        params["q"] = f"{query} after:{filter_date}"
    elif isinstance(filter_date, dict):
        date_query = query
        if filter_date.get("after"):
            date_query = f"{date_query} after:{filter_date['after']}"
        if filter_date.get("before"):
            date_query = f"{date_query} before:{filter_date['before']}"
        params["q"] = date_query
```

**Behavior:**
- String: `filter_date="2024-01-01"` searches after that date
- Dict: `filter_date={"after": "2024-01-01", "before": "2024-06-01"}` for range

## 3. >2D QD Grid Visualization

**Location:** `siare/services/qd_grid.py` (line 648)

**Problem:** `NotImplementedError` for grids with more than 2 embedding dimensions.

**Solution:** Return parallel coordinates data for N-dimensional grids.

```python
def visualize_grid(self) -> dict[str, Any]:
    if self.embedding_dimensions <= 2:
        # Existing 2D heatmap (unchanged)
        return {"type": "heatmap", "grid": grid.tolist(), ...}

    # N-dimensional: parallel coordinates
    coordinates = []
    for cell_id_str, elite in self.cells.items():
        cell_id = CellID.from_string(cell_id_str)
        coordinates.append({
            "complexity_bin": cell_id.complexity_bin,
            "diversity_bins": list(cell_id.diversity_bins),
            "quality": elite["quality"],
            "sop_id": elite["sopId"],
        })

    return {
        "type": "parallel_coordinates",
        "dimensions": ["complexity"] + [f"diversity_{i}" for i in range(self.embedding_dimensions)],
        "coordinates": coordinates,
        ...
    }
```

**Behavior:**
- 2D: Returns heatmap (backward compatible)
- N-D: Returns parallel coordinates data
- `"type"` field indicates visualization format

## 4. Publication Benchmark Evolution History

**Location:** `siare/benchmarks/scripts/run_publication_benchmark.py` (lines 613-616)

**Problem:** `evolution_history` is empty list instead of loading actual history.

**Solution:** Load from checkpoint file, with CLI option for explicit path.

```python
evolution_history: list[SOPGene] = []

checkpoint_path = Path(args.output_dir) / "evolution_checkpoint.json"
if args.checkpoint_path:
    checkpoint_path = Path(args.checkpoint_path)

if checkpoint_path.exists():
    try:
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)
        if "history" in checkpoint_data:
            evolution_history = [SOPGene(**g) for g in checkpoint_data["history"]]
            logger.info(f"Loaded {len(evolution_history)} generations from checkpoint")
    except Exception as e:
        logger.warning(f"Could not load evolution history: {e}")

if not evolution_history:
    logger.warning("No evolution history available. Learning curves will be limited.")
```

**CLI addition:**
```python
@click.option("--checkpoint-path", type=click.Path(exists=True),
              help="Path to evolution checkpoint file for learning curves")
```

## Implementation Order

1. Allowed tools validation (highest priority - security)
2. Bing date filter (simple fix)
3. >2D QD visualization (moderate complexity)
4. Publication benchmark history (depends on checkpoint format)

## Testing

Each fix should include:
- Unit test for the new behavior
- Test for edge cases (empty inputs, malformed data)
- Integration with existing test suite
