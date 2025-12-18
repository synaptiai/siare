---
layout: default
title: Prompt Engineering
parent: Guides
nav_order: 2
---

# Prompt Engineering for Multi-Agent Systems

Guide to writing effective prompts for SIARE's multi-agent pipelines.

## Overview

Multi-agent prompts differ from single-agent prompts. Each agent has:
- A specific role in the pipeline
- Inputs from upstream agents
- Outputs consumed by downstream agents
- Constraints for evolution

---

## Prompt Structure

### Recommended Template

```
You are a [ROLE DESCRIPTION].

TASK: [Specific task description]

CONTEXT:
- [What you know about the user's goal]
- [What inputs you receive]

INSTRUCTIONS:
1. [First step]
2. [Second step]
3. [Constraints/rules]

OUTPUT FORMAT:
[Exact format specification]

EXAMPLES:
[Optional: Show input/output examples]
```

### Example: Retriever Agent

```
You are a document retrieval specialist for legal contracts.

TASK: Find contract sections relevant to the user's question.

CONTEXT:
- User is asking about specific contract clauses
- You have access to a vector database of contracts
- Relevance > recency for this domain

INSTRUCTIONS:
1. Analyze the user's question to identify key legal terms
2. Search for sections containing those terms
3. Rank results by relevance to the specific question
4. Return TOP 5 most relevant sections

OUTPUT FORMAT (JSON):
{
  "relevant_sections": [
    {
      "document": "Contract name",
      "section": "Section title",
      "page": 5,
      "relevance_score": 0.95,
      "content": "Exact text from section..."
    }
  ]
}

EXAMPLE:
Question: "What are the termination clauses?"
Output: {"relevant_sections": [{"document": "ABC Contract", "section": "8.1 Termination", ...}]}
```

---

## Key Principles

### 1. Be Specific About Role

**Bad:**
```
You are a helpful assistant.
```

**Good:**
```
You are a legal document analyst specializing in contract review.
Your expertise includes identifying clauses, assessing risk, and
explaining legal terms in plain language.
```

### 2. Define Clear Boundaries

**Bad:**
```
Answer the user's question.
```

**Good:**
```
Answer ONLY based on the provided documents.
If the documents don't contain the answer, respond:
"The provided documents do not contain information about [topic]."
NEVER invent or assume information not in the documents.
```

### 3. Specify Output Format

**Bad:**
```
Return the results.
```

**Good:**
```
OUTPUT FORMAT (JSON):
{
  "answer": "Your response",
  "confidence": "high|medium|low",
  "citations": [
    {"document": "...", "section": "...", "quote": "..."}
  ]
}

Return ONLY the JSON object, no additional text.
```

### 4. Include Examples

**Bad:**
```
Format dates correctly.
```

**Good:**
```
Format dates as YYYY-MM-DD.

EXAMPLES:
- "January 15, 2025" → "2025-01-15"
- "15/01/2025" → "2025-01-15"
- "2025-01-15" → "2025-01-15" (already correct)
```

### 5. Handle Edge Cases

**Bad:**
```
Find relevant documents.
```

**Good:**
```
Find relevant documents.

EDGE CASES:
- No matches found: Return empty array with explanation
- Multiple equally relevant: Return all, sorted by recency
- Ambiguous query: Return top matches for each interpretation
- Very long documents: Return most relevant excerpt (max 500 words)
```

---

## Multi-Agent Considerations

### Input Variable Naming

Use consistent variable naming for upstream inputs:

```python
# In RoleConfig
inputs=[
    RoleInput(from_="user_input"),       # → {query}
    RoleInput(from_="retriever"),         # → {documents}
    RoleInput(from_="analyzer"),          # → {analysis}
]
```

```
# In prompt, reference as:
User Question: {query}
Retrieved Documents: {documents}
Analysis: {analysis}
```

### Chain-of-Thought Across Agents

Each agent should build on previous work:

```
# Retriever output
{
  "documents": [{"id": "doc1", "content": "..."}]
}

# Analyzer prompt references retriever output
"Analyze the following documents: {documents}"

# Synthesizer prompt references analyzer output
"Given the analysis: {analysis}, formulate a response..."
```

### Error Propagation

Handle upstream failures gracefully:

```
HANDLING UPSTREAM ERRORS:
- If {documents} is empty: Respond with "No relevant documents found"
- If {documents} contains errors: Log and skip those entries
- If upstream returned error: Respond with fallback behavior
```

---

## Domain-Specific Prompts

### Legal Domain

```
You are a legal document analyst.

DOMAIN RULES:
- Never provide legal advice (you analyze, not advise)
- Always cite specific sections and page numbers
- Flag ambiguous or potentially concerning clauses
- Use precise legal terminology where appropriate

RISK ASSESSMENT:
When asked about risks, categorize as:
- HIGH: Immediate legal/financial exposure
- MEDIUM: Potential issues requiring review
- LOW: Standard clauses, minimal concern
```

### Medical/Clinical Domain

```
You are a clinical information specialist.

DOMAIN RULES:
- Never provide medical diagnosis or treatment advice
- Always include disclaimers about consulting healthcare providers
- Cite clinical studies with proper references (Author, Year)
- Flag off-label uses or experimental treatments

SAFETY:
If a question involves patient safety concerns:
1. Provide the requested information
2. Add safety disclaimer
3. Recommend professional consultation
```

### Technical Documentation

```
You are a technical documentation specialist.

DOMAIN RULES:
- Provide code examples when helpful
- Include version numbers for libraries/tools
- Distinguish between required and optional steps
- Warn about breaking changes or deprecations

CODE EXAMPLES:
- Use syntax highlighting markers (```python)
- Include imports and dependencies
- Add comments for complex logic
- Show both success and error cases
```

---

## Evolution-Friendly Prompts

### Use Sections for Targeted Evolution

Structure prompts so SIARE can evolve specific parts:

```
## ROLE
You are a document retrieval specialist.

## TASK
Find relevant sections in the document corpus.

## INSTRUCTIONS
1. Parse the query for key terms
2. Search the vector database
3. Rank by relevance

## OUTPUT FORMAT
Return JSON with relevant_sections array.

## EXAMPLES
[Examples here]

## CONSTRAINTS
Never return more than 10 results.
```

With section markers, evolution can target specific improvements:
- `TextGrad` might improve just the INSTRUCTIONS
- `EvoPrompt` might vary the EXAMPLES
- `MetaPrompt` might restructure the TASK

### Mark Protected Content

Use constraints to protect critical instructions:

```python
from siare.core.models import RolePrompt, PromptConstraints

prompt = RolePrompt(
    id="safety_critical_prompt",
    content="""
## ROLE
You are a safety-conscious assistant.

## SAFETY POLICY (DO NOT MODIFY)
CRITICAL: Never recommend actions that could cause harm.
CRITICAL: Always suggest consulting professionals for medical/legal advice.
CRITICAL: Flag any potentially dangerous requests.

## TASK
[This section can be evolved]

## OUTPUT FORMAT
[This section can be evolved]
""",
    constraints=PromptConstraints(
        mustNotChange=[
            "SAFETY POLICY (DO NOT MODIFY)",
            "CRITICAL: Never recommend",
            "CRITICAL: Always suggest",
            "CRITICAL: Flag any potentially",
        ],
        allowedChanges=["TASK", "OUTPUT FORMAT", "examples"],
    ),
)
```

### Enable Experimentation

Leave room for evolution to explore:

```
## OUTPUT FORMAT
Return a structured response. The exact format may be refined through evolution.

Current format:
{
  "answer": "...",
  "confidence": "..."
}
```

vs. being too rigid:

```
## OUTPUT FORMAT (FIXED)
Return EXACTLY this JSON structure, no variations allowed.
```

---

## Common Patterns

### Pattern 1: Retriever Prompts

```
You are a retrieval specialist for {domain}.

TASK: Find documents relevant to the user's query.

INPUT:
- Query: {query}
- Available documents: {corpus_summary}

RETRIEVAL STRATEGY:
1. Extract key concepts from query
2. Identify document types likely to contain answers
3. Search for semantic matches
4. Rank by relevance (not recency unless specified)

OUTPUT FORMAT (JSON):
{
  "query_understanding": "Brief interpretation of what user wants",
  "relevant_documents": [
    {
      "id": "doc_123",
      "title": "Document title",
      "relevance_score": 0.95,
      "relevant_excerpt": "Most relevant passage...",
      "page_or_section": "p.5 or Section 3.2"
    }
  ],
  "search_strategy": "Brief explanation of search approach"
}

EDGE CASES:
- No matches: Return empty array with explanation
- Ambiguous query: Include documents for each interpretation
- Too many matches: Return top 10, note there are more
```

### Pattern 2: Analyzer Prompts

```
You are an analysis specialist for {domain}.

TASK: Analyze retrieved documents to extract specific information.

INPUT:
- Original query: {query}
- Retrieved documents: {documents}

ANALYSIS PROCESS:
1. Review each document for relevance to query
2. Extract specific facts, figures, or statements
3. Note any contradictions between sources
4. Assess confidence based on source quality

OUTPUT FORMAT (JSON):
{
  "key_findings": [
    {
      "finding": "Specific fact or insight",
      "source": "Document ID and location",
      "confidence": "high|medium|low",
      "supporting_quote": "Direct quote from source"
    }
  ],
  "contradictions": [
    {
      "topic": "What the contradiction is about",
      "source_a": "First perspective",
      "source_b": "Contradicting perspective"
    }
  ],
  "gaps": ["Information needed but not found in documents"]
}
```

### Pattern 3: Synthesizer Prompts

```
You are a synthesis specialist for {domain}.

TASK: Combine analysis into a coherent, accurate response.

INPUT:
- Original query: {query}
- Analysis findings: {analysis}
- Retrieved documents: {documents}

SYNTHESIS RULES:
1. Answer the query directly and completely
2. Cite sources for every factual claim
3. Acknowledge contradictions or uncertainties
4. Prioritize accuracy over completeness

OUTPUT FORMAT:
{
  "answer": "Clear, well-structured response with inline citations",
  "key_points": ["Main point 1", "Main point 2"],
  "confidence": "high|medium|low",
  "caveats": ["Any limitations or uncertainties"],
  "follow_up_questions": ["Suggested clarifying questions"]
}

CITATION FORMAT:
Use [DocID, Section] format inline.
Example: "The termination period is 30 days [ABC-Contract, §8.1]."
```

### Pattern 4: Validator Prompts

```
You are a quality validator for {domain} responses.

TASK: Validate the quality and accuracy of a generated response.

INPUT:
- Original query: {query}
- Generated response: {response}
- Source documents: {documents}

VALIDATION CHECKS:
1. ACCURACY: Does response match source documents?
2. COMPLETENESS: Does it fully answer the query?
3. CITATIONS: Are citations present and correct?
4. SAFETY: Any harmful or misleading content?
5. FORMAT: Does output match expected format?

OUTPUT FORMAT (JSON):
{
  "is_valid": true|false,
  "quality_score": 0.0-1.0,
  "checks": {
    "accuracy": {"passed": true, "issues": []},
    "completeness": {"passed": true, "issues": []},
    "citations": {"passed": false, "issues": ["Missing citation for claim X"]},
    "safety": {"passed": true, "issues": []},
    "format": {"passed": true, "issues": []}
  },
  "needs_revision": true|false,
  "revision_guidance": "Specific instructions if revision needed"
}
```

---

## Testing Prompts

### Manual Testing

Before evolution, test prompts manually:

```python
from siare.services.execution_engine import ExecutionEngine

# Create test task
test_task = Task(
    id="test_1",
    input={"query": "What are the termination clauses?"},
)

# Run single execution
trace = engine.execute(sop, genome, test_task)

# Inspect outputs
for role_id, output in trace.outputs.items():
    print(f"{role_id}: {output}")
```

### Common Issues

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Empty output | Missing output format | Add explicit format spec |
| Hallucination | No grounding constraint | Add "only use provided docs" |
| Wrong format | Vague format description | Add concrete examples |
| Truncated | Max tokens too low | Increase or summarize |
| Inconsistent | No examples | Add input/output examples |

---

## Best Practices Checklist

Before deploying a prompt:

- [ ] Role clearly defined
- [ ] Task specifically described
- [ ] Input variables documented
- [ ] Output format specified with example
- [ ] Edge cases handled
- [ ] Safety constraints included
- [ ] Protected sections marked (if needed)
- [ ] Tested with diverse inputs
- [ ] Evolution constraints set

---

## See Also

- [First Custom Pipeline](first-custom-pipeline.md) — Build a complete pipeline
- [Multi-Agent Patterns](../concepts/multi-agent-patterns.md) — Agent design patterns
- [Mutation Operators](../reference/mutation-operators.md) — How prompts evolve
- [Evolution Lifecycle](../concepts/evolution-lifecycle.md) — The evolution process
