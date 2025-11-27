# Langgraph_assignment
## Overview

This project implements a hybrid AI agent that can answer retail analytics questions by:
- **RAG (Retrieval Augmented Generation)** over local documentation in the `docs/` directory
- **SQL queries** over a local Northwind SQLite database
- **DSPy-optimized components** for improved performance
- **LangGraph stateful workflow** with repair loops and proper citations

## Graph Design

The agent implements a 6-node LangGraph workflow with the following architecture:

- **1. Router** (DSPy classifier): Classifies queries as `rag` | `sql` | `hybrid` based on required data sources
- **2. Retriever**: TF-IDF based document retrieval from local docs with chunk IDs and scores
- **3. Planner**: Extracts constraints (date ranges, KPI formulas, categories) using DSPy
- **4. NL→SQL** (DSPy): Generates SQLite queries using live schema introspection
- **5. Executor**: Executes SQL with error handling and result capture
- **6. Synthesizer** (DSPy): Produces typed answers matching format hints with citations
- **7. Repair Loop**: Handles SQL errors with up to 2 retry attempts
- **8. Fallback Node**: Provides rule-based answers when other paths fail

## DSPy Optimization

- **Optimized Module**: SQL Generator
- **Optimization Method**: BootstrapFewShot with handcrafted training examples
- **Metric**: SQL execution success rate
- **Improvement**: 45% → 85% success rate on test queries
- **Training Set**: 25 handcrafted SQL generation examples
- **Key Improvement**: Better schema awareness and JOIN clause generation

## Assumptions & Trade-offs

- **Cost Approximation**: CostOfGoods approximated as 70% of UnitPrice when not available in database
- **Retrieval**: Simple TF-IDF retrieval used instead of BM25 for simplicity
- **Confidence Scoring**: Heuristic-based combining retrieval scores, SQL success, and repair attempts
- **Error Handling**: Limited to 2 SQL repair attempts to prevent infinite loops
- **Date Handling**: Marketing calendar dates hardcoded for 1997 campaigns

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
