# Research-Work RAG Comparison Project

This repository contains a small research workflow for comparing two retrieval-augmented generation (RAG) indexing strategies over an extracted copy of the Flask codebase.

The core idea is simple:

1. Build a baseline RAG vector store using naive character-based chunking.
2. Build a second RAG vector store using Python-aware semantic chunking.
3. Run the same set of code-focused questions against both stores and compare the retrieved source snippets.

The project is organized around a paper-style experiment rather than an end-user application. The included outputs, presentation, and research paper are supporting artifacts for that comparison.

## What Is In This Repo

- `Python-Files/` contains the extracted Python source files from the Flask project.
- `1_baseline_pipeline.py` builds a Chroma database using fixed-size text chunks.
- `2_semantic_pipeline.py` builds a second Chroma database using Python-aware chunking.
- `3_evaluate.py` runs the same query set against both databases and prints the nearest retrieved chunk.
- `output_1_baseline_pipeline.txt`, `output_2_semantic_pipeline.txt`, and `output_3_evaluate.txt` capture example runs of each script.
- `chroma_db_baseline/` and `chroma_db_semantic/` are the persisted vector stores created by the two pipelines.
- `Team_32_Research_Paper.docx` and `RAG_Semantic_Chunking_Presentation.pptx` are the supporting deliverables for the research write-up and presentation.

## Research Goal

The goal of the experiment is to see whether semantic, structure-aware chunking improves retrieval quality for source-code questions compared with a simple baseline splitter.

This is especially relevant for code retrieval because naive chunk boundaries can cut through method bodies, class definitions, or docstrings, while language-aware splitting is more likely to keep related code together.

## Data Set

The indexed corpus is the `Python-Files/` directory, which contains extracted Flask source files. The files include the main application modules plus many test and support modules from the Flask source tree.

The evaluation questions are also code-centric and ask about specific Flask internals such as:

- `get_send_file_max_age`
- `open_resource`
- `create_url_adapter`
- `default_config`
- `send_static_file`
- `copy_current_request_context`
- `_AppCtxGlobals`
- `AppContext.pop()`

These questions are designed to test whether the retriever can return the most relevant source file and a coherent chunk of code.

## Pipelines

### 1. Baseline Pipeline

`1_baseline_pipeline.py` loads all `*.py` files under `Python-Files/` and splits them using `RecursiveCharacterTextSplitter` with:

- `chunk_size=1000`
- `chunk_overlap=200`

This is the intentionally simple control condition. It does not understand Python syntax, so chunks may break across functions or classes.

The chunks are embedded with `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")` and persisted to `./chroma_db_baseline`.

### 2. Semantic Pipeline

`2_semantic_pipeline.py` loads the same source files, but uses `RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, ...)`.

This makes the splitter more code-aware, so it prefers boundaries that align with Python structure such as classes and functions before falling back to character-based splitting.

The chunks are embedded with the same model and persisted to `./chroma_db_semantic`.

### 3. Evaluation Script

`3_evaluate.py` reconnects to both Chroma databases and runs a fixed list of test questions.

For each query, it prints:

- the question
- the top result from the baseline database
- the top result from the semantic database
- the matched file path
- a short content snippet from the retrieved chunk

The script uses `similarity_search(query, k=1)`, so the comparison is focused on the single most relevant hit from each index.

## Dependencies

The environment is based on the packages listed in `requirements.txt`. The key libraries are:

- `langchain`
- `langchain-community`
- `langchain-chroma`
- `langchain-huggingface`
- `langchain-text-splitters`
- `chromadb`
- `sentence-transformers`
- `torch`
- `transformers`

The recorded runs show that the embedding model downloads from Hugging Face the first time it is used.

## Setup

1. Create and activate a virtual environment if one is not already present.
2. Install the dependencies from `requirements.txt`.
3. Make sure the `Python-Files/` directory is present in the repository root.

If you are using the same workflow as the recorded outputs, the scripts are run with `uv`:

```powershell
uv run .\1_baseline_pipeline.py
uv run .\2_semantic_pipeline.py
uv run .\3_evaluate.py
```

## Reproducing the Experiment

Run the scripts in this order:

1. Build the baseline vector store.
2. Build the semantic vector store.
3. Run the evaluation script to compare retrieval results.

If you delete either `chroma_db_baseline/` or `chroma_db_semantic/`, rerun the matching build script before launching the evaluation step.

## Expected Behavior

From the captured outputs, the repository currently produces the following build statistics:

- Baseline pipeline: 83 Python files, 804 chunks
- Semantic pipeline: 861 chunks

Both pipelines use the same embedding model, `all-MiniLM-L6-v2`, and both persist their results locally in Chroma.

The evaluation output shows that both pipelines can often recover the same source file for a question, but the semantic pipeline is generally better aligned with code structure. For example, in the sample output it retrieves `Python-Files/app.py` for method-level questions where the baseline can sometimes land on a less ideal chunk.

## Sample Output Notes

The recorded runs show a Hugging Face warning about unauthenticated requests. That is expected when no `HF_TOKEN` is configured.

You may also see a model load report mentioning `embeddings.position_ids` as `UNEXPECTED`. In the recorded output this is treated as informational and does not stop the pipeline.

## Repository Layout

```text
.
├── 1_baseline_pipeline.py
├── 2_semantic_pipeline.py
├── 3_evaluate.py
├── Python-Files/
├── chroma_db_baseline/
├── chroma_db_semantic/
├── output_1_baseline_pipeline.txt
├── output_2_semantic_pipeline.txt
├── output_3_evaluate.txt
├── requirements.txt
├── RAG_Semantic_Chunking_Presentation.pptx
└── Team_32_Research_Paper.docx
```

## Practical Notes

- The repository is Windows-friendly and the captured terminal output shows the scripts being run from PowerShell.
- The `Python-Files/` directory includes many extracted Flask modules, tests, and support files. Some filenames have numeric suffixes such as `app-1.py` or `views-2.py`, which indicates duplicate extraction names rather than separate hand-written modules.
- The persisted Chroma directories are local artifacts. If you want a clean rebuild, delete them and rerun the pipeline scripts.

## Why This Setup Matters

This project is a compact demonstration of a common retrieval problem in code intelligence: the retrieval model may be strong enough to find the right topic, but the quality of the chunking strategy determines whether it returns a useful source span.

The baseline index is useful as a control, while the semantic index is the more realistic code-oriented approach. The evaluation script makes the comparison easy to inspect without requiring a full generation step.
