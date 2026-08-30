# Nokia 1830 PSS — RAG Pipeline

# A Retrieval-Augmented Generation (RAG) system built around the Nokia 1830 PSS Technical Description document.

The pipeline extracts and cleans the technical document, creates semantically meaningful chunks, builds a hybrid retrieval index, and uses **Google Gemini** to generate grounded answers with **source-page citations**.

---

## Architecture

```text
┌─────────────────────┐
│  Technical PDF      │
│  Nokia 1830 PSS     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      Chunker        │
│ Extract & clean     │
│ Detect sections     │
│ Create metadata     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      Indexer        │
│ Embeddings + TF-IDF │
│ Hybrid retrieval    │
│ Metadata filtering  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     Generator       │
│ Retrieve top-k      │
│ Build context       │
│ Query Gemini        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│       Answer        │
│ Grounded response   │
│ + page citations    │
└─────────────────────┘
```

### Pipeline Components

| Module             | Responsibility                                                                                                                          |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `src/chunker.py`   | Extracts pages 47–166, cleans document artifacts, detects section hierarchy, and creates semantically coherent chunks with metadata.    |
| `src/indexer.py`   | Generates embeddings and implements hybrid retrieval using dense similarity, TF-IDF keyword scoring, and equipment-identifier boosting. |
| `src/generator.py` | Retrieves relevant chunks and uses Google Gemini to generate grounded answers with page citations.                                      |
| `src/evaluate.py`  | Runs a predefined evaluation set of 8 technical questions through the complete RAG pipeline.                                            |

---

## Features

* PDF extraction using `pypdf`
* Semantic document chunking with configurable chunk sizes
* Hierarchical section detection
* Header/footer and document-artifact removal
* Unicode and mojibake cleanup
* Rich chunk metadata (chapter, section, page range, word count, shelf/model tags)
* Dense semantic retrieval using `all-MiniLM-L6-v2`
* TF-IDF keyword retrieval and hybrid ranking
* Equipment identifier boosting (e.g., `FAN32H`, `8DC30`)
* Metadata filtering and optional FAISS vector search
* Persistent embeddings with automatic version/hash checking
* Gemini-powered grounded generation with explicit source-page citations
* Built-in evaluation pipeline

---

## Requirements

* **Python 3.10+**
* Nokia 1830 PSS Technical Description PDF
* Google Gemini API key

The source document should be placed at:

```text
data/1830_Technical_Description.pdf
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd NRAG_venv
```

### 2. Create a virtual environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

| Package                          | Purpose                         |
| -------------------------------- | ------------------------------- |
| `pypdf >= 6.0.0`                 | PDF text extraction             |
| `sentence-transformers >= 3.0.0` | Text embeddings                 |
| `faiss-cpu >= 1.8.0`             | Optional vector search          |
| `google-genai >= 1.45.0`         | Google Gemini API               |
| `python-dotenv >= 1.0.0`         | Environment variable management |

---

## Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.6-flash
```

`GEMINI_MODEL` is optional and defaults to the configured Gemini model in the generator.

> **Security:** Never commit your `.env` file or API keys to GitHub.

Add the following to `.gitignore`:

```gitignore
.env
venv/
__pycache__/
*.pyc
```

```bash
python -m src.chunker
```

**Output:** `data/extracted_chunks.json` — an array of chunks with metadata (chapter, section, parent section, page range, shelf tags, word count, chunk text).

Each chunk contains metadata such as:

- Chapter
- Section
- Parent section
- Page range
- Shelf tags
- Word count
- Chunk text

---

### Step 2 — Build the Vector Index

Generate embeddings and build the retrieval index:

```bash
python -m src.indexer
```

The indexer automatically detects whether existing embeddings are valid and avoids unnecessary re-embedding when the source chunks have not changed.

**Output:**

```
data/embeddings.npy
data/index_meta.json
```

The module also runs built-in retrieval tests against the 8 evaluation questions and displays the top retrieved chunks.

---

## 3. Generate Answers

Run the complete RAG pipeline:

```bash
python -m src.generator
```

The generator:

1. Receives a user question.
2. Retrieves the most relevant chunks.
3. Builds a context containing the retrieved text and page metadata.
4. Sends the context to Google Gemini.
5. Generates a grounded answer and appends source-page citations.

Example response format:

```text
The PSS-32 shelf supports ... (Source: Page 83)
```

If the required information cannot be found:

```text
Not found in the provided document.
```

---

### Step 4 — Evaluate the Pipeline

Runs all 8 evaluation questions through the full RAG pipeline and prints a formatted results table.

```bash
python -m src.evaluate
```

> **Note:** There is a 60-second delay between questions to respect API rate limits.

---

## Project Structure

```
NRAG_venv/
├── data/
│   ├── 1830_Technical_Description.pdf   # Source PDF (not tracked in git)
│   ├── extracted_chunks.json            # Generated by chunker
│   ├── embeddings.npy                   # Generated by indexer
│   └── index_meta.json                  # Generated by indexer
├── src/
│   ├── chunker.py                       # PDF extraction & chunking
│   ├── indexer.py                       # Embedding & hybrid search
│   ├── generator.py                     # RAG answer generation (Gemini)
│   └── evaluate.py                      # Batch evaluation runner
├── .env                                 # API keys (create manually)
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

### File Descriptions

| File                    | Description                                                                       |
| ----------------------- | --------------------------------------------------------------------------------- |
| `chunker.py`            | PDF extraction, cleaning, section detection, and chunk generation                 |
| `indexer.py`            | Embedding generation, hybrid retrieval, metadata filtering, and index persistence |
| `generator.py`          | Retrieval + Gemini-based answer generation                                        |
| `evaluate.py`           | Automated evaluation over predefined questions                                    |
| `extracted_chunks.json` | Generated chunk dataset with metadata                                             |
| `embeddings.npy`        | Persisted normalized embeddings                                                   |
| `index_meta.json`       | Index version and model metadata                                                  |
```

---

## Pipeline Details

### Chunker (`src/chunker.py`)
- Processes pages **47–166** (Chapters 1–2).
- Cleans headers, footers, document IDs, copyright notices, and Unicode mojibake.
- Detects hierarchical section headings (chapter → section → subsection → sub-subsection).
- Splits text at heading boundaries first, then paragraphs, then sentences, with configurable word limits (default: 80–300 words per chunk).
- Merges undersized chunks with neighbors to avoid fragments.
- Tags chunks with Nokia shelf model identifiers (PSS-32, PSS-16II, PSS-8, etc.).

### Indexer (`src/indexer.py`)
- Uses **`all-MiniLM-L6-v2`** sentence-transformer for embeddings.
- Prepends section hierarchy and page metadata to chunk text before encoding (**augmented embeddings**).
- Implements **hybrid search**: `(1 − α) × dense_cosine + α × keyword_TF-IDF` with configurable alpha (default 0.3).
- **Equipment-identifier boosting**: queries mentioning specific Nokia equipment IDs (e.g., `FAN32H`, `8DC30`) get a heavy score boost on matching chunks.
- Supports **metadata filtering** by shelf tag or section name.
- Optional **FAISS `IndexFlatIP`** for comparison benchmarks.
- **Persistence with version hashing**: auto-rebuilds embeddings only when chunks change.

### Generator (`src/generator.py`)
- Retrieves the **top-15 chunks** via hybrid search.
- Builds a context prompt with page numbers for each chunk.
- Calls **Google Gemini** with a strict system prompt that enforces:
  - Answer only from provided context.
  - Append `(Source: Page [Number])` citations.
  - Reply `"Not found in the provided document."` when the answer isn't in context.

### Evaluator (`src/evaluate.py`)
- Runs **8 predefined questions** covering shelf specs, fan units, power filters, software load-lines, rack apertures, and optical reach.
- Prints results in a numbered table format.
- Includes a 60-second delay between questions for API rate-limit compliance.

---

# Technologies

* **Python**
* **pypdf**
* **Sentence Transformers**
* **FAISS**
* **NumPy**
* **Google Gemini**
* **TF-IDF**
* **Hybrid Information Retrieval**
* **Retrieval-Augmented Generation (RAG)**

---

# Key Design Decisions

### Why Hybrid Retrieval?

Pure semantic search is effective for conceptually similar questions, but technical documentation often contains exact identifiers, model numbers, and specifications. Combining semantic similarity, keyword matching, and equipment identifier boosting yields robust retrieval for both conceptual and highly specific technical queries.

### Why Page-Level Metadata?

Keeping page metadata through the pipeline allows final LLM responses to include provenance such as `(Source: Page 124)`, improving traceability.

### Why a Strict Grounding Prompt?

The generator is instructed to use only retrieved context and to return `Not found in the provided document.` when sufficient evidence is unavailable. This prioritizes faithfulness and traceability over speculative answers.
