# Nokia 1830 PSS — RAG Pipeline

A **Retrieval-Augmented Generation (RAG)** system built around the **Nokia 1830 PSS Technical Description** document.

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

* **PDF extraction** using `pypdf`
* Semantic document chunking with configurable chunk sizes
* Hierarchical section detection
* Header/footer and document-artifact removal
* Unicode and mojibake cleanup
* Rich chunk metadata including:

  * Chapter
  * Section
  * Parent section
  * Page range
  * Word count
  * Nokia shelf/model tags
* **Dense semantic retrieval** using `all-MiniLM-L6-v2`
* **TF-IDF keyword retrieval**
* Hybrid semantic + keyword ranking
* Equipment identifier boosting for terms such as `FAN32H` and `8DC30`
* Metadata filtering by shelf or section
* Optional **FAISS** vector search
* Persistent embeddings with automatic version/hash checking
* Gemini-powered grounded generation
* Explicit **source-page citations**
* "Not found" behavior when information cannot be retrieved from the document
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

---

# Usage

All modules should be executed as Python packages from the **project root**.

---

## 1. Chunk the PDF

Extracts pages **47–166**, cleans the extracted text, detects section boundaries, and generates chunks.

```bash
python -m src.chunker
```

### Output

```text
data/extracted_chunks.json
```

Each chunk contains metadata such as:

```text
- Chapter
- Section
- Parent section
- Page range
- Shelf tags
- Word count
- Chunk text
```

---

## 2. Build the Search Index

Generate embeddings and build the retrieval index:

```bash
python -m src.indexer
```

The indexer automatically detects whether the existing embeddings are still valid and avoids unnecessary re-embedding when the source chunks have not changed.

### Output

```text
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
5. Generates a grounded answer.
6. Adds source-page citations.

Example response format:

```text
The PSS-32 shelf supports ... (Source: Page 83)
```

If the required information cannot be found:

```text
Not found in the provided document.
```

---

## 4. Evaluate the Pipeline

Run the predefined evaluation set:

```bash
python -m src.evaluate
```

The evaluator runs **8 technical questions** through the complete RAG pipeline and displays the generated answers in a formatted table.

> **Note:** The evaluator includes a 60-second delay between questions to respect API rate limits.

---

# RAG Pipeline Details

## Chunker

`src/chunker.py`

The chunker processes pages **47–166**, corresponding to the relevant chapters of the Nokia 1830 PSS technical document.

### Processing steps

```text
PDF
 │
 ▼
Page extraction
 │
 ▼
Text cleaning
 │
 ├── Remove headers/footers
 ├── Remove document IDs
 ├── Remove copyright notices
 └── Fix Unicode/mojibake issues
 │
 ▼
Section detection
 │
 ├── Chapter
 ├── Section
 ├── Subsection
 └── Sub-subsection
 │
 ▼
Semantic chunking
 │
 ├── Heading boundaries
 ├── Paragraph boundaries
 └── Sentence boundaries
 │
 ▼
Metadata enrichment
 │
 ▼
extracted_chunks.json
```

Chunks are constrained by configurable word limits.

Default:

```text
Minimum: 80 words
Maximum: 300 words
```

The chunker also merges undersized chunks with neighboring chunks to avoid creating fragmented context.

Nokia equipment identifiers and shelf models such as:

```text
PSS-32
PSS-16II
PSS-8
FAN32H
8DC30
```

are extracted and stored as metadata where applicable.

---

## Indexer

`src/indexer.py`

The indexer uses:

```text
all-MiniLM-L6-v2
```

from `sentence-transformers` to generate dense vector representations of the chunks.

### Augmented Embeddings

Before embedding, the chunk content is augmented with relevant metadata such as section hierarchy and page information.

This helps the embedding model capture both the **technical content** and its **document context**.

### Hybrid Retrieval

The search engine combines two retrieval signals:

```text
Hybrid Score
     │
     ├── Dense semantic similarity
     │
     └── TF-IDF keyword similarity
```

The combined score is controlled by `alpha`:

```text
(1 − α) × dense_score + α × keyword_score
```

Default:

```text
α = 0.3
```

This gives greater weight to semantic similarity while still benefiting from exact keyword matching.

### Equipment Identifier Boosting

Technical documents frequently contain highly specific identifiers.

For example:

```text
FAN32H
8DC30
PSS-32
PSS-16II
```

When a query contains one of these identifiers, matching chunks receive an additional score boost.

This improves retrieval for equipment-specific questions where exact identifier matching is particularly important.

### Metadata Filtering

Retrieval can optionally be restricted using metadata such as:

* Shelf model
* Section name
* Equipment identifier

---

## FAISS Comparison

The project also supports optional **FAISS `IndexFlatIP`** retrieval.

FAISS can be used to compare traditional dense-vector retrieval against the custom hybrid retrieval approach.

Because the embeddings are normalized, inner-product similarity corresponds to cosine similarity.

---

## Index Persistence

The indexer stores metadata describing the current embedding state:

```text
data/index_meta.json
```

The metadata includes information such as:

* Chunk hash
* Embedding model
* Embedding dimensions

If the source chunks have not changed, the existing embeddings are reused instead of being generated again.

---

# Generator

`src/generator.py`

The generator retrieves the **top 15 chunks** for each question.

The retrieved chunks are then formatted into a context containing:

```text
Page number
Section information
Chunk text
```

Google Gemini receives this context together with a strict grounding prompt.

### Generation Rules

The model is instructed to:

1. Answer **only** using the retrieved document context.
2. Avoid introducing information from outside the document.
3. Include source-page citations.
4. Clearly indicate when the answer cannot be found.

Expected citation format:

```text
(Source: Page 102)
```

When the required information is unavailable:

```text
Not found in the provided document.
```

This reduces unsupported or hallucinated responses.

---

# Evaluation

`src/evaluate.py`

The evaluation pipeline contains **8 predefined technical questions** covering areas such as:

* Nokia shelf specifications
* Fan units
* Power filters
* Software load-lines
* Rack apertures
* Optical reach
* Equipment-specific specifications

Each question is processed through:

```text
Question
   │
   ▼
Hybrid Retrieval
   │
   ▼
Top-15 Chunks
   │
   ▼
Gemini
   │
   ▼
Grounded Answer
   │
   ▼
Page Citation
```
# Evaluation

`src/evaluate.py`

The evaluation pipeline contains **8 predefined technical questions** covering different aspects of the Nokia 1830 PSS technical documentation.

Each question is processed through the complete RAG pipeline:

```text
Question
   │
   ▼
Hybrid Retrieval
   │
   ▼
Top-15 Chunks
   │
   ▼
Gemini
   │
   ▼
Grounded Answer
   │
   ▼
Page Citation
```

## Questions & Results

### 1. 1830 PSS-8 Shelf Capacity

**Question:**

> How many slots does the 1830 PSS-8 shelf provide, and what is its rack-unit (RU) footprint?

**Answer:**

> The 1830 PSS-8 shelf provides an **8-slot SWDM platform** in a **3-RU footprint**.
> *(Source: Page 82)*

---

### 2. 1830 PSS-32 Shelf Capacity

**Question:**

> What rack-unit footprint does the 1830 PSS-32 shelf have, and how many slots does it provide?

**Answer:**

> The 1830 PSS-32 shelf has a **14-RU footprint** and provides a **32-slot core platform**.
> *(Source: Page 96)*

> The platform consists of 32 paired half-height application cards (or 16 full-height application cards), 2 equipment controllers (EC), 2 redundant power input filters, 1 user panel, and 1 fan unit.
> *(Source: Pages 96–98)*

---

### 3. Software Load-Lines

**Question:**

> What are the two software load-lines supported by the 1830 PSS system?

**Answer:**

> The two software load-lines supported by the 1830 PSS system are **SWDM software** and **OCS software**.
> *(Source: Page 47)*

---

### 4. PSS-32 Fan Units

**Question:**

> Which fan units are supported on the 1830 PSS-32 shelf?

**Answer:**

> The fan units supported on the 1830 PSS-32 shelf are **FAN** and **FAN32H**.
> *(Source: Pages 153–154)*

---

### 5. PSS-16II Fan Units

**Question:**

> Which fan unit(s) are used on the 1830 PSS-16II shelf?

**Answer:**

> The fan units used on the 1830 PSS-16II shelf include:
>
> * **16FAN2**
> * **16FAN2C** (PN 3KC49100AA)
> * **AB fan tray** (3KC48990AB), which equips the 1830 PSS-16II AB shelf (3KC48960AB)
>
> *(Source: Pages 149–150)*

---

### 6. PSS-8 Power Filter Cards

**Question:**

> Name the power filter cards supported on the 1830 PSS-8 shelf.

**Answer:**

> The power filter cards supported on the 1830 PSS-8 shelf are:
>
> * **8DC30**
> * **8DC30T**
> * **8DC30T2**
> * **8AC7**
>
> *(Source: Pages 158–160)*

---

### 7. PSS-8 Rack Aperture

**Question:**

> What is the required horizontal rack aperture for mounting a 1830 PSS-8 shelf, and which common aperture size is explicitly NOT supported?

**Answer:**

> The required horizontal rack aperture for mounting a 1830 PSS-8 shelf is **450.85 mm (17.75 in)**. Racks with a horizontal aperture of **444.5 mm (17.5 in)** are explicitly **not supported**.
> *(Source: Pages 82–83)*

---

### 8. PSS-8 Maximum Optical Reach

**Question:**

> What is the maximum optical reach, in kilometers, of the 1830 PSS-8 shelf without amplification?

**Answer:**

> **Not found in the provided document.**

This question also tests the system's **grounding and hallucination prevention** behavior. When the required information is not available in the retrieved context, the generator is expected to return:


# Project Structure

```text
NRAG_venv/
│
├── data/
│   ├── 1830_Technical_Description.pdf
│   ├── extracted_chunks.json
│   ├── embeddings.npy
│   └── index_meta.json
│
├── src/
│   ├── chunker.py
│   ├── indexer.py
│   ├── generator.py
│   └── evaluate.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
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

---

# Example Workflow

From a clean project:

```bash
# 1. Extract and chunk the document
python -m src.chunker

# 2. Build the retrieval index
python -m src.indexer

# 3. Ask questions interactively
python -m src.generator

# 4. Run the evaluation suite
python -m src.evaluate
```

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

Pure semantic search is effective for conceptually similar questions, but technical documentation often contains exact identifiers, model numbers, and specifications.

Combining:

```text
Semantic similarity
        +
Keyword similarity
        +
Equipment identifier boosting
```

provides more robust retrieval for both conceptual and highly specific technical queries.

### Why Page-Level Metadata?

The system is designed for technical documentation where answers need to be traceable back to the original source.

Keeping page metadata throughout the pipeline allows the final LLM response to provide citations such as:

```text
(Source: Page 124)
```

rather than producing an answer without provenance.

### Why a Strict Grounding Prompt?

The generator is explicitly instructed to use only retrieved context and to return:

```text
Not found in the provided document.
```

when sufficient evidence is unavailable.

This prioritizes **faithfulness and traceability** over generating an answer at all costs.

---
