# 📘 Approach Explanation: Persona-Driven Document Intelligence

Our system is engineered to function as an intelligent document analyst, adept at pinpointing and prioritizing information from a PDF collection that is most relevant to a specific user persona and their designated task. The core of our approach lies in a multi-stage process that combines classic NLP techniques with modern transformer models to deeply understand both the user's query and the documents' content.

---

## 1. 🎯 Dynamic and Context-Aware Query Generation

We use a multi-step pipeline to build a powerful search query based on the persona and task:

- **Lemmatization & Stop-word Removal**:  
  Using **spaCy**, we clean the persona and job description to extract meaningful seed keywords.

- **Vocabulary Expansion**:  
  We find semantically similar terms using word vectors (spaCy) and expand the keyword set.

- **TF-IDF Scoring**:  
  A `TfidfVectorizer` from **scikit-learn** is run on the full PDF corpus to extract statistically significant terms.

- **Query Weighting**:  
  We prioritize:
  - Terms appearing in both persona and high-TFIDF list (highest weight)
  - Persona/task-only terms
  - Other TF-IDF keywords  
  This weighted set forms the final **context-aware dynamic query**.

> **Functions Involved**:  
> `generate_dynamic_query(persona_text: str, task_text: str, corpus: List[str]) -> str`

---

## 2. 📄 Hierarchical Content Extraction from PDFs

We parse PDF files using **PyMuPDF** (`fitz`) and rebuild a logical hierarchy:

- **Header/Footer Removal**:  
  Text blocks occurring consistently across pages are ignored using frequency heuristics.

- **Font Size Baseline**:  
  A median font size is computed per document to set the “body” standard.

- **Heuristic Heading Detection**:  
  Headings are identified based on:
  - Font size relative to baseline
  - Boldness or font family
  - Short length

- **Content Grouping**:  
  When a heading is detected, it's marked as a new `section`, and all following text blocks are grouped under it as a `subsection`.

> **Functions Involved**:  
> - `parse_pdf_to_sections(pdf_path: str) -> List[Dict]`  
> - `is_heading(block, baseline_font_size) -> bool`  
> - `group_subsections(blocks: List[Dict]) -> List[Section]`

---

## 3. ⚖️ Two-Stage Hybrid Ranking Model

### 3.1 Section Ranking via Cross-Encoder
- From each document, we extract the top 5 structurally significant sections (by heading font size and page location).
- These are then reranked using a **CrossEncoder**:  
  `cross-encoder/ms-marco-MiniLM-L-6-v2`
- It compares each section title to the query for relevance.

> **Function**:  
> `rank_sections_crossencoder(query: str, section_titles: List[str]) -> List[Tuple[str, float]]`

---

### 3.2 Subsection Ranking via Semantic Search
- All content chunks (subsections) are embedded using a **bi-encoder**:  
  `sentence-transformers/all-MiniLM-L6-v2`
- We compute cosine similarity between the query and each embedded subsection.
- Top-K most relevant subsections are returned.

> **Function**:  
> `semantic_search(query: str, subsection_texts: List[str]) -> List[Tuple[str, float]]`

---

## 🧩 Tech Stack Used

| Component                         | Library / Tool                               |
|----------------------------------|----------------------------------------------|
| PDF Parsing                      | PyMuPDF (`fitz`)                             |
| NLP (tokenization, lemmatization)| spaCy                                        |
| Keyword Extraction               | TF-IDF, spaCy                                |
| Word Embeddings                  | spaCy, SentenceTransformers                  |
| Sentence Encoding (Semantic Search) | `sentence-transformers/all-MiniLM-L6-v2` |
| Cross-Encoder Ranking            | `cross-encoder/ms-marco-MiniLM-L-6-v2`       |
| Cosine Similarity                | scikit-learn                                 |
| JSON Output                      | `json`, `pydantic` or native Python `dict`   |

---

## 🧪 System Constraints Met

| Constraint                                     | Status |
|------------------------------------------------|--------|
| ✅ CPU-only inference                          | ✔️     |
| ✅ Model size under 1GB                        | ✔️     |
| ✅ Process 3–5 PDFs in <60 seconds             | ✔️     |
| ✅ JSON output with section metadata           | ✔️     |

---

## 🧾 Output Structure

The system outputs structured JSON like:

```json
{
  "metadata": {
    "input_documents": ["list"],
    "persona": "User Persona",
    "job_to_be_done": "Task description"
  },
  "extracted_sections": [
    {
      "document": "source.pdf",
      "section_title": "Title",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "source.pdf",
      "refined_text": "Content",
      "page_number": 1
    }
  ]
}
