import os
import json
import re
import fitz
import spacy
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

CONFIG = {
    "TOP_K_SECTIONS": 5,
    "TOP_K_SUBSECTIONS": 5,
    "BOLD_FLAG_MASK": 1 << 4,
    "MAX_WORKERS": os.cpu_count() or 4,
    "HEADER_FOOTER_MARGIN": 50,
    "HEADING_MAX_WORDS": 25,
}

class PDFProcessor:
    def __init__(self, persona: str, task: str, config: Dict[str, Any]):
        self.config = config
        self.persona = persona
        self.task = task
        bi_encoder_path = './local_models/bi-encoder'
        cross_encoder_path = './local_models/cross-encoder'
        try:
            self.semantic_model = SentenceTransformer(bi_encoder_path)
            self.cross_encoder = CrossEncoder(cross_encoder_path)
            self.nlp = spacy.load("en_core_web_md")
        except Exception as e:
            exit()
        self.query_text = self._build_query_text()

    def _build_query_text(self) -> str:
        all_texts = []
        pdf_dir = Path("input")
        for pdf_file in pdf_dir.glob("*.pdf"):
            try:
                with fitz.open(pdf_file) as doc:
                    all_texts.append(" ".join(page.get_text() for page in doc))
            except Exception as e:
                pass
        
        if not all_texts: return f"{self.persona} {self.task}"

        prompt_text = f"{self.persona}. {self.task}"
        prompt_doc = self.nlp(prompt_text)
        seed_keywords = {token.lemma_.lower() for token in prompt_doc if not token.is_stop and not token.is_punct and token.has_vector}

        corpus_doc = self.nlp(" ".join(all_texts))
        corpus_vocab = {token.lemma_.lower() for token in corpus_doc if not token.is_stop and not token.is_punct and token.has_vector and len(token.text) > 2}

        generated_keywords = set()
        for seed_token in self.nlp(" ".join(seed_keywords)):
            if not seed_token.has_vector or seed_token.vector_norm == 0: continue
            for vocab_token_text in corpus_vocab:
                vocab_token = self.nlp(vocab_token_text)
                if not vocab_token.has_vector or vocab_token.vector_norm == 0: continue
                if seed_token.similarity(vocab_token) > 0.7:
                    generated_keywords.add(vocab_token_text)
        
        combined_keywords = seed_keywords.union(generated_keywords)

        vectorizer = TfidfVectorizer(vocabulary=list(combined_keywords), use_idf=True, norm=None)
        try:
            vectorizer.fit(all_texts)
            idf_scores = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
            min_idf = np.percentile(list(idf_scores.values()), [25])[0] if idf_scores else 1
            important_keywords = {word for word, score in idf_scores.items() if score >= min_idf}
        except ValueError:
             important_keywords = seed_keywords

        final_query_parts = []
        for word in seed_keywords:
            if word in important_keywords: final_query_parts.extend([word] * 3)
        for word in important_keywords:
            if word not in seed_keywords: final_query_parts.extend([word] * 2)
        if not final_query_parts: final_query_parts = list(seed_keywords)
        final_query = " ".join(final_query_parts)
        return final_query

    def _detect_headers_footers(self, doc: fitz.Document) -> Tuple[set, set]:
        headers, footers = set(), set()
        try:
            header_candidates, footer_candidates = defaultdict(int), defaultdict(int)
            margin = self.config['HEADER_FOOTER_MARGIN']
            page_threshold = min(2, doc.page_count // 2) if doc.page_count > 3 else 1
            if page_threshold > 0:
                for page in doc:
                    for block in page.get_text("blocks", clip=(0, 0, page.rect.width, margin)):
                        if text := block[4].strip(): header_candidates[text] += 1
                    for block in page.get_text("blocks", clip=(0, page.rect.height - margin, page.rect.width, page.rect.height)):
                        if text := block[4].strip(): footer_candidates[text] += 1
                headers = {text for text, count in header_candidates.items() if count >= page_threshold and len(text.split()) < 10}
                footers = {text for text, count in footer_candidates.items() if count >= page_threshold and len(text.split()) < 10}
        except Exception as e:
            pass
        return (headers, footers)

    def _get_body_font_size(self, doc: fitz.Document, headers: set, footers: set) -> float:
        sizes = Counter()
        for page in doc:
            for block in page.get_text("dict").get("blocks", []):
                block_text = "".join(span["text"] for line in block.get("lines", []) for span in line.get("spans", [])).strip()
                if block.get("type") == 0 and block_text not in headers and block_text not in footers:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            sizes[round(span["size"])] += 1
        return sizes.most_common(1)[0][0] if sizes else 10.0

    def _extract_content_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            return []
        headers, footers = self._detect_headers_footers(doc)
        body_size = self._get_body_font_size(doc, headers, footers)
        extracted_content = []
        current_subsection_text = []
        current_heading_info = None
        for page_num, page in enumerate(doc, 1):
            blocks = sorted(page.get_text("dict").get("blocks", []), key=lambda b: b['bbox'][1])
            for block in blocks:
                if block.get("type") != 0 or not block.get("lines"): continue
                spans = [span for line in block.get("lines", []) for span in line.get("spans", [])]
                if not spans: continue
                block_text = " ".join(s["text"] for s in spans).strip()
                if not block_text or block_text in headers or block_text in footers: continue
                first_span = spans[0]
                font_size = round(first_span["size"])
                is_bold = (first_span["flags"] & self.config["BOLD_FLAG_MASK"]) > 0
                word_count = len(block_text.split())
                is_heading = (
                    (font_size > body_size or (is_bold and font_size >= body_size)) and
                    (word_count <= self.config["HEADING_MAX_WORDS"]) and
                    (not block_text.strip().startswith(('•', '*', '-', '–'))) and
                    (not block_text.endswith(('.', ':', ';'))) and
                    (not re.match(r'^\d+\.', block_text.strip()))
                )
                if is_heading:
                    if current_heading_info and current_subsection_text:
                        extracted_content.append({"type": "subsection", "text": " ".join(current_subsection_text), "document": pdf_path.name, "page_number": current_heading_info["page"]})
                    current_subsection_text = []
                    current_heading_info = {"title": block_text, "page": page_num, "style": (font_size, is_bold)}
                    extracted_content.append({"type": "section", "text": block_text, "document": pdf_path.name, "page_number": page_num, "style": (font_size, is_bold)})
                elif current_heading_info:
                    current_subsection_text.append(block_text)
        if current_heading_info and current_subsection_text:
            extracted_content.append({"type": "subsection", "text": " ".join(current_subsection_text), "document": pdf_path.name, "page_number": current_heading_info["page"]})
        doc.close()
        return extracted_content

    def run(self, pdf_dir: Path, files: List[str]) -> Dict[str, Any]:
        all_content = []
        with ThreadPoolExecutor(max_workers=self.config["MAX_WORKERS"]) as executor:
            pdf_paths = [pdf_dir / f for f in files if (pdf_dir / f).exists() and f.lower().endswith('.pdf')]
            results = list(executor.map(self._extract_content_from_pdf, pdf_paths))
        for content_list in results: all_content.extend(content_list)
        if not all_content: return {}
        sections = [c for c in all_content if c['type'] == 'section']
        subsections = [c for c in all_content if c['type'] == 'subsection']

        candidates_by_doc = defaultdict(list)
        first_sections = {}
        for sec in sections:
            doc_name = sec['document']
            if doc_name not in first_sections:
                first_sections[doc_name] = sec
        for sec in sections:
            font_size, _ = sec.get('style', (10, False))
            is_main_title = sec is first_sections.get(sec['document']) and sec['page_number'] == 1
            primacy_bonus = 1000 if is_main_title else 0
            importance_score = ((1 / np.log(sec['page_number'] + 1.1)) * font_size) + primacy_bonus
            sec['importance_score'] = importance_score
            candidates_by_doc[sec['document']].append(sec)
        diversified_candidates = []
        for doc_name, doc_sections in candidates_by_doc.items():
            top_sections_from_doc = sorted(doc_sections, key=lambda s: s['importance_score'], reverse=True)[:5]
            diversified_candidates.extend(top_sections_from_doc)
        if not diversified_candidates: return {}

        cross_encoder_pairs = [[self.query_text, cand['text']] for cand in diversified_candidates]
        rerank_scores = self.cross_encoder.predict(cross_encoder_pairs, show_progress_bar=False)
        for candidate, score in zip(diversified_candidates, rerank_scores):
            candidate['rerank_score'] = score
        final_sections = sorted(diversified_candidates, key=lambda s: s.get('rerank_score', 0), reverse=True)
        
        all_subsection_texts = [c['text'] for c in subsections]
        if all_subsection_texts:
            query_embedding = self.semantic_model.encode(self.query_text, show_progress_bar=False)
            sub_embeddings = self.semantic_model.encode(all_subsection_texts, show_progress_bar=False)
            sub_scores = cosine_similarity([query_embedding], sub_embeddings).flatten()
            for i, sub in enumerate(subsections): sub['score'] = sub_scores[i]
            final_subsections = sorted(subsections, key=lambda s: s["score"], reverse=True)
        else: final_subsections = []

        top_sections = [{"document": s["document"], "section_title": s["text"], "importance_rank": i + 1, "page_number": s["page_number"]} for i, s in enumerate(final_sections[:self.config["TOP_K_SECTIONS"]])]
        top_subsections = [{"document": s["document"], "refined_text": s["text"], "page_number": s["page_number"]} for s in final_subsections[:self.config["TOP_K_SUBSECTIONS"]]]
        return {"metadata": {"input_documents": files, "persona": self.persona, "job_to_be_done": self.task, "processing_timestamp": datetime.now().isoformat()}, "extracted_sections": top_sections, "subsection_analysis": top_subsections}


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    input_json_path = Path("input_json/challenge1b_input.json")
    pdf_dir = Path("input")
    output_path = Path("output/challenge1b_output.json")
    try:
        with open(input_json_path, "r", encoding='utf-8') as f: data = json.load(f)
    except FileNotFoundError:
        return
    except json.JSONDecodeError:
        return
    persona_text = data.get("persona", {}).get("role", "")
    task_text = data.get("job_to_be_done", {}).get("task", "")
    file_list = [doc.get("filename") for doc in data.get("documents", []) if doc.get("filename")]
    if not persona_text or not task_text or not file_list:
        return
    processor = PDFProcessor(persona=persona_text, task=task_text, config=CONFIG)
    output_data = processor.run(pdf_dir=pdf_dir, files=file_list)
    if not output_data:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding='utf-8') as f: json.dump(output_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()