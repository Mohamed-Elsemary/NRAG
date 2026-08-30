import os
import json
import re
from pathlib import Path
from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = str(BASE_DIR / "data" / "1830_Technical_Description.pdf")
OUTPUT_CHUNKS_PATH = str(BASE_DIR / "data" / "extracted_chunks.json")

START_PAGE = 47
END_PAGE = 166

# Shelf model identifiers to auto-tag
SHELF_PATTERNS = [
    "PSS-32", "PSS-16II", "PSS-16", "PSS-8",
    "PSI-8L", "PSI-4L", "PSS-36", "PSS-64",
]

# ---------------------------------------------------------------------------
# Unicode / mojibake cleanup
# ---------------------------------------------------------------------------
UNICODE_REPLACEMENTS = {
    "ΓÇó": "•",
    "ΓÇ£": '"',
    "ΓÇ¥": '"',
    "ΓÇô": "–",
    "ΓÇö": "—",
    "ΓÇÖ": "'",
    "ΓÇ¿": "»",
    "ΓêÆ": "→",
    "Γëñ": "≤",
    "Γëá": "≥",
    "┬⌐": "©",
    "┬░": "°",
    "Γûí": " ",
    "\u00a0": " ",       # non-breaking space
    "\u2002": " ",       # en-space
    "\u2003": " ",       # em-space
}

def fix_unicode(text: str) -> str:
    for bad, good in UNICODE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = re.sub(r"  +", " ", text)
    return text


# ---------------------------------------------------------------------------
# Header / footer cleaning
# ---------------------------------------------------------------------------
HEADER_FOOTER_PATTERNS = [
    # Document title block (multi-line)
    re.compile(
        r"Nokia\s+1830\s+PSS[^\n]+(?:Release\s+[\d.]+)?", re.IGNORECASE
    ),
    # Copyright notice
    re.compile(r"©\s*\d{4}\s*Nokia[^\n]*", re.IGNORECASE),
    # Confidentiality
    re.compile(r"Nokia\s+Confidential\s+Information[^\n]*", re.IGNORECASE),
    re.compile(
        r"Use\s+(?:pursuant\s+to\s+applicable|subject\s+to\s+agreed)[^\n]*",
        re.IGNORECASE,
    ),
    # Document ID
    re.compile(r"3KC[-\d]+QAAA[-\w]*", re.IGNORECASE),
    # Issue / date footers
    re.compile(r"(?:Issue\s+\d+|June\s+\d{4})", re.IGNORECASE),
    # Standalone page numbers (with optional "Issue 1" prefix)
    re.compile(r"^\s*(?:Issue\s+\d+\s+)?\d{1,4}\s*$", re.MULTILINE),
    # Repeated section breadcrumbs at page bottom
    re.compile(
        r"^Shelves\s+and\s+common\s+equipment/cards\s*$", re.MULTILINE
    ),
    re.compile(r"^(?:Shelves|Fan\s+units|Power\s+filters|Equipment\s+controllers)\s*$", re.MULTILINE),
    re.compile(r"^Release\s+\d+\.\d+\s*$", re.MULTILINE),
]


def clean_page_text(text: str) -> str:
    text = fix_unicode(text)
    for pattern in HEADER_FOOTER_PATTERNS:
        text = pattern.sub("", text)
    # Remove blank lines but preserve paragraph structure
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------
def get_heading_level(line: str):
    """
    Return (level, heading_text) if this line is a section heading, else None.

    Levels:
      1 → chapter   (e.g. "1 System concept")
      2 → section   (e.g. "2.5 1830 PSS-32 shelf")
      3 → subsection (e.g. "2.5.3 Slot numbering")
      4 → sub-subsection (e.g. "2.5.3.1 Detail")
    """
    line = line.strip()
    if not line or len(line) > 100:
        return None
    # Reject sentence-like fragments
    if line.endswith((".", ",", ":", ";")):
        return None

    # Numbered sections: "1.2 Title", "2.5.3 Title", etc.
    # The leading number must be a plausible section number (1-99),
    # NOT a product model (1830) or legend number (like "3 Rotary Switch").
    m = re.match(r"^(\d+(?:\.\d+)+)\s+([A-Z].*)", line)
    if m:
        number_part = m.group(1)
        first_num = int(number_part.split(".")[0])
        if first_num <= 99:
            dots = number_part.count(".")
            level = min(dots + 1, 4)
            return (level, line)

    # Top-level chapter headings: "1 System concept", "2 Shelves and..."
    # Must be a single digit 1-9 followed by a short, title-like phrase
    m = re.match(r"^(\d)\s+([A-Z][A-Za-z ]{2,50})$", line)
    if m:
        return (1, line)

    # "Chapter N ..." style
    m = re.match(r"^Chapter\s+(\d+)\s+(.*)", line, re.IGNORECASE)
    if m:
        return (1, line)

    # Component category headers (standalone breadcrumbs used as section dividers)
    if re.match(
        r"^(?:Fan\s+[Uu]nits?|Power\s+[Ff]ilters?|Equipment\s+[Cc]ontrollers?|Wavelength\s+[Rr]outers?)$",
        line,
    ):
        return (2, line)

    return None


# ---------------------------------------------------------------------------
# Shelf tag detection
# ---------------------------------------------------------------------------
def detect_shelf_tags(text: str) -> list:
    tags = set()
    text_upper = text.upper()
    for shelf in SHELF_PATTERNS:
        if shelf.upper() in text_upper:
            tags.add(shelf)
    return sorted(tags)


# ---------------------------------------------------------------------------
# Sentence-aware text splitting
# ---------------------------------------------------------------------------
def split_into_sentences(text: str) -> list:
    # Split on period followed by space and capital letter, or on bullet points
    parts = re.split(r"(?<=\.)\s+(?=[A-Z•])", text)
    return [p.strip() for p in parts if p.strip()]


def split_text_into_chunks(text: str, max_words: int = 300, min_words: int = 80) -> list:
    """
    Split a block of text into word-count-bounded chunks.
    Tries paragraph boundaries first, then sentence boundaries.
    """
    # Check whether the text is empty.
    if not text.strip():
        return []

    words = text.split()
    if len(words) <= max_words:
        return [text.strip()]

    # Try splitting by paragraphs (double newline)
    paragraphs = re.split(r"\n\s*\n", text)
    if len(paragraphs) > 1:
        return _merge_segments(paragraphs, max_words, min_words)

    # Fall back to sentence splitting
    sentences = split_into_sentences(text)
    if len(sentences) > 1:
        return _merge_segments(sentences, max_words, min_words)

    # Last resort: hard word split with overlap
    chunks = []
    overlap = 30
    i = 0
    while i < len(words):
        end = min(i + max_words, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)

        i = end - overlap if end < len(words) else end
    return chunks


def _merge_segments(segments: list, max_words: int, min_words: int) -> list:
    chunks = []
    current = []
    current_wc = 0

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        seg_wc = len(seg.split())

        if current_wc + seg_wc > max_words and current_wc >= min_words:
            chunks.append("\n\n".join(current))
            current = [seg]
            current_wc = seg_wc
        else:
            current.append(seg)
            current_wc += seg_wc

    if current:
        # If the last chunk is too small, merge with previous
        last_text = "\n\n".join(current)
        if current_wc < min_words and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + last_text
        else:
            chunks.append(last_text)

    return chunks


# ---------------------------------------------------------------------------
# Main extraction + chunking pipeline
# ---------------------------------------------------------------------------
def extract_and_chunk(
    pdf_path: str = PDF_PATH,
    output_path: str = OUTPUT_CHUNKS_PATH,
    min_words: int = 80,
    max_words: int = 300,
    overlap_words: int = 30,
):
    """
    Extract text from the PDF, detect section boundaries, and produce
    semantically coherent chunks with rich metadata.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at '{pdf_path}'")

    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    print(f"[Chunker] Processing pages {START_PAGE} to {END_PAGE} "
          f"(of {total_pages} total)...")

    # ------------------------------------------------------------------
    # Pass 1: Extract all page texts with heading annotations
    # ------------------------------------------------------------------
    page_contents = []  # list of (page_num, cleaned_text)
    for page_num in range(START_PAGE, min(END_PAGE + 1, total_pages + 1)): # min just for safety limit
        raw_text = reader.pages[page_num - 1].extract_text() or ""
        cleaned = clean_page_text(raw_text)
        page_contents.append((page_num, cleaned))

    # ------------------------------------------------------------------
    # Pass 2: Walk through lines, tracking section hierarchy and building
    #          section blocks (text between two headings)
    # ------------------------------------------------------------------
    section_blocks = []   # list of {section, parent, chapter, page_start, page_end, lines}
    # Section hierarchy stack: [chapter, section, subsection, sub-subsection]
    hierarchy = ["", "", "", ""]

    current_block_lines = []
    current_block_page_start = START_PAGE
    current_block_page_end = START_PAGE

    def _flush_block(): # Save the accumulated block as a section block.
        
        nonlocal current_block_lines, current_block_page_start, current_block_page_end
        text = "\n".join(current_block_lines).strip()
        if text and len(text.split()) > 15:
            # Determine section info from hierarchy
            section = (hierarchy[3] or hierarchy[2] or
                       hierarchy[1] or hierarchy[0] or "Unknown")
            parent = ""
            chapter = hierarchy[0]
            if hierarchy[3]:
                parent = hierarchy[2] or hierarchy[1] or hierarchy[0]
            elif hierarchy[2]:
                parent = hierarchy[1] or hierarchy[0]
            elif hierarchy[1]:
                parent = hierarchy[0]

            section_blocks.append({
                "section": section,
                "parent_section": parent,
                "chapter": chapter,
                "page_start": current_block_page_start,
                "page_end": current_block_page_end,
                "text": text,
            })
        current_block_lines = []

    for page_num, page_text in page_contents:
        lines = page_text.splitlines()
        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_block_lines.append("")  # preserve paragraph breaks
                continue

            heading_info = get_heading_level(stripped)
            if heading_info:
                level, heading_text = heading_info
                # Flush previous block before starting new section
                _flush_block()
                current_block_page_start = page_num
                current_block_page_end = page_num

                # Update hierarchy
                hierarchy[level - 1] = heading_text
                # Clear lower levels
                for i in range(level, 4):
                    hierarchy[i] = ""
                continue

            current_block_lines.append(stripped)
            current_block_page_end = page_num

    # Flush the last block
    _flush_block()

    print(f"[Chunker] Found {len(section_blocks)} section blocks.")

    # ------------------------------------------------------------------
    # Pass 3: Split large section blocks into retrieval-sized chunks
    # ------------------------------------------------------------------
    raw_chunks = []

    for block in section_blocks:
        sub_chunks = split_text_into_chunks(
            block["text"], max_words=max_words, min_words=min_words
        )

        for sub_text in sub_chunks:
            word_count = len(sub_text.split())
            shelf_tags = detect_shelf_tags(sub_text)
            # Also check section/parent for shelf tags
            for ctx in [block["section"], block["parent_section"], block["chapter"]]:
                for tag in detect_shelf_tags(ctx):
                    if tag not in shelf_tags:
                        shelf_tags.append(tag)
            shelf_tags.sort()

            raw_chunks.append({
                "section": block["section"],
                "parent_section": block["parent_section"],
                "chapter": block["chapter"],
                "page_start": block["page_start"],
                "page_end": block["page_end"],
                "shelf_tags": shelf_tags,
                "word_count": word_count,
                "text": sub_text,
            })

    # ------------------------------------------------------------------
    # Pass 4: Merge small chunks into neighbors
    # Chunks below min_words are merged with the NEXT chunk if they share
    # ------------------------------------------------------------------
    merged = []
    for chunk in raw_chunks:
        if (merged
                and merged[-1]["word_count"] < min_words
                and (merged[-1]["section"] == chunk["section"]
                     or merged[-1]["parent_section"] == chunk["parent_section"]
                     or merged[-1]["parent_section"] == chunk["section"])):
            # Merge the small previous chunk into this one
            prev = merged.pop()
            chunk = {
                **chunk,
                "text": prev["text"] + "\n\n" + chunk["text"],
                "word_count": prev["word_count"] + chunk["word_count"],
                "page_start": min(prev["page_start"], chunk["page_start"]),
                "page_end": max(prev["page_end"], chunk["page_end"]),
                "shelf_tags": sorted(set(prev["shelf_tags"] + chunk["shelf_tags"])),
            }
        merged.append(chunk)

    # Final pass: merge any remaining trailing small chunk into previous
    if len(merged) > 1 and merged[-1]["word_count"] < min_words:
        last = merged.pop()
        merged[-1]["text"] += "\n\n" + last["text"]
        merged[-1]["word_count"] += last["word_count"]
        merged[-1]["page_end"] = max(merged[-1]["page_end"], last["page_end"])
        merged[-1]["shelf_tags"] = sorted(
            set(merged[-1]["shelf_tags"] + last["shelf_tags"])
        )

    # Assign final IDs
    chunks = []
    for i, c in enumerate(merged):
        c["id"] = i
        chunks.append(c)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Print statistics
    word_counts = [c["word_count"] for c in chunks]
    sections = set(c["section"] for c in chunks)
    pages = set()
    for c in chunks:
        pages.update(range(c["page_start"], c["page_end"] + 1))

    print(f"[Chunker] Built {len(chunks)} chunks.")
    print(f"  Word count: min={min(word_counts)}, avg={sum(word_counts)/len(word_counts):.0f}, "
          f"max={max(word_counts)}")
    print(f"  Unique sections: {len(sections)}")
    print(f"  Pages covered: {min(pages)}-{max(pages)} ({len(pages)} pages)")

    return chunks


if __name__ == "__main__":
    chunks = extract_and_chunk()
    # Show a few sample chunks
    print("\n--- Sample chunks ---")
    for c in chunks[:5]:
        print(f"\n  #{c['id']:3d} | pg {c['page_start']}-{c['page_end']} | "
              f"{c['word_count']}w | {c['shelf_tags']}")
        print(f"  Section: {c['section']}")
        print(f"  Parent:  {c['parent_section']}")
        print(f"  Text:    {c['text'][:150]}...")