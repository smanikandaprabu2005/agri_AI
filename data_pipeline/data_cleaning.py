"""
data_cleaning.py
=================
Comprehensive data cleaning script for agricultural text datasets.

This script performs thorough cleaning of raw text data to remove noise,
normalize content, and prepare it for tokenization and model training.

Features:
- Remove HTML tags and entities
- Remove URLs and email addresses
- Normalize whitespace and punctuation
- Remove excessive special characters and noise
- Filter by sentence length and quality
- Remove duplicates
- Agriculture domain filtering (optional)
- Language detection (basic English/Hindi check)

Usage:
    python data_cleaning.py --input input_file.txt --output cleaned_output.txt
    python data_cleaning.py --input input_file.jsonl --output cleaned_output.jsonl --format jsonl
"""

import re
import json
import argparse
from pathlib import Path
from collections import Counter
import unicodedata

# Pipeline default raw data locations
RAW_DIR = Path("data_pipeline/data/raw")
PIPELINE_CLEAN_CONFIG = [
    {
        "name": "instruction dataset",
        "input": RAW_DIR / "final_agriculture_training_dataset.jsonl",
        "output": RAW_DIR / "cleaned_final_agriculture_training_dataset.jsonl",
        "type": "jsonl",
    },
    {
        "name": "ICAR JSON",
        "input": RAW_DIR / "ICAR_Text_Extracted.json",
        "output": RAW_DIR / "cleaned_ICAR_Text_Extracted.json",
        "type": "json",
    },
    {
        "name": "research text",
        "input": RAW_DIR / "research_wikipedia_final_dataset.txt",
        "output": RAW_DIR / "cleaned_research_wikipedia_final_dataset.txt",
        "type": "text",
    },
    {
        "name": "BSc Agri text",
        "input": RAW_DIR / "bsc agri.txt",
        "output": RAW_DIR / "cleaned_bsc_agri.txt",
        "type": "text",
    },
]

# Configuration
MIN_SENTENCE_LENGTH = 20  # Minimum characters per sentence
MAX_SENTENCE_LENGTH = 1000  # Maximum characters per sentence
MIN_WORDS_PER_SENTENCE = 5  # Minimum words per sentence
MAX_WORDS_PER_SENTENCE = 100  # Maximum words per sentence

# Agriculture keywords for domain filtering
AGRI_KEYWORDS = {
    'crop', 'soil', 'water', 'fertilizer', 'pesticide', 'harvest', 'yield',
    'plant', 'seed', 'irrigation', 'farming', 'agriculture', 'cultivation',
    'paddy', 'rice', 'wheat', 'maize', 'cotton', 'tomato', 'potato',
    'pest', 'disease', 'weed', 'manure', 'organic', 'farm', 'farmer'
}

def clean_text(text: str) -> str:
    """Perform comprehensive text cleaning."""
    if not text:
        return ""

    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)

    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)

    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)

    # Remove page markers and headers
    text = re.sub(r'---\s*Page\s*\d+\s*---', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page\s*\d+', '', text, flags=re.IGNORECASE)

    # Remove excessive punctuation
    text = re.sub(r'[!?]{3,}', '!', text)
    text = re.sub(r'[.]{5,}', '.', text)
    text = re.sub(r'[-]{3,}', '-', text)
    text = re.sub(r'[*#=]{5,}', '', text)

    # Remove repeated characters (more than 3)
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)

    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\x0A\x0D\u0900-\u097F]+', ' ', text)  # Keep English, Hindi, newlines

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    return text

def is_agriculture_related(text: str) -> bool:
    """Check if text contains agriculture-related keywords."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    return bool(words & AGRI_KEYWORDS)

def is_english_or_hindi(text: str) -> bool:
    """Basic check for English or Hindi text."""
    # Count English and Hindi characters
    english_chars = sum(1 for c in text if ord(c) < 128)
    hindi_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars = len(text.replace(' ', ''))

    if total_chars == 0:
        return False

    # At least 30% English or Hindi characters
    return (english_chars / total_chars > 0.3) or (hindi_chars / total_chars > 0.3)

def split_into_sentences(text: str) -> list:
    """Split text into sentences."""
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def filter_sentence(sentence: str) -> bool:
    """Filter sentences based on quality criteria."""
    if not sentence:
        return False

    # Length checks
    if len(sentence) < MIN_SENTENCE_LENGTH or len(sentence) > MAX_SENTENCE_LENGTH:
        return False

    words = sentence.split()
    if len(words) < MIN_WORDS_PER_SENTENCE or len(words) > MAX_WORDS_PER_SENTENCE:
        return False

    # Check for too many numbers (likely tables or OCR noise)
    numbers = sum(1 for word in words if re.match(r'^\d+(\.\d+)?$', word))
    if numbers / len(words) > 0.3:
        return False

    # Check for too many uppercase words (likely headers)
    uppercase_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    if uppercase_words / len(words) > 0.5:
        return False

    # Language check
    if not is_english_or_hindi(sentence):
        return False

    return True

def remove_duplicates(sentences: list) -> list:
    """Remove duplicate sentences."""
    seen = set()
    unique = []
    for sent in sentences:
        normalized = sent.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique.append(sent)
    return unique

def clean_file(input_path: str, output_path: str, format_type: str = 'text',
               agriculture_filter: bool = False):
    """Clean a file and save the result."""
    print(f"Cleaning {input_path}...")

    sentences = []

    if format_type == 'jsonl':
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    for key in ['instruction', 'output', 'text']:
                        if key in data and data[key]:
                            text = clean_text(data[key])
                            sentences.extend(split_into_sentences(text))
                except json.JSONDecodeError:
                    continue
    else:
        with open(input_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
            cleaned_text = clean_text(text)
            sentences = split_into_sentences(cleaned_text)

    # Filter sentences
    filtered_sentences = [s for s in sentences if filter_sentence(s)]

    # Agriculture filter
    if agriculture_filter:
        filtered_sentences = [s for s in filtered_sentences if is_agriculture_related(s)]

    # Remove duplicates
    unique_sentences = remove_duplicates(filtered_sentences)

    # Save output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if format_type == 'jsonl':
        # For JSONL, we need to reconstruct or save as text
        with open(output_path, 'w', encoding='utf-8') as f:
            for sent in unique_sentences:
                f.write(sent + '\n')
    else:
        with open(output_path, 'w', encoding='utf-8') as f:
            for sent in unique_sentences:
                f.write(sent + '\n')

    print(f"Original sentences: {len(sentences)}")
    print(f"After filtering: {len(filtered_sentences)}")
    print(f"After deduplication: {len(unique_sentences)}")
    print(f"Cleaned data saved to {output_path}")


def clean_jsonl_dataset(input_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with input_path.open("r", encoding="utf-8", errors="replace") as src, \
            output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            for key in ["instruction", "output", "input", "text"]:
                if key in item and isinstance(item[key], str):
                    item[key] = clean_text(item[key])

            if not item.get("instruction", "").strip() or not item.get("output", "").strip():
                continue

            dst.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    return written


def clean_json_file(input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8", errors="replace") as src:
        data = json.load(src)

    def _clean_value(value):
        if isinstance(value, str):
            return clean_text(value)
        if isinstance(value, dict):
            return {k: _clean_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_clean_value(v) for v in value]
        return value

    cleaned = _clean_value(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as dst:
        json.dump(cleaned, dst, ensure_ascii=False)

    return 1


def clean_text_file(input_path: Path, output_path: Path):
    with input_path.open("r", encoding="utf-8", errors="replace") as src:
        text = src.read()
    cleaned = clean_text(text)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as dst:
        dst.write(cleaned)
    return 1


def clean_pipeline_files(force: bool = False):
    print("\n[Step 0] ── Cleaning pipeline raw files")
    for cfg in PIPELINE_CLEAN_CONFIG:
        input_path: Path = cfg["input"]
        output_path: Path = cfg["output"]
        if not input_path.exists():
            print(f"[Step 0] Skipping missing source: {input_path}")
            continue
        if output_path.exists() and not force:
            print(f"[Step 0] Skipping existing cleaned file: {output_path}")
            continue

        print(f"[Step 0] Cleaning {cfg['name']}: {input_path} → {output_path}")
        if cfg["type"] == "jsonl":
            written = clean_jsonl_dataset(input_path, output_path)
            print(f"           Records written: {written}")
        elif cfg["type"] == "json":
            clean_json_file(input_path, output_path)
            print(f"           JSON cleaned: {output_path.name}")
        else:
            clean_text_file(input_path, output_path)
            print(f"           Text cleaned: {output_path.name}")

    print("\n[Step 0] ✓ Pipeline raw cleaning complete.")


def main():
    parser = argparse.ArgumentParser(description="Clean agricultural text datasets")
    parser.add_argument('--input', help='Input file path')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--format', choices=['text', 'jsonl'], default='text',
                       help='Input file format')
    parser.add_argument('--agriculture-filter', action='store_true',
                       help='Filter for agriculture-related content only')
    parser.add_argument('--clean-pipeline-files', action='store_true',
                       help='Clean the default pipeline raw files before preprocessing')
    parser.add_argument('--force', action='store_true',
                       help='Overwrite existing cleaned pipeline files')

    args = parser.parse_args()

    if args.clean_pipeline_files:
        clean_pipeline_files(force=args.force)
        return

    if not args.input or not args.output:
        parser.error('--input and --output are required unless --clean-pipeline-files is used')

    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist")
        return

    clean_file(args.input, args.output, args.format, args.agriculture_filter)

if __name__ == '__main__':
    main()