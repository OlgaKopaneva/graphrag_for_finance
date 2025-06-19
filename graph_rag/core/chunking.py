import re
from typing import List

def clean_text(text: str) -> str:
    """
    Removes unnecessary characters and CFA copyright lines
    """
    text = re.sub(r'© CFA Institute\. For candidate use only\. Not for distribution\.?', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_chapters(text: str) -> str:
    """
    Extracts the contents of the chapters between INTRODUCTION и PRACTICE PROBLEMS
    """
    result_text = ""
    
    reading_pattern = r'R?EADING\s+\d+'
    readings = re.split(reading_pattern, text)
    num_chapters = len(readings) - 1
    
    for reading in readings[1:]:
        # Skip the first part (up to the first chapter)
        intro_match = re.search(r'INTRODUCTION', reading)
        if not intro_match:
            continue
        
        # find chapter boarders
        chapter_text = reading[intro_match.end():]
        practice_match = re.search(r'PRACTICE PROBLEMS', chapter_text)
        if practice_match:
            chapter_text = chapter_text[:practice_match.start()]
        
        result_text += " " + chapter_text.strip()
    
    print(num_chapters)
    return result_text.strip()

def process_book(text: str) -> str:
    """
    Book processing
    """
    chapters_text = extract_chapters(text)
    cleaned_text = clean_text(chapters_text)
    
    return cleaned_text

def split_into_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks with specified length and overlap.
    chunk_size — chunk length (in words)
    overlap — number of words overlapping between chunks
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks