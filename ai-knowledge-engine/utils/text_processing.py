import os
import re

def load_text_file(filepath: str) -> str:
    """Read a text file and return its content."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def clean_text(text: str) -> str:
    """Clean the text by removing extra whitespaces and newlines."""
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split the text into smaller chunks of approximately `chunk_size` characters,
    with an awareness of word boundaries to prevent splitting words in half.
    The `overlap` parameter ensures contiguous context across chunks.
    """
    cleaned_text = clean_text(text)
    
    chunks = []
    i = 0
    text_length = len(cleaned_text)
    
    while i < text_length:
        end_idx = min(i + chunk_size, text_length)
        
        # If we are not at the end of the text, try to break at the last space
        # within the proposed chunk to preserve whole words.
        if end_idx < text_length and cleaned_text[end_idx] != ' ':
            last_space = cleaned_text.rfind(' ', i, end_idx)
            if last_space != -1:
                end_idx = last_space
        
        chunk = cleaned_text[i:end_idx].strip()
        if chunk:
            chunks.append(chunk)
            
        i = end_idx - overlap
        
        # Fallback to prevent infinite loop if overlap is too large or chunk is too small
        if end_idx >= text_length:
            break
        if i <= end_idx - chunk_size:
            i = end_idx

    return chunks

def process_document(filepath: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    End-to-end text processing for a single document.
    Loads the document, cleans it, and splits it into discrete vectorizable chunks.
    """
    raw_text = load_text_file(filepath)
    return chunk_text(raw_text, chunk_size, overlap)
