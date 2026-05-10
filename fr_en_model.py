
import re
from transformers import MarianMTModel, MarianTokenizer
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# TRANSLATION MODEL — Helsinki-NLP/opus-mt-fr-en
# ~300MB, downloads once, runs on CPU or GPU, far better than LLM translation
# ─────────────────────────────────────────────────────────────────────────────

print("Loading translation model (Helsinki-NLP/opus-mt-fr-en)...")
TRANS_MODEL_NAME = "Helsinki-NLP/opus-mt-fr-en"
trans_tokenizer  = MarianTokenizer.from_pretrained(TRANS_MODEL_NAME)
trans_model      = MarianMTModel.from_pretrained(TRANS_MODEL_NAME)

trans_model = trans_model.to(DEVICE)
print(f"Translation model loaded on {DEVICE}")


def translate_fr_to_en(text: str) -> str:
    """Translate French text to English using opus-mt-fr-en."""
    if not text or not text.strip():
        return ""
    # opus-mt has a ~512 token limit per input, so split long texts
    sentences = split_long_text(text, max_chars=400)
    translated_parts = []
    for chunk in sentences:
        inputs = trans_tokenizer(chunk, return_tensors="pt",
                                 padding=True, truncation=True,
                                 max_length=512).to(DEVICE)
        with torch.no_grad():
            output = trans_model.generate(**inputs, max_length=512)
        decoded = trans_tokenizer.decode(output[0], skip_special_tokens=True)
        translated_parts.append(decoded)
    return " ".join(translated_parts)


def split_long_text(text: str, max_chars: int = 400) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    current = ""
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        if len(current) + len(sentence) > max_chars and current:
            chunks.append(current.strip())
            current = sentence
        else:
            current = current + " " + sentence if current else sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]

