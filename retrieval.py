"""The R in RAG: split documents into chunks, then find the chunks that best match a question.

This is keyword search with a simplified BM25 score. Larger systems usually search
with embeddings instead, but the steps are the same: chunk, score, keep the best few.
The course page runs a JavaScript copy of this file so you can watch it work.
"""
import math
import re

STOPWORDS = {
    "a", "about", "all", "also", "am", "an", "and", "any", "are", "as", "at", "be",
    "been", "but", "by", "can", "could", "did", "do", "does", "ever", "for", "from",
    "had", "has", "have", "he", "her", "hers", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "just", "many", "me", "more", "most", "much", "my",
    "of", "on", "or", "our", "she", "should", "so", "some", "tell", "than", "that",
    "the", "their", "them", "then", "there", "they", "this", "to", "us", "use",
    "used", "using", "very", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "would", "you", "your",
}


def stem(word):
    """Strip plural endings so 'dashboards' matches 'dashboard'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text):
    """Lowercase words, minus common filler words, with plurals stripped."""
    words = re.findall(r"[a-z0-9+#]+", text.lower())
    return [stem(w) for w in words if len(w) > 1 and w not in STOPWORDS]


def build_chunks(documents):
    """Split each document on blank lines. A short line on its own (a heading) joins the next chunk."""
    chunks = []
    for source, text in documents.items():
        carry = ""
        for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
            block = block.strip()
            if not block:
                continue
            if len(block) < 60:
                carry = f"{carry}\n{block}" if carry else block
                continue
            chunks.append({"source": source, "text": f"{carry}\n{block}" if carry else block})
            carry = ""
        if carry:
            chunks.append({"source": source, "text": carry})
    for chunk in chunks:
        chunk["terms"] = tokenize(chunk["text"])
    return chunks


def search(question, chunks, k=4, ignore=()):
    """Score every chunk against the question and return the best k.

    A word found in fewer chunks is worth more (it says more about where the answer is).
    A word repeated in a chunk counts a little more each time, with diminishing returns.
    Words in `ignore` (the candidate's name) are skipped: they appear in most questions
    but say nothing about which chunk holds the answer.
    """
    q_terms = sorted(set(tokenize(question)) - set(ignore))
    n = len(chunks)
    df = {t: sum(1 for c in chunks if t in c["terms"]) for t in q_terms}

    scored = []
    for i, chunk in enumerate(chunks):
        score = 0.0
        for t in q_terms:
            tf = chunk["terms"].count(t)
            if tf:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * tf / (tf + 1)
        scored.append((score, i))

    scored.sort(key=lambda s: (-s[0], s[1]))
    best = [chunks[i] | {"score": round(s, 2)} for s, i in scored[:k] if s > 0]
    # Nothing matched (for example "tell me about them"): fall back to the top of the documents.
    return best or [c | {"score": 0} for c in chunks[:k]]
