
import re
import nltk
from typing import Optional
from config import FORWARD_LOOKING_KEYWORDS, BACKWARD_LOOKING_KEYWORDS


def ensure_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def split_transcript_sections(content: str) -> dict:
    content = content.strip()

    qa_markers = [
        r"(?i)(?:analyst|operator).*?(?:q\s*(?:&|and)\s*a|question(?:s)?(?:\s+and\s+answer)?)",
        r"(?i)q\s*(?:&|and)\s*a\s*(?:session|segment|portion)",
        r"(?i)we (?:will|would) (?:now )?(?:like to |)(?:open|take|begin).*?(?:question|q&a)",
        r"(?i)(?:let'?s|we'?ll)\s+(?:now\s+)?(?:open|take|begin).*?question",
        r"(?i)Analyst Q&A",
        r"(?i)Question-and-Answer Session",
    ]

    split_idx = None
    for pattern in qa_markers:
        match = re.search(pattern, content)
        if match:
            split_idx = match.start()
            break

    if split_idx and split_idx > len(content) * 0.15:
        prepared = content[:split_idx].strip()
        qa = content[split_idx:].strip()
    else:
        split_point = int(len(content) * 0.6)
        para_break = content.rfind("\n\n", 0, split_point + 200)
        if para_break > split_point - 200:
            split_point = para_break
        prepared = content[:split_point].strip()
        qa = content[split_point:].strip()

    return {
        "prepared_remarks": prepared,
        "qa": qa,
        "full": content,
    }


def sentence_tokenize(text: str) -> list[str]:
    ensure_nltk_data()
    text = re.sub(r"\s+", " ", text).strip()
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def classify_sentence(sentence: str) -> str:
    lower = sentence.lower()

    forward_count = sum(1 for kw in FORWARD_LOOKING_KEYWORDS if kw in lower)
    backward_count = sum(1 for kw in BACKWARD_LOOKING_KEYWORDS if kw in lower)

    if forward_count > backward_count:
        return "forward"
    elif backward_count > forward_count:
        return "backward"
    return "neutral"


def process_transcript(content: str) -> dict:
    sections = split_transcript_sections(content)

    all_sentences = []

    for section_name in ["prepared_remarks", "qa"]:
        text = sections[section_name]
        sentences = sentence_tokenize(text)
        for sent in sentences:
            all_sentences.append({
                "text": sent,
                "section": section_name,
                "temporal": classify_sentence(sent),
            })

    return {
        "sections": sections,
        "sentences": all_sentences,
    }
