import re
import nltk
from dataclasses import dataclass
from nltk.tokenize import sent_tokenize
from transformers import pipeline

nltk.download("punkt_tab", quiet=True)

_nli = pipeline("text-classification", model="cross-encoder/nli-deberta-v3-base")


@dataclass
class HallucinationReport:
    confidence: float
    verdict: str
    sentence_labels: list[dict]
    flagged: list[str]


def _clean_sentence(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^\d+\.\s*", "", text)
    return text.strip()


def _get_verdict(score: float) -> str:
    if score >= 0.6:
        return "grounded"
    elif score >= 0.35:
        return "mostly_grounded"
    return "hallucinated"


def score_answer(answer: str, retrieved_chunks: list[str]) -> HallucinationReport:
    lines = [line.strip() for line in answer.split("\n") if line.strip()]
    sentences = []
    for line in lines:
        sentences.extend(sent_tokenize(line))
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    flagged, unverified, per_sentence = [], [], []
    entailed_count = 0

    for sentence in sentences:
        clean = _clean_sentence(sentence)
        scores = {"ENTAILMENT": 0.0, "NEUTRAL": 0.0, "CONTRADICTION": 0.0}
        seen = {"ENTAILMENT": False, "NEUTRAL": False, "CONTRADICTION": False}

        for chunk in retrieved_chunks:
            inp = f"{chunk} [SEP] {clean}"
            result = _nli(inp, truncation=True, max_length=512)[0]
            label = result["label"].upper()
            score = result["score"]

            if not seen[label] or score > scores[label]:
                scores[label] = score
                seen[label] = True

            if label == "ENTAILMENT":
                break  # top priority found, no need to keep scanning

        if seen["ENTAILMENT"]:
            best_label = "ENTAILMENT"
        elif seen["NEUTRAL"]:
            best_label = "NEUTRAL"
        elif seen["CONTRADICTION"]:
            best_label = "CONTRADICTION"
        else:
            best_label = "NEUTRAL"

        best_score = scores[best_label]

        if best_label == "ENTAILMENT":
            entailed_count += 1
        elif best_label == "CONTRADICTION":
            flagged.append(sentence)
        else:
            unverified.append(sentence)

        per_sentence.append({"sentence": sentence, "label": best_label, "score": round(best_score, 3)})

    confidence = round(entailed_count / len(sentences), 3) if sentences else 0.0
    return HallucinationReport(
        confidence=confidence,
        verdict=_get_verdict(confidence),
        sentence_labels=per_sentence,
        flagged=flagged,
    )
