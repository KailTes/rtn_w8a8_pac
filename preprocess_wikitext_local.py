"""
process_results for wikitext_local task.
doc field: 'text' (document-level, already merged)
"""
import re


def process_results(doc, results):
    (loglikelihood,) = results
    text = doc["text"]
    _words = len(re.split(r"\s+", text))
    _bytes = len(text.encode("utf-8"))
    return {
        "word_perplexity": (loglikelihood, _words),
        "byte_perplexity": (loglikelihood, _bytes),
        "bits_per_byte": (loglikelihood, _bytes),
    }
