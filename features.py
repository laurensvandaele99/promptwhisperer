from __future__ import annotations

import math
import re
from collections import Counter

import nltk
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

PROMPT_COL = "evidence_first_prompt_excerpt"


def _load_vader() -> SentimentIntensityAnalyzer:
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        ok = nltk.download("vader_lexicon", quiet=True)
        if not ok:
            raise RuntimeError(
                "NLTK VADER lexicon is missing and could not be downloaded. "
                "Run: python -m nltk.downloader vader_lexicon"
            )
    return SentimentIntensityAnalyzer()


_SIA: SentimentIntensityAnalyzer | None = None


def _get_sia() -> SentimentIntensityAnalyzer:
    global _SIA
    if _SIA is None:
        _SIA = _load_vader()
    return _SIA

def word_count(text):
    return len(re.findall(r"\b\w+\b", str(text)))

def char_count(text):
    return len(str(text))

def sentence_count(text):
    ss = [s.strip() for s in re.split(r"[.!?]+", str(text)) if s.strip()]
    return max(len(ss), 1)

def avg_word_length(text):
    words = re.findall(r"\b\w+\b", str(text))
    return 0 if not words else sum(map(len, words)) / len(words)

def simple_readability(text):
    words = re.findall(r"\b\w+\b", str(text))
    if not words:
        return 0
    return 100 - (len(words) / sentence_count(text) * 1.5) - (avg_word_length(text) * 5)

def question_rate(text):
    text = str(text)
    return text.count("?") / max(len(text), 1)

def exclamation_rate(text):
    text = str(text)
    return text.count("!") / max(len(text), 1)

def caps_ratio(text):
    letters = [c for c in str(text) if c.isalpha()]
    return 0 if not letters else sum(c.isupper() for c in letters) / len(letters)

def has_url(text):
    return int(bool(re.search(r"http[s]?://|www\.", str(text).lower())))

def has_code(text):
    signals = [
        "```", "def ", "function", "class ", "import ", "return ",
        "console.log", "print(", "for ", "while ", "if ", "else",
        "{", "}", ";", "<html", "</", "select ", "from "
    ]
    lower = str(text).lower()
    return int(any(sig in lower for sig in signals))

def has_numbers(text):
    return int(bool(re.search(r"\d", str(text))))

def constraint_count(text):
    lower = str(text).lower()
    patterns = [
        "must", "should", "only", "do not", "don't", "without",
        "include", "exclude", "use", "avoid", "make sure",
        "format", "in bullet", "in table", "no comments", "step by step"
    ]
    return sum(lower.count(p) for p in patterns)

def polite_flag(text):
    lower = str(text).lower()
    return int(any(t in lower for t in ["please", "could you", "would you", "can you", "thanks", "thank you"]))

def urgent_flag(text):
    lower = str(text).lower()
    return int(any(t in lower for t in ["urgent", "asap", "quickly", "right now", "immediately", "fast"]))

def hedge_rate(text):
    lower = str(text).lower()
    words = re.findall(r"\b\w+\b", lower)
    if not words:
        return 0
    terms = ["maybe", "perhaps", "possibly", "probably", "somewhat", "kind of", "sort of", "i think", "i guess", "might", "could"]
    return sum(lower.count(t) for t in terms) / len(words)

def vague_pronoun_rate(text):
    words = re.findall(r"\b\w+\b", str(text).lower())
    if not words:
        return 0
    terms = {"it", "this", "that", "they", "them", "thing", "stuff", "something"}
    return sum(w in terms for w in words) / len(words)

def negation_rate(text):
    words = re.findall(r"\b\w+\b", str(text).lower())
    if not words:
        return 0
    terms = {"not", "no", "never", "none", "nothing", "without", "cannot", "can't", "dont", "don't"}
    return sum(w in terms for w in words) / len(words)

def subordinate_clause_rate(text):
    words = re.findall(r"\b\w+\b", str(text).lower())
    if not words:
        return 0
    markers = {"because", "although", "since", "while", "whereas", "unless", "if", "when", "after", "before", "that", "which"}
    return sum(w in markers for w in words) / len(words)

def shannon_entropy(text):
    text = str(text)
    if not text:
        return 0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())

def keyword_density(text, keywords):
    words = re.findall(r"\b\w+\b", str(text).lower())
    if not words:
        return 0
    return sum(w in keywords for w in words) / len(words)

code_keywords = {"code","script","function","class","python","java","javascript","html","css","sql","debug","error","api","json","loop","variable","compile","runtime"}
creative_keywords = {"write","story","poem","essay","creative","imagine","roleplay","character","dialogue","scene","lyrics","novel","prompt","midjourney"}
factual_keywords = {"what","why","how","which","when","where","explain","define","compare","difference","information","facts","describe"}
emotional_keywords = {"feel","sad","happy","angry","lonely","afraid","anxious","love","hate","worried","stress","upset","excited"}
task_keywords = {"write","create","make","give","generate","solve","fix","analyze","summarize","translate","calculate","find","list"}

def prompt_type_flag(text):
    lower = str(text).lower().strip()
    if "?" in lower or any(lower.startswith(q) for q in ["what","why","how","which","when","where","who"]):
        return "question"
    starts = ["write","create","make","give","generate","solve","fix","analyze","summarize","translate","calculate","find","list","explain"]
    if any(lower.startswith(w) for w in starts):
        return "instruction"
    return "other"

def count_regex(text, pattern):
    return len(re.findall(pattern, str(text), flags=re.IGNORECASE))

def line_count(text):
    return len([line.strip() for line in str(text).splitlines() if line.strip()])

def bullet_count(text):
    return count_regex(text, r"(^|\n)\s*[-*•\d]+[.)]?\s+")

def quoted_string_count(text):
    return count_regex(text, r"[\"“”'‘’][^\"“”'‘’]{2,}[\"“”'‘’]")

def command_verb_count(text):
    lower = str(text).lower()
    verbs = ["write","create","make","generate","give","list","explain","summarize","analyze","compare","translate","solve","fix","debug","calculate","find","classify","extract","rewrite","rephrase","improve"]
    return sum(len(re.findall(rf"\b{re.escape(v)}\b", lower)) for v in verbs)

def output_format_count(text):
    lower = str(text).lower()
    formats = ["table","bullet","bullets","list","json","csv","markdown","code","essay","paragraph","email","title","caption","summary","only code","no comments"]
    return sum(lower.count(f) for f in formats)

def constraint_marker_count(text):
    lower = str(text).lower()
    markers = ["must","should","need to","have to","only","without","do not","don't","never","always","include","exclude","at least","at most","exactly","no more than","less than"]
    return sum(lower.count(m) for m in markers)

def multi_task_marker_count(text):
    lower = str(text).lower()
    return sum(lower.count(m) for m in [" and "," also "," plus "," then "," after that "," as well as "])

def explicit_quantity_count(text):
    return count_regex(text, r"\b\d+\b|\b(one|two|three|four|five|six|seven|eight|nine|ten|twenty|hundred)\b")

def question_word_count(text):
    lower = str(text).lower()
    return sum(len(re.findall(rf"\b{q}\b", lower)) for q in ["what","why","how","which","when","where","who"])

def proper_noun_proxy_count(text):
    return len(re.findall(r"\b[A-Z][a-z]{2,}\b", str(text)))

def typo_gibberish_proxy(text):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", str(text).lower())
    if not words:
        return 0
    bad = 0
    for w in words:
        vowels = sum(ch in "aeiou" for ch in w)
        if vowels == 0 or vowels / len(w) < 0.15:
            bad += 1
    return bad / len(words)

def short_prompt_flag(text): return int(word_count(text) <= 3)
def very_long_prompt_flag(text): return int(word_count(text) >= 150)

def no_clear_task_flag(text):
    lower = str(text).lower().strip()
    task_words = ["write","create","make","generate","give","list","explain","summarize","analyze","compare","translate","solve","fix","debug","calculate","find","classify","extract","rewrite","rephrase","what","why","how","which","when","where","who"]
    return int(not any(re.search(rf"\b{re.escape(w)}\b", lower) for w in task_words))

def quiz_multiple_choice_flag(text):
    lines = [l.strip() for l in str(text).splitlines() if l.strip()]
    option_like = sum(1 for l in lines if re.match(r"^([A-Da-d][.)]|[-*•]|\d+[.)])\s+", l) or len(l.split()) <= 8)
    return int(option_like >= 3 and "?" in str(text))

def translation_flag(text):
    lower = str(text).lower()
    return int(any(w in lower for w in ["translate","translation","traduc","перев","翻译"]))

def summarization_flag(text):
    lower = str(text).lower()
    return int(any(w in lower for w in ["summarize","summary","summarise","tl;dr"]))

def roleplay_flag(text):
    lower = str(text).lower()
    return int(any(w in lower for w in ["pretend","roleplay","act as","you are","character"]))

def coding_task_flag(text):
    lower = str(text).lower()
    return int(any(w in lower for w in ["code","script","python","java","javascript","lua","html","css","sql","debug","error","function","class","api","json"]))

def writing_task_flag(text):
    lower = str(text).lower()
    return int(any(w in lower for w in ["essay","article","paragraph","email","letter","story","poem","rewrite","rephrase","caption","blog","speech"]))

def build_handcrafted_features(frame):
    x = frame.copy()
    text = x[PROMPT_COL].fillna("").astype(str)

    x["prompt_word_count"] = text.map(word_count)
    x["prompt_char_count"] = text.map(char_count)
    x["prompt_sentence_count"] = text.map(sentence_count)
    x["prompt_avg_word_length"] = text.map(avg_word_length)
    x["prompt_readability_score"] = text.map(simple_readability)
    x["prompt_question_rate"] = text.map(question_rate)
    x["prompt_exclamation_rate"] = text.map(exclamation_rate)
    x["prompt_caps_ratio"] = text.map(caps_ratio)
    x["prompt_has_url"] = text.map(has_url)
    x["prompt_has_code"] = text.map(has_code)
    x["prompt_has_numbers"] = text.map(has_numbers)
    x["prompt_constraint_count"] = text.map(constraint_count)
    x["prompt_polite_flag"] = text.map(polite_flag)
    x["prompt_urgent_flag"] = text.map(urgent_flag)
    x["prompt_hedge_rate"] = text.map(hedge_rate)
    x["prompt_vague_pronoun_rate"] = text.map(vague_pronoun_rate)
    x["prompt_negation_rate"] = text.map(negation_rate)
    x["prompt_subordinate_clause_rate"] = text.map(subordinate_clause_rate)
    x["prompt_first_turn_length"] = x["prompt_char_count"]

    vader_analyzer = _get_sia()
    vader = text.map(lambda t: vader_analyzer.polarity_scores(t))
    x["prompt_vader_compound"] = vader.map(lambda d: d["compound"])
    x["prompt_vader_positive"] = vader.map(lambda d: d["pos"])
    x["prompt_vader_neutral"] = vader.map(lambda d: d["neu"])

    x["prompt_char_shannon_entropy"] = text.map(shannon_entropy)
    x["prompt_code_density"] = text.map(lambda t: keyword_density(t, code_keywords))
    x["prompt_creative_density"] = text.map(lambda t: keyword_density(t, creative_keywords))
    x["prompt_factual_density"] = text.map(lambda t: keyword_density(t, factual_keywords))
    x["prompt_emotional_density"] = text.map(lambda t: keyword_density(t, emotional_keywords))
    x["prompt_task_oriented_density"] = text.map(lambda t: keyword_density(t, task_keywords))
    x["prompt_type_flag"] = text.map(prompt_type_flag)

    x["prompt_line_count"] = text.map(line_count)
    x["prompt_bullet_count"] = text.map(bullet_count)
    x["prompt_quoted_string_count"] = text.map(quoted_string_count)
    x["prompt_comma_count"] = text.map(lambda s: str(s).count(","))
    x["prompt_colon_count"] = text.map(lambda s: str(s).count(":"))
    x["prompt_semicolon_count"] = text.map(lambda s: str(s).count(";"))
    x["prompt_parenthesis_count"] = text.map(lambda s: str(s).count("(") + str(s).count(")"))
    x["prompt_slash_count"] = text.map(lambda s: str(s).count("/") + str(s).count("\\"))
    x["prompt_newline_count"] = text.map(lambda s: str(s).count("\n"))
    x["prompt_command_verb_count"] = text.map(command_verb_count)
    x["prompt_output_format_count"] = text.map(output_format_count)
    x["prompt_constraint_marker_count"] = text.map(constraint_marker_count)
    x["prompt_multi_task_marker_count"] = text.map(multi_task_marker_count)
    x["prompt_explicit_quantity_count"] = text.map(explicit_quantity_count)
    x["prompt_question_word_count"] = text.map(question_word_count)
    x["prompt_proper_noun_proxy_count"] = text.map(proper_noun_proxy_count)
    x["prompt_typo_gibberish_proxy"] = text.map(typo_gibberish_proxy)
    x["prompt_short_prompt_flag"] = text.map(short_prompt_flag)
    x["prompt_very_long_prompt_flag"] = text.map(very_long_prompt_flag)
    x["prompt_no_clear_task_flag"] = text.map(no_clear_task_flag)
    x["prompt_quiz_multiple_choice_flag"] = text.map(quiz_multiple_choice_flag)
    x["prompt_translation_flag"] = text.map(translation_flag)
    x["prompt_summarization_flag"] = text.map(summarization_flag)
    x["prompt_roleplay_flag"] = text.map(roleplay_flag)
    x["prompt_coding_task_flag"] = text.map(coding_task_flag)
    x["prompt_writing_task_flag"] = text.map(writing_task_flag)

    return x
