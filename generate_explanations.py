import json
import re
import sys
import requests

MODEL = "hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ3_XXS"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Words that must never appear (the emotion itself and close derivatives),
# checked case-insensitively against the model's output.
FORBIDDEN_EXTRA = {
    "admiration": ["admire", "admiring", "admired"],
    "amusement": ["amuse", "amusing", "amused", "funny", "hilarious"],
    "anger": ["angry", "angrily", "enrage", "furious", "fury"],
    "annoyance": ["annoy", "annoyed", "annoying", "irritate", "irritated", "irritating"],
    "approval": ["approve", "approving", "approved"],
    "caring": ["care", "cares", "cared"],
    "confusion": ["confuse", "confused", "confusing"],
    "curiosity": ["curious"],
    "desire": ["desirous", "desiring", "desired"],
    "disappointment": ["disappoint", "disappointed", "disappointing"],
    "disapproval": ["disapprove", "disapproving", "disapproved"],
    "disgust": ["disgusted", "disgusting"],
    "embarrassment": ["embarrass", "embarrassed", "embarrassing"],
    "excitement": ["excite", "excited", "exciting"],
    "fear": ["afraid", "scared", "frightened", "fearful", "terrified"],
    "gratitude": ["grateful", "thankful"],
    "grief": ["grieve", "grieving", "grieved"],
    "joy": ["joyful", "joyous"],
    "love": ["loving", "loved", "lovingly"],
    "nervousness": ["nervous", "nervously"],
    "optimism": ["optimistic"],
    "pride": ["proud", "proudly"],
    "realization": ["realize", "realized", "realizing", "realise", "realised", "realising"],
    "relief": ["relieved", "relieving"],
    "remorse": ["remorseful"],
    "sadness": ["sad", "sadly"],
    "surprise": ["surprised", "surprising", "surprisingly"],
    "neutral": ["neutrality"],
}

SYSTEM_PROMPT = (
    "You are a precise writing assistant. You write exactly two short sentences "
    "that EXPLAIN what a specific emotional/mental state MEANS, the way a dictionary "
    "or encyclopedia entry would: describe its nature, what typically causes or "
    "triggers it, and how it is generally understood. Write in general, conceptual "
    "terms - NOT a story, NOT a specific character, NOT a scene, NOT sensory imagery "
    "about one person in one moment. Do not use narrative devices. Do NOT ever use "
    "the target word or any of its word-forms (noun, verb, adjective, adverb) or "
    "obvious synonyms that would directly give it away. Do not use quotation marks. "
    "Do not add a preamble or label. Output ONLY the two sentences, on two separate "
    "lines, nothing else."
)


def build_prompt(word, definition, pos):
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f'The state to explain is: "{word}" ({pos}), dictionary definition: "{definition}".\n'
        f'Write two conceptual, definition-like sentences that explain what "{word}" is '
        f'and when/why it arises, without writing the word "{word}" or any of its forms '
        f"anywhere in your answer.\n"
    )


def call_model(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "think": False,
        "stream": False,
        "options": {"temperature": 0.7},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=300)
    r.raise_for_status()
    return r.json()["response"].strip()


def extract_sentences(raw):
    lines = [l.strip(" -\t") for l in raw.splitlines() if l.strip()]
    lines = [re.sub(r"^\d+[\.\)]\s*", "", l) for l in lines]
    if len(lines) == 1:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", lines[0])
        if len(parts) >= 2:
            lines = parts
    elif len(lines) >= 2:
        expanded = []
        for l in lines:
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", l)
            expanded.extend(parts)
        lines = expanded
    return [l.strip() for l in lines if l.strip()]


def contains_forbidden(text, word, extra_terms):
    lower = text.lower()
    terms = {word.lower()} | {t.lower() for t in extra_terms}
    for term in terms:
        if re.search(r"\b" + re.escape(term) + r"\b", lower):
            return term
    return None


def generate_for(word, definition, pos, max_attempts=4):
    extra_terms = FORBIDDEN_EXTRA.get(word, [])
    last_raw = None
    last_bad = None
    sentences = []
    for attempt in range(1, max_attempts + 1):
        prompt = build_prompt(word, definition, pos)
        if attempt > 1:
            reason = f"leaked a forbidden word ('{last_bad}')" if last_bad else "did not return two clean sentences"
            prompt += (
                f"\nIMPORTANT: your previous attempt {reason}. Try again with "
                "completely different phrasing that avoids it and all its forms, "
                "stay conceptual/definitional (not a scene), and return exactly two "
                "sentences.\n"
            )
        raw = call_model(prompt)
        last_raw = raw
        sentences = extract_sentences(raw)
        bad = None
        for s in sentences:
            bad = contains_forbidden(s, word, extra_terms)
            if bad:
                last_bad = bad
                break
        if not bad and len(sentences) >= 2:
            return sentences[:2], attempt, raw
    return sentences[:2] if sentences else [raw], max_attempts, last_raw


def main():
    with open("definitions.json", encoding="utf-8") as f:
        entries = json.load(f)

    results = []
    for e in entries:
        word = e["word"]
        definition = e.get("definition", "")
        pos = e.get("part_of_speech") or "noun"
        if not definition:
            print(f"SKIP {word}: no definition", file=sys.stderr)
            continue
        sentences, attempts, raw = generate_for(word, definition, pos)
        ok = attempts <= 4 and not any(
            contains_forbidden(s, word, FORBIDDEN_EXTRA.get(word, [])) for s in sentences
        )
        print(f"=== {word} (attempt {attempts}, clean={ok}) ===")
        for s in sentences:
            print("  ", s)
        results.append(
            {
                "word": word,
                "definition": definition,
                "part_of_speech": pos,
                "sentences": sentences,
                "attempts": attempts,
                "clean": ok,
            }
        )
        with open("explanations.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nDone. Wrote explanations.json")


if __name__ == "__main__":
    main()
