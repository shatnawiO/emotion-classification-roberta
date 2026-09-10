import json
import re
import time
import requests
from bs4 import BeautifulSoup

EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
BASE = "https://www.oxfordlearnersdictionaries.com/definition/english/"

# fallback spellings for the OED (British) site when the American form 404s
ALT_SPELLING = {"realization": "realisation"}


def fetch_definition(word):
    for candidate in (word, ALT_SPELLING.get(word)):
        if not candidate:
            continue
        url = BASE + candidate
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
        except requests.RequestException as e:
            return {"word": word, "error": str(e), "url": url}
        if r.status_code != 200:
            continue
        soup = BeautifulSoup(r.text, "html.parser")
        senses = soup.find_all("li", class_="sense")
        if not senses:
            continue
        first_def = None
        pos = None
        pos_tag = soup.find("span", class_="pos")
        if pos_tag:
            pos = pos_tag.get_text(strip=True)
        for sense in senses:
            def_span = sense.find("span", class_="def")
            if def_span:
                first_def = def_span.get_text(" ", strip=True)
                first_def = " ".join(first_def.split())
                first_def = re.sub(r"\s+([,.;:])", r"\1", first_def)
                break
        if first_def:
            return {"word": word, "definition": first_def, "part_of_speech": pos, "url": url}
    return {"word": word, "error": "definition not found", "url": BASE + word}


def main():
    results = []
    for w in EMOTIONS:
        res = fetch_definition(w)
        results.append(res)
        status = res.get("definition", "ERROR: " + str(res.get("error")))
        print(f"{w:15s} -> {status}")
        time.sleep(1.0)
    with open("definitions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
