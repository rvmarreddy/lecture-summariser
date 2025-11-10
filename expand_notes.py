import os
import re
import requests
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL_ID")  # optional: e.g., ft:gpt-4o-mini:org::id
DEFAULT_MODEL = "gpt-4o-mini"

WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"

def _wiki_snippets(query: str, max_chars: int = 600) -> str:
    """Very light enrichment: grab a short summary from Wikipedia (no key needed)."""
    try:
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": query,
            "redirects": 1,
        }
        r = requests.get(WIKI_SEARCH_URL, params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return ""
        page = next(iter(pages.values()))
        extract = page.get("extract", "")
        return extract[:max_chars].strip()
    except Exception:
        return ""

def _guess_keywords(text: str) -> List[str]:
    """Heuristic: pick likely key terms (capitalised phrases / headings)."""
    candidates = set()
    # headings-like lines
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.match(r"^(#+|[A-Z][A-Za-z0-9 \-]{3,})$", line):
            if len(line.split()) <= 6:
                candidates.add(re.sub(r"^#+\s*", "", line))
    # top few unique words (very rough)
    words = re.findall(r"[A-Za-z][A-Za-z\-]{3,}", text)
    freq = {}
    for w in words:
        lw = w.lower()
        freq[lw] = freq.get(lw, 0) + 1
    common = sorted(freq.items(), key=lambda x: -x[1])[:5]
    candidates.update(w for w, _ in common if len(w) > 4)
    # limit
    return list(candidates)[:3]

def expand_slide_content(
    slide_data: List[Dict],
    model_name: Optional[str] = None,
    web_enrich: bool = False,
    extra_transcript: str = ""
) -> List[Dict]:
    """Expand each slide into detailed, Notion-formatted Markdown notes."""
    model_to_use = model_name or FINE_TUNED_MODEL or DEFAULT_MODEL
    expanded_notes = []

    for entry in slide_data:
        slide_num = entry["slide"]
        text = entry.get("text", "")
        images = entry.get("images", [])

        web_context = ""
        if web_enrich and text:
            # Pull 1–3 short Wikipedia extracts based on guessed keywords
            bits = []
            for q in _guess_keywords(text)[:3]:
                snippet = _wiki_snippets(q)
                if snippet:
                    bits.append(f"• {q}: {snippet}")
            if bits:
                web_context = "External context:\n" + "\n".join(bits)

        system_style = (
            "You produce concise, well-structured lecture notes in Notion-flavoured Markdown. "
            "Use clear headings, bullet points, math as $$...$$, and insert image placeholders where helpful. "
            "Prefer short, precise sentences and highlight key definitions."
        )

        prompt = f"""
### Slide {slide_num} — Raw Slide Text
{text if text else '[No text detected on slide]'}

### Optional Transcript Context
{extra_transcript if extra_transcript else '(none)'}

### Images on this slide
{', '.join(images) if images else '(none)'}

{web_context if web_context else ''}

### Output Requirements
1) Start section with: ### 🧩 Slide {slide_num}
2) Keep it concise but complete; explain equations, add intuition.
3) Prefer bullet points where appropriate.
4) For each image, insert a placeholder line:
   ![Slide {slide_num} - Image k](./slides_output/{'{image_filename}'})
5) End the section with a horizontal rule: ---
"""

        messages = [
            {"role": "system", "content": system_style},
            {"role": "user", "content": prompt},
        ]

        completion = client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            temperature=0.2,
        )

        expanded_notes.append({
            "slide": slide_num,
            "markdown": completion.choices[0].message.content.strip()
        })

    return expanded_notes

def save_markdown(expanded_notes: List[Dict], output_path: str = "lecture_notes.md") -> str:
    """Save the final expanded notes to a Markdown file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for note in expanded_notes:
            f.write(note["markdown"])
            f.write("\n\n")
    return output_path