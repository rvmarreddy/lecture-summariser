# Lecture Summariser

Generates LaTeX revision notes from a lecture slide PDF, using a local
[Ollama](https://ollama.com) model and a textbook for grounding. Runs entirely offline,
so no API keys are needed and nothing leaves your machine.

## Pipeline

The architecture lives in `run_extraction.run_pipeline`. For each deck:

1. **Extract & clean** — pull slide text with PyMuPDF and strip `emoji_xxx` placeholders
   that break LaTeX (`_clean_slide_text`).
2. **Plan an outline** (`_topic_groups`) — group slides by their `Section: Subtitle`
   header into one topic each, dropping syllabus/objectives/feedback/recap slides. Every
   topic becomes its own `\section`, so none can be silently dropped.
3. **Per topic, a staged Content → Review → Structure pipeline** (`src/note_writer.py`),
   each stage a focused **Qwen-via-Ollama** call (`OLLAMA_MODEL`, default `qwen2.5:7b`;
   use `qwen2.5:14b` for real quality):
   - **Scope-linked grounding** (`_ground` + `verify_relevant`) — per-slide MiniLM
     retrieval from a local textbook index, then a **local-LLM scope gate** that keeps
     only passages whose *primary subject* is the topic (kills the bi-encoder's
     surface-word matches). Set `SCOPE_MODEL`; the 14B judges scope far better than 7B.
   - **Protected spans** (`_protected_block`) — the slide's worked examples / vectors /
     similarity matrices / stem tables are **extracted and frozen** as the opaque token
     `[[EXAMPLES]]`, deduped across topics, and substituted back as real LaTeX only at
     the very end, so no stage can re-author or break them.
   - **Content** (`content_points`) → grounded numbered points (coverage + correctness).
   - **Review** (`review_points`) → fact-check each point against the source.
   - **Structure** (`structure_points`) → arrange into styled LaTeX, taught by gold
     few-shot exemplars in `src/structure_fewshot.tex`.
4. **Deterministic dedup & cleanup** — merge same-titled sections, drop duplicate boxes
   (`dedup_sections`/`dedup_blocks`), fix box titles, and a math-aware `_sanitize_body`
   (escape stray `_`/`#`/`%`/`&`/currency-`$` in prose, balance environments).
5. **Render** — wrap in the muted `src/note_preamble.tex` template → `xelatex` → PDF.

## Setup & usage

Requires the project virtualenv (`.venv/bin/python`), `xelatex`, and a local Ollama
server with the model pulled. Run from the repo root so `inputs/`, `models/` and
`outputs/` resolve.

```sh
ollama serve & ollama pull qwen2.5:7b        # or qwen2.5:14b for quality
.venv/bin/python src/book_retrieval.py        # build the textbook index (set BOOKS_DIR)
.venv/bin/streamlit run app.py                # UI
.venv/bin/python run_extraction.py <slides.pdf> [transcript.txt]   # CLI
OLLAMA_MODEL=qwen2.5:14b .venv/bin/python run_extraction.py <slides.pdf>   # higher quality
```

## Performance

Notes are scored 1–10 on three criteria by an independent judge (Claude) reading the
**source slides** and the **generated notes** side by side.

The tuned development deck (**L2 — Language Preprocessing**) climbed from **2.67 → 5.33**
(Coverage 6 / Correctness 5 / Structure 5, vs a 9.0 frontier-authored gold in a blind A/B
panel) across the staged-pipeline rewrite. Held-out decks the pipeline was *not* tuned on
generalise to similar quality, confirming the gains come from the architecture rather than
overfitting:

| Deck (held-out)                       | Coverage | Correctness | Structure |
| ------------------------------------- | :------: | :---------: | :-------: |
| **L3** — BoW / TF-IDF / OOV / Search  |  **8**   |    **4**    |   **5**   |
| **L5** — Word embeddings / Word2Vec   |  **7**   |    **3**    |   **6**   |

**Coverage and rendering are the strong axes** (7–8): topics are fully enumerated, the
LaTeX compiles cleanly, and definitions/cards/equations render correctly. **Correctness is
the weak axis** (3–4) — see Limitations.

## Limitations

- **Factual correctness is the ceiling, and it lives in the model weights.** The local
  model still hallucinates content that is *not* on the slides — e.g. it invented a
  TF-IDF cosine-similarity formula (L3) and described positional encoding as sinusoidal
  when the slides use plain one-hot (L5), and it substituted a textbook-standard vector
  analogy (`king − man + woman = queen`) for the slides' own example. The local-LLM
  **Review** stage does not reliably catch these (this is NLI-model territory).
  Orchestration buys completeness, structure and clean rendering — **not** factual
  reliability. A bigger local model helps; more prompting does not.
- **Worked-example boxes can contain undigested raw vectors.** The protected-span design
  places slide examples verbatim rather than letting the (unreliable) model narrate them,
  so number grids and character-n-gram fragments sometimes appear without interpretation.
  This is the deliberate trade-off against the model garbling them.
- **Occasional over-boxing / fragmentation.** Despite the consolidation few-shot, some
  topics still stack Definition + Properties + Pitfall + Worked-trace boxes where prose
  or a single table would read better.
- **Image/diagram extraction is not yet implemented.** Phase 2 of `PROPOSAL.md`
  (multimodal grounding — extracting slide figures into the notes) remains future work;
  `src/diagrams.py` only feeds the legacy pandoc path.

For development context and project conventions, see `CLAUDE.md`; for the roadmap, see
`PROPOSAL.md`.
