# Lecture Summariser

Generates LaTeX revision notes from a lecture slide PDF, using a local
[Ollama](https://ollama.com) model and a textbook for grounding. Runs entirely offline,
so no API keys are needed and nothing leaves your machine.

## Pipeline

The architecture lives in `run_extraction.run_pipeline`. For each deck:

1. **Extract & clean**: pull slide text with PyMuPDF and strip `emoji_xxx` placeholders
   that break LaTeX (`_clean_slide_text`).
2. **Plan an outline** (`_topic_groups`): group slides by their `Section: Subtitle`
   header into one topic each, dropping syllabus/objectives/feedback/recap slides. Every
   topic becomes its own `\section`, so none can be silently dropped. Adjacent groups
   sharing a section prefix merge when a subtitle is vague ("The Problem"), so one slide
   theme is not split across sections; standalone vague subtitles get qualified by their
   prefix.
3. **Per topic, a staged Content, Review, Structure pipeline** (`src/note_writer.py`),
   each stage a focused **Qwen-via-Ollama** call (`OLLAMA_MODEL`, default `qwen2.5:7b`;
   use `qwen2.5:14b` for real quality):
   - **Scope-linked grounding** (`_ground` + `verify_relevant`): per-slide MiniLM
     retrieval from a local textbook index, then a **local-LLM scope gate** that keeps
     only passages whose *primary subject* is the topic (kills the bi-encoder's
     surface-word matches). The "different subject" contrast is derived from the deck's
     own other topics, so the gate is not hard-wired to one course. Set `SCOPE_MODEL`.
   - **Protected spans** (`_protected_block`): the slide's worked examples, vectors,
     similarity matrices and stem tables are **extracted and frozen** as the opaque token
     `[[EXAMPLES]]`, deduped across topics, and substituted back as real LaTeX only at
     the very end, so no stage can re-author or break them.
   - **Content** (`content_points`): grounded numbered points (coverage + correctness).
   - **Review** (`review_points`): the model fact-checks each point against the source.
   - **Independent NLI check** (`nli_filter`): a cross-encoder entailment model — a
     different architecture and failure-mode than the writer — drops points the source
     *contradicts* (the swapped-definition / invented-number class the same-model review
     misses). Set `NLI_MODEL`; disable with `NLI_ENABLE=0`.
   - **Structure** (`structure_points`, deterministic): the model emits a *structured
     JSON* section (typed blocks — definitions, comparison tables, fact cards, traps,
     prose) and a Python renderer (`_render_section`) turns it into guaranteed-clean
     LaTeX — comparisons always become real tables, boxes are always balanced and
     correctly labelled, nothing nests, empty columns are pruned. Formatting is off the
     model, so structure can't be the limiting factor. `STRUCTURE_MODE=llm` forces the
     old free-LaTeX path (taught by `src/structure_fewshot.tex`).
4. **Deterministic dedup & cleanup**: merge same-titled sections, drop duplicate boxes
   (`dedup_sections`/`dedup_blocks`), fix box titles, and a math-aware `_sanitize_body`
   (escape stray `_`/`#`/`%`/`&`/currency-`$` in prose, balance environments).
5. **Render**: wrap in the muted `src/note_preamble.tex` template, then run `xelatex` to PDF.

## Setup & usage

Requires the project virtualenv (`.venv/bin/python`), `xelatex`, and a local Ollama
server with the model pulled. Run from the repo root so `inputs/`, `models/` and
`outputs/` resolve.

```sh
ollama serve & ollama pull qwen2.5:7b        # default; qwen3:8b measured best for correctness
.venv/bin/python src/book_retrieval.py        # build the textbook index (set BOOKS_DIR)
.venv/bin/streamlit run app.py                # UI
.venv/bin/python run_extraction.py <slides.pdf> [transcript.txt]   # CLI
OLLAMA_MODEL=qwen3:8b .venv/bin/python run_extraction.py <slides.pdf>      # higher correctness (see Performance)
```

The first run downloads the NLI cross-encoder (`NLI_MODEL`, ~440 MB) once, then runs offline.
Toggles: `STRUCTURE_MODE=llm` (old free-LaTeX path), `NLI_ENABLE=0` (skip the NLI check),
`NLI_ENTAIL_MIN` / `NLI_CONTRA_MIN` (NLI drop thresholds).

## Performance

Notes are scored 1 to 10 each on Coverage / Correctness / Structure by an independent
frontier-model judge reading the **source slides** and the **generated notes** side by
side. The reproducible harness lives in `eval/` (`run_eval.py` generates, `structure_metrics.py`
counts objective defects, `judge_workflow.js` runs a blind multi-judge panel).

**A/B (held-out L3, blind 3-judge panel, /30 total):** the deterministic structure renderer
plus the independent NLI fact-check, then a newer model, both improve output:

| L3 configuration                                  | Coverage | Correctness | Structure | Total |
| ------------------------------------------------- | :------: | :---------: | :-------: | :---: |
| OLD (LLM-authored LaTeX, no NLI)                  |   7.3    |     3.0     |    4.3    | 14.6  |
| **NEW** (deterministic structure + NLI, qwen2.5-7b) | 6.3    |   **4.3**   |  **5.0**  | **15.6** |
| **NEW + qwen3-8b**                                |   6.3    |   **5.0**   |    4.7    | **16.0** |

Objective (non-LLM) structure metrics on L3: comparison **tables 3 → 6** with **zero**
LaTeX defects (no column-count mismatches, leaked labels, markdown-bullet leaks, or broken
rules — the whole LLM-LaTeX defect class is gone). Held-out **L5** generalises (15.0/30,
Structure 5.3) — the renderer is a pure function of the JSON, so it is deck-independent.
The cost is **~1 point of Coverage** (the NLI filter and JSON renderer trade breadth for
fidelity). See Limitations.

## Limitations

- **Correctness improved but the ceiling persists, and it lives in the model weights.**
  The independent NLI check now drops the *contradiction* class (swapped definitions,
  inverted formulas, garbled numbers), but by design it keeps *neutral* content — so
  **fabrication-by-omission survives**: when a slide range has no extractable text, the
  model can invent a plausible section the source cannot contradict. Cross-topic textbook
  bleed (e.g. n-gram language-model material in a bag-of-n-grams deck) survives for the
  same reason. The next lever is a groundedness/*support* gate (drop a block when nothing
  entails it), not only a contradiction gate. A newer/bigger local model also helps.
- **Worked-example boxes contain verbatim, unnarrated vectors.** The protected-span
  design (`_protected_block`, outside the structure stage) places slide examples exactly
  as printed rather than letting the unreliable model narrate them, so number grids appear
  without per-row labels; attaching slide labels at capture time is the next refinement.
- **Coverage regressed ~1 point** under the NLI filter + JSON renderer — the breadth-for-
  fidelity trade. Tuning `NLI_ENTAIL_MIN`/`NLI_CONTRA_MIN` is the lever.
- **Cross-section repetition survives.** Generation is per-topic, so no stage sees two
  sections at once; only exact duplicates are removed deterministically.
- **Image/diagram extraction is not implemented** and is now scoped as optional/future
  (see the proposal); `src/diagrams.py` only feeds the legacy pandoc path.

For the project roadmap, see `lecture-summariser-proposal.tex` / `.pdf`.
