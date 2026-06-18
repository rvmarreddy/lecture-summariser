# Lecture Summariser

**Status:** shipped · **Category:** Apps · **Updated:** 2026-06-02
**Project folder:** `../../Apps/Lecture Summariser/`

## Purpose

A **local, offline** pipeline that turns lecture slides (PDF) + a transcript into structured, revision-ready notes (Markdown → LaTeX → PDF) with grounded figures. No external APIs. The one project with a public GitHub repo, linked on the CV.

## Findings

- Core pipeline works end-to-end (slides + transcript -> structured notes -> LaTeX/PDF), fully local, no external APIs. Public repo on the CV.
- Open differentiator: multimodal grounding quality (which figures matter, and their context).

## Ideas to improve first

The multimodal grounding is the differentiator, and the next improvements are agentic:

- **Diagram / image extraction tool** — pull useful diagrams and figures from the slides, **rank them by importance**, and embed the important ones inline in the notes (filtering decorative visuals and logos).
- **Make it agentic** — an agent that decides which visuals matter, where to place them, and how to caption them, for better outputs.
- **Internet agent** — a web-research agent that fetches better explanations and extra context to enrich the notes beyond the slides and transcript.
- Whether the fine-tuned note style generalises beyond the NLP course it was trained on.

## Next actions
- [ ] Build the diagram-extraction + importance-ranking agent; prototype the internet research agent

## Schedule
| Phase | Target | Status |
|---|---|---|
|  |  |  |

---

## Proposal

*Mirror of `../../Apps/Lecture Summariser/PROPOSAL.md` (canonical lives in the project folder).*

# Lecture Summariser — Proposal

> Canonical proposal. A mirror lives in `Project Hub/projects/proposal_Lecture Summariser.md` for planning.
> Converted from `LLM lecture note taker.docx` on 2026-06-02.

---

## **Lecture Summariser --- Project Roadmap**

### **Phase 1 --- MVP Foundation**

The goal here is a working end-to-end pipeline with no bells and
whistles.

**Inputs:** PDF slide deck + plain text transcript

**Deliverables:**

1.  Slide parser --- converts PDF pages to images and extracts raw text
    per slide

2.  Visual region detector --- identifies large non-text regions
    (diagrams, figures, charts) worth keeping

3.  Transcript chunker --- splits transcript into segments that can be
    assigned to slides

4.  Basic aligner --- maps transcript chunks to slides using keyword
    overlap and slide order

5.  Note generator --- prompts an LLM to produce structured notes per
    slide section using the aligned text

6.  LaTeX exporter --- outputs a compilable .tex file with sections,
    text, and embedded slide images

**Success criterion:** Given a lecture, produce a readable, structured
.tex document that a student could use for revision.

### **Phase 2 --- Multimodal Grounding**

The goal here is making the image-context link meaningful, which is the
core novelty of the project.

1.  Improved visual region detection --- filter out decorative elements,
    logos, and backgrounds; retain only pedagogically useful visuals

2.  Per-figure context extraction --- for each extracted figure,
    retrieve the transcript span that explains it and generate a short
    caption or description

3.  Semantic alignment --- replace keyword-based alignment with
    embedding similarity so the system handles cases where the lecturer
    paraphrases or elaborates beyond the slide text

4.  Text-image grounding in notes --- figures appear inline next to the
    explanation that corresponds to them, not just appended at the end
    of a section

**Success criterion:** Each extracted figure in the output has a
contextually accurate explanation drawn from the transcript, not just
the slide bullet text.

### **Phase 3 --- Richer Output and Quality**

1.  Equation extraction and formatting --- detect mathematical content
    on slides and format it properly in LaTeX

2.  OCR for diagram labels --- read axis labels, annotations, and
    callouts within figures to improve captions

3.  Note quality refinement --- add key takeaways, definitions, and
    example callouts where the transcript supports them

4.  Revision mode vs full notes mode --- shorter bullet-driven output vs
    longer prose explanation, selectable at export time

### **Phase 4 --- Extension (if time allows)**

1.  Flashcard generation --- derive question-answer pairs from grounded
    notes for spaced repetition

2.  Multi-lecture support --- handle a series of lectures and produce a
    unified document with consistent section numbering

3.  Slide format flexibility --- support .pptx input directly, not just
    PDF

### **Core Technical Challenges (in priority order)**

1.  Deciding which visual regions are worth extracting

2.  Aligning transcript spans to the correct slide and figure

3.  Generating captions that reflect what the lecturer said, not just
    what the slide shows

4.  Keeping the LaTeX output clean and compilable across varied inputs

### **Module Summary**

  -------------------------------------
  **Module**            **Phase**
  --------------------- ---------------
  Slide parser          1

  Visual region         1 → refined in
  detector              2

  Transcript chunker    1

  Transcript-slide      1 → semantic in
  aligner               2

  Per-figure context    2
  extractor             

  Note generator        1 → richer in 3

  LaTeX exporter        1

  Equation / OCR        3
  extraction            

  Flashcard generator   4
  -------------------------------------
