# Lecture Summariser

Turns a lecture slide PDF into LaTeX revision notes. A local model served by [Ollama](https://ollama.com) writes the notes, and passages from a textbook index keep them grounded. Everything runs offline, so no API keys are needed.

## How it works

The pipeline is `run_extraction.run_pipeline`. For each deck it does the following.

1. Extract the slide text with PyMuPDF and strip placeholders that break LaTeX.
2. Group slides into topics by their `Section: Subtitle` header, and drop syllabus, objectives and recap slides. Each topic becomes one `\section`, so no topic is lost. A group with a vague subtitle ("The Problem") is merged into the section it belongs to.
3. Write each topic in stages, each a separate call to the model (`OLLAMA_MODEL`, default `qwen2.5:7b`). The code is in `src/note_writer.py`.
   - Grounding. MiniLM retrieves textbook passages for each slide, then the model keeps only the passages whose main subject is the topic. This removes passages that only share a few words with the slide.
   - Protected examples. Worked examples, vectors and tables from the slides are taken out before writing and put back as LaTeX at the end, so no stage can rewrite them.
   - Content. Numbered points grounded in the slides and passages.
   - Review. The model checks each point against the source.
   - NLI check. A cross-encoder entailment model drops points the source contradicts. It is a different model from the writer, so it catches swapped definitions and invented numbers that the review misses. Set `NLI_MODEL`, or turn it off with `NLI_ENABLE=0`.
   - Structure. The model returns JSON blocks (definitions, comparison tables, fact cards, common mistakes, prose) and Python renders them to LaTeX. Comparisons always become real tables and boxes are always balanced. `STRUCTURE_MODE=llm` uses the older path where the model writes the LaTeX itself, taught by `src/structure_fewshot.tex`.
4. Merge sections with the same title, remove duplicate boxes and escape stray LaTeX characters.
5. Wrap the result in `src/note_preamble.tex` and build the PDF with `xelatex`.

## Setup

You need Python, `xelatex`, and Ollama with a model pulled. Run everything from the repo root.

    python -m venv .venv
    .venv/bin/pip install -r requirements.txt
    ollama pull qwen2.5:7b
    BOOKS_DIR=path/to/books .venv/bin/python src/book_retrieval.py   # builds the textbook index

Then use the web app or the command line:

    .venv/bin/streamlit run app.py
    .venv/bin/python run_extraction.py slides.pdf [transcript.txt]

`OLLAMA_MODEL=qwen3:8b` gave the most correct notes in the tests below. The first run downloads the NLI model (about 440 MB) once.

## Results

A larger model acting as judge scored the notes 1 to 10 on coverage, correctness and structure, reading the source slides and the notes side by side. It ran three times per setup and was not told which setup made the notes. `eval/run_eval.py` generates the notes and `eval/structure_metrics.py` counts structural defects. Scores are out of 30, on lecture 3, which was held out during development.

| Setup | Coverage | Correctness | Structure | Total |
|---|:-:|:-:|:-:|:-:|
| Model writes the LaTeX, no NLI check | 7.3 | 3.0 | 4.3 | 14.6 |
| JSON structure and NLI check, qwen2.5:7b | 6.3 | 4.3 | 5.0 | 15.6 |
| Same, qwen3:8b | 6.3 | 5.0 | 4.7 | 16.0 |

On the same lecture the JSON renderer doubled the comparison tables from 3 to 6, with no LaTeX defects. Lecture 5, also held out, scored 15.0 with structure 5.3. The cost is about one point of coverage, because the NLI filter and the renderer drop some material to stay faithful.

## Limitations

- The NLI check only drops points the source contradicts. When a slide has no text to check against, the model can still invent a plausible section, and textbook material from a neighbouring topic can slip in the same way. A check that drops points nothing supports would be the next step.
- Worked examples are copied as printed, so grids of numbers appear without row labels.
- Each topic is written on its own, so the same point can appear in two sections. Only exact duplicates are removed.
- Diagrams are not extracted. `src/diagrams.py` only serves an older pandoc path.

## Licence

MIT
