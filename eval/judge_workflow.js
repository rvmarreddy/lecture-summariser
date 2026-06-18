export const meta = {
  name: 'lecture-notes-judge-panel',
  description: 'Multi-judge blind scoring of generated lecture notes vs source slides (Coverage/Correctness/Structure)',
  phases: [
    { title: 'Judge' },
    { title: 'Synthesize' },
  ],
}

const OUT = '/Users/rithvikmarreddy/Projects/Apps/Lecture Summariser/eval/out'

// Comparison set. tag_deck.tex is the generated note; deck_slides.txt is the source.
const ITEMS = args && args.items ? args.items : [
  { tag: 'old', deck: 'L3', label: 'OLD: qwen2.5-7b, LLM-structure, no-NLI' },
  { tag: 'new', deck: 'L3', label: 'NEW: qwen2.5-7b, JSON-structure, NLI' },
  { tag: 'q3', deck: 'L3', label: 'NEW+Q3: qwen3-8b, JSON-structure, NLI' },
  { tag: 'new', deck: 'L5', label: 'NEW on held-out L5: qwen2.5-7b, JSON-structure, NLI' },
]
const JUDGES = (args && args.judges) || 3

const SCORE_SCHEMA = {
  type: 'object',
  properties: {
    coverage: { type: 'integer', minimum: 1, maximum: 10, description: 'how completely the notes cover the slide content a student must revise' },
    correctness: { type: 'integer', minimum: 1, maximum: 10, description: 'factual fidelity to the slides; penalise slide-contradictions and fabrications. Plausible enrichment from a standard textbook is acceptable, NOT a fabrication.' },
    structure: { type: 'integer', minimum: 1, maximum: 10, description: 'organisation/formatting: real tables for comparisons, sensible boxes, no fragmentation, clean headings, readable as revision notes' },
    correctness_errors: { type: 'array', items: { type: 'string' }, description: 'specific factual errors found (slide-contradictions, swapped definitions, invented numbers/formulas)' },
    structure_defects: { type: 'array', items: { type: 'string' }, description: 'specific structural defects (comparison flattened into box stack, leaked/duplicate titles, broken table, nested boxes, empty headings)' },
    notes: { type: 'string', description: 'one-sentence overall judgement' },
  },
  required: ['coverage', 'correctness', 'structure', 'correctness_errors', 'structure_defects', 'notes'],
}

function judgePrompt(item, j) {
  return `You are an independent, strict examiner (judge #${j}) scoring AI-generated revision notes for a student.
Use the SOURCE SLIDES as ground truth.

Read these two files with the Read tool:
- SOURCE SLIDES: ${OUT}/${item.deck}_slides.txt
- GENERATED NOTES (LaTeX): ${OUT}/${item.tag}_${item.deck}.tex

Score the notes 1-10 on Coverage, Correctness, and Structure (definitions in the schema). Be exacting and
concrete: list the actual correctness errors (slide-contradictions, swapped definitions, invented
numbers/formulas) and structural defects you find. Plausible enrichment from a standard NLP textbook
(Jurafsky & Martin) is acceptable and must NOT be counted as a fabrication. Judge only what is in the files.`
}

phase('Judge')
const scored = await pipeline(
  ITEMS,
  (item) => parallel(
    Array.from({ length: JUDGES }, (_, j) => () =>
      agent(judgePrompt(item, j + 1), { label: `judge${j + 1}:${item.tag}_${item.deck}`, phase: 'Judge', schema: SCORE_SCHEMA })
        .catch(() => null)
    )
  ).then((verdicts) => {
    const v = verdicts.filter(Boolean)
    const avg = (k) => v.length ? Math.round((v.reduce((s, x) => s + (x[k] || 0), 0) / v.length) * 10) / 10 : null
    return {
      ...item,
      n: v.length,
      coverage: avg('coverage'),
      correctness: avg('correctness'),
      structure: avg('structure'),
      total: v.length ? Math.round((avg('coverage') + avg('correctness') + avg('structure')) * 10) / 10 : null,
      all_correctness_errors: v.flatMap((x) => x.correctness_errors || []),
      all_structure_defects: v.flatMap((x) => x.structure_defects || []),
      per_judge: v.map((x) => ({ cov: x.coverage, cor: x.correctness, str: x.structure, note: x.notes })),
    }
  })
)

log('Scores: ' + scored.map((s) => `${s.tag}_${s.deck} cov ${s.coverage}/cor ${s.correctness}/str ${s.structure} (n=${s.n})`).join(' | '))

phase('Synthesize')
const digest = scored.map((s) =>
  `${s.label}\n  Coverage ${s.coverage} | Correctness ${s.correctness} | Structure ${s.structure} | Total ${s.total} (n=${s.n})\n  correctness errors flagged: ${s.all_correctness_errors.slice(0, 6).join('; ') || 'none'}\n  structure defects flagged: ${s.all_structure_defects.slice(0, 6).join('; ') || 'none'}`
).join('\n\n')

const SYNTH_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', description: 'Did the changes improve output? Compare OLD vs NEW (structure+NLI, same model) on L3, and NEW vs NEW+Q3 (model swap) on L3. Be quantitative.' },
    structure_isolated: { type: 'string', description: 'Did the deterministic structure renderer make structure stop being the limiting factor? cite the structure scores and defects.' },
    correctness_isolated: { type: 'string', description: 'Did the NLI filter / newer model move correctness? cite scores and the errors flagged.' },
    recommendation: { type: 'string', description: 'Which configuration should be the default, and what is the next highest-value change.' },
  },
  required: ['verdict', 'structure_isolated', 'correctness_isolated', 'recommendation'],
}

const synthesis = await agent(
  `Below are multi-judge averaged scores for generated revision notes under different pipeline configurations.
The experiment isolates two changes vs an OLD baseline: (1) deterministic JSON structure rendering + an
independent NLI fact-check (NEW, same qwen2.5-7b model), and (2) additionally a newer model (NEW+Q3 = qwen3-8b).
L5 is a held-out deck the NEW pipeline was not tuned on.

${digest}

Synthesize whether the changes improved output, isolating the structure change and the correctness change.`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA }
)

return { scored, synthesis }
