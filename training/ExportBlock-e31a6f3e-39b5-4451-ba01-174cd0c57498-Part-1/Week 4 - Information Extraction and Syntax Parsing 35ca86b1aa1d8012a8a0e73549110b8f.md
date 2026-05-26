# Week 4 - Information Extraction and Syntax Parsing

So far we’ve been focused on embeddings and using text to perform tasks. This week we will be looking at how we can extract additional information from text

# 1. Information Extraction - the why and the pipeline

> **Information Extraction (IE).** The process of extracting structured information from raw text rather than just embedding or classifying it. Outputs feed knowledge graphs, business analytics or intent detection.
> 

The lecturer introduces IE with three applications archetypes:

## 1.1 Three motivating applications

**Knowledge graphs.** “Josie joined the University of Bath in 2013, and the University of Bristol in 2020” becomes a structure triple:

`Josie -joined -> Uni of Bath (date: 2014)`

A knowledge graph then composes many such triples into a navigable network for downstream reasoning, search or recommendation. 

- **Nodes.** are entities (people, organisations, locations, events)
- **Edges.** are typed relations (joined, employs, is_in)

Using this graph you can answer queries like `“which Bath alumni now work at Google”` by graph traversal. 

**Business analytics.** "GWR's quarterly revenue rose 8% to £13.6m" yields:

`Metric: Quarterly Revenue
Change: +8%
Value: £13.6m
Period: Q?`

- Auto-extracted from earnings report at scale, this feeds dashboards and trading signals.
- The schema is fixed up front (here: metric, change, value, period); IE is the process of mapping free text into the schema slots.

**Intent understanding.** “Who is the president of the USA?” is parsed into:

`Entity: USA
Intent: Search
Slot: President`

This slot-filled representation is what dialogue systems and search engines build on. 

## 1.2 The standard IE pipeline

![diagram.jpg](Week%204%20-%20Information%20Extraction%20and%20Syntax%20Parsing/diagram.jpg)

**Raw Text → Tokenisation:** this has been discussed in prior lectures slicing raw text into distinct manageable tokens without removing grammatical signals like stopwords. 

**POS Tagging:** Evaluates morphological context to assign an underlying grammatical class to each token (e.g. `NNS` for plural noun).

**NER:** Groups tokens into meaningful spans and categorises them by entity type, such as People (`PER`), Organisations (`ORG`), or locations (`GPE`).

**Relationship Extraction (RE):** Utilises the POS structure and NER tags to identify connnections between entities, formatting them into structured `(subject, relation, object)` triples. 

**Triples / Knowledge Graph:** The final structured output. These connected nodes and edges can now be easily queried for downstream business analytics, intent detection, or personalised recommendations without requiring further language model processing.

# 2. POS Tagging

> **Part-of-Speech (POS) tagging.** Use grammatical and morphological context to assign each token its underlying word-class.
> 

## 2.1 Three classes of POS

- **Open-class** (admits new members over time): *Adjective, Adverb, Noun, Proper Noun, etc.* New nouns constantly enter English constantly.
- **Closed-class** (fixed inventory): Adposition (in, on, at), Auxillary (have, will), Coordinating Conjunction (and, but), Determiner (the, a), Numeral, Particle, Pronoun, Subordinating Conjunction (because, although).
- **Other:** punctuation, symbols

## 2.2 Penn Treebank tag inventory

The **Penn Treebank** convention uses 36 fine-grained tags. The most common ones:

`NN (Singular noun), NNS (Plural Noun), JJ (adjective), JJR (adjective, comparitive)`

## 2.3 Why context matters

**Worked example: Same word, four POS tags**
• "Alice turned on the **light**" → "light" = Noun (`NN`)
• "Alice's candle will **light** the cave" → "light" = Verb (`VB`)
• "Alice carried a **light** backpack" → "light" = Adjective (`JJ`)
• "Alice travels **light**" → "light" = Adverb (`RB`)

Word form is insufficient - the surrounding tokens determine the tag. Rules out lookup-based approaches. 

## 2.4 Three POS-tagging methods

| Method | Pros | Cons |
| --- | --- | --- |
| Rule-based (grammars) | Interpretable: no training data needed | Cannot scale; lacks domain transfer |
| Deep Learning (e.g. fine-tuned ELMo BiLSTM) | Generalisation: high precision | Lacks interpretability; requires large training data |
| Probabilistic (HMM) | Works with limited data; computationally efficient | Independence assumptions; requires smoothing for unseen tokens |

**Rule-based example.** e.g. Any word starting with a capital letter not following a full-stop is proper noun. Broken “…? She’s …” she’s is not a proper noun.

- impractical maintaining a rule set across domains is impractical.

### **HMM formalism**

The HMM approach maximises the joint probability of a sequence T given a word sequence W:

$$
\arg \max_{T} P(T,W) = \arg \max_{T} \prod_{i} P(t_{i} | t_{i-1}) \cdot P(w_{i} | t_{i})
$$

The transition probabilities $P(t_i | t_{i+1})$ and emission probabilities $P(w_i|t_i)$ are estimated from a tagged training corpus, with smoothing

**Viterbi algorithm.** Finding the most probable tag sequence requires search over $|T|^n$ possibilities (exponential in sentence length). 

- The Solution: By saving the best path so far it throws away all the terrible guesses and avoids recalculating. This is an effective $O(n \cdot |T|^2)$.

<aside>
💡

MCQ trap:

Benefits of rule based POS tagging. The correct answer is interpretability. Rule based does not generalise since you hard code rules. The headline tradeoff is interpretability vs scaling.

- An issue with POS is that it might tag incorrectly. For example “Obama visited New York in 2010”. “Obama” as NNP, “New” as NNP, “York” as NNP. But “New York” is one entity. POS doesn’t know that.
</aside>

# 3. Named Entity Recognition (NER)

> **NER.** Identify singular or grouped tokens that designate name entities, and assign each entity a type.
> 
- Common labels: [PER] People, [ORG] Organisations, [LOC] Locations, [GPE] Geo Political Entity.
- These can be custom entities for certain corpuses.

## 3.1 Three tagging schemes (IO / BIO / BIOES)

The schemes differ in how they encode entity boundaries

|  | Andy | lives | in | Swindon | by | the | River | Ray |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **IO** | I-PER | O | O | I-GPE | O | O | I-LOC | I-LOC |
| **BIO** | B-PER | O | O | B-GPE | O | O | B-LOC | I-LOC |
| **BIOES** | S-PER | O | O | S-GPE | O | O | B-LOC | E-LOC |

B-X is the beginning of your tag, I-X is inside, E-X is the end of the tag and S-X is a singular entity. O is outside not an entity. 

| **Scheme** | **Trade-off** |
| --- | --- |
| IO | Easy to encode (binary classification) But cannot separate adjacent same-type entities; start indices unknown.  |
| BIO | Adds boundary detection (B marks the start); endings still unclear |
| BIOES | Very clear boundaries; separates independent entities. More labels and modelling complexity used |
- IO is fine where adjacency is rare. BIO is the production default - good in most cases.
- BIOES give the cleanest boundaries but cost of more labels and more training data needed to learn them. CRFs commonly use BIOES for sequence labelling because explicit end markers help the transition model.

# 4. Relationship Extraction

Once entities are tagged, RE finds the relation between them: so triples of the form (subject, relation, object). 

- Subject: Anglo Saxons, Relation: entered, Object: Exeter.

## 4.1 Three RE methods

### Rule-based

Extracts information by defining specific structural patterns using POS and NER tags.

**How it works**

- RegExp or Regular expressions act as the search formulas for these rules.
- Instead of looking at the raw words, RegExp takes the sequence of assigned tags, squashes them into a single string, and searches for exact pattern matches.

**Example pattern `[PER] + [VERB] + [GPE]`**

Find a Person entity (subject), followed by a verb (relation), followed by a Geo-political entity (object).

- Subject: Anglo Saxons, Relation: entered, Object: Exeter.

**Takeaway**

It is highly precise for the exact formulas you write, but brittle this specific rule would fail if the sentence was rephrased to “Exeter was entered by the Anglo Saxons”.

### **Supervised Classification**

Instead of manually writing rules, you train a ML models to predict the relationship between entities.

**How it works**

- You provide the model with a passive dataset of pre-labelled triples.
- Modern standard: Current models use BERT-pair embeddings as features to understand the deep context surrounding entity pairs.

**Takeaway**

- **High precision:** as long as the training data is large and high-quality the classifier will be very accurate.
- **Expensive labelling:** it requires a human annotated data to train effectively.
- **Struggles with rare relations:** Needs many examples to learn a pattern.

### Distant supervision

Instead of using manually labelled data, you leverage an existing database of known facts to automatically generate training examples from unlabelled text.

**How it works**

- Start with a known database of triples `(England, is_in, United Kingdom)`
- Search through unlabelled text and **blindly assume** that any sentence containing both of these entities expresses the exact relationship.

**Takeaway**

- **Cheap and scalable:** it bypasses the cost of human annotation, allowing you to easily scale up and extract massive amounts of training data from large unlabelled corpora.
- **High noise:** it assumes every co-occurence implies the relation even if the sentence mentions two entities for a completely different reason.
- **Multiple relation conflicts:** it struggles when an entity pair shares multiple relations
    - e.g. Bill Gates and Microsoft are associated via founded, owns, works_at, and donated to.

**Potential fixes**

- **Syntactic filtering:** You can clean up the noisy data by applying rules to the matched sentences - such as restricting the syntactic distance between two entities or requiring that at least one verb appears between them.

# 5. Why we need Context-Free Grammars

Regular expressions are designed to formalise **regular languages.** They are excellent for flat pattern-matching tasks in information extraction. They rely on combination of symbols to match linear strings:

- Character classes
- Sets
- Qualifiers

- Logical
- Grouping

e.g. great at capturing email addresses or phone numbers but natural language is rarely a flat sequence. 

## 5.1 The limits of regular languages

Regular expressions strictly handle flat, sequential patterns and structurally fail when faced with recursion or memory-dependent pairings.

- **Matched balanced patters:** Languages such as $L=a^n b^n | n>0$ (e.g. depth of a’s and b’s compressing repeated characters)
- **Arbitrary nesting:** balanced boolean queries with unpredictable depths such as `((A AND B) OR C) AND (D OR E)`
- **Recursive grammar:** matched brackets in arithmetic expressions, nested HTML tags, or syntactic nesting in natural language aren’t understood by RegExp.

RegExp cannot count or recurse - it has no memory beyond its current state. It is a finite-state machine and cannot keep an unbounded counter for matched depth because it has no memory beyond its current state.

### The Chomsky hierarchy

Languages partition into hierarchy by the formal power of the grammars that recognise them:

Regular $⊂$ Context-Free $⊂$ ****Context-Sensitive $⊂$ Recursively Enumerable

## 5.2 Context-Free Grammars

Context free grammars sit a step above regular expressions because they support recursion, they can represent structurally nested and balanced patterns. 

> **Context-Free Grammar (CFG):** A 4-tuple $\textbf{G}=(\textbf{V}, \sum, \textbf{P}, \textbf{S})$
> 
> - $\textbf{V}$ **(non-terminals)** = finite set of syntactic variables
> - $\sum$ **(terminals)** = finite set of terminal symbols (the final tokens that appear in the input)
> - $\textbf{P}$ **(production rules)** = A finite set of production rules of the form $A \rightarrow \beta$ where $A \in V$ and $\beta$ is a string over $V \cup \sum$.
> - $\textbf{S}$ **(starting symbol)** = the starting symbol must be in V

### Power of “Centre Embedding” (Memory)

To understand how CFGs support recursion, look at the classic rule used to capture balanced pairs $a^n b^n$:

$$
L \rightarrow aLB|\epsilon
$$

**Decoding the rule:**

- L: This is a **non-terminal (a variable)**. It is what needs to be expanded.
- aLb: This is the recursive part. it means you replace the variable L with the terminal character a, followed by a new variable L, followed by the terminal character b.
- | : This logical OR operator gives the grammar a choice of which rule to apply
- $\epsilon$: This represents the **empty string.** This is the base case that stops the recursion. it means “replace L with nothing.”

L → aLb → a(aLb)b → aa(L)bb → aa($\epsilon$)bb → aabb

### Example: CFG email validation

CFGs validate as well as parse, which is why they are used in query languages and compilers. If a string cannot be derived from the starting symbol, it is rejected

- S      → Name '@' Domain
- Name   → Name '.' Word | Word
- Domain → Domain '.' Word | Word
- Word   → WordChar | Char
- Char   → 'a' | 'b' | ... | 'Z' | '0' | ... | '9'

![diagram.jpg](Week%204%20-%20Information%20Extraction%20and%20Syntax%20Parsing/diagram%201.jpg)

- Derivation tree for parsing `a1@g.uk`

So an input like a@gm@hm.uk fails because you cannot derive the second @, so the grammar rejects the string. 

### Example: Parse tree for “lucy works at Google”

So instead of raw words, NLP uses CFGs build over POS tags to expose syntactic structure. Once the parse tree is build, we can travers it with extraction rules of find relationships. 

![diagram.jpg](Week%204%20-%20Information%20Extraction%20and%20Syntax%20Parsing/diagram%202.jpg)

## 5.3 Ambiguity in CFGs

“I shot an elephant in my pajamas”. This sentence has two valid parses:

![diagram.jpg](Week%204%20-%20Information%20Extraction%20and%20Syntax%20Parsing/diagram%203.jpg)

![diagram.jpg](Week%204%20-%20Information%20Extraction%20and%20Syntax%20Parsing/diagram%204.jpg)

- Is the elephant wearing pajamas

So we can have two versions of the prepositional phrase that requires semantic/world knowledge to fix. This is a syntactical ambiguity when a sentence can have two different meanings due to its grammar. 

# 6. Two parsing strategies

A parser is the engine that takes a CFG and an input sentence and figures out how to build the final parse tree. Two strategic choices: top down vs bottom up.

## 6.1 Top-down (Recursive Descent)

> **Recursive descent parsing.** A top-down parser: start at S and recursively expand non-terminals using production rules until terminals match the input. Builds the tree from root to leaves: backtracks on failure.
> 
- **Backtracking.** Slow and inefficient.
- **The Left recursion death trap.** If a grammar has a left-recursive rule like Name → Name.[cite start]Word, a top-down parser will read the first Name, expand it to another Name, and get stuck in an infinite loop without ever checking the actual input.

**Variant: LL parsers like LL(1).** A top-down parser that looks 1 token ahead to decide which production to apply eliminating backtracking. Used for many programming language compilers; requires the grammar to be LL(1) compatible which natural language grammars rarely are.

## 6.2 Bottom up (Shift Reduce)

> **Shift-reduce parsing.** A bottom-up parser: start with input tokens (already POS-tagged and progressively combine adjacent symbols by reverse-applying production rules until reaching S.
> 

Bottom-up parsers use a temporary holding area called a **Stack** and perform only two actions:

1. **Shift:** Pick up the next token from the input sentence and move it onto the Stack.
2. **Reduce:** Look at the top items sitting on the Stack. If they perfectly match the right side of a CFG rule (e.g. you have PER on the stack and the rule is NP → PER), you “reduce” them by swapping them out for the left side of the rule (NP).

**LR Parsers:** Production grade parsers like **LR(1)** use lookahead to decide whether it is faster to Shift the next token or Reduce what is currently on the stack. 

| Step | Stack (Holding Area) | Action Taken | Why? |
| --- | --- | --- | --- |
| 0 | (empty) | Shift PER | Move first word to stack.  |
| 1 | PER | Reduce $NP \rightarrow PER$ | PER perfectly matches an NP rule.  |
| 2 | NP | Shift ADV | Move next word.  |
| 3 | NP, ADV | Shift V | Move next word.  |
| ... | ...skipping to end... | ... | ... |
| 9 | NP, VP | Reduce S → NP VP | The whole sentence is found!  |
| 10 | S | Accept | Parsing successful.  |

## 6.3 Comparing the two strategies

| **Feature** | **Top-Down (Recursive Descent)** | **Bottom-Up (Shift-Reduce)** |
| --- | --- | --- |
| **Direction** | S → leaves | Tokens → S |
| **Mechanic** | Uses a function per non-terminal | Uses a Stack + Transition table |
| **Backtracking?** | Yes (explores wrong paths) | No (more efficient) |
| **Left-Recursion** | **Fails:** Loops forever | **Succeeds:** Handles naturally |

<aside>
💡

Pitfalls

**Recursive descent = top-down** (Start at S, expand toward leaves). **Shift-reduce = bottom-up** (start at tokens, reduce up to S).

- **R**ecursive **D**escent dives **D**own from S”; “**S**hift-**R**educe **R**olls up”.

Some inputs have multiple valid parse trees under the same grammar. Real CFGs include disambiguation rules or use probabilistic CFGs (PCFGs). 

</aside>

## 6.4 From parse tree to relationship

Our goal in Relationship Extraction (RE) is to pull out a (Subject, Relation, Object) triple. The parse tree gives us the exact syntactic map to find these pieces:

- **The subject:** Usually the Noun Phrase that is the immediate child of the root sentence S.
- **The object:** often the Noun Phrase that is tucked inside the Prepositional Phrase PP.
- **The Relation:** determined by the base form of the main Verb V.

### Dependency parsing

Up until know we looked at **constituency parsing** (CFGs building those abstract NP and VP phrase boxes). **Dependency parsing** is a different strategy that skips the abstract boxes entirely and draws arrows directly between tokens. 

**How it works**

Instead of nesting phrases, a dependency tree connects a **head word** to its **dependents** using specific grammatical labels. 

e.g. “Lucy works at Google”

works["works (Head Verb)"]

```
works -- nsubj (Nominal Subject) --> Lucy["Lucy"]
works -- prep (Preposition) --> at["at"]
at -- pobj (Object of Preposition) --> Google["Google"]
```

Extraction rules are more direct and efficient.

# 7. Lab 4

**1. Why doesn't the code remove stopwords/punctuation here?** Because both stopwords (like "at", "in", "the") and punctuation are essential for determining the grammatical structure of the sentence. A dependency parser needs them to figure out how phrases connect to one another.

**2. What happens if you remove the "." after "Mt"?**It causes a failure in the very first step: tokenization. The tokenizer gets confused, which leads to the system mis-tagging the entire entity "Mt Snowdon" during the NER stage.

**3. What are the limitations of the simple RE function?** The simple function extracts relationships purely based on physical adjacency (pairing nouns that happen to be next to a verb). It doesn't understand the actual grammatical roles (who is doing the action vs. who is receiving it), leading to wildly incorrect triples on complex sentences. Real systems must use dependency relations instead.

**4. How could dependency graphs aid sentence splitting?** A dependency graph connects all the words in a single sentence back to one primary "root" word (usually the main verb). If you map out a chunk of text and find two completely separate, unconnected graphs with two different roots, you can reliably infer that you are looking at two distinct sentences!

**5. Compare dependency graphs across languages - what challenges arise for multilingual approaches?**
Languages structure thoughts differently. As seen with the French example, grammar isn't just a 1-to-1 word swap; the actual dependency tree changes shape because of different auxiliary verbs, word orders, and grammatical rules.