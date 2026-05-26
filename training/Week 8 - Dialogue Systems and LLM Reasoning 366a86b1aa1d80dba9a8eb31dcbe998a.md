# Week 8 - Dialogue Systems and LLM Reasoning

# 1. The three components of any dialogue system

<aside>
💡

**NLU (Natural language Understanding) -** determines what was said, what the intent is, what context applies. 

**DM (Dialogue Management) -** decides what to resolve, how, and what to remember.

**NLG (Natural language generation) -** produces the system’s response

</aside>

Two architectures combine this: **frame based** (deterministic, rule-driven) and **Dialogues-state** (probabilistic, flexible).

# 2. Frame based dialogue

> **Frame** is a predetermined structure representation, capturing all the information needed to perform a task. Three properties:
> 
> - Finite, Stored, Required

e.g. Frame for hotel booking might include required properties such as date, duration, location, people. 

## 2.1 Frame-based NLU

NLU for frame-based dialogue uses POS tagging plus a matcher (keyword regex, CRF, HMM or DL classifier) to map utterance token into slot value. 

- Each slot is filled when matched with partially filled frames triggering follow-up turns

## 2.2 Slot-filling mechanics

**Rule-based:** Regex / hand-written patterns for date/number/location etc.

- **Precise** but **brittle** (no handling of typos, slang, misformats)
- **Cheap** to start but expensive to maintain doesn’t scale well.

**CRF:** Sequence labelled trained on (token, slot-tag) pairs; same family as BIO/BIOES NER taggers.

- Needs labelled training data per domain.
- Generalises within distribution but degrades on out of distribution phrasings
- Mid cost

**LLM-based:** Prompt a GPT style LLM with structured output mode: The LLM extracts the slots and outputs a JSON object for the task.

- **Flexible** but **expensive**
- **non-deterministic** the same input might emit different JSON across calls

A hybrid is used in most production models. 

- Rule based for reliable slots (dates, numbers, currency). Regex
- CRF or LLMs then used for natural language slots like cuisine type, complaint, intent etc.
- Rule first then classifier as fallback keeping the cost down.

## 2.3 Frame-based DM (rule-based)

A small set of IF-rules:

1. If a slot is missing → ask for it
2. If all slots are filled → confirm with the user
3. if confirmed → act
4. If not confirmed → recheck/re-ask

## 2.4 Frame-based NLG

Slot substitution strings “What {requirement} would best suit your needs?| Templates minimise hallucinations, support localisation and ensure consistency. 

- Deterministic error messages
- But at the cost of rigid formulaic responses.

## 2.5 Pros and Cons

pros are deterministic outputs, testable finite states. Robust so can predict failures and cheap to build.

Cons are they are brittle to unexpected inputs. Poor linguistic scalability adding domains is expensive. Inflexible meaning it is hard to extend and no reasoning or uncertainty handling. 

# 3. Dialogue-state Architectures

> **Dialogue state:** a richer JSON style representation tracking current goal, constraints, preferences, prior actions, unanswered questions. and domain context.
> 
- **Interpretable, persistent, actionable.**

## 3.1 Belief state

> **Belief state** is a probability distribution over candidate slot values, allowing the system to act on confidence and handle miultiple hypotheses simultaneously.
> 
- Dialogue management can act to ask to disambiguate if the confidence is not high enough

## 3.2 Belief update - Bayesian filtering

A proper dialogue-state tracker updates the belief at each turn using Bayesian filtering. For a slot s taking value v, with new observation o from the user’s utternace:

$$
P(s=v|o) \propto P(o|s=v) \cdot P(s=v)
$$

The intuitation behind this is that given our prior about the slot $s$ we want to know the probability giving our observed. The idea is we are carrying forward the belief and accumulating evidence. 

## 3.3 Dialogue state DM

Dialogue management. Here we decide which slot to ask next, when to confirm, when to call APIs when to switch domain., how to minimise uncertainty. The policy can be implemented in several ways. 

## 3.4 Dialogue policy types.

- **Rule-based.** The same IF rules as frame based but operating oer the richer state that includes belief uncertainty. This could be a simple threshold if the probability is above a margin.
- **RL.** Learn a policy from interactions: the reward function is shaped by user satisfaction or task completion. The agent treats dialogues as a Markov Decision Process: state = belief state, actions = ask/confirm/recommend, reward=task success.
- **LLM-based.** Prompt as LLM with the current state and ask the next action. The LLM picks from a predefined menu and supplies arguments. More flexible to cases but non-deterministic and expensive per turn. Dominant in modern systems.
- **Hybrid.** Rules for safety critical actions (refund processing, account deletion, age verification); LLM for everything else. Production systems almost universally take this shape because the cost of an LLM error on a refund is asymmetric problematic.

## 3.5 Dialogue state NLGs

Mirrors the frame approach or hands off to an LLM. Hybrid is common and templates for rote response. LLM for free-form clarifications.

## 3.6 Pros and cons

context tracking across turns and natural interactions. Improves with data.

Cons are cost, data hungry needs dialogue corpora to train. hard to debug - non-deterministic flow.

## 3.7 Frame-based vs Dialogue state

Dialogue management logic. Frame based uses more IF rules over slots, Dialgoue state uses LLM and RL policies. 

Multi goal. Only one frame at a time. Alternative allows for parallel goals across fomains. 

Uncertainty. None can only do deterministic slots. Belief states used in dialogue state using probability distributions. 

Frame based best for narrow and well defined domains like customer support ticket triage. Dialogue state best for open-ended context rich conversations. Like a travel agent requires booking flight + restaurant + hotel. 

<aside>
💡

Don’t claim that DS is always better because Frame-based wins on control testability cost robustness in narrow domains. DS wins on natrual interaction, multi goal handling and uncertainty but at the cost of more data, higher cost and harder debugging. 

So narrow vs open datasets is the key discriminator. 

</aside>

## 3.8 Dialogue evaluation metrics

**Frame-based metrics**

- Task completion rate
- Turn count. How many turns to completion (lower generally better)
- Slot accuracy.

**Dialogue state metrics**

- Slot accuracy (per-slot accuracy) e.g. for a particular tag like location how accurate was it
- Joint accuracy scores 1 if all the slots in that turn were accurate so the average over that
- Success rate is what fraction of dialogues achieved the user’s goal.
- Human satisfaction - post call surveys explain the helpfulness ratings.

DS systems are hard to evaluate because it is hard to track where the error came from. Some benchmarks are able to decouple this. 

The best method is to let the model’s own state propogate and measure end-to-end success rate. 

- Modern dialogue evaluation often uses a strong LLM to score otuputs against a rubric. Cheaper than human evaluation but biased judges have a preference that need calibration.

# 4. The six reasoning types

| **Type** | **Mechanism** | **Example** |
| --- | --- | --- |
| **Deductive** | General rule $\rightarrow$ specific certain conclusion | "All A are B; Z is A; therefore Z is B" |
| **Inductive** | Pattern $\rightarrow$ generalisation (probabilistic) | "Climate has been warming over decades $\rightarrow$ likely hot in 300 years" |
| **Abductive** | Observation $\rightarrow$ most likely explanation | "Street is wet $\rightarrow$ probably rained" |
| **Analogical** | Comparison: A is to B as X is to Y | "Batteries:headlamps :: petrol:cars" |
| **Commonsense** | Assumption-based | "Dropping glass breaks it; ice-cream melts in a cone" |
| **Formal** | Puzzle / proof solving | "Solve $4x + 5 = 25$"; theorem proving |

**Deductive** is general → specific and certain. The conclusion is guaranteed

**inductive** is pattern → generalisation probabilistic more data can change the conclusion.

**Abductive** is observation → best-fit explanation also probabilistic but chooses the most likely cause from many. 

## 4.1 Per-type where LLMs succeed and fail

LLMs do not truly reason. The richer examinable picture is how LLMs perform on each of the six tasks. 

## 4.2 Tree of thoughts

CoT generates one linear reasoning chain. **Tree of thoughts** generates multiple chains in parallel and evaluates each at intermediate steps, and prunes the unpromising branches. Analogous to beam search but at the level of reasoning steps rather than tokens.

**Example:** “Schedule three meetings (A,B,C) on Tuesday given constraints”

**CoT:** produces one ordering attempt, If the first ordering hits a constraint violation, the model has to backtrack within the same chain, often poorly.

**ToT:** generates 3-4 candidates in parallel. After each scheduling step, an evaluator scores partial schedules; low scoring branches are pruned and remaining continue. Final answer is whichever branch reaches a complete valid schedule first or with the highest score. 

## 4.3 Self consistency vs CoT

- **Chain of thought.** Generates one resoning path. “Think step by step”.
- **Self-consistency.** Generates many reasoning paths by sampling with temperature. Each path produces an answer: take the majority answer as the final output.
- **Self-consistency of CoT.** The composition each the N samples is itself a CoT chain this is the canonical strong baseline on arithmetic and logic benchmarks.

## 4.4 Emergent abilities

> **Emergent ability:** a capability that appears suddenly once an AI model reaches a certain scale. Rather than a gradual improvement there is a threshold the causes rapid growth for the model.
> 

# 5. LLMs do not truly reason

LLMs are statistical next token predictors. They mimic reasoning rather than perform it. Any answer claiming genuine reasoning misses an examinable framing.

## 5.1 Internal mechanisms that approximate reasoning.

- **Transformers:** pattern matching over the training distribution. emergent abilities appear suddenly when scaled passed a certain parameter count
- **Self-attention -** relates premises to conclusion within a sequence
- **World models -** implicit fragemented statistical regularities about how things behave; not localised in any single parameter but emerge from large scale training.
- **Analogy -** embedding space similarity captures analyogical relationships.

## **5.2** External mechanisms that enhance reasoning

- **Tool use** - hand off arithmetic to a calculator, hand off facts to a RAG or web search
- **Self-consistency sampling -** generate multiple reasoning paths take majority answer
- **In-context learning -** few shot examples as a temporary task specific dataset; sensitive to wording and order.
- **Chain of Thought (CoT) -** prompt the model to produce intermediate reasoning steps before its final answer. The steps re-enter the model\s generative component biasinng later tokens toward consistency with earlier reaosning.

## 5.3 Model level enhancement

- Scaling laws - abilities appear suddenly with scale
- RLHF human feedback in an RL system
- **Fine -tuning on reasoning datasets**
- Distillation of reasoning traces - Outputs of another ai become inputs of the new model.

## 5.4 Process vs outcome supervision

OpenAI’s labelling every reasoning step (process supervision) beats labelling only the final answer (outcome supervision) - because outcome supervision rewards right answer via wrong method.