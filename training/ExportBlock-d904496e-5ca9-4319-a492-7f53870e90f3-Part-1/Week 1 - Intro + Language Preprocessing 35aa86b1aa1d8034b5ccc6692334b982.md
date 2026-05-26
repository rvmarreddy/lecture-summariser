# Week 1 - Intro + Language Preprocessing

# 1. What NLP is and why is it hard

NlP sits at the intersection of computer science, linguistics and artificial intelligence. 

> **Natural Language Processing** is a field of AI that enables computers to understand, interpret, and generate human language, converting it into something a machine can act on.
> 
- e.g. conversational AI operates a continuous loop: **speech recognition** converts audio into data, **Language Processing** interprets the intent, and **Text/Speech Production** delivers the final output back to the user.

## 1.1 The components of language

| **Tier** | **What it covers** | **Example** |
| --- | --- | --- |
| Pragmatics | Contextual meaning of phrases | “Can you open the door?” - request, not a query about ability |
| Semantics | Literal meaning of phrases | "I saw a duck” - the bird not the action. |
| Syntax | Sentences and phrases combining words | Subject - verb - object structure |
| Morphology | Individual words and internal components | run + ing → running |
| Phonetics | Combinations of sounds | How “th” is pronounced |
| Phonemes | Speech sounds | /b/, /æ/, /t/ in “bat” |

## 1.2 The four core challenges

> **Creativity:** Poetic, metaphorical, or otherwise non-literal use of language.
> 
> - “The rain and wind argues all night”, “The server was drowning in requests”.

Literal-meaning models cannot unpack metaphor without world-knowledge bridges

> **Diversity:** Dialects, code-switching (mixing languages within a single prompt), and cultural variation in expression.
> 
> - “Oi be goin’ down t’farm later, me lad” (West Country),
> - “Je vais à meeting with my boss” (code-switching).
> - “The nurse and doctor were running before she tripped” (cultural assumption: which one is “she”?)

A model trained on standardised English will not recognise the same intent in dialect

> **Common Knowledge:** Common sense, world understanding facts humans assume but machines don’t have.
> 
> - “Alice put ice-cream in the cone, and it melted” - what melted Alice or ice-cream due to world knowledge.
> - “It hasn’t rained in weeks! My plants are dry” relies on the implicit causal linked.

> **Ambiguity:** Uncertainty in meaning that cannot be resolved from the words alone.
> 

| **Sub-type** | **What is ambiguous** | **Example** |
| --- | --- | --- |
| Lexical | Single word has multiple meanings | “River bank or financial bank” |
| Syntactic | Grammatical structure is ambiguous | “I saw a duck with binoculars.” (who has the binoculars) |
| Semantic | Phrase parses but means several things | “The old women and men went to the beach” (are the men old too?) |
| Pragmatic | Literal meaning clear, intent isn’t | “Can you open the door?” (Request not query) |
| Referential | Pronouns could refer to multiple entities | “Anna told Jess she passed the test” (Who passed the test) |

### MCQ traps

- **Slang → creativity** **(accept ambiguity but not diversity).** Slang is a novel coinage; dialect is diversity
- **Code-switching → Diversity (NOT ambiguity)**. Mixing language is form variation, not meaning-uncertainty.
- **“I saw a duck with binoculars” → Syntactic ambiguity (NOT common knowledge).** Even though that is common sense, the underlying error for the machine is grammatical. The prepositional phrase with can attach to either the verb “saw” or the noun “duck”.
- **“Can you open the door?” → Pragmatic ambiguity (NOT Syntactic).** Pragmatics deal with standard everyday language where the literal text differs from the social intent.
- **Pronoun resolution (e.g. “Anna told jess she passed”) → Referential ambiguity** **(NOT Syntactic).** Do not confuse this with a broken sentence structure. The syntax here is solid; the ambiguity is just a pointer issue - the machine doesn’t know which entity the variable “she” is referring to.

# 2. The nine types of NLP task

| **Task** | **What it does** |
| --- | --- |
| Conversational Agents | Dialogue systems that converse in natural language |
| Machine Translation | Translate between languages |
| Question Answering | Answer questions posed in natural language |
| Text Summarisation | Produce a short summary of a long document |
| Information Extraction | Pull structure facts from text |
| Text Classification | Assign a label to a document. i.e. genre |
| Topic Modelling | Extract topics from large document collections for text mining |
| Information Retrieval | Find documents relevant to a query |
| Language Modelling | Predict the next token - foundation of modern LMMs |

## 2.1 Real world Examples

**Conversational Agents:** ChatGPT and Claude (Open-domain LMM assistants), Alexa and Siri (slot-filing task agents). Hard because they require NLU, dialogue-state tracking, response generation, and a memory of the conversation so far.

**Machine Translation:** Google Translate, built on seq2seq transformer architectures with multilingual training.

**Question Answering:** Google’s featured-snippet QA or a RAG pipeline that retrieves passages and feed them to a generator. 

**Text Summarisation:** extractive (pick the most important sentences) and abstractive (generate new sentences - fine-tuned BART or a prompted LLM). e.g. news-app, TL;DRs or auto summary. 

**Information Extraction:** Bloomberg runs an industrial NER + RE pipeline over earnings filing to populate trading dashboards.

**Text Classification:** Gmail’s spam classifier started life as Naive-Bayes model over bag-of-words features and still uses Naive Bayes as one input to a larger ensemble. 

**Topic Modelling:** The algorithm Latent Dirichlet Allocation (LDA); applied at scale to news archives it creates topic clusters. Used for content analysis, audience segmentation and as a feature for downstream classifiers. 

**Information Retrieval:** Given a query, return ranked documents. Google search, whose ranking historically combined BM25 (a TF-IDF refinement) with PageRank and now blends in BERT-based dense retrieval. 

**Language Modelling:** Foundation of every modern LLM; drives autocomplete in spelling an grammar corrections.

# 3. The three families of NLP approaches

## 3.1 Heuristic / rule-based

Hand-crafted rules and lexical resources. WordNet-style relations

- **Synonyms**
- **Hyponyms -** specific instances (sport → tennis)
- **Meronyms -** part-whole (arm → hand → finger)

They are interpretable, deterministic, no training data but they won’t scale very similar responses (lack creativity) and brittle.

## 3.2 Classical machine learning

Statistical models that learn from labelled data:

- **Naive Bayes -** Text classification. Bag-of-features; strong independence assumption.
- **HMM (Hidden Markov Model) -** Sequence prediction. Slide example: “Jenny lost her ____ “ → predict dog. Use for POS tagging.
    - POS tagging is assigning a grammatical label (like Noun adjective, Verb) e.g. “The duck”: “The” [Determiner] and “duck” [Noun].
- **CRF (Conditional Random Field)** - Sequence labelling. e.g. “He is in Bristol” → location = Bristol. Used for NER and chunking.
    - NER (Named Entity Recognition) identifies real world structures in unstructured text. e.g. Person. Organisation, Date, etc.

## 3.3 Deep learning

Neural network families:

- **CNNs** - text classification (treating inputs as 1D sequence). convolutional filters slide over a fixed width window of tokens, capturing local n-gram-like patterns (e.g. filter learns to fire on “Not very good”)
- **RNN/LSTM -** sequential modelling. A recurring hidden state propagates left to right allowing the model to extract sequential context so the prediction at position i depends on everything that came before.
    - LSTMs add a gated memory cell that mitigates the vanishing gradient problem of vanilla RNNs
- **Transformers -** the dominant modern architecture: self attention lets every token directly attend to every other token in on step, giving parallelism (no sequential bottleneck like RNNs) and long-range dependencies (no decay over distance like RNNs).
- **Auto encoders** - representation learning. an encoder-decoder trained to reconstruct its input forces the bottleneck to learn a compressed semantic representation; useful for unsupervised feature learning, denoising and pertaining objectives like masked-LM.

## 3.4 Why we don’t always use deep learning

- Prone to overfitting on small datasets: Deep learning needs a lot of data
- Interpretable models: sometimes you need to explain the decision (legal, medical)
- Stationary knowledge: DL models freeze knowledge at training at training time
- Domain Adaptation: Models trained on one dataset will not work on another
- Lack of common sense: DL doesn’t have world knowledge unless trained in
- Cost: Compute, electricity and environmental impact

# 4. The end-to-end NLP pipeline

![diagram.jpg](Week%201%20-%20Intro%20+%20Language%20Preprocessing/diagram.jpg)

1. **Data acquisition.** Where from? Right format? Consent? non-stationarity?
    1. You can augment via **synonym swapping, back-translation, adding noise, or replacing entities.**
2. **Text cleaning.** Parse HTML if web-scraped; normalise via Unicode; decide on spelling correction. 
    1. parse means to remove these identifier tags for example <div> or <p> you might find in html.  
3. **Pre-processing.** Sentence segmentation, tokenisation, stopwords, stemming or lemmatisation, punctuation.
4. **Feature engineering.** One-hot encoding, BoW, TF-IDF. 
    1. Classical ML often designed: N-gram patterns local sequences (e.g. “not very good”). 
    2. Linguistic Signals: Using capitalisation as a feature for Named Entity Recognition (NER). All caps words as “sentiment intensity”. 
    3. Overlap features: CRFs use feature like word prefixes and suffixes (matching words against a known list/dictionary)
5. **Modelling.** Pick approach (e.g. heuristic) consider stacking or ensembles. 
6. **Evaluation.** Use both **intrinsic** (accuracy, F1 on held out data) and **extrinsic** (does it improve downstream business outcome) metrics. 
    1. intrinsic captures how well the model performs on the **held-out or testing dataset**
    2. Extrinsic: does it actually help the user or save the company time. e.g. in the uber case study does it reduce the time it takes for a human agent to solve the customer’s problem?
7. **Deployment.** Roll out scheme?
8. **Monitoring and maintenance.** Watch usage and long-term effectiveness? Processes in place for updating the model.

<aside>
💡

Pipeline is iterative, not one-shot

The pipeline contains a feedback arrow from evaluation back to pre-processing (not modelling). Often other fields treat pre-processing as fixed first step but in practice the bigger gains usually come from changing how you clean and tokenise your inputs. 

</aside>

## 4.1 Data Augmentation techniques

**Synonym Swapping.** Replacing each with a WordNet synonym. (e.g. “the film was excellent” → “the movie was superb”)

- Risk: “a great director” → “a big director” mean different things. Heavy synonym swapping can drift meaning.

**Back-translation.** Translate sentence to a pivot language and then translate back to the source. The round-trip produces a natural paraphrase that preserves meaning while changing the surface form. 

- Strong for QA, classification and seq2seq tasks because the augmented text reads like human writing.
- Risk: translation systems hallucinate, especially on idioms and named entities; filter on a similarity threshold against the original.\

**Adding noise.** Inject realistic typos, character swaps, deletion or keyboard-adjacency substitiutions (”excellent → “excelent”) Trains the model to be robust to the dirty inputs from real users. 

- Strong for chat / social media / search-query data.
- Risk: too much noise makes input unreadable and model can learn nonsense.

**Replacing entities.** Swap named entities of the same type drawn from a gazetteer. (”Alice met Bob in London” → “Carol met Dave in Edinburgh”). Increases entity diversity and reduces the model’s tendency to memorise specific names from the training set. 

- particularly useful for NER and RE training, where a small dataset can overfit to the handful of names.
- Risk: entity-type mismatches (swapping a sports star for politician) can destroy the sentence’s semantic plausibility. Swaps must be constrained to the same fine-grained type.

## 4.2 The Uber Customer Care case study.

![Screenshot 2026-05-09 at 15.50.40.png](Week%201%20-%20Intro%20+%20Language%20Preprocessing/Screenshot_2026-05-09_at_15.50.40.png)

The COTA system’s goal was to help resolve many issues by recommending the top 3 most likely issue types and solutions.

- The NLP pipeline:
    - feature engineering LSI and TF-IDF to extract topic vectors
    - Mathematical optimisation instead of raw vectors calcualtes the cosine similarity between tickets and historical solution sets to reduce the feature space.
    - The Model uses a pointwise ranking algorithm (random forest based) to score matches between tickets and possible resolutions.

Note LSI is latent semantic indexing: TF-IDF produces sparse matrices, LSI compresses this into a smaller “Topic Vector” that captures the meaning of the ticket rather than just the exact words used. 

# 5. Document Preprocessing

<aside>
💡

Definitions

**Document -** a self-contained piece of text (e.g. a single reddit post)

**Corpus** - a collection of documents (e.g a collection of comments on that reddit post)

**Corpora** - a collection of collections. (e.g. Reddit itself)

</aside>

## 5.1 Document processing flow

![Screenshot 2026-05-09 at 16.02.30.png](Week%201%20-%20Intro%20+%20Language%20Preprocessing/Screenshot_2026-05-09_at_16.02.30.png)

- order for the first 3 steps can be changed but encoding is last.
- Stemming and lemmatisation are alternatives - pick one.

## 5.2 Normalisation

**Case folding.** Reduce everything to lowercase (or uppercase). “Do you like the Black Rose café in Cardiff?” → “do you like the black rose café in cardiff?” 

<aside>
💡

Pitfall

Black Rose Café (Proper noun, the cafe name). colour and flower are part of the name collapsing that loses the entity unless you do NER (named entity recognition) first.

</aside>

**Whitespace removal.** Strip hidden characters: tabs, newlines, section breaks, page breaks. 

- Appear from formatting

**Diacritics.** café → cafe. Don’t do this for other languages or code-mixed multilingual text where the accent changes meaning. 

**Numbers.** Three options: spell out, keep as digits (3 or 3.00); remove entirely if irrelevant.

<aside>
💡

101 spelled out becomes “one hundred and one” - three separate tokens, breaking the integer’s identity. If you downstream task cares about the actual number, **keep it as digits.** 

</aside>

## 5.3 Tokenisation

> **Tokenisation:** The process of separating a string into a list of tokens (words, punctuation, or sub-word units).
> 
- splitting is critical because downstream models cannot perform mathematical operations like embedding lookups or probability calculations on a raw string. (Requires discrete units to function).

**what goes wrong without good tokenisation.** 

- Naive whitespace split keeps punctuation glued. “cat.”. “cat,” and “cat” are three different tokens meaning a much larger vocabulary.
- Aggressive whitespace split breaks compounds. “state-of-the-art” fragments into multiple tokens. Treating the hypen group as one token preserves this.
- Contraction splitting is task-dependent. “don’t” tokenised as one token loses the negation; tokenised as “ do + n’t” surfaces it for a sentiment classifier but creates a non-word in the vocabulary.

**Three sub-steps:**

- **Word detection.** Split on whitespace; NLTK uses TreeBank + sentence tokenisers.
- **Punctuation handling.** Separate punctuation, but care with hyphens
- **Special tokens.** handling emojis or @.

## 5.3 Stop word removal

> **Stop word** is a frequently occuring word that can usually be removed without losing meaning (the, is, of, to). NLTK has 127 in its English list.
> 

**Use if for…**

- Information retrieval
- Topic modelling
- Keyword extraction
- Genre classification

**Don’t use if for…**

- Sentiment analysis
- Question-answering
- Legal / medical /financial documents
- Anything where structure matters: referring to the relationship between words itself is important.

## 5.5 Stemming and lemmatisation

Both methods purpose is to reduce inflected or derived words to a common base or root form. Doing so decreases the complexity of data (the vocabulary size). 

> **Stemming** is a brute force approach that chops off the end of words using fixed rules to find a common base.
> 

> **Lemmatisation** us a smart approach that uses a dictionary and grammar rules to find the true root of a word.
> 

| **Aspect** | **Stemming** | **Lemmatisation** |
| --- | --- | --- |
| Method | Strips suffixes by rule | Vocabulary + grammar lookup |
| Output a real word? | Not always | Always |
| Speed | Fast | Slow |
| Context preserved? | Generally not | Yes |
| Resources needed | Algorithm only | Vocabulary + POS info |
| Example (Study, run, leaves) | studi, ran, leav | study, run, leaf |

## 5.6 One-hot encoding and cosine similarity

> **One-hot encoding.** Builds a vocabulary V. Each document becomes a vector of length $|V|$ where position $i$ is 1 if vocabulary token $i$ appears in the document, 0 otherwise.
> 

<aside>
💡

**Why one-hot loses?**

Word order is gone (the bag-of-words property). Frequency is also gone: a document with the once and a document with five times look the same. 

</aside>

**Cosine similarity** compares two encoded vectors:

$$
cos(\theta) = \frac{A \cdot B}{||A|| ||B||}
$$

This captures how similar documents are since the dot product essentially counts the number of matches between the two documents and normalises it (compared to the size of the document). 

- A perfectly matched document correlates to $\theta = \pi/2$

<aside>
💡

Cosine vs Euclidean

Cosine ignores the magnitude - a long document with the same word-mix as a short one stays close. 

- it normalises these compared to the length of the documents and only considers the direction.

Euclidean punishes length differences. $||A-B||$. 

- if the document A is 50 words vs B 500 words, there will be a much greater distance regardless if all the words in A match B - misleading.
</aside>

## 5.7 Preprocessing trade-offs by task

A single preprocessing recipe does not generalise to all NLP tasks. Summarising all the methods:

**Case folding (lowercasing).** Aggressive - you lose all capitalisation signals in exchange for a smaller vocabulary.

- Topic classification: apply liberally. Categories are robust to casing (Apple the company vs apple the fruit are usually distinguishable from context and the topic classifier doens’t care which sense is meant)
- Named Entity Recognition: **do not lowercase.** Capitalisation is the strongest surface cue for proper nouns costs the NER model its most-discriminative feature.
- Sentiment analysis: mixed. “HORRIBLE” in all-caps is a sentiment-intensity signal; lowercasing destroys it. A common comprimise is to add an ALL_CAPS feature before lowercasing.

**Stop word removal.** Aggresive - you delete entire token classes. The underlying principle is that stop-word removal helops when you only care about topical content and hurts when structure or modality matters. 

- Information Retrieval: remove. Stop words inflate the vocabulary, dominate term frequencies and add no retrieval signal. BM25 traditionally drops them or weights them near zero.
- Sentiment anlaysis: **do not remove.** Stop words contain important negations that invert the meaning.
- Question answering: do not remove. The interrogative function words (”who”, “where”, “when”) are the question type itself.
- Machine translation: do not remove. Function-words carry grammatical information that must survive into the target language.

**Stemming and lemmatisation.** Pict at most one - they are alternatives that collapse variants to a base form. 

- Topic modelling: stem aggressively. LDA is happier when you have less token - topic word distribution is then sharper.
- Information retrieval: stem. Recall matters more than precision; you want a query for “running” to also match “runs”.
- Question-answering and exact match search: **neither.** Exact wording is part of the answer.
- Sentiment analysis: lemmatise. Lemmatisation preserves real words so downstream can still look words up.
- NER and IE pipelines: **neither.** POS tagging, dependency parsing and entity recognition all rely on the original surface form.

The general principle: the more downstream care about syntax, modality, structure or exact wording, the lighter the preprocessing should be. 

# 6. The Britannia Hotel use case

Britannia International Hotel had over 5000 reviews and wanted automated sentiment analysis. 

## 6.1 Why preserve punctuation in this use case?

For sentiment analysis, punctuation carries semantic structure. The excalmation mark is a sentiment-intensity signal; the question mark in “Is this what £200 a night gets you?” is a rhetorical disappointment signal. Stripping punctuation discards both. 

This is an example here of pragmatic ambiguity the question mark tells the reader and the model that the literal reading interpretation is probably wrong. 

## 6.2 Is “add a period before each capital letter” a reasonable approach

**No.** It fails on lists of names, on sentiment based all caps, and on proper nouns. This is a brittle rule-based approach.

## 6.3 Which techniques reduced skewed word distributions like “hotel”, “the”, “in”?

Stop word removal strips the words like “the”, “in” but also strips “not” which is critical for sentiment analysis.

**Stemming and lemmatisation** collapses morphological variants but doesn’t address hotel.

**Domain stop words** add “hotel” to a custom stop word list since it appears in every review and carries no signal for this corpus. 

# 7. Lab 1

Processing text: full preprocessing pipeling from scratch on the 20 News-groups dataset. The goal is topic classification. 

**Q: Why is it OK to do punctuation AND stopword removal in this scenario (20 Newsgroups)?**
**A:** Because this is a **topic classification** task. Punctuation and stopwords (e.g., "the", "and") carry grammatical structure but no topical signal. Removing them reduces noise and sharpens the vocabulary for the classifier.
**Q: Why is it OK at this stage to replace apostrophes with an empty string?**
**A:** It prevents the tokeniser from fragmenting contractions (e.g., "don't" splitting into "do" and "n't"). For topic clustering, collapsing it into a single token ("dont") is simpler and avoids vocabulary bloat without losing topical meaning.
**Q: Is there a case for leaving punctuation in during tokenisation?**
**A:** Yes. Tasks like **Sentiment Analysis** need punctuation for emotional intensity (e.g., "!"), and **Information Extraction / NER** pipelines require punctuation for accurate sentence boundary detection and grammar parsing.
**Q: What influence does vocabulary size have on the feature vectors?**
**A:** They are directly proportional. A vocabulary of size $N$ will result in every document being encoded as a feature vector of length $N$.
**Q: What happens if a word appears in the test dataset that wasn't in the training vocabulary (OOV)?**
**A:** The encoder will ignore/drop the unseen word. Since the feature vector dimensions are locked to the training vocabulary, the model has no designated input slot for the new word.