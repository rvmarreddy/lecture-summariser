# Week 2 - Document Representations and Sparsity

# 1. Bag of Words to TF-IDF

> **Bag of Words** is a vector representation of a document where each entry is the count of how many times a vocabulary token appears in the document.
> 
> - Binary bag of words was what was discussed in one hot encoding. BoW adds the frequency now.

**BoW limitations**

- **No length normalisation.** A long aCV and a short CV containing the same words end up with very different magnitudes, biasing similarity comparisons. (Mitigation: use cosine similarity, which divides by product of magnitudes or normalised vectors.
- **No word order.** the order of words doesn’t matter may have two different meanings “the cat sat on the dog” and “the dog sat on the cat”.
- **No semantic similarity.** Cat and kitten are different vocabulary slots; their vectors are orthogonal. BoW has no notion that two distinct tokens might mean similar things. (motivates word embeddings)
- **Sparsity.** Each document is a V-dimensional vector with mostly entries zero. This is wasteful and breaks distance metrics.

## 1.1 Bag of N-Grams (BoN)

> **N-gram** is a contiguous sequence of N tokens. **BoN** represents a document as a count vector over all its n-grams vocabulary, partially preserving word order.
> 
- **Sentence A:** *"the cat sat on the dog"*
    - **Trigrams:** `{the cat sat, cat sat on, sat on the, on the dog}`
    - **Vector:** `[1, 1, 1, 1, 0, 0]`
- **Sentence B:** *"the dog sat on the cat"*
    - **Trigrams:** `{the dog sat, dog sat on, sat on the, on the cat}`
    - **Vector:** `[0, 0, 1, 0, 1, 1]`

**Vocabulary explosion.** The number of distinct n-grams grows as roughly $|V|^n$ in the worst case. 

- Means that n-grams are sparser than unigram representations: more features, fewer non-zero observations per feature.

**Benefits:** Bigrams capture local negations and idiomatic phrases that unigrams flatten. not bad is a positive sentiment not caputred by a unigram. 

- N-grams here would have the tokens for `{not, bad, not bad}` and be able to learn to flip the sign.

## 1.2 TF-IDF

> **TF-IDF (Term Frequency - Inverse Document Frequency)** reweighs BoW or BoN entries by how rare a term is across the corpus, down weighting common words and amplifying distinct ones.
> 

### **Term frequency**

1. **Standard convention**

$TF(t, d) = \frac{\text{count}(t \in d)}{\text{tokens in } d}$

1. **Log Normalisation**

$TF(t, d) = 1 + \log(\text{count}(t, d))$

Used to penalise high-frequency terms so a single word doesn’t dominate the vector

1. **Double normalisation (**$\eta$**)**

$\eta\text{-}TF(t, d) = 0.5 + 0.5 \cdot \frac{\text{count}(t, d)}{\max_{t' \in d} \text{count}(t', d)}$

Scales TF relative to the most frequent token to protect against length bias. 

### Inverse Document Frequency (IDF)

IDF measures how rare a term is across the entire corpus. A term appearing in every document results in an IDF of 0. 

$IDF(t) = \log_e \frac{N}{df(t)}$

where:

- $N$ is the total number of documents in the corpus.
- $df(t)$ is the number of documents containing the term $t$.

### TF-IDF Weight

$TFIDF(t, d) = TF(t, d) \cdot IDF(t)$

**Design Choices**

- Why the log? Without it IDF would scale linearly with document frequency ratio. The log gives diminishing returns: doubling the corpus from 2 to 4 documents containing a term shouldn’t shift its rarity weight as drastically as doubling from 1 to 2 (where a term goes from unique to shared)
- N/df is used because we want common terms, terms with a high df to be down-weighted . Taking the inverse keeps the size positive while gating the magnitude.

## 1.3 Comparison of representations

**BoW**

- Limited technical skill required to implement and interpret
- Computationally Efficient
- Loses order and context
- Sensitive to common tokens

**BoN**

- Captures context
- Domain-specific expressions (e.g. machine learning model)
- Vocabulary size scales rapidly
- Requires more memory to compute

**TF-IDF**

- Can be used in conjunction with BoW or BoNG.
- Reduces noise from stop words
- Ignores semantics
- Sparse and high-dimensional feature vectors

**Use cases:**

- Text Classification
- Topic Modelling
- Small datasets
- Finding a quick baseline

**Bag of N-Grams**

- Word order matters
- Large dataset available
- Named Entity Recog.
- Short Documents

**TF-IDF**

- Keyword identification
- Text classification
- Search / Querying
- Clustering

## 1.4 Out-of-Vocabulary (OOV) and mitigations

> **OOV** is a token in the test set with no entry in the training vocabulary. It cannot be encoded and is silently dropped, losing information.
> 
- **Preprocessing -** lemmatisation/stemming/stopwords shrink the variability that creates OOV.
- **<UNK> tokens** - map all unknowns to a single slot and count them. Reduces dimensionality but loses information.
- **Character n-grams** - “languages” → [lan, gua, ges]. Handles misspellings; loses semantic meaning; vector size explodes.
    - Doesn’t mean they are cheaper they are computationally less efficient but in some use cases it is necessary.
- Other options: **Hashing trick,** combining word and character n-grams.

# 2. Sparsity

> A representation is **sparse** when most entries of the feature matrix $F$ (size $n \times |V|$) are zero.
> 

Some causes include:

- discrete representation (one dim per token)
- Model choice (n-grams)
- Token variability (poor preprocessing)
- Short documents relative to $|V|$
- Human language contains a small set of frequently used words and a large number of rare words.

## **2.1 Three reasons sparsity matters:**

1. **Memory and computational Overhead -** Storing zeros wastes RAM; distance treats every dimension equally.
2. **Curse of dimensionality -** distance metrics lose meaning in very high dimensions.
3. **Probabilistic models -** multiplying many small probabilities. Also any single zero collapses to 0. 

### Curse of Dimensionality - Geometric intuition

In low dimensions, “near” and “far” are intuitive. As dimensionality grows, this stops being true. 

Take the example of a 50,000 dimension space, because there is so much room, a**lmost every wallet is roughly the same huge distance away from each other wallet.** In this example these wallets represents your tokens. 

## 2.2 Naive Bayes recap

> **Naive Bayes classifier** is a probabilistic classifier built on Bayes’ rule with a strong independence assumption: each feature is conditionally independent given the class. The class score is:
> 
> 
> $$
> P(\text{class} \mid \text{features}) \propto P(\text{class}) \cdot \prod_{i}P(\text{feature}_{i} \mid \text{class})
> $$
> 
> $\text{Predict } \arg\max_{c} \left( P(c) \cdot \prod_{i}P(f_{i} \mid c) \right)$
> 

The conditional independence assumption is the “naive” part. Take the example of detecting a spam email: the words *bitcoin* and *wallet* co-occur in (they’re correlated) but the model multiplies their probabilities as if they’re independent. 

- This is wrong but typically still a useful classifier - the ranking of classes survives even when the absolute probabilities are off.

**Smoothing:** will be explored later but if any probability is zero. One unseen word in the test document sends the joint probability to zero. Smoothing solves this issue. 

**Log-space trick:** In practice you compute $P(c)+\sum_i logP(f_i|c)$ rather than the raw product. 

1. Avoids underflow when multiplying many tiny probabilities
2. turns multiplication into addition which is easier cheaper and numerically stabler.

Argmax is preserved because log is monotonic. Smoothing still matters $log(0) = -\infty$ collapses the sum just as $0$ collapses the product.

## 2.3 Vocabulary Pruning

Five families of pruning, each with trade-offs. 

1. **Preprocessing -** lowercasing, lemmatisation, stemming, stopword removal, punctuation stripping. 
    1. Collapses surface variants and removes function words
    2. Appropriate when downstream tasks doesn’t need word form distinction and when formal text where morphology is regular
    3. Trade off: stemming over-collapses; stopword lists are domain-dependent
2. **Frequency / TF-IDF threshold -** drop tokens below a count or TF-IDF cut-off. 
    1. Removes words appearing once and ultra-rare typos. 
    2. Appropriate when large corpora where vocabulary is large and produces little signals
    3. Trade off: loses rare but meaningful tokens - medical a single mention of anaphylaxis in a medical corpus. 
3. **Subword / byte-level / character tokenisation -** BPE (Byte pair encoding) and WordPiece that find the statistical middle ground between whole words and single letters
    1. Starts with individual characters and greedily merges the most frequent adjacent pairs into reusable subword chunks (e.g. unbelievable → un, believ, able)
    2. Appropriate in modern deep learning models and completely eliminates OOV errors because any unknown word, made up words or typos can be decomposed into known subwords.
    3. Trade-off: loses human interpretability and results in slightly longer token sequences than whole-word methods. 
4. **Named Entity Normalisation (NER) -** Replacing specific entities (names, dates, amounts) with generic placeholder tags like `<PERSON>`or`<MONEY>.` 
    1. Collapses an infinite, open ended class of tokens into a single, shared vocabulary slot. 
    2. Appropriate when the type of entity matters more than its specific identity, such as intent classification.
    3. Trade off: requires running a separate NER model first and can introduce ambiguity (is “Apple” a fruit or a company?)
5. **Compound splitting -** notebook → note + book; particularly important for languages with productive compounding like German. 
    1. Appropriate: non-English corpora where compounds inflate vocabulary 5-10x
    2. Trade-off: struggles with non-compositional compounds requires a morphological analyser tuned for the language

# 3. Smoothing

## 3.1 Laplace / Lidstone (add $\alpha$)

> **Laplace / Lidstone smoothing** adds a constant $\alpha$ to every count so no probability is zero. **Lidstone: $0 < \alpha ≤ 1.$ Laplace: $\alpha=1$.**
> 

$$
𝑃(𝑤 ∣ 𝑐) = \frac{count(𝑤, 𝑐) + 𝛼}{T_c + \alpha|V|}
$$

Where $T_c= \text{total tokens in class } c, |V|= \text{vocabulary size}$.

### Example

- Building a Naive-Bayes spam classifier using unigrams.
    - Spam: [‘buy’, ‘cheap’, ‘medicines’], [‘cheap’, ‘bitcoin’]
    - Not spam: [‘client’, ‘meeting’, ‘morning’], [‘cheap’, ‘meeting’]

| **Token** | **Spam** | **Not Spam** |
| --- | --- | --- |
| buy | 1 | 0 |
| cheap | 2 | 1 |
| medicines | 1 | 0 |
| bitcoin | 1 | 0 |
| client | 0 | 1 |
| meeting | 0 | 2 |
| morning | 0 | 1 |

The purpose of Naive Bayes is to predict the most likely category for an input by calculating which class yields the highest property given specific features (tokens) observed. 

- In this example we are finding which class: “Spam” or “Not Spam” maximises the probability of our set of tokens being there.

Test for [’bitcoin’, ‘meeting’]

- P(’bitcoin’|Spam) = 0.2
- P(’bitcoin’|Not Spam) = 0
- P(’meeting’|Spam) = 0
- P(’meeting’|Not Spam) = 0.33

Using Smoothing

- P(’bitcoin’|Spam) = (1+1)/(5+7)= 0.167
- P(’bitcoin’|Not Spam) = (0+1)/(5+7) = 0.083
- P(’meeting’|Spam) = (0+1)/(5+7) = 0.083
- P(’meeting’|Not Spam) = (2+1)/(5+7)=0.25

P(Spam|’bitcoin meeting’) $\propto$ P(c) $\cdot$ P(’bitcoin’|Spam) $\cdot$ P(’meeting’|Spam) = 0.5 $\cdot$ 0.2 $\cdot$ 0 = 0

P(Not Spam|’bitcoin meeting’) $\propto$ P(c) $\cdot$ P(’bitcoin’|Not Spam) $\cdot$ P(’meeting’|Not Spam) = 0.5 $\cdot$ 0 $\cdot$ 0.33

= 0.5 $\cdot$ 0.167 $\cdot$ 0.083 = 0.00694

= 0.5 $\cdot$ 0.083 $\cdot$ 0.25 = 0.01041

<aside>
💡

Pitfall: why both classes collapsed pre-smoothing

Both probabilities went to 0 because the test pair contained one token unseen in each class. It means the classifier is not valid and can’t make a valid prediction. Smoothing is a crucial step.

</aside>

## 3.2 Written-Bell

> **Written-Bell smoothing.** Splits probability mass between seen and unseen tokens using a per-class weight $\lambda_c$ derived from the ratio of total to unique tokens in the class.
> 

Calculate the probability mass weight $\lambda_c$ for seen tokens in a class:

$$
\lambda_{c} = \frac{T_{c}}{T_{c} + U_{c}}
$$

Then calculate the conditional probability $P(w | c)$ using a piecewise function depending on if the word was seen or unseen in that class during training:

$$
P(w \mid c) = \begin{cases} \lambda_{c} \cdot \frac{\text{count}(w,c)}{T_{c}} & \text{if } w \text{ seen in } c \\ \frac{1 - \lambda_{c}}{Z_{c}} & \text{if } w \text{ unseen in } c \end{cases}
$$

- $T_{c}$: Total tokens (instances) in class $c$
- $U_c$: Unique tokens (types) in class c
- $Z_c$: Number of vocabulary words not present in class $c$

### Example

| **Token** | **Spam** | **Not Spam** | **P(w|S)** | **P(w|NS)** |
| --- | --- | --- | --- | --- |
| buy | 1 | 0 | 0.56*0.2=0.112 | 0.15 |
| cheap | 2 | 1 | 0.56*0.4=0.224 | 0.56*0.2=0.112 |
| medicines | 1 | 0 | 0.56*0.2=0.112 | 0.15 |
| bitcoin | 1 | 0 | 0.56*0.2=0.112 | 0.15 |
| client | 0 | 1 | 0.15 | 0.56*0.2=0.112 |
| meeting | 0 | 2 | 0.15 | 0.56*0.4=0.224 |
| morning | 0 | 1 | 0.15 | 0.56*0.2=0.112 |
| T_c | 5 | 5 |  |  |
| U_c | 4 | 4 |  |  |
| Lambda | 0.56 | 0.56 |  |  |
| P(unseen|c) | 0.15 | 0.15 |  |  |

# 4. Lab 2

The lab implements all three representations from scratch then trains Naive-Bayes on each. Testing unigrams and trigrams with and without TF-IDF.

**Why the Trigram Model Collapses (The data per feature argument)**

- The illusion of feature count: Trigrams (70k features) might only seem slightly larger than unigrams but they suffer from severe sparsity.
- Low Reusability: Unigrams repeat constantly across documents, giving the model plenty of data per feature. Trigrams rarely repeat.
- Because there are so few non-zero observations for each trigram, **smoothing dominates the actual evidence.**
- Fix: either need exponentially more data or a switch to dense representations (embeddings).

**Why TF-IDF Barely helps**

- Reweighting ≠ New info: TF-IDF only scales existing features. Downweighting common words doesn’t magically make rare words statistically informative if they only appear a few times.
- Task mismatch: TF-IDF is highly effective for search/IR (where you want distinctive terms to bubble up in rankings). However, it doesn’t automatically win at classification on small, sparse corpora where smoothing has already levelled out the probabilities.

**Takeaways**

- The trap of complexity: adding more representational power actively hurts performance if you dataset doesn’t scale to support it.
- **Actionable Rules:** 1. Always match your representation’s complexity to your corpus size. 2. If your corpus is fixed and small, abandon sparse metrics entirely. Use **dense representations** that share statistical strength across features rather than paying a sparsity tax.