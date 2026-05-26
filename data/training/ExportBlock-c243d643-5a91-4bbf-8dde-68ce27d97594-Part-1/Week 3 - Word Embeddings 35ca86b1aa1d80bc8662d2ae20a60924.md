# Week 3 - Word Embeddings

# 1. Why frequency representations are not enough

- **Same word, different meaning.** Lexical ambiguity means that words that mean the same return the same vector.
- **No expressivity of similarity.** “heating” sits no closer to “lighting” than to “turn” - encoding has no notion that two words might mean similar things.
- **Sparsity.** at V dimensions per word

## 1.1 One-hot positional encoding

Instead of aggregating into a count vector, one-hot positional encoding represents each token as a separate $|V|$-dimensional vector preserving order.

<aside>
💡

Example

“Turn off the heating” → [turn, off, heating] → three 6-dim one-hot vectors. 

turn: [0,0,0,1,0,0]

…

</aside>

Order is preserved but the same two limitations remain.

# 2. Word embeddings: the concept

> **Word embedding.** A dense, low dimensional, real-valued vector per word that captures meaning so similar words sit close in vector space.
> 
- Typically 100-1000 dimensions vs $|V|$ dimnesions for one-hot.

**Semantic similarity -** cosine in embedding space tracks meaning. 

- Word2Vec captures semantic similarity only. The embedding distance reflects meaning.

**Analogical Reasoning -** vector arithmetic captures relations: France - Paris + Stockholm = Sweden; kitten - cat + dog = puppy. 

**Clustering -** semantically related words form regions in the embedding space. 

# 3. Word2Vec - CBOW vs Skip-gram

**Word2Vec** is a shallow neural network trained on a self-supervised prediction task (guessing words from their context). We don’t actually care about the network’s final predictions; we only care about the weights it learns in its first layer, $W_1$.

1. The Weight Matrix $(W_1)$
    1. **Shape: $|V| \times d$  (**vocabulary size x Embedding dimension)
    2. **Contents:** Contain dense real numbers, with each row representing the learned embedding vector for each specific word.
2. Matrix Multiplication
    1. one-hot input x $W_1$ acts like a lookup table outputting the embeddings of that token.
    2. Once the model finishes playing its prediction game you discard the rest of the network. 
    3. You only keep $W_1$ to use as your static word embedding lookup table. 

Breaking down the structure we train a neural network to output a probability of what the next word is. We then extract the hidden layer which represents the embeddings relation and delete the rest. 

- Self supervised because we train the neural network on our corpus.

## 3.1 Two variants

> **CBOW (Continuous Bag of Words).** Predict the centre word from its surround context words.
> 
- Take the one-hot vectors for all words in your window, average (or sum) them together, and feed that combined $|V|$ dimensional vector into the network
- **Hidden layer ( $W_1$)** this is d-dimensional layer of all the encodings of the vocabulary we are interested in.
- **Output layer.** A softmax function that calculates the probabilities across the entire vocabulary to make a single prediction: what is the missing centre word?

> **Skip-gram.** Predict the surrounding context words from a single centre word.
> 
- Now have only one, one-hot vector for the centre word. Using the exact same hidden layer.
- **Output layer.** The network uses a softmax function over the vocabulary, but does it multiple times to predict each surrounding context position separately.

## 3.3 Trade-off

CBOW is faster to train (univariate prediction).

Skip-gram is slower but typically better for rare words and large corpora because each centre word generates multiple training examples. 

- In the example that we have rare vocabulary, say the vocab appears twice in the entire corpus. In Skip-gram (Window of 5): the embedding is updated 20 times.
    - 5 words each side and twice because of the appearances in the corpus.

<aside>
💡

Mnemonic: **C**BOW takes a **C**ollection of context; **S**kip-gram **S**tarts from a **S**ingle word. 

</aside>

## 3.4 Training Mechanics

The output layer is a softmax over the entire vocabulary. This means every gradient update touches 100,000 x 300 = 30M parameters - ad you do this for every training example. **Two efficiency tricks:**

- **Negative sampling.** Instead of softmaxing over all words, sample $k$ negative words (random non-context words, typically $k$ (5-20) and train binary classifiers.
    - **The task:** out of this tiny group of words which one is real and which one is fake.
    - **Benefit:** instead of updating 100,000 rows in the output matrix, the network only updates k+1 rows (the real word + 5 fake words).
    - $P(w)^{0.75}$ distortion. When picking those fake negative words, the model distorts the odds so it doesn’t just accidently pick the wor “the” every time. It artificially boosts the chances of picking rare words to keep the trianing balanced.
- **Hierarchical Softmax.** Instead of a true/false test, this trick organisises the entire vocabulary into a massive **binary tree.** Common words near the top.
    - How it works. Instead of evaluating all the vocab at once, the network starts at the top and makes a series of binary decisions to navigate down the branch until it hits its target word.
    - **Benefit.** Reduces the no. of calculations by log_2. 100,000 words means only 17 decisions.

## 3.5 Window size effect

The window size - how many surrounding tokens count as “context”

- **Small window (1-2).** Captures **syntactical similarity.** Words that fill the same grammatical slot end up nearby. e.g. all common nouns are generally closer together or followed by “the”. Useful for POS-style downstream tasks.
- **Large window (5-10).** Captures **semantic / topic similarity:** words that co-occur in the same broad topic become neighbours. e.g. king to throne, palace, crown (regardless of grammatical category). Useful for topic modelling and document retrieval.

## 3.6 Embedding dimension trade-off

- **Too few dimensions.** Loses information - the model cannot fit enough orthogonal directions to separate semantic clusters. Cosine similarity becomes noisy.
- **Too many dimensions.** Overfits the training corpus, slows down inference, and (above some threshold0 gives no further accuracy gain. With insufficient data, extra dimenions add noise rather than signal.
- **100-300 sweet spot**

Modern contextual embeddings push this further BERT base uses d = 768, BERT large d = 1024. They can do this because they train on much more data. Also they share parameters across positions via the transformer, so the per-word effective parameter count is smaller than it seems. 

## 3.7 Distributional hypothesis - the assumption underneath everything

All methods discussed relies on the **distributional hypothesis.** Words that appear in similar contexts have similar meanings. 

This is an empirical claim, not a mathematical theorem. The limitations:

- **Antonyms cluster together.** hot and cold appear in identical contexts, so the distributional hypothesis treats them as similar - you have to inject explicit antonym supervision to separate them.
- **Rare words get poor embeddings.** A word that apperas 5 time in the corpus has 5 contexts to learn from. Frequent stopwords end up with the most precisely-trained embeddings.

Concluded: if the corpus doesn’t contain the word in informative contexts, the embedding will not capture its meaning. 

# 4. Document embeddings via aggregation

The simplest way to get a document embedding from word embeddings: average (or sum) the word vectors.

**Trade-offs.** Average smooths noise but lets frequent words dominate. TF-IDF weighting on top can downweight common words; this is the basis of weighted embedding aggregation.

## 4.1 GloVe - another popular static embedding

- **Word2Vec** is a predictive model - it learns by predicting the middle word from its context using a sliding window over the corpus. Local in scope; sees only the words within the window.
- **GloVe** is a count-based / matrix factorisation model - it builds the global word-word co-occurence matrix $X$ (entry $X_{ij}$ is how often word i appear in the context of word j across the entire corpus) and learns vectors that minimise a weighted reconstruction error on $logX_{ij}$. Global in scope; sees the whole corpus’ co-occurence statistics at once.
    - So matrix of $|V| \times |V|$ gets squashed by matrix factorisation to a lower dimensional $|V| \times d$.
    - It looks at global statistics of the entire corpus all at once.

## 4.2 Why “static” is a limitation

The same word always retrieves the same vector regardless of the sentence. The five failure points:

1. **Context-dependent sentiment.** “I literally can’t even” vs “they literally ran a marathon” frustration vs factual
2. **Grammatical role change.** “book a meeting” vs “reading a book” verb vs noun.
3. **Negation.** “not good” vs “good” the negation flips meaning but the embedding doesn’t see it.
4. **Domain-specific meaning. “**Trained an ML model” vs “is a fashion model”
5. **long-range dependencies.** “the apple wasn’t put in the basket because it was too small”

# 5. Contextual embeddings: ELMo

> **Contextualised embedding.** An embedding function that takes the whole sentence as input, so the vector for a token depends on its neighbours.
> 

ELMo is build from **two independent unidirectional LSTM language models** - a Forward LM (left to right) and a Backward LM (right to left) - whose hidden states are concatenated layer by layer. 

## 5.1 Per-direction stack

![diagram.jpg](Week%203%20-%20Word%20Embeddings/diagram.jpg)

1. **Inside each layer**

Step 1: Character-CNN $(F_0$ and $B_0)$ 

- Instead of looking up a whole word in a dictionary the model reads the word character by character. A CNN processes these characters to understand the word’s surface form.
    - This learns morphology (spelling, prefixes and suffixes). It helps handle typos or entirely new words.

Step 2: The first LSTM layer $(F_1$ and $B_1)$

- The output moves into the first Long Short-Term Memory (LSTM) layer. This layers starts looking at the surrounding context.
    - This learns: Syntax. It figures out the grammar.

Step 3: The Residual Shortcut

- Skipping past the first LSTM. This takes the raw word shape data from the character CNN and adds it directly to the output of the first LSTM ensuring that the original spelling and form of the word aren’t forgotten

Step 4: The Second LSTM layer $(F_1$ and $B_2)$

- The data now contains both raw word features and syntax
    - This layer now learns semantics. It learns the actual meaning of the word in this exact context.
1. **Training Step: Softmax**
- At the end of the individual stacks, the model tries to guess the next word or previous word for the forward and backward passes respectively.
    - This part is thrown away it is just used to train the embeddings.
1. **Stitching it together**

ELMo extracts the hidden states from each level of the stack and **concatenates** them together. This happens in distinct levels: 

- $I_0 = [F_0, B_0]$: Stitched Character-CNN outputs. This form the base layer containing morphological data.
- $I_1 = [F_1, B_1]$: The stitched first LSTM outputs. This forms the middle layer containing syntactical data.
- $I_2 = [F_2, B_2]$: The stitched second LSTM outputs. This forms the top layer containing semantic data.

1. **Final output**

ELMo doesn’t give you one final vector instead gives the concatenated layers. 

When you use ELMo for a specific task (like sentiment analysis or POS tagging), your downstream model will learn to apply weights to these layers, deciding which one is most useful. 

## 6.3 Three step pipeline

1. **Pre-train** the BiLSTMs on a large generic global corpus. 
2. **Extract and concatenated** internal states for each word in the task corpus.
3. **Train** the downstream classifier and fine-tune $s_n, \gamma$ jointly via backpropagation.

$$
ELMo_{task} = \gamma \sum_{n} s_n I_n
$$

**At inference,** for new sentence, run the BiLSTMs forward, concatenate per-layer states, apply the learned weights, feed to the task model. 

<aside>
💡

ELMo is not a single bidirectional model

ELMo is bidirectional vis two independent unidirectional LMs whose hidden states are concatenated. The forward LM only ever sees left etc - only outputs are stitched. 

BERT is an example of a single jointly-bidirectional model that sees both sides at once.

</aside>

## **6.4-6.5 missed**

# 7. Pre-trained embeddings: pros and cons

**Pros:**

- **Linguistic generalisation -** large generic corpus captures broad language properties; mitigating overfitting on small task datasets.
- **Reduced data needs -** frozen weights mean the task model has fewer parameters to fit, faster convergence
- **Modularisation -** plug and play, no DL training overhead per task.

**Cons:**

- **Domain mismatch -** general model embedding may be wrong for ML/fashion contexts. Domain specific training helps but is expensive
- **Embedded bias propagation -** no control over training corpus; biases (gender, race, geographical) progagate to every downstream task.
- **Compute cost -** large memory footprint, slow inference even when frozen.

<aside>
💡

Static vs contextual

| Feature | Static Embeddings | Contextual Embeddings |
| --- | --- | --- |
| Profile | Cheap to query (single lookup) and small footprint | Slower (forward pass per sentence) and larger footprint |
| Capabilities | Fixed vector; no context awareness | Recovers semantics across contexts |
| When to use |   • Simple high throughput tasks
  • No disambiguation needed
  • Tight memory constraints |   • Ambiguity sensitive tasks
  • GPU inference is available
  • Accuracy is prioritised |
</aside>