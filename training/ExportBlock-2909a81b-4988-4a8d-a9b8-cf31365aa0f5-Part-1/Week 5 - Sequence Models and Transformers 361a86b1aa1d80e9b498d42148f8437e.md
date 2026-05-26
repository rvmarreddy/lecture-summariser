# Week 5 - Sequence Models and Transformers

# 1. The five types of text generation

## 1.1 Free-form generation

- **Story generation.** (”Write me a story about a woman who is late for work”). This is long-form and requires plot consistency, character coherence and a narrative arc.
- **Dialogue generation.** (”Give an example conversation between a dog and a cat”). This requires turn-taking, a consistent character voice and plausible exchanges.
- **Expository generation.** (”Explain how stars are made to a child”). This requires accurate factual generation and an audience appropriate style meaning the model must adapt its vocabulary and depth to the reader.

**Challenges:** Bias, ethical concerns, coherence, and creativity. Long-form generation is particularly where hallucinations and stylistic drift causes the most harm.

## 1.2 Summarisation

This invovles compressing information.

- **Extractive paradigm:** Picks exact spans of text from the source. It is safer but less natural.
- **Abstractive paradigm:** Rephrases the text. It is more natural but carries a higher risk of hallucinations.

**Challenges:** Accuracy, abstraction (don’t just extract verbatim), domain specific, and redundancy (don’t repeat the same fact in different ways). 

## 1.3 Translation

This is the canonical seq2seq task. The original 2017 transformer was built as a translation model. Sub-types of language translation and dialect translation.

**Challenges:** Ambiguities are amplified across language, low-resource representation for certain languages, tracking long sequences and word rarity.

## 1.4 Question-Answering

This category involves providing answers to user queries.

- **Internal-knowledge:** This answer is drawn directly from the model’s training corpus (or hallucinated if it doesn’t know).
- **External-knowledge:** Requires external retrieval systems (Like RAG) to pull inn outside facts.

**Challenges:** Reasoning capabilities, knowledge limits, context limits and lexical gap (when wording of the word differs significantly from the wording in the source text.

# 2. The three methods of text generation

## 2.1 Rule-based generation

This approach uses decision logic, fill in the blanks and templates (e.g. If time == night → “Bit late to be wanering around”). This was used heavily in early video games.

- Pros: Predictable, cheap to implement, requires no large dataset and works well on closed domains.
- Cons: Requires manual creation, lacks flexibility, cannot generalise to unseen phrasings and is brittle to small input variations.

Rule based systems are unsuitable for dynamic tasks like sports translation. The number of rules required for the cross product of vocabulary, grammar and idioms across multiple languages would be enormous making it impossible to handle unseen phrasings or dialect.

## 2.2 N-gram generalisation

> **N-gram language model:** A probabilistic model where the probability of the next token depends only on the previous N-1 tokens (Markov assumption of order N-1.
> 

**Conditional probability**

$$
P(w_{n}|w_{n-(N-1)},...,w_{n-1}) = \frac{count(w_{n-(N-1)},...,w_{n})}{count(w_{n-(N-1)},...,w_{n-1})}
$$

**Unigram, bigram, trigram.** A unigram model has $P(w_n)= count(w_n)/N$ - ignores all context. A bigram conditions on one previous token; a trigram on two; a 4-gram on three. 

**Generation procedure.** To start generating, the model needs something to look at first. A specific word or phrase you provide to kick of the story. A special marker <SOS> is used during training to denote the start of a sentence. Conditional distribution is used as above with probabilities calculated depending on the amount of context we are considering. <EOS> denotes the end of sequence, a specific character is used to predict when it thinks the sentence is finished. 

- **Greedy Picking:** Always picking the word with the highest probability but leads to repetitive boring text. Best for factual/code tasks.
- **Sampling:** Choosing a word based on its weight. If “reads” has a 50% chance, the model will pick it roughly half the time. This adds **variety and creativity.** Best for story telling.

**Pros:** Explainable, fast inference (it acts as a simple lookup) and serves as a standard baseline.

**Cons:** Limited creativity, can produce ungrammatical sentences, and cannot capture long range dependencies beyond its fixed window

<aside>
💡

**Pitfall:** The main limitation of N-grams is **sparsity,** not compute. As the window size grows, the vocabulary of possible contexts grow combinatorially, meaning most contexts have zero counts in the training data. 

Smoothing helps but doesn’t fully solve this.

</aside>

## 2.3 LSTM generation

> **Long Short-Term Memory (LSTM)** networks are neural sequence models. Text is chunked, passed through an embedding layer, processed by an LSTM layer, and fed into a dense layer with a softmax function over vocabulary to predict the next token.
> 
- <pad> used for shorter inputs makes the input chunk a standard length.

**Internals.** Use three gates (forget, input , output) to decide what to discard , add and expose at each timestep. The **forget gate** specifically allows LSTMs to handle long-range dependencies that vanilla RNNs lose due to vanishing gradients. 

**Pros.** Handles long-term dependencies, require no manual rule specification, produces more natural output than n-grams, and learns from raw data.

**Cons.** Computationally expensive, data-hungry, and its sequential processing makes training slow. 

### Exam question

Illustrate how an LSTM could perform translation.

**Input and embeddings.** Source-language tokens are fed into the embedding layer which maps each word into a dense vector.

**The encoder.** These vectors flow into an encoder LSTM. The encoder processes the sequence and produces a final hidden state.

**Information bottleneck.** This hidden state is crucial because it encodes the meaning of the entire full source sequence into one place.

**The decoder.** LSTM takes that hidden state, along with start-of-sequence token and begins its work. 

**Autoregressive generation.** The decoder predicts each target-language token via a softmax layer over the target vocabulary.

**The loop.** It operates autoregressively, meaning it feeds each predicted token back into the decoder as the input for the next step.

**Halt.** This loop continues until the model predicts an end-of-sequence token.

![diagram.jpg](Week%205%20-%20Sequence%20Models%20and%20Transformers/diagram.jpg)

# 3. Decoding strategies

Decoding strategies are the actual rules used to convert those distributions into final text sequences. These strategies balance between coherence, making sense and diversity (being creative). 

## 3.1 Greedy decoding

This method picks the `argmax P(w)` (the single highest probability token) at every single step.

- **Characteristics:** it is always deteministic (the same input will always yield the exact same output0 and it operates very fast.
- **The flaw:** it commits early to a token that looks “locally good: but might lead to a “globally poor” sequence overall, because it cannot look ahead.

## 3.2 Random sampling

Instead of always picking the top token, thi smethod samples a word w based on the model’s**The flaw: i** weighted probability distribution.

- **Characteristics:** it is always non-deterministic and produces much more diverse text than greedy decoding.
- **The flaw:** it can produce complete nonsense if low-probaibliity tokens happen to get “unlucky” and are chosen during a step.

## 3.3 Beam search

Beam search tracks multiple likely sequences (called “beams”) in parallel to prevent the early commitment mistakes made by greedy decoding. 

**Algorithm:** 

1. Define a beam width n 9e.g. n=2). A larger n yields better quality but requires more compute.
2. The model gives P(2) for the first word. keep the top-n tokens.
3. For each kept beam run the model on the extended sequence compute the joint probability of every 2 step extension.
4. You are then keeping the top-n of the joint sequences and repeat until you reach the <EOS> token is predicted or max depth. 
5. Return the sequence with the highest total joint probability

**Note.** So beam search tracks multiple likely sequences at each step, computes joint probability across steps, compares alternatives, and picks the most probable sequences without committing too early.

- **Costs vs benefit.** It is n times slower than greedy decoding because it runs the model n times per step. The benefit is robustness to local optima
    - Production systems typically use widths of 4-8
    - **Limitation:** it tends to produce repetitive safe outputs because it optimises strictly.

## 3.4 Temperature

> **Temperature.** A softmax-modifying parameter T that flattens or sharpens the nex-token distribution.
> 

**Standard softmax:**

$$
P_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

**Temperature softmax:**

$$
P_i = \frac{e^{z_i}/T}{\sum_j e^{z_j}/T}
$$

Dividing the logits by T before the exponential rescales the difference. 

- Low T: widens the gap between high and low probability tokens (sharper distribution)
    - risk averse. Coherent, predictable and reproducible. Good for code generation, factual QA, classification.
    - Repetitive / loops
- High T (greater than 1) narrows the gap (flatter distribution).
    - risk tolerant. Creative, diverse. Good for brainstorming, story generation
    - Incoherent or nonsense output.

## 3.5 Top-k vs Top-p (nucleus) sampling

**Top-k.** Restrict sampling to the k most likely tokens; renormalise their probabilities; sample from the truncated distribution. Excludes the long tail of low-probability nonsense. Typically k=40-50 words. 

**Top-p.** Restricts sampling to the smallest set of tokens whose cumulative probability exceeds p (e.g. p=0.9). Adapts the cut-off dynamically.

**Combining.** Production LLM APIs typically expose all three: temperature, top-k and top-p applied in that order. 

# 4. Transformers

The transofrmer replaced recurrent architectures for most NLP tasks. Its biggest breakthrough was removing the sequential bottleneck of RNNs by computing all token representations in parallel using a attention.

## 4.1 Architecture

The Transformer consists of stacked encoders feeding into stacked decoders. e.g. BERT uses 12 layers, 96+ for GPT-3. 

- **Encoder Block:** Self-attention → add & Norm → Feed-Forward MLP → Add & Norm.
- **Decoder Block:** Masked Self-Attention → Add & Norm → Encoder-Decoder Attention → Add & Norm → Feed-forward MLP → Add & Norm
- Add & Norm refers to a residual connection plus layer normalisation

![image.png](Week%205%20-%20Sequence%20Models%20and%20Transformers/image.png)

## 4.2 Intuition

Self attention allows each token to look at other tokens in the sequence to disambiguate context. For example, in the sentence “The animal didn’t cross the street because it was too fired”, the word “it” attends most strongly to “animal” to resolve the pronoun.

The Mechanics (Q,K,V):

Each token produces three vectors:

- Query (Q): represents what the token is looking for
- Key (K): represents what the token offers
- Value (V): represents what the token actually contributes if it gets attended to.

So the attention score between token i and token j is the dot product of $Q_i \cdot K_j$. A softmax function converts these into weights, and the final output for token i is the weighted sum of all $V_j$. These weights learn entirely from data.

**Special types of attention:**

- **Encoder-decoder attention:** This is the bridge between the input and output. In this layer the decoder attends not just to past tokens, but directly to the Keys and Values outputted byt he encoder. This step ensures that the text being generated remains firmly grounded in  the original source sequence (which is crucial for tasks like translation).
- **Masked Self Attention (Decoder Only:** During training, the decoder sees the entire target sentence at once. To prevent it from “cheating” by looking at future words it is supposed to predict, masking is applied. Masking sets attention scores to -infinity for future positions before the softmax is applied.

<aside>
💡

The conceptual flow of text generation in a transformer.

Source encoding: source tokens flow through a stack of encoder blocks. Each token uses self attention to attend directly to every other token in the sequence, deeply dismabiguating context.

Key and Value Handoff. The final layer of the encoder stack produces comprehensive key and vlaue matrices that represent the fully procesed context of the source sequence and passes these to every layer in the decoder

Autoregressive initilisation. The decoder generates text autoregressively one token at a time beginning its sequence with a start of sequence token <SOS>

Masked self attention within the decoder blocks, tokens pass through masked self-attnetion. This prevvents the model from cheating by setting attention scores for future positions. Setting the scores to negative infinity forces it to only attend to earlier tokens. 

Encoder-decoder attention: the decoder layers generate their own queries to attend to the keys and values provided by the encoder. This crucial step grounds the generation in the original source text.

Softmax prediction and feedback. The final decoder output is passed through a dense layer and a softmax function to produce a probability distribution over the target vocabulary. The chosen word is then appended to the decoder’s input to preduct the subsequent tokesn, repeating until an <EOS> token is reached.

</aside>

## 4.3 Why Transformers Won

LSTMs must process t before token t+1. Transformers compute positions simultaneously, leading to much better GPU utilisation and vastly faster training. **Parallelism**

**Long range dependencies:** Self-attention creates direct connections from any token to any other token (constant path length). LSTMs have an $O(n)$ path length meaning they lose information across long distances even with their special gates. 

**Scalability**: The combination of parallelism and the right inductive bias allows transformers to scale to billions of parameters. 

## 4.4 Tokenisation in modern transformers

Modern transformers don’t tokenise on whitespace. They use sub-word tokenisation (BPE = Byte Pair Encoding, WordPiece) that handles rare and outof vocabulary words by breaking them into known sub word pieces. 

# 5. Fine-tuning, transfer learning, pre-training

Pre-training is often quite expensive. So Transfer learning is an alternative. It works by freezing all but the final layer, Train only the final layer on a new task. 

- This is a cheaper alternative and only need a small target dataset.

**Fine-tuning.** Take a pre-trained model and update all layers for the new task. Mid cost moderate target dataset produces a model adapted to the new task.  

# BERT vs GPT

| **Feature** | **BERT** | **GPT** |
| --- | --- | --- |
| **Architecture** | Encoder-only | Decoder-only |
| **Direction** | Bidirectional (sees both sides of a token) | Causal / Autoregressive (only sees left context) |
| **Training Task** | Masked Language Modeling (MLM) | Next-token prediction |
| **Best For** | Classification, NER, Embeddings | Generation, dialogue, instruction-following |
| **Lineage** | ELMO → BERT → RoBERTa | GPT-1 → GPT-4, Claude, Gemini |
| **Size Example** | BERT-base: 110M parameters | GPT-3: 175B parameters |

<aside>
💡

One key distinction is regarding the task. Encoder is essentially the way you contextualise mathematical representations between input tokens. 

Decoder takes information from the output of the encoder but doesn’t update these ideas and the goal is next-token prediction. 

</aside>

## 6.1 BERT’s masked language modelling

Instead of predicting the next word, BERT randomly masks 15% of the input tokens during pre-traiing to guess what they are. Because it looks at the surrounding words on both sides to make its guess, it is a naturally **bidirectional** encoder. 

- The purpose of this mask is to speed up training time but not remove too much of the context to actually solve the mask.
- Of that 15% it is split 80/10/10.
    - 80% replaced with [MASK]
    - 10% are replaced with a random wrong token: forces model to trust the surrounding context rather tahn blindly accepting the token it sees.
    - 10% are kept as the original correct token: This keeps the model on its toes, forcing to. maintain sharp, useful representations at every point because it never knows if ta normal looking token is actually a test.

## 6.2 GPTs Autoregressive Modelling

GPT predicts the next token given all previous tokens. Curcially it applies **causal masking,** meaning each position can only attend to early positions.

- GPT is naturally generative, allowing you to sample on token at a time fluently. BERT cannot generate fluently because its bidirectional training never asked it to practice left-to-right writing.