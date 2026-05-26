# W1 L1: Introduction

29 September 2025 

[Week1Lec1Introduction.pdf](W1%20L1%20Introduction/Week1Lec1Introduction.pdf)

## Introduction to Deep Learning

- From 1950s - late 1980s, early AI used rule-based or expert systems.
    - follows explicit programmed rules rather than learning from the data
    - i.e. if and else statements
- However it is unable to figure out explicit rules for most real-life problems.
    - e.g. how does a toddler know that a giraffe is a giraffe
- Instead you provide some examples and let the algorithm learn the rules.

![Screenshot 2025-10-07 at 16.51.05.png](W1%20L1%20Introduction/Screenshot_2025-10-07_at_16.51.05.png)

<aside>

> **Deep Learning:** is a branch of machine learning that teaches computers to learn patterns and representations directly from raw data by using multi-layered neural networks.
> 
</aside>

### Why Deep Learning?

![Screenshot 2025-10-07 at 17.00.05.png](W1%20L1%20Introduction/Screenshot_2025-10-07_at_17.00.05.png)

- The above pipeline was how we traditionally approached making learning systems.
    - Here the domain expert would define the important features
- Deep learning however takes the data and would automaticaly learn the rules
    - **The system is an end-to-end process**
    - It learns the **hierarchical features** from the raw data **automatically**
    - We do not have access to the rules used to make a decision so the system becomes more opaque.
- The hierarchical structure represents **different levels** of features
    - Each layer learns to recognise features at a level of abstraction

**Example**

| **Layer** | **What it learns** |
| --- | --- |
| 1st (lowest layer) | Simple patterns like edges, colors, corners |
| 2nd layer | combination of edges - shapes, textures |
| 3+ layers | Meaningful objects - faces, animals, letters |
- Each layer **builds on the features learned by previous layers**, forming a hierarchy.

### Landmarks in Deep Learning

![Screenshot 2025-10-07 at 17.12.09.png](W1%20L1%20Introduction/Screenshot_2025-10-07_at_17.12.09.png)

## Types of Learning

![Screenshot 2025-10-07 at 18.01.48.png](W1%20L1%20Introduction/Screenshot_2025-10-07_at_18.01.48.png)

### Examples of Supervised Learning

<aside>

> **Supervised learning:** is a type of machine learning in which a model is trained using labelled data - each input inn the training set is paired with a known correct output
> 
> - learns to map inputs to outputs by finding patterns in the examples provided.
>     - Predicts the output for new, unseen inputs.
</aside>

![image.png](W1%20L1%20Introduction/image.png)

<aside>
💡

**Terms**

- Regression = continuous numbers as ouput
- Classification = discrete classes as output
- Two class and multi-class classification treated differently
- Univariate = one output
- Multivariate = more than one output
</aside>

**a) Regression**

- Univariate regression problem (one output, real value)
- Work with continuous outputs
- Regression involves continuous variables
- *Fully connected network*

All inputs influence all the outputs

- for non-linear regression you have multiple layers

![image.png](W1%20L1%20Introduction/image%201.png)

Understanding each layer in the example of house prices:

- Layer 0 - raw features: size, location, bedrooms, age, etc.
- Layer 1 - bigger house tend to cost more, newer houses are more expensive, good location increase price, etc.
- Layer 2 - large and new houses in good areas are premium, small houses in poor locations are cheap.
- Layer 3 - complex iteractions price rises fast until around 200 $m^2$, then slows.

**b) Graph regression**

- Multivariate regression problem (>1 output, real value)
- Graph neural network (not looking in this unit)

**c) Text Classification**

- Binary classification problem (two discrete classes)
    - e.g. positive or negative
- Discrete outputs
- Solved using Transformer network

**d) Music genre classification**

- Multi-class classification problem (discrete classes, >2 possible values)
- Recurrent neural networks (RNNs)

**e) Image classification**

- Multi-class classification problem (discrete classes, >2 possible values)
    - input here is even larger than music genre
- Convolution network

### **Supervised learning model structure**

![Screenshot 2025-10-14 at 12.36.36.png](W1%20L1%20Introduction/Screenshot_2025-10-14_at_12.36.36.png)

- Deep neural networks are jsut a very flexible family of equations
    - Deep learning algorithm finds the appropriate set of equations to describe the data
- **Process of fitting deep neural networks = Deep learning**

### **More complex examples**

![Screenshot 2025-10-14 at 12.50.28.png](W1%20L1%20Introduction/Screenshot_2025-10-14_at_12.50.28.png)

<aside>
💡

Key Terms

**Structured output:** An output that has internal relationships patterns among its elements

**Aligned ouput:** Output that corresponds spatially or positionally to the input - same width, height or sequence index.

**Local Correlation:** The idea that nearby elements are statistically related or similar

**High-dimensional output:** The model’s output has many elements or coordinates

**Ambiguity:** There can be several correct answers for the same input

**Spatial structure:** Arrangement and relationship between neighbouring positions in an image or space

**Temporal structure:** Relationships over time between sequential elements

**Sequence-to-Sequence models:** A model that maps one sequence to another possibly of different length.

</aside>

**a) Multivariate Binary Classification - Semantic Segmentation**

- Each pixel in the input image gets a binary label (e.g. cow vs background)
- Output: binary image of the same size as the input
    - is high-dimensional and structured
- Convolutional encoder-decoder network

**b) Multivariate regression - Monocular depth estimation**

- Input: RGB street-scene image
- Output: continuous depth value for each pixel (continuous regression)
- Convultional encoder-decoder network

**d) Sequence-to-Sequence - machine translation**

- input: english text string
- output: french translation
- output structure follows linguistic rules
- multiple valid outputs may exist for each input

**e) Generative mapping - image synthesis**

- input: descriptive text
- ouput: generative image matching that description
- Many possible valid outputs per caption

### Challenges

- very complex relationship between input and output
- sometimes may be many possible valid answers
- Outputs and inputs may need to obey some rules
    - language rules, natural images

## Unsupervised Learning

Learning about a dataset without labels. Different types include:

- clustering
    - DeepCluster (Caron et al., 2018)
- Finding outliers
- Generating new examples
    - generative adversarial networks: generates fake data
- Filling in missing data
- Learns distributions over data
    - **Variational autoencoders:** learns to encode data into a smooth latent space and decode it back, generating new samples probabiltistically.
    - **Normalising flows:** learns an exact, invertible mapping betweena. simple distriution and comples real data.
    - **diffusion models:** learns to gradually denoise random noise step-by-step to generate realistic data such as images
- Conditional synethesis: generating new data while controlling specific attributes or conditions.
    - e.g. erase part of the image and ask the model to repaint it without the obstructution.

---

> **Latent variables** are hidden representations learned in unsupervised models that capture underlying data structure and enable generation of new, realistic samples.
> 

---

- Deep learning work since most problems have some underlying regularity
    - e.g. face we have 42 face muscles hence a defined number of possibilities.

## Reinforcement learning

- a set of states → actions → rewards loop.

> **Goal:** take actions to change the state so that you recieve rewards
> 

You dont recieve any data - you have to explore the environment yourself to gather data as you go

- example: chess
    - Actions at a given time are valid possible moves
    - Possible rewards for taking pieces, negative for losing
    - These rewards may not be immediate