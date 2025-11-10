# W2 L1 - Decision Trees

9 October 2025 

Decision trees - used for supervised classification

- Random forest which builds on decision trees were the best approach before deep learning

## What is the goal?

$\textbf{x} \in R^n, \text{ n-dimensional feature vector}$

$y \in \{0, ..., K-1 \}, \text{ is the output class when their are K possible output classes}$

<aside>

**Goal:** to learn $y = f(x)$ from the data

</aside>

- We can’t consider every possible $f(x)$ since there are infinitely many that fit any data set
- Instead we assume functions $f_{\theta}(x)$ that belong to a space of functions parameterised by $\theta$.
    - $\theta$ is a vector of parameters

We learn the parameter $\theta$ in $y = f_{\theta}(x)$ from the initial data or training set:

- $\{(\mathbf{x_1}, y_1), (\mathbf{x_2}, y_2), ..., (\mathbf{x_N}, y_N) \}$
    - each $\mathbf{x_i}$ is an n-dimensional vector.

### How do we know which $\theta$ is best?

- **Loss function: $L(y_i, f_{\theta}(x_i))$**
- **Total loss: $\sum_{i=1}^{N}L(y_i, f_{\theta}(x_i))$**
    - Compares your prediction $f_{\theta}(x_i)$ with the actual value $y_i$.

<aside>

**Goal:** find $\theta$ that minimises the total loss using optimisation techniques. 

</aside>

## Decision Stump

Idea that one feature that could explain whether an animal is a vertebrate or a invertebrate is the so called decision stump.

- Parameters: $\theta = \{\text{feature}, \text{match} \}$
- Function: $f_{\theta (x)} = \delta(x_{\text{feature}},\text{match})$
    - Tells you whether this feature matches the Boolean variable or not

$$
L(y_i, f_{\theta}(x)) =
\begin{cases}
    0 & \text{if } y_i=f_{\theta}(x_i) \\
    1 & \text{otherwise}
\end{cases}
$$

- called **0-1 loss**

### **Learning process:**

- We are trying to learn $\theta$ to see whih feature and match has the lowest loss.
    
    
    | **Animal** | **Has Backbone** | **Has Shell** | **Label** |
    | --- | --- | --- | --- |
    | Shark | 1 | 0 | 1 |
    | Octopus | 0 | 0 | 0 |
    | Crab | 0 | 1 | 0 |

$\text{Testing: }\theta = \{\text{Has Shell},1\}$

- Shark $\rightarrow \delta(0,1)=0\rightarrow L(1,0)=1$
- Octopus $\rightarrow \delta(0,0)=1\rightarrow L(1,1)=0$
- Crab $\rightarrow \delta(1,1)=1\rightarrow L(0,1)=1$
- Total Loss = 2

$\text{Testing: }\theta = \{\text{Has Backbone},1\}$

- Shark $\rightarrow \delta(1,1)=1\rightarrow L(1,1)=0$
- Octopus $\rightarrow \delta(0,1)=0\rightarrow L(0,0)=1$
- Crab $\rightarrow \delta(0,1)=0\rightarrow L(0,0)=0$
- Total Loss = 0

Hence the function learned with the lowest loss: $f_{\theta (x)} = \delta(\text{Has Backbone},1)$

- Find the input feature which is most similar to the output or the inverse is most similar

## Continuous Decision Stump

- The $x$ and $y$ axes represent the features, which are continuous inputs
- The colour of the point in the scatter graph is the binary ouput.
    - For example angle, distance from goal. Output is (on target).

Since the inputs are continuous, we could instead **split** the space:

$$
f_{\theta}(x) =
\begin{cases}
    0 & \text{if } x_{feature}<\text{split} \\
    1 & \text{if } x_{feature}≥\text{split} \end{cases}
$$

- similary can use $y_{feature}$ for the split

![Screenshot 2025-11-06 at 12.50.10.png](W2%20L1%20-%20Decision%20Trees/Screenshot_2025-11-06_at_12.50.10.png)

### **How to find best parameters?**

- Brute force varying the split value across each feature and evaluate the loss.

![Screenshot 2025-11-06 at 13.06.30.png](W2%20L1%20-%20Decision%20Trees/Screenshot_2025-11-06_at_13.06.30.png)

- The left graph is varying the $x_{feature}$ and the right is varying the $y_{feature}$
- Each graph has the feature on the x axis and is measuring the accuracy

## Decision Trees

For a more complex example we require decision stumps to be performed recursively.

- As before we choose the first split to minimise the loss or maximise the accuracy
- **Greed optimisation** of parameters
    - brute force method at each node

**Decision tree** = recursive splitting

![Screenshot 2025-11-06 at 13.11.55.png](W2%20L1%20-%20Decision%20Trees/Screenshot_2025-11-06_at_13.11.55.png)

![Screenshot 2025-11-06 at 13.12.05.png](W2%20L1%20-%20Decision%20Trees/Screenshot_2025-11-06_at_13.12.05.png)

![Screenshot 2025-11-06 at 13.20.11.png](W2%20L1%20-%20Decision%20Trees/Screenshot_2025-11-06_at_13.20.11.png)

![Screenshot 2025-11-06 at 13.12.15.png](W2%20L1%20-%20Decision%20Trees/Screenshot_2025-11-06_at_13.12.15.png)

![Screenshot 2025-11-06 at 13.20.26.png](W2%20L1%20-%20Decision%20Trees/Screenshot_2025-11-06_at_13.20.26.png)

![Screenshot 2025-11-06 at 13.12.24.png](W2%20L1%20-%20Decision%20Trees/Screenshot_2025-11-06_at_13.12.24.png)

![Screenshot 2025-11-06 at 13.20.32.png](W2%20L1%20-%20Decision%20Trees/Screenshot_2025-11-06_at_13.20.32.png)

![Screenshot 2025-11-06 at 13.12.34.png](W2%20L1%20-%20Decision%20Trees/Screenshot_2025-11-06_at_13.12.34.png)

continuing like this

### Multiple ouputs

<aside>

**Terminology:**

**Features:** The input variables (x values) that describe each data point

**Classes:** Output labels (y values) that indicate whihc category each input belongs to

</aside>

**Example:** deciding a loan application, 

- Three classes - approve (0), reject (1), and hold (2).
- Four features - age, salary/income, marital status, profession.

![Screenshot 2025-11-07 at 00.11.41.png](W2%20L1%20-%20Decision%20Trees/Screenshot_2025-11-07_at_00.11.41.png)

> **A Node** represents a decision point in the structure of a decision tree. There are three main types:
> 
- **Root node:** the top of the tree, representing the entire dataset before any splits
- **Internal nodes:** points where the data is split based on a feature value ****
- **Leaf nodes:** end point of the tree containing the final prediction or class label