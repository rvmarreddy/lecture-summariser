### 🧩 Slide 1

- **Outputs**: The results produced by a system or process.
- **Context**: Outputs can refer to various fields such as mathematics, engineering, and computer science.

#### Key Concepts
- **Definition of Output**: The final product or result generated from an input or a series of processes.
- **Mathematical Representation**: 
  - If $$ f(x) $$ is a function, then the output is $$ f(x) $$ for a given input $$ x $$.
  
#### Intuition
- Outputs are essential for evaluating the effectiveness of a system.
- Understanding outputs helps in optimizing processes and improving performance.

![Slide 1 - Image 1](./slides_output/image_1)

---

### 🧩 Slide 2

#### Regression
- **Definition**: A statistical method for predicting continuous outcomes based on input variables.

#### Classification
- **Purpose**: Assign labels to data points based on their features.
  
- **Splitting Criteria**:
  - **Gini Impurity**:
    - Measures the impurity of a dataset.
    - Formula: 
      $$ Gini = 1 - \sum (p_i^2) $$
      where \( p_i \) is the probability of class \( i \).
    - Aim: Minimize Gini impurity to create pure nodes.
  
  - **Information Gain**:
    - Measures the reduction in entropy after a split.
    - Formula:
      $$ IG = H(parent) - \sum \left( \frac{N_{child}}{N_{parent}} H(child) \right) $$
      where \( H \) is entropy.
    - Aim: Maximize information gain to improve classification accuracy.

- **Leaf Nodes**:
  - Provide the final classification.
  - Output the most common class among the data points reaching that leaf.

![Slide 2 - Image 1](./slides_output/image1)

---

### 🧩 Slide 3

#### Regression vs. Classification

- **Classification**:
  - **Objective**: Assign data points to discrete classes.
  - **Splitting Criteria**:
    - Minimise **Gini impurity**: Measures the impurity of a dataset.
    - Maximise **Information Gain**: Measures the reduction in uncertainty.
  - **Leaf Node Output**: Most common class among data points in the leaf.

- **Regression**:
  - **Objective**: Predict continuous values.
  - **Splitting Criteria**:
    - Minimise **Variance**: Measures the spread of data points.
    - Maximise **Information Gain**: Similar to classification but focuses on continuous outcomes.
  - **Leaf Node Output**: Mean value of data points in the leaf.
    - **Note**: Using the **median** can be advantageous in the presence of outliers, as it is less sensitive to extreme values.

- **Key Insight**: 
  - Both methods use similar structures but differ in their objectives and output types.

![Slide 3 - Image 1](./slides_output/image_1)

---

### 🧩 Slide 4

#### Variance Reduction
- **Definition**: Variance measures the spread of output values. A lower variance indicates that the output is more consistent.
- **Objective**: Choose splits in data that minimize variance to improve model accuracy.

#### Key Concepts
- **Node Variance**: 
  - A node's variance is calculated based on the outputs of the data points it contains.
  - The goal is to create child nodes with lower variance than the parent node.

- **Splitting Criteria**:
  - When deciding how to split data, select the option that results in the least variance for the resulting nodes.
  - This leads to more homogeneous groups, enhancing predictive performance.

#### Mathematical Representation
- Variance for a set of values can be calculated as:
  $$
  \text{Variance} = \frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2
  $$
  where:
  - \( N \) = number of observations
  - \( x_i \) = individual observation
  - \( \bar{x} \) = mean of observations

![Slide 4 - Image 1](./slides_output/image_1)

---

### 🧩 Slide 5

#### Variance Reduction
- **Definition**: Variance measures the consistency of output in a dataset.
- **Goal**: Choose splits that minimize variance to improve prediction accuracy.

#### Variance Calculation
- **Left Node Variance**:
  $$ \sigma^2_l = E\left[(Y_l - E[Y_l])^2\right] $$
  - Where:
    - \( Y_l \) = output values in the left node
    - \( E[Y_l] \) = expected value (mean) of \( Y_l \)
  
- **Right Node Variance**:
  $$ \sigma^2_r = E\left[(Y_r - E[Y_r])^2\right] $$
  - Where:
    - \( Y_r \) = output values in the right node

#### Intuition
- A lower variance indicates that the outputs are more consistent.
- By minimizing the variance in both nodes after a split, we enhance the model's predictive power.

![Slide 5 - Image 1](./slides_output/image_1)

---

### 🧩 Slide 6

#### Variance Reduction
- **Definition**: Variance measures the consistency of output in a node.
- **Goal**: Choose splits that minimize variance to improve model accuracy.

#### Variance Calculation
- **Left Node Variance**:
  $$\sigma^2_l = E\left[(Y_l - E[Y_l])^2\right]$$
  - Where \(Y_l\) represents the outputs of data going to the left node.
  
- **Right Node Variance**:
  $$\sigma^2_r = E\left[(Y_r - E[Y_r])^2\right]$$
  - Where \(Y_r\) represents the outputs of data going to the right node.

#### Weighted Combination of Variance
- **Objective**: Minimize the weighted combination of variances from both nodes.
- **Formula**:
  $$L(\text{split}) = \frac{n_l}{n} \sigma^2_l + \frac{n_r}{n} \sigma^2_r$$
  - \(n\): Total number of exemplars.
  - \(n_l\): Number of exemplars in the left branch.
  - \(n_r\): Number of exemplars in the right branch.

#### Intuition
- A split that results in lower variance in both child nodes indicates a better decision point.
- Reducing variance leads to more reliable predictions.

![Slide 6 - Image 1](./slides_output/image_1)

---

### 🧩 Slide 7

#### Information Gain
- **Definition**: Information gain measures the reduction in uncertainty about a random variable after observing another variable.
- **Application**: Commonly used in classification tasks, but not typically for regression tasks with decision trees.

#### Key Points
- **Regression Tasks**: 
  - In regression, the goal is to predict continuous outcomes, making information gain less applicable.
  - Instead, metrics like Mean Squared Error (MSE) or Mean Absolute Error (MAE) are preferred.

#### Intuition
- Information gain quantifies how much knowing the value of one variable improves our prediction of another.
- It is calculated using entropy, which measures the unpredictability of information content.

#### Formula
- Information Gain can be expressed as:
  $$ IG(Y|X) = H(Y) - H(Y|X) $$
  - Where:
    - $$ H(Y) $$ = Entropy of the target variable before observing $$ X $$
    - $$ H(Y|X) $$ = Conditional entropy of $$ Y $$ given $$ X $$

![Slide 7 - Image 1](./slides_output/image1.png)

---

### 🧩 Slide 8

#### Information Gain in Regression
- **Information Gain**: A metric typically used for classification tasks, not directly applicable to regression.
  
#### Steps to Apply Information Gain for Regression
1. **Fit Gaussian Distribution**: 
   - Fit a Gaussian distribution to the output variable.
  
2. **Compute Entropy**:
   - The entropy for a Gaussian distribution is given by:
     $$ 
     H = \frac{1}{2} \log(2\pi e \sigma^2) 
     $$
   - Here, $\sigma^2$ is the variance of the distribution.

#### Information Gain Formula
- The information gain for a split is calculated as:
  $$
  I(split) = \frac{1}{2} \log(2\pi e \sigma^2_p) - \frac{n_l}{2n} \log(2\pi e \sigma^2_l) - \frac{n_r}{2n} \log(2\pi e \sigma^2_r)
  $$
  - **Variables**:
    - $p$: Parent node
    - $l$: Left child node
    - $r$: Right child node
    - $n_l$: Number of samples in left child
    - $n_r$: Number of samples in right child
    - $n$: Total number of samples

#### Intuition Behind the Formula
- The formula compares the entropy of the parent node with the weighted entropies of the child nodes.
- A higher information gain indicates a better split, as it suggests that the child nodes are more informative than the parent.

![Slide 8 - Image 1](./slides_output/image_1)

---

### 🧩 Slide 9

#### From Decision Trees to Random Forests via Bias-Variance Tradeoff

- **Decision Trees**:
  - Simple, interpretable models.
  - Prone to overfitting (high variance).
  - Low bias but high variance.

- **Bias-Variance Tradeoff**:
  - **Bias**: Error due to overly simplistic assumptions in the learning algorithm.
  - **Variance**: Error due to excessive sensitivity to fluctuations in the training set.
  - Goal: Minimize total error, which is the sum of bias and variance.

- **Random Forests**:
  - Ensemble method combining multiple decision trees.
  - Reduces overfitting by averaging predictions.
  - Balances bias and variance:
    - Increased bias (due to averaging) but significantly reduced variance.

- **Mathematical Representation**:
  - Total Error: 
    $$ \text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error} $$

- **Intuition**:
  - Single decision tree = high variance, low bias.
  - Random forest = moderate bias, low variance.
  - Tradeoff leads to better generalization on unseen data.

![Slide 9 - Image 1](./slides_output/image1.png)

---

### 🧩 Slide 10

#### Bias-Variance Tradeoff

- **Definition**: The bias-variance tradeoff is a fundamental concept in machine learning that describes the tradeoff between two types of errors that affect model performance:
  - **Bias**: Error due to overly simplistic assumptions in the learning algorithm. High bias can cause an algorithm to miss relevant relations (underfitting).
  - **Variance**: Error due to excessive sensitivity to fluctuations in the training data. High variance can cause an algorithm to model the random noise in the training data (overfitting).

#### Key Points

- **Bias**:
  - Represents the error introduced by approximating a real-world problem with a simplified model.
  - A model with high bias pays little attention to the training data and oversimplifies the model.

- **Variance**:
  - Represents the error introduced by the model's sensitivity to small fluctuations in the training set.
  - A model with high variance pays too much attention to the training data and captures noise as if it were a true pattern.

- **Tradeoff**:
  - As model complexity increases, bias decreases and variance increases.
  - The goal is to find a balance that minimizes total error, which is the sum of bias squared, variance, and irreducible error.

#### Visual Representation
![Slide 10 - Image 1](./slides_output/slide_10_img_1.png)

---

