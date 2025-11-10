# W2 L2 - Decision Trees cont.

10 October 2025 

- repeated splitting to make a diagonal decision boundary

<aside>

**Key point:**

In decision trees, splits are always **axis-aligned,** either horizontal or vertical along individual features.

</aside>

![image.png](W2%20L2%20-%20Decision%20Trees%20cont/b61f660f-6bd0-4eee-a8ae-685fd3a95677.png)

- In the example above when the data can be separated by a simple linear seperator, the decision tree cannot directly make that diagonal split.
- Through recursive binary splits, the tree gradually forms a staircase boundary.
- The higher the depth of the decision tree, the closer it tends to the true diagonal separator.

**Question: why does a decision tree only split on one feature at a time?**

- Computationally explosive, if there are 100 features and checking all pairs means testing $\binom{10}{2}$pairs at each node.
- Checking triples or higher-order sets would exponentially grow this value further

**Question: Can we reduce dimensionality to make separation easier?**

- Dimensionality reduction uisng methods like PCA compress data by projecting onto a lower dimensional space (keeping variance).
    - But this comes at the cost of losing features that differentiate classes and does not necessarily improve spearability.
- Higher dimensional spaces often make separation easier, intuitively add extra derived features opens more possiblities for complex class boundaries.
    - The **kernel method** uses this to transform each input into a richer feature space

## Decision trees - Loss fn