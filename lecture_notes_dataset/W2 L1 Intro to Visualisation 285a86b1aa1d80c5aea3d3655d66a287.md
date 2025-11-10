# W2 L1: Intro to Visualisation

6 October 2025 

[Lecture 5, Intro to Visualization.pdf](W2%20L1%20Intro%20to%20Visualisation/Lecture_5_Intro_to_Visualization.pdf)

<aside>
💡

**Information Visualisation:** The use of computer generated, interactive, visual representations of data to amplify cognition

</aside>

- There is a lot of data and many ways to show it
    - Choosing which data to present is part of good visualisations.
- Analysis is a cycle:
    - Gathering data → apply some statistical tools → construct graphics to answer these questions → inspect answers → assess new questions

### Data Pipeline examples

| **Type of Pipeline** | **Purpose** | **Example Output** |
| --- | --- | --- |
| Classification | Assigns data to discrete categories | Spam / Not spam |
| Regression | Predicts a continuous numeric value | House price vs features |
| Clustering | Groups similar data points | Customer segments |
| Dimensionality reduction | Compresses features into fewer dimensions | Principal components |
| Anomaly detection | Identifies unusual patterns | Fradulent transactions |

<aside>
💡

**Black box pipeline:** is a data-processing or ML system where the input and output are known, but the internat decision making process is not interpretable.

</aside>

### **Why do we need to visualise data ?**

- Data does not directly give answers
    - You need to clearly present your conclusions
- There is too much data to show
- Data science pipeline
    - We can report and visualise data at every stage
        - **Help** to analyse the data
        - **Inform** others
        - **Persuade** others
- Human vision is **preattentive**
    - Brains can detect visual properties almost instantly
    - Reduces the effort the viewer needs to understand the information.

<aside>
💡

**Exploratory Data Analysis:** is a type of data analysis focused on looking at the data directly (through graphs, summaries, plots), finding unexpected patterns or relationships, detecting outliers and generating new hypotheses.

- Here we are learning about the data directly rather than using some pre-defined model.
</aside>

## Data visualisation tips

### Purpose of the visualisation

- For **whom** are you creating the visualisation?
- What is the **purpose** of it?
    - Can determine the validity of some visualisation by understanding if the source has vested interest. e.g. political campaigns
- How do you **effectively** deliver the information to them?

### Visualisation for analysis

Visualisaition allows us to inspect and understand data before processing it. i.e. checking for outliers

It takes advantage of our ability to:

- **Absorb** more information visually
- **Encode** and interpret appters efficiently
- **Recognise** trends and anomalies intuitively

### Post Ingestion

- May cause you to question whether there is a problem with ingestion/wrangling of the data?
- Is this outlier real

**Example:** involving the classification of apples and oranges.  ****

![Screenshot 2025-10-07 at 15.33.29.png](W2%20L1%20Intro%20to%20Visualisation/Screenshot_2025-10-07_at_15.33.29.png)

**k-Nearest Neighbours:** is a non-parametric classification algorithm that assigns a label to a new data point based on the majority of its k closest points in the training data. 

- an example of an algorithm for classification
- for example the plot could be weight vs diameter with the the use of colour to see if it is an apple or orange.
- Use the visualisation to see how apples and oranges are spread in feature space
- We notice if they form distinct clusters or overlap

<aside>
📎

The main part is how a visualisation is able to notice these ideas and suggest problems in your data. 

</aside>

### Data understanding and validation

- Visualisation enable data **‘sanity check’**
    - e.g. object detection model binary: simply outputting a number is not really intuitive
- Build trust in underlying processes and analysis
    - e.g. in a black box pipeline where you are unaware of the internal workings having a visualisation at each step will build trust in your process.
    - e.g. ChatGPT and Gemini show how they perform on different benchmarks to buld this trust
- Beware: design parameters of visualisations can obscure data quality issues
    - e.g. in the histogram example you cannot see the outlier when choosing a larger bin size
- Visualisation can enable generation of intial hypotheses or questions to explore
- Data analysis often involves iterative process of visual inspection and refinement of questions to investigate

### Dimensions

- All the data we have looked at has a limited number of dimensions we are attempting to visualise
- Large datasets come with the challenge of increased dimensions
    - Deciding which are important is nto easy
    - may perform analysis to decide
- Need to consider how we encode the infromation
    - We can use more than just too dimensions of the page
    - e.g. the colour, shape, intensity (to give a third dimension)
    - e.g. in contour plots or a rain map with intensity showing heavier rainfall
    - Should try be consistent with the method to not confuse the audience
- Need to consider the level of detail
    - **Too much** and we **overwhelm the user**
    - **Too little** and we **miss the important insight**

## Data Visualisation examples

### Box Plots

![Screenshot 2025-10-07 at 15.03.45.png](W2%20L1%20Intro%20to%20Visualisation/Screenshot_2025-10-07_at_15.03.45.png)

### Scatter plot

![Screenshot 2025-10-07 at 15.42.42.png](W2%20L1%20Intro%20to%20Visualisation/Screenshot_2025-10-07_at_15.42.42.png)

- Often used to represent 2D plots
- Can reduce multidimensional data using methods such as:
    - Principal Component Analysis (PCA)
    - Linear Discriminant Analysis (LDA)

### Histograms

![Screenshot 2025-10-07 at 15.58.02.png](W2%20L1%20Intro%20to%20Visualisation/Screenshot_2025-10-07_at_15.58.02.png)

- bin size is the width of the bar

### Simple Graphs

![Screenshot 2025-10-07 at 16.04.04.png](W2%20L1%20Intro%20to%20Visualisation/6a2bc6ad-1d1d-4ce6-a109-bdef94a97034.png)

- for one dimensional data

### Pie Charts

![Screenshot 2025-10-07 at 16.06.14.png](W2%20L1%20Intro%20to%20Visualisation/Screenshot_2025-10-07_at_16.06.14.png)

### Chord Diagrams

![Screenshot 2025-10-07 at 16.06.44.png](W2%20L1%20Intro%20to%20Visualisation/Screenshot_2025-10-07_at_16.06.44.png)

- Show the inter-relationships between data in a matrix
    - Slightly misleading Large fries appear as though the proportion of vitamin C is greater then fats. Vitamin C would. Be measured in miligrams

### Network Diagrams

![Screenshot 2025-10-07 at 16.07.13.png](W2%20L1%20Intro%20to%20Visualisation/Screenshot_2025-10-07_at_16.07.13.png)

### Word Clouds

![Screenshot 2025-10-07 at 16.07.45.png](W2%20L1%20Intro%20to%20Visualisation/Screenshot_2025-10-07_at_16.07.45.png)

### Animations

### Interactivity

![Screenshot 2025-10-07 at 16.22.47.png](W2%20L1%20Intro%20to%20Visualisation/Screenshot_2025-10-07_at_16.22.47.png)

This representation allows you to toggle the different catagories

- The ability to explore the data by varying the parameters yourself
- Important if we don’t know what we are looking for
- Easy way to compact a lot of data

- **Scikit-learn** to generate some of these visualisations