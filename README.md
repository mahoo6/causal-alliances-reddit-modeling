**Is the Enemy of my Enemy my Friend?**

**Abstract**

This project investigates whether online communities form alliances through shared hostility. Using Reddit as a case study, we model the temporal evolution of inter-community relations based on sentiment-labeled interactions.
To support our analysis, we developed an interactive and dynamic map to visualize all our results, from intermediate to final outcomes, and illustrate how inter-community relationships evolve over time.
Since our analysis relies on the accuracy of these sentiment labels, we first assess and refine the dataset’s labeling process. We aim to detect hidden hostilities (interactions labeled as non-negative but that implicitly convey hostility) in order to expand and strengthen our dataset.
Finally, we test the hypothesis “the enemy of my enemy is my friend”, examining whether communities that co-attack the same targets are more likely to develop new positive relations, while accounting for the effect of other confounding factors such as pre-existing similarities between subreddits.


**Proposed additional datasets**

In addition to our dataset, we use subreddit embeddings containing one 300-dimensional vector per subreddit, with closeness capturing user-base similarity.

**Research Questions**

- Does the current labeling of inter-community interactions hide undetected hostilities, and how can it be improved?
  - Can temporal patterns in sentiment shifts between communities reveal potential labeling errors? 
  - How can linguistic features help identify mislabeled interactions?
  - Can community embeddings confirm or refine the detection of hidden hostilities?


- Does sharing a common enemy lead two subreddits to form new causal friendships?
  - How does the level of positive interaction between subreddit pairs change before and after a shared conflict?
  - Does co-attacking a common target causally increase the likelihood of future friendship, or is it mere correlation?
  - How robust is this effect to unobserved confounders such as shared users or ideology?







**Methodology**
	
  **1- Hidden hostilities**

The original labeling of post sentiments (source) was obtained from a model with an accuracy ≃ 0.80, trained on only 1,020 manually labeled posts taken solely from the source. In addition, positive and neutral sentiment classes were merged, and the label detected is only as accurate as the sentiment is explicit. Such limitations make false positives and mislabeled cases plausible, motivating our attempt to detect potential hidden hostilities. 
We approach this problem through three complementary axes:

- **The temporal dimension:**
    We analyze how often and how quickly sentiment shifts occur between community pairs to detect suspicious transitions. For each interaction (positive or negative), we     check whether an opposite-sentiment link between the same communities appears soon after the event. A positive link occurring shortly before or after a negative one may indicate hidden hostility that was initially mislabeled as non-negative. We will study the delay of these switches relative to the usual frequency of exchanges to identify serious candidate mislabels.

- **Linguistic analysis:**
    We examine whether lexical and emotional markers in the source posts align with their assigned sentiment labels.
We focus on key features (ex: VADER compound, LIWC_Anger, LIWC_Swear, LIWC_Dissent, LIWC_You/They) which capture polarity, hostility, and outgroup targeting.
Outliers in these distributions are flagged as potential mislabels.
These candidates will then be cross-checked through temporal analysis or embedding similarity (Section 3) to validate hidden hostility patterns.

- **Community Embedding and Hostility Correlation:**
    We integrate the embedding space of subreddits to jointly consider the source and target in sentiment labeling (unlike the original approach). Since hostility often depends on the relation between communities, we analyze cosine similarities between their embeddings. A preliminary result shows that cosine similarity distributions are different for negative and non-negative interactions. We therefore plan to leverage this feature on our candidate interactions (identified beforehand) to help estimate the likelihood of mislabeling and refine the sentiment classification.






**2- Is the enemy of my enemy my friend?**

To answer our problem, we then proceed as follows:

- **Descriptive Co-Attack Analysis:**
    First, a descriptive analysis identifies for every target C, all the pairs of subreddits (A,B) that co-attacked it within a certain time frame (monthly). For every (A,B,C) and every month, we store the number of negative links A->C and B->C. Then, for each pair (A,B), we define the earliest month in which they started co-attacking C as conflict_start, and we study the evolution of the number of positive links A<->B by comparing them before and after this conflict to identify how many new friendships were formed post-conflict.

- **Causal Inference:**
    This section applies a matching-based causal inference approach to test whether co-attacking a shared target truly causes later friendship formation.
The treated group includes subreddit pairs that co-attacked a common target without prior friendship, while the control group includes similar pairs that never co-attacked and were also not friends before that time.

    We control for three confounders: topical similarity (cosine similarity between embeddings), activity level (log of total outgoing links), and aggressiveness (ratio of negative to total outgoing links). A logistic regression estimates the probability of being treated given these confounders (the propensity score). Using nearest-neighbor matching, each treated pair is matched with control pairs of similar propensity.
  
    Finally, we compare the change in friendship—the increase in mutual positive links before vs. after conflict—between treated and matched controls. A larger increase among treated pairs suggests that shared conflict causally promotes alliance.

- **Sensitivity Analysis:**
    This section tests whether the causal effect from Section 2 holds up under possible unobserved confounders. Using Rosenbaum sensitivity analysis, we simulate how a hidden factor might bias treatment assignment.
Starting from the matched treated and control pairs, we first compute the observed effect (difference in friendship change). Then, we introduce a bias parameter Γ (Gamma) that represents how much a hidden factor could increase a pair’s odds of co-attacking. For each Γ ≥ 1, we recompute the p-value bounds of the treatment effect.
If the effect stays significant up to high Γ (e.g., Γ ≈ 2), it’s robust — it would take a strong hidden bias to remove it. If it fails at low Γ (≈ 1.1), it’s sensitive, meaning unobserved factors might explain the result.


**Proposed Timeline and Team Organization**

<img width="661" height="620" alt="image" src="https://github.com/user-attachments/assets/92c3cc77-0b32-401a-b04f-7a54676143ef" />

## Quickstart

```bash
# clone project
git clone https://github.com/epfl-ada/ada-2025-project-adanalysts.git
cd ada-2025-project-adanalysts

# create conda environment (if needed)
conda create -n ada25 python=3.11 -y
conda activate ada25


# install requirements
pip install -r pip_requirements.txt
```



### How to use the library
Note that throughout the project, we make use of 3 datasets. The hyperlinks title dataset, hyperlinks body dataset, and subreddit embeddings dataset. Due to their big size, we do not upload them directly to the data folder in Github. Instead, we load the datasets in our results notebook directly from the online path to these datasets. Therefore, one can run the code in the notebook to have the datasets loaded without any other step required.



## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```

