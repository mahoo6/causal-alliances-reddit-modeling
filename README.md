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

To study whether sharing a common enemy causally increases the likelihood of friendship between subreddits, we follow three main steps.

### Descriptive Co-Attack Analysis
We first aggregate all subreddit interactions at a **monthly resolution**. For each unordered pair of subreddits (A,B) and each month, we compute a **Friendship Score** that combines sentiment balance and interaction volume. Rather than relying on ad-hoc cutoffs, we learn **enemy, neutral, and friend thresholds** directly from the data by applying K-Means clustering to the distribution of monthly Friendship Scores.

We define **co-attacks** as instances where two subreddits A and B both post negative links toward the same target subreddit C within the same month. To focus on meaningful conflicts, we further require that both A and B are classified as **enemies of C** in that month. For each such trio (A,B,C), we identify the **start and end months** of the conflict period and record how long the co-attack remains active.

### Causal Inference
To move from descriptive patterns to causal estimation, we use a **propensity-score matching framework**.

The **treated group** consists of subreddit pairs (A,B) that ever engaged in a strong co-attack, with the **event time** defined as the first month of such a co-attack. The **control group** consists of pairs that never co-attacked any common target at any time.

For each control pair, we assign a **pseudo-conflict window** by:
- sampling a conflict start month from the empirical distribution of treated conflict start times, and
- sampling a conflict duration from the empirical distribution of treated conflict durations,
and setting the pseudo conflict end accordingly. Control pairs that already formed a friendship before their pseudo event are excluded.

The outcome is defined as whether a **new strict friendship** between A and B appears within the window  
[conflict_start, conflict_end + 1],  
which captures the period in which a causal effect of co-attacking is most plausible.

We control for pre-conflict confounders that may jointly affect co-attacking and friendship formation: topical similarity (cosine similarity of embeddings), activity level (log total outgoing links), aggressiveness (ratio of negative to total outgoing links), and prior hostility between the pair. These confounders are used in a logistic regression model to estimate **propensity scores**, after which treated pairs are matched to control pairs using nearest-neighbor matching.

### Sensitivity Analysis
Finally, we assess robustness to **unobserved confounding** using **Rosenbaum sensitivity analysis** on the matched pairs. This analysis quantifies how strongly an unobserved factor would need to influence treatment assignment within matched pairs to alter the conclusions of the matched comparison.

The full methodology is applied both to the original sentiment labels and to an alternative specification in which potentially implicit negative interactions are relabeled as negative, allowing us to assess how sensitive the causal framework is to the operationalization of negative interactions.


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

