**Is the Enemy of my Enemy my Friend?**

## Abstract

This project investigates whether online communities form alliances through shared hostility. Using Reddit as a case study, we model the temporal evolution of inter-community relations based on sentiment-labeled interactions.
To support our analysis, we developed an interactive and dynamic map to visualize all our results, from intermediate to final outcomes, and illustrate how inter-community relationships evolve over time.
Since our analysis relies on the accuracy of these sentiment labels, we first assess and refine the dataset’s labeling process. We aim to detect hidden hostilities (interactions labeled as non-negative but that implicitly convey hostility) in order to expand and strengthen our dataset.
Finally, we test the hypothesis “the enemy of my enemy is my friend”, examining whether communities that co-attack the same targets are more likely to develop new positive relations, while accounting for the effect of other confounding factors such as pre-existing similarities between subreddits.


## Proposed additional datasets

In addition to our dataset, we use subreddit embeddings containing one 300-dimensional vector per subreddit, with closeness capturing user-base similarity.

## Research Questions

- Does the current labeling of inter-community interactions hide undetected hostilities, and how can it be improved?
  - Can temporal patterns in sentiment shifts between communities reveal potential labeling errors? 
  - How can linguistic features help identify mislabeled interactions?
  - Can community embeddings confirm or refine the detection of hidden hostilities?


- Does sharing a common enemy lead two subreddits to form new causal friendships?
  - How does the level of positive interaction between subreddit pairs change before and after a shared conflict?
  - Does co-attacking a common target causally increase the likelihood of future friendship, or is it mere correlation?
  - How robust is this effect to unobserved confounders such as shared users or ideology?





## Methodology
	
## 1 - Hidden hostilities
Our goal is to identify a specific phenomenon we call hidden hostility: interactions that are labeled as non-negative but whose language and subsequent dynamics strongly suggest an underlying hostile intent.
We use a two-step approach that separates what is said from what happens next.

1. Language-based Detection

We first analyze the content of posts to estimate whether an interaction sounds hostile, even when it is not explicitly labeled as such.
Using linguistic and emotional features combined with the relationship between the two communities, we assign each interaction a hostility likelihood score.

2. Temporal Corroboration

Language alone can be noisy. To avoid over-interpreting ambiguous cases, we then check whether the interaction is followed by a rapid negative response between the same communities.
Fast sentiment reversals are treated as external evidence that the initial interaction may have carried latent hostility.

3. Decision Logic

An interaction is flagged as hidden hostility if: 
- the language is unambiguously hostile,
- or the language is suspicious and the negative reaction occurs unusually fast.
Additional safeguards are applied to avoid noise and over-representation of very large or inactive communities.


## 2 – Is the enemy of my enemy my friend?

To study whether sharing a common enemy causally increases the likelihood of subsequent friendship, we follow three main steps.

### Descriptive Co-Attack Analysis
We first aggregate all subreddit interactions at a monthly resolution. For each unordered pair of subreddits (A,B) and each month, we compute a **Friendship Score** that reflects both how positive or negative their interactions are and how often they interact. We then use **K-Means clustering** on the distribution of monthly Friendship Scores to learn data-driven thresholds that classify relationships as **enemy**, **neutral**, or **friend**.

We define co-attacks as situations where two subreddits A and B both post negative links toward the same target subreddit C within the same month. To focus on meaningful conflict episodes, we further require that A and B are both classified as enemies of C at that time, qualifying the pair as 'strong co-attackers' of C. For each such trio (A,B,C), we identify when the conflict starts and ends and record how long the strong co-attack remains active.

### Causal Inference
Descriptive patterns alone cannot establish causality. To estimate whether strong co-attacking (co-attacking C and being ennemies with C) actually causes subsequent friendship formation, we use a propensity-score matching framework.

The **treated group** consists of subreddit pairs (A,B) that engaged in at least one strong co-attack, with the event time defined as the first month of such an attack. The **control group** consists of pairs that never co-attacked any common target.

For each control pair, we construct a pseudo-conflict window by:
- sampling a conflict start month from the empirical distribution of treated conflict start times, and
- sampling a conflict duration from the empirical distribution of treated conflict durations,
then setting the pseudo conflict end accordingly. Control pairs that formed a friendship before their pseudo event are excluded to ensure comparable pre-treatment histories.

The outcome is defined as whether a friendship start between A and B appears within the window [conflict_start, conflict_end + 1 month], which corresponds to the period in which a causal effect of co-attacking is most plausible.

We control for pre-conflict factors that may influence both co-attacking and friendship formation: topical similarity (cosine similarity of embeddings), activity level (log total outgoing links), aggressiveness (ratio of negative to total outgoing links), and prior hostility between the pair. These confounders are used in a logistic regression model to estimate propensity scores, after which treated pairs are matched to similar control pairs using nearest-neighbor matching.

The causal effect is summarized using the Average Treatment Effect on the Treated (ATT). For each matched pair, we compute the within-pair difference between the treated and control outcomes (friendship formed vs. not formed). The ATT is then obtained by averaging these differences across all matched pairs, yielding an estimate of how strong co-attacking affects the probability of subsequent friendship formation among pairs that did co-attack.

### Sensitivity Analysis
Finally, we evaluate how sensitive the matched comparison is to unobserved confounding using **Rosenbaum sensitivity analysis**. This analysis measures how strongly an unobserved factor would need to affect treatment assignment within matched pairs to change the conclusions of the causal comparison.

Starting from the matched treated and control pairs, we first compute the observed effect (difference in friendship outcome). Then, we introduce a bias parameter Γ (Gamma) that represents how much a hidden factor could increase a pair’s odds of co-attacking. For each Γ ≥ 1, we recompute the p-value bounds of the treatment effect.
If the effect stays significant up to high Γ (e.g., Γ ≈ 2), it’s robust — it would take a strong hidden bias to remove it. If it fails at low Γ (≈ 1.1), it’s sensitive, meaning unobserved factors might explain the result.


The entire causal study is applied first using the dataset with the original sentiment labels, then using the modified dataset containing potential implicit negatives as actual '-1' links. This allows us to assess how sensitive the causal framework is to how negative interactions are defined and detected.


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

