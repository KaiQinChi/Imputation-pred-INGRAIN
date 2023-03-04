# Multiple-level Point Embedding for Solving Human Trajectory Imputation with Prediction (INGRAIN)
## Introduction
The work is to explore whether the learning process of imputation and prediction could benefit from each other to
achieve better outcomes. And the question will be answered by studying the coexistence patterns between
missing points and observed ones in incomplete trajectories. More specifically, the proposed model develops
an imputation component based on the self-attention mechanism to capture the coexistence patterns between
observations and missing points among encoder-decoder layers. Meanwhile, a recurrent unit is integrated
to extract the sequential embeddings from newly imputed sequences for predicting the following location.
Furthermore, a new implementation called Imputation Cycle is introduced to enable gradual imputation
with prediction enhancement at multiple levels, which helps to accelerate the speed of convergence. 


<img decoding="async" src="https://raw.githubusercontent.com/KaiQinChi/Imputation-pred-INGRAIN/main/output/fig1.png" width="70%">


<img decoding="async" src="https://raw.githubusercontent.com/KaiQinChi/Imputation-pred-INGRAIN/main/output/fig2.png" width="70%">

## Requirements
- Numpy
- Pytorch 1.6.0
- Python 3.7

## Contributions
* We propose one new framework that integrates both autoregressive and non-autoregressive components to impute missing points in human trajectories and predict future movements.
* A model is established for trajectory imputation with prediction based on gradual amelioration at multiple levels via setting the granularity of imputing points. It is applicable to different human mobility datasets.
* Comprehensive evaluations are conducted to show the efficiency and effectiveness of our model on three real-world human trajectory datasets. The paper provides insights into how the method satisfies accurate estimations on missing points and next positions and how trade-offs could be handled in this type of cooperative learning.

## Reference
Kyle K. Qin, Yongli Ren, Wei Shao, Brennan Lake, Filippo Privitera, and Flora D. Salim. 2023. Multiple-level Point Embedding for Solving Human Trajectory Imputation with Prediction. ACM Trans. Spatial Algorithms Syst. Just Accepted (February 2023). https://doi.org/10.1145/3582427

