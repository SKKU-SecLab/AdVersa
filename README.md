# AdVersa: Adversarially-Robust and Practical Ad and Tracker Blocking in the Wild

## Abstract
While machine learning has significantly advanced ad and tracker detection, existing systems face critical challenges in practice. They are vulnerable to adversarial attacks (57-92% evasion rates), fail to generalize to unseen domains due to data contamination, and suffer performance degradation over time, requiring costly retraining. To address these challenges, we present AdVersa, a client-side framework for robust and practical ad and tracker blocking. AdVersa leverages novel, hard-to-perturb latent features from code and URL embeddings to deliver state-of-the-art performance. On a 2.06M-request dataset, our results show that AdVersa achieves a 98.23% F1 score, twice the robustness against adversarial attacks, and strong generalization to unseen domains (91.47% F1 score). For sustainable protection, we demonstrate that a low-cost pseudo-labeling strategy can maintain near-optimal accuracy, reducing maintenance overhead by over 99.8% compared to filter-list curation. Finally, we implement AdVersa as a lightweight, standalone client-side application that ensures user privacy by operating without external dependencies. 

The demo of *AdVersa* blocking ads and trackers in the wild is available in <a href="https://www.youtube.com/watch?v=3Ld1CnOjEDo">this link</a>. 

[<img src="https://i.ytimg.com/vi/3Ld1CnOjEDo/maxresdefault.jpg" width=720 height=450/>](https://www.youtube.com/watch?v=3Ld1CnOjEDo)


## Prerequisites
### 0. Configuration
This study has been run and tested in both *Ubuntu 20.04.6 LTS (Focal Fossa)* and *Windows 11 Education 24H2*. 
1. Python
    We strongly recommend configuring the environment using Python of versions *<=10* and *>=8*. The specific version used in the study is *3.10.18*.

2. Java
    
    This repository uses the <a href="https://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html#requirements">H2O Python package</a> which requires Java of versions *<=17* and *>=8*. 

3. Git LFS

    This repository requires Git LFS (Large File System). Enabling LFS is described <a href="https://git-lfs.com/">here</a>.

### 1. Clone Repository
Click the "Download Repository" button in the upper right corner of this anonymous repository. 
Unzip the contents to the directory you want. 

### 2. Install Dataset


### 3. Configure Environment


