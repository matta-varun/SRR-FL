# Towards Accurate and Stronger Local Differential Privacy for Federated Learning with Staircase Randomized Response: SRR-FL

This repository contains the source code of the paper titled "Towards Accurate and Stronger Local Differential Privacy for Federated Learning with Staircase Randomized Response", published in CODASPY 2024. 
For more details, you can read the paper at https://yhongcs.github.io/pub/codaspy24.pdf

If you find that this work is related to your work, please cite our work as:
 
```
 @inproceedings{10.1145/3626232.3653279,
author = {Varun, Matta and Feng, Shuya and Wang, Han and Sural, Shamik and Hong, Yuan},
title = {Towards Accurate and Stronger Local Differential Privacy for Federated Learning with Staircase Randomized Response},
year = {2024},
isbn = {9798400704215},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3626232.3653279},
doi = {10.1145/3626232.3653279},
abstract = {Federated Learning (FL), a privacy-preserving training approach, has proven to be effective, yet its vulnerability to attacks that extract information from model weights is widely recognized. To address such privacy concerns, Local Differential Privacy (LDP) has been applied to FL: perturbing the weights trained for the local model by each client. However, besides high utility loss on the randomized model weights, we identify a new inference attack to the existing LDP method, that can reconstruct the original value from the noisy values with high confidence. To mitigate these issues, in this paper, we propose the Staircase Randomized Response (SRR)-FL framework, which assigns higher probabilities to weights closer to the true weight, reducing the distance between the true and perturbed data. This minimizes the noise for maintaining the same LDP guarantee, leading to better utility. Compared to existing LDP mechanisms (e.g., Generalized Randomized Response) on the FL, SRR-FL can further provide a more accurate privacy-preserving training model, and enhance the robustness against the inference attack while ensuring the same LDP guarantee. Furthermore, we also use the parameter shuffling method for privacy amplification. The efficacy of SRR-FL has been validated on widely used datasets MNIST, Medical-MNIST and CIFAR-10, demonstrating remarkable performance. Code is available at https://github.com/matta-varun/SRR-FL.},
booktitle = {Proceedings of the Fourteenth ACM Conference on Data and Application Security and Privacy},
pages = {307–318},
numpages = {12},
keywords = {client-level ldp, federated learning, local differential privacy},
location = {Porto, Portugal},
series = {CODASPY '24}
}
```

## The datasets
MNIST Dataset
Medical MNIST Dataset
CIFAR-10 Dataset
Download the Medical MNIST dataset from https://www.kaggle.com/datasets/andrewmvd/medical-mnist

## Requirements 
- numpy==1.24.4
- torch==1.12.1
- matplotlib==3.7.3

Note: The python version used for the experiments: Python 3.8.5

## Files
- DATASET_LDPFL.ipynb: The main program that demonstrates our FL on respective datasets. This file includes the main flow of LDPFL.
- DATASET_FL.py : Codifies the federated learning setup for respective datasets.
- LDP_Functions.py : Defines all the LDP algorithm related classes and methods.
- MNIST_Attack.ipynb & InferenceAttackTests.ipynb : Outline various tests carried out as part our attack investigations.

## Usage
1. Install the requirements. 
2. Check if path to datasets is properly configured.
3. Run the respective LDPFL.ipynb file, commenting and uncommenting sections/statements as per requirements.

