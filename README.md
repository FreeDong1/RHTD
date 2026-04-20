# RHTD


The code in this project primarily encompasses the detection model, performance evaluation, feature ablation, and attack scripts within the RHTD framework.

## Attack Script
This folder primarily contains scripts for 11 types of web attacks along with their corresponding payloads.

## Data
This folder contains PCAP packets of 11 types of web attacks in the HTTP/3 environment.

In addition, the project also uses the H23Q and VisQUIC datasets.

H23Q：[A hands-on gaze on HTTP/3 security through the lens of HTTP/2 and a public dataset](https://www.sciencedirect.com/science/article/pii/S0167404822004436)

VisQUIC：[Exploring QUIC Dynamics: A Large-Scale Dataset for Encrypted Traffic Analysis](https://ieeexplore.ieee.org/abstract/document/11104435)


## Data Process
This folder contains the code for data augmentation, simulating changes in HTTP/3 traffic under network fluctuation conditions.

## Detected Mode
This folder contains the code corresponding to the dual-modal prototype network detection model.

## GPT
This folder contains the large model invocation code for both few-shot prompting and fine-tuning approaches, which are used to conduct feature incremental experiments to verify feature effectiveness and cross-model generalization capability.

## Evaluate
This folder contains the code for feature ablation, validation of data augmentation model effectiveness, and model robustness verification.

