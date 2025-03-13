# Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs
<a href="https://arxiv.org/abs/2503.02597" target="_blank">
    <img alt="AKI arXiv" src="https://img.shields.io/badge/arXiv-AKI_Paper-003366?logo=arxiv&logoColor=003366" height="25"/>
</a>
<a href="https://huggingface.co/Sony/AKI-4B-phi-3.5-mini" target="_blank">
    <img alt="HF Model: AKI-4B" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model-AKI--4B_Model-003366?color=ffc107&logoColor=white" height="25"/>
</a>

This repo contains an official PyTorch implementation of [Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](https://arxiv.org/abs/2503.02597) by Wei-Yao Wang, Zhao Wang, Helen Suzuki, Yoshiyuki Kobayashi.

## Overview
Existing efforts to address vision-language misalignment have focused on developing specialized vision-language connectors or leveraging visual instruction tuning from diverse domains.
In this paper, we tackle this issue from a fundamental yet unexplored perspective by revisiting the core architecture of MLLMs.
Most MLLMs are typically built on decoder-only LLMs consisting of a causal attention mechanism, which **limits the ability of earlier modalities (e.g., images) to incorporate information from later modalities (e.g., text)**.
To address this problem, we propose <i class="fa-brands fa-canadian-maple-leaf" style="color: #ff0000;"></i>AKI, a novel MLLM that unlocks causal attention into modality-mutual attention (MMA) to enable image tokens to attend to text tokens.
This simple yet effective design allows AKI to achieve superior performance in 12 multimodal understanding benchmarks (+7.2\% on average) without introducing additional parameters and increasing training time.
Our MMA design is intended to be generic, allowing for application across various modalities, and scalable to accommodate diverse multimodal scenarios.

![Framework](./framework.png)

## Usage
### Prerequisites
Our environment is Python 3.12 with PyTorch >= 2.0.1. For more details, please check `create_env.sh`.
1. Clone the repo
    ```
    git clone https://github.com/sony/aki.git && cd aki
    ```
2. Install the corresponding packages
    ```
    bash codes/open_flamingo/scripts/create_env.sh
    ```

### Pre-Training
Need to run `cd codes/open_flamingo` first.
1. Prepare datasets in the `webdataset` format. In this paper, we adopt the pre-training datasets from BLIP-3, including [BLIP3-Kale](https://huggingface.co/datasets/Salesforce/blip3-kale) and [BLIP3-OCR-200m](https://huggingface.co/datasets/Salesforce/blip3-ocr-200m).
2. Start pre-training
    ```
    bash scripts/run_train.sh
    ```

### Instruction Finetuning
Need to run `cd codes/open_flamingo` first.
1. Prepare SFT datasets with the original formats
2. Start instruction finetuning
    ```
    bash scrips/run_sft.sh
    ```

### Evaluations
1. CV-Bench
    
    The benchmark dataset is fetched from the [official release](https://huggingface.co/datasets/nyu-visionx/CV-Bench).
    ```
    python3.12 eval_cv_bench/eval.py {model_path}
    ```

2. Other VLM Benchmarks

    Under construction to create a PR to VLMEvalKit.

### Local Demonstration
Need to run `cd codes/open_flamingo` first.

Start the local demo
```
python3.12 local_demo.py
```

## Results
### Main Comparisons with the Same Configurations (Table 1)
|                          | MME<sup>P</sup> | MME<sup>C</sup> | MMB  | SEED<sup>I</sup> | LLaVA<sup>W</sup> | MMMU  | MathV<sup>mini</sup> | POPE  | MM-Vet | RealWorldQA | CV-Bench<sup>2D</sup> | CV-Bench<sup>3D</sup> |
|--------------------------|------------|------------|------|-------------|---------------|------|----------------|------|-------|------------|-----------------|-----------------|
| (I&T)<sub>PT</sub> + (I&T)<sub>SFT</sub>  | 1226.3     | 258.2     | 64.9 | 64.1        | 47.0          | 31.1 | 24.2           | 79.8 | 24.3  | 50.6       | 45.2            | 54.3            |
| CCA [Xing et al., 2024]  | 1212.7     | 243.6     | _67.4_ | _65.3_      | _54.0_        | _34.6_ | _25.6_         | _81.9_ | _29.0_  | **52.7**  | _56.0_          | 62.8            |
| (w/o T&I)<sub>PT</sub>   | 1046.3     | 226.4     | 31.7 | 45.1        | 38.1          | 27.2 | 23.8           | 65.0 | 17.2  | 40.1       | 53.2            | 54.8            |
| (w/o I&T)<sub>PT</sub>   | 1013.2     | 208.6     | 32.0 | 43.3        | 37.9          | 27.7 | 22.4           | 70.4 | 20.6  | 39.5       | 55.4            | 53.0            |
| (w/o T&I)<sub>SFT</sub>  | 1194.8     | _289.3_   | 58.5 | 61.1        | 40.2          | 28.0 | 21.9           | 79.0 | 22.8  | 47.8       | 41.4            | _63.0_          |
| (w/o I&T)<sub>SFT</sub>  | 1166.2     | 264.3     | 58.4 | 60.8        | 36.9          | 26.7 | 23.1           | 76.8 | 20.4  | 46.9       | 43.3            | 61.2            |
| DOT (Ours)              | _1267.8_   | 251.4     | 43.8 | 54.7        | 47.5          | 30.7 | _25.6_         | **82.7** | 25.0  | 50.5       | 52.2            | 58.1            |
| MMA (Ours)              | **1363.7** | **315.4** | **71.8** | **67.1**  | **59.6**      | **37.3** | **26.4** | **82.7** | **30.2**  | _52.3_ | **57.8** | **64.1** |
| **Improvements**        | 10.9%      | 29.5%      | 4.3%  | 2.8%        | 10.4%         | 7.8%  | 3.1%          | 1%   | 4.1%  | -          | 3.2%            | 2.1%            |

### AKI-4B (Table 2)
|                          | MME<sup>P</sup> | MME<sup>C</sup> | MMB  | SEED<sup>I</sup> | LLaVA<sup>W</sup> | MMMU  | MathV<sup>mini</sup> | POPE  | MM-Vet | RealWorldQA | CV-Bench<sup>2D</sup> | CV-Bench<sup>3D</sup> |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AKI-4B              | **1491.9** | **362.9** | **73.1** | **69.4**  | **74.6**      | **38.7** | **32.1** | **86.9** | **40.8**  | **58.9** | **62.1** | **71.8** |

## Contact
For any questions or issues, pleasefeel free to open an issue/PR or reach out: wei-yao.wang@sony.com.

## Citation
If you found this repository is relevant or useful to your research, please consider citing our paper:
```
@misc{wywang2025AKI,
      title={Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs}, 
      author={Wei-Yao Wang and Zhao Wang and Helen Suzuki and Yoshiyuki Kobayashi},
      year={2025},
      eprint={2503.02597},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.02597}, 
}
```

## Acknowledgements
The training code is based on the [open_flamingo repo](https://github.com/mlfoundations/open_flamingo), and the evaluation code is based on the [VLMEvalKit repo](https://github.com/open-compass/VLMEvalKit).
The SFT config file is built on top of the [HoneyBee repo](https://github.com/khanrc/honeybee/tree/main).
Thank you for making your codes public!
We also thank the [XGen-MM repo](https://github.com/salesforce/LAVIS/tree/xgen-mm) as we use their released data for pre-training and to take inspiration from their model implementation.