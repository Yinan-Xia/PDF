# Predictive-Dynamic-Fusion
This is the official implementation for [Predictive Dynamic Fusion](https://arxiv.org/pdf/2406.04802) (ICML 2024) by Bing Cao, Yinan Xia, Yi Ding, Changqing Zhang, and Qinghua Hu.

<p align="center">
<img src="https://github.com/YinanXia2023/Predictive-Dynamic-Fusion/blob/main/frame.png" width="850" height="500">
</p>

## Abstract
Multimodal fusion is crucial in joint decision-making systems for rendering holistic judgments. Since multimodal data changes in open environments, dynamic fusion has emerged and achieved remarkable progress in numerous applications. However, most existing dynamic multimodal fusion methods lack theoretical guarantees and easily fall into suboptimal problems, yielding unreliability and instability. To address this issue, we propose a predictive dynamic fusion (PDF) framework for multimodal learning. We proceed to reveal the multimodal fusion from a generalization perspective and theoretically derive the predictable Collaborative Belief (Co-Belief) with Mono- and Holo-Confidence, which provably reduces the upper bound of generalization error. Accordingly, we further propose a relative regularization strategy to calibrate the predicted Co-Belief for potential uncertainty. Extensive experiments on multiple benchmarks confirm our superiority. 

## Environment Installation
```
numpy==1.21.6
Pillow==9.4.0
Pillow==10.3.0
pytorch_pretrained_bert==0.6.2
scikit_learn==1.0.2
torch==1.11.0+cu113
torchvision==0.12.0+cu113
tqdm==4.65.0
```
## Dataset preparation

  Step 1: Download [food101](https://www.kaggle.com/datasets/gianmarco96/upmcfood101) and [MVSA_Single](https://www.kaggle.com/datasets/vincemarcs/mvsasingle) and put them in the folder *datasets*.

  Step 2: Prepare the train/dev/test splits jsonl files. We follow the [MMBT](https://github.com/facebookresearch/mmbt) settings and provide them in corresponding folders.

  Step 3 (optional): If you want use Glove model for Bow model, you can download [glove.840B.300d.txt](https://www.kaggle.com/datasets/takuok/glove840b300dtxt) and put it in the folder *datasets/glove_embeds*. For bert model, you can download [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) and put in the root folder *bert-base-uncased/*.


## Train
```
bash ./shells/batch_train_latefusion_pdf.sh
```

## Test
```
bash ./shells/batch_test_latefusion_pdf.sh
```

## Citation
```
@inproceedings{cao2024predictive,
  title={Predictive Dynamic Fusion},
  author={Cao, Bing and Xia, Yinan and Ding, Yi and Zhang, Changqing and Hu, Qinghua},
  booktitle={International conference on machine learning},
  year={2024},
  organization={PMLR}
}
```
