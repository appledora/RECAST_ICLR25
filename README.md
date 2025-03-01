# RECAST: Reparameterized, Compact weight Adaptation for Sequential Tasks
> Accepted as a conference paper at ICLR 2025

## Abstract
Incremental learning aims to adapt to new sets of categories over time with minimal computational overhead. Prior work often addresses this task by training efficient task-specific adaptors that modify frozen layer weights or features to capture relevant information without affecting predictions on any previously learned categories. While these adaptors are generally more efficient than finetuning the entire network, they still can require tens to hundreds of thousands task-specific trainable parameters even for relatively small networks, making it challenging to operate on resource-constrained environments with high communication costs like edge devices or mobile phones. Thus, we propose Reparameterized, Compact weight Adaptation for Sequential Tasks (RECAST), a novel method that dramatically reduces the number of task-specific trainable parameters to fewer than 50 – several orders of magnitude less than competing methods like LoRA. RECAST accomplishes this efficiency by learning to decompose layer weights into a soft parameter-sharing framework consisting of a set of shared weight templates and very few module-specific scaling factors or coefficients. This soft parameter-sharing framework allows for effective task-wise reparameterization by tuning only these coefficients while keeping templates frozen. A key innovation of RECAST is the novel weight reconstruction pipeline called Neural Mimicry, which eliminates the need for pretraining from scratch. This allows for high-fidelity emulation of existing pretrained weights within our framework and provides quick adaptability to any model scale and architecture. Extensive experiments across six diverse datasets demonstrate RECAST outperforms the state-of-the-art by up to 3% across various scales, architectures, and parameter spaces. Moreover, we show that RECAST’s architecture-agnostic nature allows for seamless integration with existing methods, further boosting performance.


In this repository, we provide the code related to the paper "RECAST: Reparameterized, Compact weight Adaptation for Sequential Tasks" by the authors. The code is organized as follows:

- 'models': contains the implementation of the different RECAST models used in the paper. This demonstrates the flexibility of RECAST to be applied to different architectures. These include: `RECAST-Att.`, `RECAST-MLP`, `RECAST-Att. + MLP`, and `RECAST-resnet`.
- 'adapters': contains the implementation of the various adapter model we have combined with RECAST to enhance their performance. These include: `RECAST + Piggyback`, `RECAST + CLR`, `RECAST + MeLo`, and `RECAST + AdaptFormer`.
- 'notebooks': contains sample classification notebooks used to train and evaluate the models.  They  contain the dataloaders we have used in our experiments - however, they are not exhaustive and may need to be modified to work with your datasets. We also include the end-to-end code for reconstructing and evaluating the weights for both ViT and ResNet. We also added the the code for reconstructing with VAE, which we have used for some of our experiments. You can easily generate any model's weights using the provided code without any GPU requirements.


> We didn't provide the direct script for training the adapter models, as they follow the same training procedure as the classification task. The main difference is the model architecture, and the weight to be loaded. 

> Ann difference in the training loop exists for `RECAST + Piggyback` for which we have used the training script provided in the `Piggyback` repository. We provided the training script we have used in the [additional scripts](additional_scripts) folder.

> The hyperparameters are the same as the classification task (except in `Piggyback`). We used `AdamW` optimizer with a learning rate of `2e-3` and a batch size of `128`. We used a stepwise learning rate scheduler with a decay factor of `0.1` at each 33 epochs. We trained the models for 100 epochs.

> The results reported in the paper are averaged over 3 runs with the following seeds `[42, 1998, 3142592]`. We used the same seeds for all the experiments to ensure a fair comparison. To generate our models, we have used seedvale `42`.
