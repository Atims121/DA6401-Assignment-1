# DA6401-Assignment-1
Implementing a FeedForward Neural Network with Backpropagation from scratch alongwith few optimizers and loss functions on **Fashion-MNIST data**.

## Optimizers implemented: <br>
**SGD** - Stochastic Gradient Descent <br>
**Momentum** - Momentum SGD <br>
**NesterovAG** - Nesterov Accelerated Gradient <br>
**RMSProp** - Root Mean Square Propagation<br>
**Adam** - Adaptive Moment Estimation<br>
**Nadam** - Nesterov Adaptive Moment Estimation<br>

## How to run? <br>
``` bash
python train.py -wp <wandb_project_name> -we <wandb_entity_name> 
```
To run the file with custom values,
```bash
python train.py [-h] -wp [WANDB_PROJECT] -we [WANDB_ENTITY] [-e EPOCHS]
                [-b BATCH_SIZE] [-l LOSS] [-o OPTIMIZER] [-lr LEARNING_RATE]
                [-m MOMENTUM] [-beta BETA] [-beta1 BETA1] [-beta2 BETA2]
                [-eps EPSILON] [-w_d WEIGHT_DECAY] [-w_i WEIGHT_INIT]
                [-nhl NUM_LAYERS] [-sz HIDDEN_SIZE] [-a ACTIVATION]
```
