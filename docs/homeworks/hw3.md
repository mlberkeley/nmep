---
title: Homework 3
parent: Homeworks
nav_order: 2
layout: home
---

#  Homework 3: Model Zhu

In this homework, you will be implementing a few popular computer vision models, and training them on both CIFAR-10 and on a custom dataset we created. You will be using PyTorch for this homework.

You will be using a medium-sized [repository](https://github.com/mlberkeley/fa25-nmep-hw3/) which mimics that of a standard codebase which you might find for modern projects. Don't be intimidated! We will walk you through all of the parts of it, and hopefully after this homework you will be more confident working with codebases like this. We believe this is a realistic representation of what you may do in the future, and we hope you will find it useful.

We would recommend you first set up the repository ASAP on honeydew and try running it out of the box to see how it trains, and only afterwards focus on understanding all parts of the code. For your benefit, the codebase works out of the box, and you should be able to train a model on CIFAR-10 with no changes. Throughout the assignment, you will need to make some changes to models/alexnet.py and models/resnet.py, for which you will find the provided implementations of other models in models/ to be helpful.

All of the assignment details are provided in this spec, you will need to fill in some answers on Gradescope and make code changes. 

Best of luck, and we hope you enjoy it!

## Setup 

To get started, you will need to fork and clone [the repository](https://github.com/mlberkeley/fa25-nmep-hw3/) (clone on honeydew!) and install the dependencies, preferably in a conda or uv environment. Standard instructions are provided below.

```bash
ssh honeydew
git clone git@github.com:mlberkeley/fa25-nmep-hw3.git
cd fa25-nmep-hw3
conda env create -f env.yml
conda activate vision-zoo
CUDA_VISIBLE_DEVICES=0 python main.py --cfg=configs/lenet_base.yaml
```

This should begin a download and training of a LeNet(ish) model on CIFAR-10. You should see all of the output files in ```output/lenet```, but you can specify exactly where in the configs (more on that in a second).

## Overview of Project Structure

The project is organized roughly as follows:

```bash
configs/            # contains all of the configs for the project
  resnet/           # you can organize configs by model
  ...
data/               # contains all of the data related code
  build.py          # contains the data loader builder
  datasets.py       # where dataset loaders are defined
  ...
models/             # contains all of the model related code
  build.py          # contains the model builder
  resnet.py         # ResNet definition
  ...
utils/              # misc. utils for the project
  ... 
config.py           # contains the config parser; define defaults here!!
main.py             # main training loop
optimizer.py        # optimizer definition
```

You'll notice that the main subfolders all have a ```build.py``` file. This is a common pattern in codebases, and is used to build the model and data loaders using the configs. Generally all the config parameters are handled in the build files, which then call the appropriate class to build the model or data loader. They're kind of a liaison between the configs and the actual code, so that the code can be written free of any config dependencies.

## Configs

Speaking of configs, most projects you'll come across will use a config system to specify hyperparameters and other settings. This is a very common practice, and is used in many of the projects you'll see in the future. We've provided a simple config parser for you to use, which you can find in ```config.py```. You can see how it's used in ```main.py```, where we parse the config and then pass it to the model and data loader builders. Notably, configs are defined and given defaults in ```config.py```, and then can be overridden using yaml files in ```configs/```. This particular system is nested, so for example your configs will look something like this. 

```yaml
# configs/resnet18_cifar.yaml
...
MODEL:
  NAME: resnet18
  NUM_CLASSES: 10
  ...
DATA:
  BATCH_SIZE: 128
  DATASET: cifar10
    ...
```

You'll need to chase them down in the code to understand the exact impact of the settings, but these are useful because they allow you to easily change hyperparameters and settings without having to change the code.
Plus, for experimentation, it's nice to be able to keep track of all of the settings you used for a particular run and have everything you need to reproduce them whenever you want.

However for when you're hacking or just testing things quickly, it's useful to not have to create a new config for everything. Hence we've also provided the option of using a few command line arguments to override the configs. You can see how this is done in ```main.py```, where we parse the command line arguments and then override the configs with them. Throw these together in a shell script to keep track of everything, and you're good to go!

## Tips

Don't try to understand everything at once, it's daunting! Treat this like you would a large class project or a software engineering project, and work in small chunks (it's why we've cleanly factored the code into modules). Ask questions, don't be afraid to test things out in jupyter notebooks or use the pdb debugger (```breakpoint()``` or ```import pdb; pdb.set_trace()```). These are all good skills to learn to become a great machine learning engineer.

Let's get started!

<br>

## Part 1
Complete the `HW 3 Model Zhu [Part 1]` assignment on Gradescope. This should give you some intuition behind this codebase.


## Part 2
This part will walk you through implementing AlexNet and ResNet, two other influential CNN architectures. 

As you complete sections, you will see `WRITTEN ANSWER HERE` flags pop up to indicate that your answer should be filled out on the corresponding part of the `HW 3 Model Zhu [Part 2]` assignment on Gradescope.

## 2.1: AlexNet

**Implement AlexNet.** Feel free to use the provided LeNet as a template. For convenience, here are the parameters for AlexNet:

```
Input NxNx3 # For CIFAR 10, you can set img_size to 70
Conv 11x11, 64 filters, stride 4, padding 2
MaxPool 3x3, stride 2
Conv 5x5, 192 filters, padding 2
MaxPool 3x3, stride 2
Conv 3x3, 384 filters, padding 1
Conv 3x3, 256 filters, padding 1
Conv 3x3, 256 filters, padding 1
MaxPool 3x3, stride 2
nn.AdaptiveAvgPool2d((6, 6)) # https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
flatten into a vector of length x # what is x?
Dropout 0.5
Linear with 4096 output units
Dropout 0.5
Linear with 4096 output units
Linear with num_classes output units
```

> ReLU activation after every Conv and Linear layer. DO **NOT** Forget to add activations after every layer. Do not apply activation after the last layer.

### 2.1.1 
How many parameters does AlexNet have? How does it compare to LeNet? With the same batch size, how much memory do LeNet and AlexNet take up while training? 
> (hint: use `gpuststat`)

`WRITTEN ANSWER HERE`

### 2.1.2
**Train AlexNet on CIFAR10.** What accuracy do you get?

Report training and validation accuracy on AlexNet and LeNet. Report hyperparameters for both models (learning rate, batch size, optimizer, etc.). We get ~77% validation with AlexNet.

> You can just copy the config file, don't need to write it all out again.
> Also no need to tune the models much, you'll do it in the next part.

`WRITTEN ANSWER HERE`



## 2.2: Weights and Biases

Parts 2 and 3 are independent. Feel free to attempt them in any order you want.

> Background on W&B. W&B is a tool for tracking experiments. You can set up experiments and track metrics, hyperparameters, and even images. It's really neat and we highly recommend it. You can learn more about it [here](https://wandb.ai/site).
> 
> For this HW you have to use W&B. The next couple parts should be fairly easy if you setup logging for configs (hyperparameters) and for loss/accuracy. For a quick tutorial on how to use it, check out [this quickstart](https://docs.wandb.ai/quickstart). We will also cover it at HW party at some point this week if you need help.

### 2.2.0 
Setup plotting for training and validation accuracy and loss curves. Plot a point every epoch.

`PUSH YOUR CODE TO YOUR OWN GITHUB :)`

### 2.2.1 
Plot the training and validation accuracy and loss curves for AlexNet and LeNet. Attach the plot and any observations you have.

`WRITTEN ANSWER HERE`

### 2.2.2
For just AlexNet, vary the learning rate by factors of 3ish or 10 (ie if it's 3e-4 also try 1e-4, 1e-3, 3e-3, etc) and plot all the loss plots on the same graph. What do you observe? What is the best learning rate? Try at least 4 different learning rates.

`WRITTEN ANSWER HERE`

### 2.2.3
Do the same with batch size, keeping learning rate and everything else fixed. Ideally the batch size should be a power of 2, but try some odd batch sizes as well. What do you observe? Record training times and loss/accuracy plots for each batch size (should be easy with W&B). Try at least 4 different batch sizes.

`WRITTEN ANSWER HERE`

### 2.2.4 
As a followup to the previous question, we're going to explore the effect of batch size on _throughput_, which is the number of images/sec that our model can process. You can find this by taking the batch size and dividing by the time per epoch. Plot the throughput for batch sizes of powers of 2, i.e. 1, 2, 4, ..., until you reach CUDA OOM. What is the largest batch size you can support? What trends do you observe, and why might this be the case?
You only need to observe the training for ~ 5 epochs to average out the noise in training times; don't train to completion for this question! We're only asking about the time taken. If you're curious for a more in-depth explanation, feel free to read [this intro](https://horace.io/brrr_intro.html). 

`WRITTEN ANSWER HERE`

### 2.2.5 
Try different data augmentations. Take a look [here](https://pytorch.org/vision/stable/transforms.html) for torchvision augmentations. Try at least 2 new augmentation schemes. Record loss/accuracy curves and best accuracies on validation/train set.

`WRITTEN ANSWER HERE`

### 2.2.6 (optional) 
Play around with more hyperparameters. I recommend playing around with the optimizer (Adam, SGD, RMSProp, etc), learning rate scheduler (constant, StepLR, ReduceLROnPlateau, etc), weight decay, dropout, activation functions (ReLU, Leaky ReLU, GELU, Swish, etc), etc.

`WRITTEN ANSWER HERE`



## 2.3: ResNet

### 2.3.1
**Implement and train ResNet18**

In the `models/*` directory, we provided some skelly/guiding comments to implement ResNet. Implement it and train it on CIFAR10. Report training and validation curves, hyperparameters, best validation accuracy, and training time as compared to AlexNet. 

`WRITTEN ANSWER HERE`

### 2.3.2 (optional) 

Visualize a couple of the predictions on the validation set (20 or so). Be sure to include the ground truth label and the predicted label. You can use `wandb.log()` to log images or also just save them to disc any way you think is easy.

`WRITTEN ANSWER HERE`


## 2.4 - Kaggle submission

To make this more fun, we have scraped an entire new dataset for you! 🎉

We called it MediumImageNet. It contains 1.5M training images, and 190k images for validation and test each. There are 200 classes distributed approximately evenly. The images are available in 224x224 and 96x96 in hdf5 files. The test set labels are not provided :). 

The dataset is downloaded onto honeydew at `/data/medium-imagenet`. Feel free to play around with the files and learn more about the dataset.

For the kaggle competition, you need to train on the 1.5M training images and submit predictions on the 190k test images. You may validate on the validation set but you may not use is as a training set to get better accuracy (aka don't backprop on it). The test set labels are not provided. You can submit up to 10 times a day (hint: that's a lot).

Your Kaggle scores should approximately match your validation scores. If they do not, something is wrong.

(Soon) when you run the training script, it will output a file called `submission.csv`. This is the file you need to submit to Kaggle. You're required to submit at least once. 

### Kaggle writeup

We don't expect anything fancy here. Just a brief summary of what you did, what worked, what didn't, and what you learned. If you want to include any plots, feel free to do so. That's brownie points. Feel free to write it below or attach it in a separate file.

**REQUIREMENT**: Everyone in your group must be able to explain what you did! Even if one person carries (I know, it happens) everyone must still be able to explain what's going on!

Now go play with the models and have some competitive fun!
