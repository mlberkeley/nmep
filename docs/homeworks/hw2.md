---
title: Homework 2 - Word Embeddings
parent: Homeworks
nav_order: 2
layout: home
---

# Homework 2: Word Embeddings

In this homework, you will be experimenting with word embeddings. Clone the skeleton code at [TBD] to begin.

## Part 1: GloVe Embeddings Exercise

We'll be using GloVe embeddings for this exercise, which you can download [here](https://drive.google.com/file/d/1U8Ytt3hqUfU29wjYo9qnFZJ2DTWMAPHJ/view?usp=drive_link). Feel free to change it later on to get better accuracy.

GloVe stands for global vectors. It is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

For example, the embeddings for king, man, queen, and woman might relate linearly as king - man + woman = queen.

For this exercise, all you need to know is that GloVe embeddings are a way of representing words as vectors. The vectors are learned from a large corpus of text. The vectors are dense, meaning that each word is represented by a vector of a fixed size. The vectors are learned in such a way that words that appear in similar contexts have similar vectors. This means that the vectors can be used to capture semantic meaning.

For the specific GloVe embeddings we have provided, the embedded vectors have dimension `50`. This means that words are captured by vectors that lie in a `50d` vector space.

### GloVe Embeddings Interface

Install the python package [gensim](https://pypi.org/project/gensim/) which will act as an interface for the GloVe embeddings.

The following code is provided to you in `word_embeddings.py` to initialize the GloVe model:
```python
from gensim.models import KeyedVectors

EMBEDDING_MODEL_PATH = ...
EMBEDDING_MODEL_NAME = "glove.6B.50d.txt"

class WordEmbeddings:
    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format(EMBEDDING_MODEL_PATH + EMBEDDING_MODEL_NAME)
```

KeyedVectors is a data structure that allows querying of vectors keyed by lookup tokens (e.g., strings). 

You can check if a word is a valid key in KeyedVectors by using `in`. 

You can access the value associated with a key just like a typical python dictionary `vector = self.model["word"]` which returns an ndarray representing the embedded vector.

Check the [documentation](https://radimrehurek.com/gensim/models/keyedvectors.html) to find all methods that KeyedVectors supports.


### Task

Implement the `embed` function in the `WordEmbeddings` class in `word_embeddings.py`. `embed` takes in a list of documents and returns the average word embedding for each document using a pre-trained GloVe model.

```python
...
def embed(self, documents: list[str]) -> np.ndarray
```

For the examples below, we should expect these shapes:
```python
emb = embed(["I like goats"])
print(emb.shape) # expecting (1, 50)
emb = embed(["I like goats", "I hate pizza"])
print(emb.shape) # expecting (2, 50)
```

Hint: Consider that a "document" is just a string of space separated words. How can we obtain the word embeddings for each word in a document and then average them out?


### Concept Checks

1. What are the pros of this embedding approach (to embed an entire document)
2. What are the cons of this embedding approach?



## Part 2: MLP Sentiment Classifier

Download the dataset [tweets.csv](https://drive.google.com/file/d/1bnq3apLWG5jcs2Kv3XpXZO_K8wAz5-ju/view?usp=drive_link) which contains data on various tweets to airlines, the sentiment of the tweets, and various other specifics about the tweets.

You are to build an MLP (Multi Layer Perceptron) to predict airline tweet sentiments.

Questions to ask yourself: What does an MLP take as input? What data do I have? What sort of outputs am I seeking?

### Data Handling

We will work through implementing the `def prepare_data()` function in `main.py`.

Begin by using Pandas to read the dataset into a dataframe. Look at some examples of the data that you have.

Which features may be useful for us in predicting airline sentiment?

For this exercise, we will only use `text`. 

First, drop all columns except `text` and `airline_sentiment`. 

Next, let's convert airline sentiment to a numeric value to train on: `negative = 0, neutral = 1, positive = 2`. Why should we do this?

Next, we ideally want to take a piece of text (a string) as input and pass this into our MLP and have it predict a sentiment `{0,1,2}`. But simply passing a string into our MLP will not work, why?

Once you realize the answer to this question, transform your data, the `text` column to convert strings into a form that your MLP can actually take as input. *Hint: You have already written a function to convert strings into their mathematical representations.*

### Training and Test Splits

Why must we separate our data into training and test splits? Would it be erroneous to simply train on our entire dataset?

Use sklearn's train_test_split function to perform a train/test split. Set the random_state to 42 so the results are reproducible.

```python
from sklearn.model_selection import train_test_split
```

### Torch Dataset and DataLoader

Now that our data is cleaned and formatted in quantities that our MLP can take as input and use to predict, let's wrap them with pytorch's dataset and dataloader classes to train on.

Specifically, use [TensorDataset](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset) to convert both your training and validation splits into pytorch datasets.

*Hint: How can you extract the embedding and sentiment columns from the train and test dataframes to pass into `TensorDataset`? You ultimately want something like:*

```python
dataset = TensorDataset(x_tensor, labels_tensor)
```

Finally, use a `DataLoader` to create train and validation dataloaders. Check the [documentation](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for dataloaders. You want to specify at least `batch_size` and `shuffle`.



### The MLP

Create an MLP! Ideally you make it easy to change the number of layers and the size of each layer, but you don't have to

Hint: Create a class that inherits from torch.nn.Module. What methods do you need to implement? (giveaway: `__init__` and `forward`)

Hint: use torch.nn.Linear to create a linear layer. Why?

Hint: you need an activation function after each linear layer. Why?

Hint: you need a softmax activation function on the output layer. Why?

Hint: use torch.nn.Sequential to create a sequential model that's easy to execute in forward



### Train One Epoch

Implement the function:

```python
def train_one_epoch(model, loss_fn, train_loader, optimizer)
```


### Validation

Implement a validation function:

```python
def validate(model, loss_fn, val_loader)
```


### Full Training Loop

Finally, implement the full training loop:

```python
def train(model, train_loader, val_loader, epochs)
```


Then, define and train your model!

### Logging

Plot your train and validation loss on the same plot using `plt.plot`. Make sure to label your lines.

Then, plot your train and validation accuracy on another figure. Make sure to label your lines.

Finally, save your plots for submission to Gradescope.
### Evaluating

Write a short script to evaluate the model on a piece of text you write.

Remember to set the model to eval mode, and to convert the text to an embedding.

Result should be a string either "negative", "neutral", or "positive".

Implement this:

```python
def evaluate(trained_model, sample_text):
```


### Optional Tasks:
1. Write a short script to evaluate the model on the validation set, and  print out the tweets that were misclassified

2. A confusion matrix shows  what classes the model is likely to make mistakes.
Write a few lines of code to plot a confusion matrix for the validation set using
`from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay`


## Results and Submission
The dataset is not perfect, but your model should be able to get over **70%** on the validation set. We do not have a hold out test set for this exercise.

### Credit
For full credit, you must have a model that achieves at least **70%** accuracy on the validation set. 

All functions and classes mentioned above should be implemented as instructed, including:

1. WordEmbeddings: `def embed(self, documents: list[str]):`
2. `def prepare_data():`
3. the `MLP` model
4. `def train_one_epoch(model, loss_fn, train_loader, optimizer):`
5.  `def validate(model, val_loader, loss_fn):`
6. `def train(model, train_loader, val_loader, epochs):`
7. `def evaluate(trained_model, sample_text):`

Additionally, you should download these plots and submit them for full credit:

1. Train and Validation Loss Plot
2. Train and Validation Accuracy Plot


### Experimentation
To achieve a higher accuracy, feel free to change the way you featurize. Some ideas for you to try:

Try larger models (wider layers, more layers)

Play around with activation functions, dropout, batch norm

Remove stop words (the, a, in, is, etc) (hint: take a look at nltk)

Use a different way of featurizing. For example, instead of taking the average you could concatenate the min, mean, and max embeddings. Remember that embeddings for all documents must be the same dimensions, so you cannot just concatenate the embeddings of each word. (unless you're training an LSTM :eyes:)

Use TF-IDF to take a weighted average of the embeddings. (haven't tried it, but wouldn't it be interesting)

Use a different pre-trained model. I haven't tried it but you could try: w2v, fasttext, BERT, Univseral Sentence Encoder, CLIP, OpenAI embedding models.

Deal with the class imbalance - you can try oversampling or undersampling the data. You can also try using a weighted loss function.

