# Yelp Review Classification

A controlled comparison of four neural architectures on the same text classification
task, built for the Neural Networks course (750039M) at Universidad del Valle.
Individual work, July 2025.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![KerasHub](https://img.shields.io/badge/KerasHub-RoBERTa-D00000.svg?logo=keras&logoColor=white)](https://keras.io/keras_hub/)
[![Report](https://img.shields.io/badge/report-37%20pages-success.svg)](report/informe-final.pdf)

The task is predicting a Yelp review's star rating from its text, framed two ways: as
binary sentiment and as 5-class rating. Four architectures were trained on both, under
three different treatments of class imbalance.

**The full write-up, with every confusion matrix and learning curve, is in
[`report/informe-final.pdf`](report/informe-final.pdf) (Spanish, 37 pages).**

## The finding worth reading

Class weighting did not work.

Weighting the loss inversely to class frequency is the standard first answer to
imbalanced data. Across every architecture tested here it **failed to remove the bias
toward majority classes, and in some configurations made it worse**. The confusion
matrices show it directly.

What did shift the behaviour was undersampling, and it came at a price: fairness across
classes improved, overall accuracy dropped. That tradeoff only disappeared at scale.
Once the training set grew from ten thousand reviews to millions, the usual penalty for
balancing stopped applying, and the models got both fairer and more accurate.

The negative result is reported rather than buried, because the experiment was set up to
answer the question either way.

## Architectures

Each one was chosen to fix a specific limitation of the one before it.

| | Architecture | What it adds | What broke |
|---|---|---|---|
| 1 | **MLP** | baseline | No notion of word order. Fit the training set almost perfectly and failed to generalize |
| 2 | **SimpleRNN** | temporal memory | Beat the MLP, but unstable learning curves and severe bias on long sequences |
| 3 | **LSTM** | long-range dependencies | More stable and more accurate than the RNN; still overfit on small data |
| 4 | **RoBERTa** | pretrained language knowledge | Best overall. Still confused adjacent star ratings |

RNN and LSTM are built from scratch in Keras: `Embedding` → `SpatialDropout1D` →
recurrent layers → `BatchNormalization` / `LayerNormalization` → dense head. The
transformer is `keras_hub.models.RobertaTextClassifier` from the `roberta_base_en`
preset, fine-tuned at a learning rate of 3e-5.

## Results

Star ratings collapse to two classes cleanly and to five classes badly. That gap is the
main quantitative result.

| Task | Best configuration | Result |
|---|---|---|
| Binary | RoBERTa, balanced, scaled dataset | **macro F1 89.8%**, precision and recall above 90% on both classes |
| 5-class | scaled dataset | **61.5%** accuracy |
| 5-class | base configurations | 43% to 52.7% accuracy |

> **Read the dataset column before quoting a number.** The 89.8% is on the balanced
> multi-million-review set, not on the 10,000-review sample the project starts from. On
> the small set the same task lands far lower. The numbers are not comparable across
> scales and the report keeps them separate.

## Method

**Preprocessing.** Keras `TextVectorization` for standardization and index tokenization,
vocabulary capped at 20,000, spaCy for lemmatization. Sequences truncate at **348
tokens, chosen as the 95th percentile of the length distribution** so that 95% of
reviews survive intact rather than at a round number picked by hand. Split 70/10/20.

**Imbalance.** The 5-class distribution is heavily skewed: 748 one-star reviews against
3,525 four-star. Three treatments were run as a declarative experiment matrix over
`{mode, balance, use_weights}`:

1. **Raw** — original distribution, used as the reference
2. **Balanced** — majority-class undersampling to equal counts
3. **Weighted** — original data, `sklearn` balanced class weights applied in the loss

**Hyperparameters.** Keras Tuner `BayesianOptimization`, 30 trials, maximizing
validation accuracy.

**Scaling.** The full Yelp dump does not fit in a Colab session. It is streamed as
NDJSON in 100,000-line chunks, each written out as a Parquet shard with Polars, then
read back as a single frame. That is what makes the multi-million-row runs possible.

## Layout

```
notebooks/
  01-rnn-lstm-baseline.ipynb    RNN, LSTM and BiLSTM by hand on the 10k sample
  02-rnn-lstm-tuned.ipynb       the same models scaled up, with Bayesian tuning
  03-transformer-roberta.ipynb  RoBERTa fine-tuning through KerasHub
report/
  informe-final.pdf             the full write-up
```

The notebooks were written for Colab: they install their own dependencies, pull the
dataset through `kagglehub`, and read and write under a mounted Drive. Paths need
adjusting to run anywhere else.

## Scope

Coursework, not a library. The MLP notebook from the first deliverable was not kept;
that stage survives only in the report, which covers it in full.
