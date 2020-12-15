- [Introduction to NLP](#introduction-to-nlp)
  - [main approaches in NLP](#main-approaches-in-nlp)
    - [deep learning vs. traditional NLP](#deep-learning-vs-traditional-nlp)
  - [brief overview of the next weeks](#brief-overview-of-the-next-weeks)
    - [week1](#week1)
    - [week2](#week2)
    - [week3](#week3)
    - [week4](#week4)
    - [week5](#week5)
  - [linguistic knowledge in NLP](#linguistic-knowledge-in-nlp)
    - [libraries and tools](#libraries-and-tools)
- [How to: from plain texts to their classification](#how-to-from-plain-texts-to-their-classification)
  - [text preprocessing](#text-preprocessing)
  - [transfor tokens into features](#transfor-tokens-into-features)
    - [bag of words (BOW), loose word order](#bag-of-words-bow-loose-word-order)
- [linear model for sentiment analysis](#linear-model-for-sentiment-analysis)
  - [sentiment classification](#sentiment-classification)
- [hashing trick in spam fitering](#hashing-trick-in-spam-fitering)
  - [mapping n-grams to feature indices](#mapping-n-grams-to-feature-indices)
  - [spam filtering is a huge task](#spam-filtering-is-a-huge-task)
  - [vowpal Wabbit](#vowpal-wabbit)

# Introduction to NLP

## main approaches in NLP

- rule-based mathods
  - regular expressions
  - context-free grammars(CFG)
- probabilistic modeling and machine learning
  - likelihood maximization
  - linear classifiers
- deep learning
  - recurrent neural networks
  - convolutional neural networks

### deep learning vs. traditional NLP

why do we need to study traditional NLP?
- perform good enough in many tasks. e.g. sequence labeling
- allow us not ot be blinded with the hype. e.g. word2vec/distributional semantics
- can help to further improve DL models. e.g. word alignment priors in machine translation

## brief overview of the next weeks

### week1

text classification tasks:
- predict some tags or categories
- predict sentiment for a review
- filter spam emails

### week2

- how to predict word sequences
  - language models are needed in chat-bots, speech recognition, machine translation, summarization...
- how to predict tags for the word sequences
  - part-of-speech tags
  - named entities
  - semantic slots

### week3

how to represent a meaning of a word, a sentence, or a text?
- word embeddings
- sentence embeddings
- topic models

### week4

sequence to sequence tasks:
- machine translation
- summarization, simplification
- conversational chat-bot

### week5

dialogue agents
- goal-oriented
- conversational

## linguistic knowledge in NLP

NLP pyramid:
pragmatics(highest abstraciton) -> semantics(meaning) -> syntax(relationship btw. words) -> morphology(words)

### libraries and tools
- NLTK
- Stanford parser
- sapCy
- Gensim
- MALLET

# How to: from plain texts to their classification

## text preprocessing

- tokenization

```python
nltk.tokenize.WordPunktTokenizer()
nltk.tokenzie.TreebankWordTokenizer()
nltk.tokenize.WhitespaceTokenizer()
```

- token normalization: stemming & lemmatization
  - Porter stemmer
  - WordNet lemmatizer

- further normalization
  - normalizing capital letters
  - acronyms

- summary
  - we can think of text as a sequence of tokens
  - tokenization is a process of extracting those tokens
  - we can normalize tokens using stemming or lemmatizaation
  - we can also normalize casing and acronyms
  - transform extracted tokens into features for our model

## transfor tokens into features

### bag of words (BOW), loose word order

- preserve some ordering, too many features
  - 1-grams for tokens
  - 2-grams for token pairs
  - ...

- remove some n-grams

  remove some n-grams from features based on occurrence frequency in documents of our corpus

  - high frequency n-grams: stop-words
  - low frequeny n-grams: typos, rare n-grams
  - medium frequency n-grams: good n-grams

  there're a lot of medium frequency n-grams

- TF-IDF

  - term freqency (TF)

  $$
  \begin{array}{|c|c|}\hline \text { weighting scheme } & {\text { TF weight }} \\ \hline \text { binary } & {0,1} \\ \hline \text { raw count } & {f_{t, d}} \\ \hline \text { term frequency } & {f_{t, d} / \sum_{t^{\prime} \in d} f_{t^{\prime}, d}} \\ \hline \text { log normalization } & {1+\log \left(f_{t, d}\right)} \\ \hline\end{array}
  $$

  - inverse document frequency (IDF)

  $$
  \begin{array}{l}{\cdot N=|D|-\text { total number of documents in corpus }} \\ {\cdot|\{d \in D : t \in d\}| \text { - number of documents where the }} \\ {\text { term } t \text { appears }} \\ {\cdot \operatorname{idf}(t, D)=\log \frac{N}{|\{d \in D : t \in d\}|}}\end{array}
  $$

  - TF-IDF

  $$
  \begin{array}{l}{\cdot \text { tfidf }(t, d, D)=\operatorname{tf}(t, d) \cdot \operatorname{idf}(t, D)} \\ {\cdot \text { A high weight in TF-IDF is reached by a high }} \\ {\text { term frequency (in the given document) and a low }} \\ {\text { document frequency of the term in the whole }} \\ {\text { collection of documents }}\end{array}
  $$

- better BOW

  - replace counters with TF-IDF
  - normalize the result row-wise (divided by l2-norm)

  ```python
  from sklearn.feature_extraction.text import TfidfVectorizer
  texts = []
  tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))
  features = tfidf.fit_transform(texts)
  ```

# linear model for sentiment analysis

## sentiment classification

- IMDB movie reviews dataset

- model: logistic regression
  - can handle sparse data
  - fast to train
  - weights can be interpreted

- how to make it better
  - play ground with tokenization
  - try to normalize tokens
  - try different models
  - throw BOW away and use Deep Learning

# hashing trick in spam fitering

## mapping n-grams to feature indices

  ```python
  sklearn.feature_extraction.text.HashingVectorizer
  # implemented in vowpal wabbit library
  ```

## spam filtering is a huge task

$$
\begin{array}{l}{\cdot \phi(x)=\operatorname{hash}(x) \% 2^{b}} \\ {\cdot \text { For } b=22 \text { we have } 4 \text { million features }} \\ {\cdot \text { That is a huge improvement over } 40 \text { million features }} \\ {\cdot \text { It turns out it doesn't hurt the quality of the model }}\end{array}
$$

$$
\begin{array}{l}{\operatorname{hash}(s)=s[0]+s[1] p^{1}+\cdots+s[n] p^{n}} \\ {p-\text { fixed prime number }} \\ {s[i]-\text { character code }}\end{array}
$$

- personalized tokens trick

$$
\begin{array}{l}{\cdot \phi_{o}(\text {token})=\text { hash }(\text {token}) \% 2^{b}} \\ {\cdot \phi_{u}(\text {token})=\operatorname{hash}(u+\sqrt[n]{\text { n }}+\text { token }) \% 2^{b}} \\ {\text { . We obtain } 16 \text { trillion pairs (user, word) but still } 2^{b} \text { features }}\end{array}
$$

- why personalized features work
  - personalized features capture "local" user-specific preference
  - learn better "global" preference having personalized features which learn "local" user preference

## vowpal Wabbit

- a popular machine learning library for training linear models
- uses feature hashing internally
- has lots of features
- really fast and scales well

