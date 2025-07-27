---
layout: post
title: A primer on Vector embeddings 
categories: AI
tags: [NLP, machine-learning, vector-embeddings]
math: true
---

  

# A primer on Vector embeddings.

  

Let's rollback to the times when transformers didn't exist. finding the contextual meaning of a word in a sentence

was a much harder task. Getting contextual similarity between two words or a word and a sentence for that matter

falls in domain of Information and retrieval which has proposed multiple approaches for the task. (A separate blog on those models *Reminder* :P)

  

---

  

## What do we mean by word similarity?

  

A lot about word similarity comes from the 'meaning' of that word. Imagine a system/machine/model which could grasp

the meaning of the word, differentiate between homonyms like we do. Such a system would be much more capable of answering

our questions that the current version of chatGPT. Let's back up a little from this futuristic thought and dive into how

current systems 'understand' the 'meaning' of a word.

  

Languages are complicated. Despite having grammer, languages are extremely complex for a machine to understand.

> "Bow and Arrow" and "Bow down" have a same word 'bow' but mean differently (homonyms) .

  

> "happy" and "delighted" are different words but same meaning (synonyms)

  

> 'Coffee beans' are two different words with no similarity in meaning but still they are **related**.

  

**Language** in itself is complicated. Human intelligence shapes and forms the way languages are interpreted. But how do we **represent** words for a system to understand or maybe generate new words for us?

  

One possible way is to represent words as _vectors_ or more specifically 1-hot vectors.

  

For eg,

  

$$Word_1 (W_1)= \begin{bmatrix}x_{1}\\

x_{2}\\

x_{3}

\end{bmatrix} \rightarrow \begin{bmatrix}0\\

1\\

0\end{bmatrix}$$

  

and

  

$$Word_2 (W_2) = \begin{bmatrix}1\\

0\\

0\end{bmatrix}$$

  

Since we made representation as column vectors, each word vector can be differentiated easily.

But does the meaning, semantics or context of the word is represented here?

  

Let's say if we use dot product as a similarity measure $$W _1^TW_2 = W_2^TW_1 = 0$$

Both the words are equally dissimilar or have 0 similarity between them. But is that the real case?

How about the words 'tea' and 'coffee' or maybe 'cup' and 'mug' .

  

Mathematically speaking this makes sense though as the word vector for $W_1\text{ }and\text{ }W_2$

are orthogonal to each other i.e. each vector has magnitude 1 in Y and X axes respectively.

  

Both the words are equally dissimilar or have 0 similarity between them. But is that the real case?

How about the words 'tea' and 'coffee' or maybe 'cup' and 'mug' ? The problem is the length of these vectors is as long as the vocabulary size which changes over time. Words are added and phased out from languages all the time. In 2023, 'rizz' was added to the Oxford dictionary $($ also the word of the year ;$)$ $)$ . So each time a new word is added, it will require changes in all thousands and lakhs of word vectors.

Secondly, these 1-hot vectors do not capture the meaning anyway

  

This forces us to think of an alternative representation of words.

  
  

How about annotating each word based on each meaning, plurality and the grammatical information? I am not going to dive much into this but WordNet <Miller, 1995 > annotates for synonyms, hyponyms, and other semantic relations; UniMorph <Batsuren et al., 2022> annotates for morphology (subword structure) information across many languages. These methods are also costly when they are to be updated with recently added words.

  
  

---

  

### Vector semantics (Distributional Semantics)

  

This is where things start to get interesting. So when we say the word 'meaning' of a word, what do we actually mean?

  

A meta thought would be that our vocal chords produce a sound in a particular fashion for aa word which is perceived as 'something' in our minds. But how do we perceive what that word means?

  

Well, the discussion is overly complicated and beyond the scope here. So I am just going to start with an analogy. As a human baby, how do you think we understood language?

Dekang Lin [^1] argues that 'The meaning of an unknown word can often be inferred from its context'. Consider the following sentences [^1] :

  

1. A bottle of **tezgiiino** is on the table.

2. Everyone likes **tezgiiino**.

3. **Tezgiiino** makes you drunk.

4. We make **tezgiiino** out of corn.

  

After noticing the way a particular word (tezgiino) is used in different contexts, we get some idea about it's meaning. The contexts in which the word tezgiiino is used suggest that tezgiiino may be a kind of alcoholic beverage made from corn mash.

  

You can think of this as the words adjacent or near to 'tezgiino' provides clarity on it's meaning. This idea is well articulated by Firth $($ 1957 $)$ as

  

> You shall know a word by the company it keeps.

  

So the distribution of words around 'tezgiino' will be similar to the distribution of words around let's say 'wine' or 'beer'. This is the **distributional hypothesis**.

  

Great, so how do we formulate or generalise this hypothesis?

  

**Vector semantics** is used to determine the meaning/context of a word considering the neighbouring semantic space (distribution).Semantic space can be thought as domain or field under which a words occurs. So in general, **if we put information about the context or neighbourhood of the word into it's representation, we would be able to capture it's meaning.**

  

"Bow and arrow" and "Bow down" will have different semantic space for the word 'bow'

  

There are different designs to implement this concept of distributional semantics (putting context into word representations)

  

-----

  

### Sparse Vector Methods

  

Below are some common sparse vector methods.

  

1.**Co occurence matrix and term document matrix**

  

Easiest way to infuse some context into word is by constructing a co occurence matrix or term term matrix.

Let's say we have 1000 documents with 500 unique words (vocabulary 'V'). Then we can simply construct a 500 x 500 matrix with dimensionality |V|×|V| and each cell records the number of times the row (target) word and the column (context) word co-occur in some context in some training corpus. This will be very skewed with most of entries being 0.

  

Another way is to create a term document matrix, where the rows are words and dimensions or columns are documents. So a 500 x 1000 matrix can be constructed with each row containing the number of times a word V occurs in doc D.

  

2.**Tf-IDF scheme**

  

I remember making use of Tf-IDF for one of my NLP projects at my workspace. Having no idea about it's inner workings, I was kind of amazed with the results even though much better techniques have been around.

It was only during Master's when I learned about this algorithm in Information Retrieval.

So, Tf-IDF was introduced to this world by Karen Spark Jones in a paper titled [A statistical interpretation of term specificity and its application in retrieval](https://www.staff.city.ac.uk/~sbrp622/idfpapers/ksj_orig.pdf ) .

  

The idea behind Tf-IDF is fairly simple. Tf-IDF is expanded as **Term Frequency - Inverse Document Frequency** . Term Frequency is the raw frequency of each word in a document. So Term frequency or TF signifies how many times a term T occured in each document. This is a row vector with length equal to the number of documents in corpus.

IDF or inverse document frequency is calculated as below $$idf_t = log \frac{N}{df_t}$$ with the final weight (contextual weight in our case) being $$w_{t,d} = tf_{t,d} * idf_t$$

where N is the total number of documents and $df_t$ is the document frequency or the numnber of documents a term appeared in. I'll cover Tf-IDF separately in a post but the bottomline is TF tells us how **common** a word is within corpus and IDF signifies how **unique** a word is to the document. So if a word 'the' is highly common across the docs, the Tf component will be high but the idf will tend to a lower value.

  

Tf-IDF scheme was introduced initially for Information retrieval to fetch related documents for a query and was later used in word representation in NLP. Also Tf-IDF is used when dimensions are represented by documents.

  

So what's the issue here? Well, like I mentioned before, the sparse vector schemes tend to grow exponentially with growing vocabulary. Imagine a 10000 document corpus with around 50000 vocabulary after applying Zipf's law (yeah I like to throw big words here and there).

  

3. **PMI (Pointwise Mutual Information)**

  

For a (word, context word) pair to occur together in a sliding window of length L, the PMI is defined as

  

$$PMI(w,c) = \log[2]{(\frac{P(w,c)}{P(w)P(c)})}$$

  

where numerator tells us how often two words w,c are observed together.The denominator tells us how often we would expect the two words to co-occur assuming they each occurred independently; recall that the probability of two independent events both occurring is just the product of the probabilities of the two events. Thus, the ratio gives us an estimate of how much more the two words co-occur than we expect by chance. PMI is a useful tool whenever we need to find words that are strongly associated.

  

The PMI can have values ranging from negative to positive infinity. However, negative values do not exactly tell us if the two words are 'unrelated'. This is because two words having negative PMI does not necessarily be unrelated unless we have searched entire corpus (which is quite impossible). So we clip the negative values by calculating **PPMI(Positive Pointwise Mutual Information)**

  

$$PPMI(w,c) = max(PMI(w,c) , 0)$$

  

Even PPMI tends to be biased for rare words, as rarer the word, the more value of PPMI is.

This bias is removed by changing the computation of $P(c)$ slightly as :

  

$$P_\alpha(c) =\frac{count(c)^\alpha}{\sum_ccount(c)^\alpha}$$

  

$$PPMI_\alpha(w,c) = max(\log[2]{(\frac{P(w,c)}{P(w)P_\alpha(c)})} , 0)$$

  

where $\alpha$ is a constant paramter by which the Probability is raised to for unfrequent words. Laplace smoothing is another way to discount the non zero values by adding a small constant $$k \in (0,1.3)$$

  

The words are are represented as 'vector' in these semantic spaces which are known as **embeddings**. Embedding are nothing but fixed length, dense and short vectors shorter than vocabulary size mostly. Most commonly use embedding scheme

is called **word2Vec**. Embeddings mathematically signify mapping of one space to another not ensuring the retention of meaning of each space. I am going to implement this from scratch. 

  

---
### Word2Vec



  
  
  
  
  

  
  
  

--------

  

[^1]: Dekang Lin “[Automatic Retrieval and Clustering of Similar Words](https://aclanthology.org/C98-2122.pdf).”