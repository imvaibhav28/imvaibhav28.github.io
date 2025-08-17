---
layout: post
title: A Primer on Vector Embeddings
categories: AI
tags: [NLP, machine-learning, vector-embeddings]
math: true
---

# A Primer on Vector Embeddings

Let's roll back to the times before transformers existed. Finding the contextual meaning of a word in a sentence was a much harder task. Getting contextual similarity between two words—or even between a word and a sentence—falls under the domain of **Information Retrieval**, which has proposed multiple approaches for the task. (A separate blog on those models *Reminder* :P)


# Vector Representations

We routinely use ordered lists of numbers, or vectors, to describe objects of
any shape or form. Examples abound. Any geographic location on earth can
be recognized as a vector consisting of its latitude and longitude. A desk can
be described as a vector that represents its dimensions, area, color, and other
quantifiable properties. A photograph as a list of pixel values that together
paint a picture. A sound wave as a sequence of frequencies.


Vector representations of objects have long been an integral part of the
machine learning literature. Indeed, a classifier, a regression model, or a rank-
ing function learns patterns from, and acts on, vector representations of data.
In the past, this vector representation of an object was nothing more than
a collection of its features. Every feature described some facet of the object
(for example, the color intensity of a pixel in a photograph) as a continuous
or discrete value. The idea was that, while individual features describe only a
small part of the object, together they provide sufficiently powerful statistics
about the object and its properties for the machine learnt model to act on. [^3]

---

## What do we mean by word similarity?

A lot about word similarity comes from the "meaning" of that word. Imagine a system which could grasp meaning the way humans do, and differentiate between homonyms. Such a system would be much more capable of answering our questions than the current version of ChatGPT. But before we go futuristic, let’s see how current systems try to "understand" the meaning of a word.

Languages are complicated. Despite having grammar, they are extremely complex for a machine to parse.  

- **Homonyms:** "Bow and Arrow" vs "Bow down" — same word, different meanings.  
- **Synonyms:** "happy" and "delighted" — different words, same meaning.  
- **Related terms:** "Coffee beans" — words not similar but still related.

So how do we represent words for a system to understand?

---

### One-hot vectors

One way is to represent words as **one-hot vectors**.

For example:

$$
Word_1 (W_1)= \begin{bmatrix}0 \\ 1 \\ 0\end{bmatrix}, \quad
Word_2 (W_2)= \begin{bmatrix}1 \\ 0 \\ 0\end{bmatrix}
$$

This makes each word unique and differentiable. But does it capture meaning or context? No.

If we compute similarity via dot product:

$$
W_1^T W_2 = 0
$$

So the words are equally dissimilar. But intuitively, "tea" and "coffee" are closer in meaning than "tea" and "car". One-hot vectors fail here.  

Worse, their length equals the vocabulary size, which grows and changes over time (e.g., Oxford Dictionary added *rizz* in 2023 as "word of the year"). Adding a new word requires updating all vectors.  

Thus, one-hot vectors are sparse, large, and semantically meaningless.

---

### Alternatives: Annotated lexical resources

Another approach is to annotate words with linguistic information.  
- **WordNet** (Miller, 1995) annotates for synonyms, hyponyms, and semantic relations.  
- **UniMorph** (Batsuren et al., 2022) annotates morphological info across languages.  

These are useful but costly to maintain and don’t scale easily when new words emerge.

---

## Vector Semantics (Distributional Semantics)

So what do we mean by the "meaning" of a word?  

Dekang Lin [^1] argues: *"The meaning of an unknown word can often be inferred from its context."*

Example sentences:  

1. A bottle of **tezgiiino** is on the table.  
2. Everyone likes **tezgiiino**.  
3. **Tezgiiino** makes you drunk.  
4. We make **tezgiiino** out of corn.  

From context, **tezgiiino** is likely an alcoholic beverage made from corn.  

This idea—*"You shall know a word by the company it keeps"* (Firth, 1957)—is the **distributional hypothesis**.

---

### Sparse Vector Methods

1. **Co-occurrence / Term-document matrix**  
   Construct a |V|×|V| matrix where each cell counts co-occurrence of (word, context). Sparse, mostly zero.

2. **Tf–IDF**  
   Balances frequency with uniqueness.  
   $$idf_t = \log \frac{N}{df_t}$$  
   $$w_{t,d} = tf_{t,d} \times idf_t$$  
   Still sparse and grows with vocabulary.

3. **PMI / PPMI**  
   $$PMI(w,c) = \log_2 \frac{P(w,c)}{P(w)P(c)}$$  
   Positive PMI (PPMI) is often used. Rare words cause bias, mitigated with smoothing.

---

## Dense Embeddings

Sparse vectors are replaced with **dense embeddings**: fixed-length, low-dimensional continuous vectors.  

Advantages:  
- Compact and efficient.  
- Capture similarity better (e.g., "car" and "automobile").  

The most famous method: **Word2Vec**.

---

### Word2Vec

Word2Vec introduced the idea of learning embeddings directly from data. It has two main architectures:  
- **CBOW (Continuous Bag of Words):** Predicts a word from its context.  
- **Skip-gram:** Predicts context words from a target word.  

Both learn embeddings such that words appearing in similar contexts have similar vectors. This was a breakthrough in NLP and inspired further models like GloVe [^5] , FastText, and later transformer-based embeddings.

---

### Additional Reads

[^1]: Dekang Lin. “[Automatic Retrieval and Clustering of Similar Words](https://aclanthology.org/C98-2122.pdf).”  
[^2]: Jurafsky & Martin. “[Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/6.pdf).”

[^3]: Sebastian Bruch "[Foundations of Vector Retrieval](https://arxiv.org/abs/2401.09350)"

[^4]: Mikolov et al. “[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546).”

[^5]: Pennington et al. “[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf).”

[^6]: Bojanowski et al. “[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606).”
[^4]: [Word2Vec Explained](https://arxiv.org/abs/1301.3781)