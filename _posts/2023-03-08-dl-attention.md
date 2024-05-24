---
layout: distill
title: 8. Transformer Pt. 1 – Cross-Attention
date: 2023-03-08
description: ⏮️ <b><i>I recommend reading the RNN post first for the encoder-decoder architecture.</i></b>
#categories: deep-learning
#categories: deep-learning
tags: dl-fundamentals neural-net-archetype attention transformer generative-models
giscus_comments: true
related_posts: true
bibliography: references.bib

authors:
  - name: Tobias Stenzel
    url: "https://www.tobiasstenzel.com"
    affiliations:
      name: N/A

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Attention
    subsections:
    - name: Attention
    - name: Key, Value and Query
    - name: Multi-Head Attention
    - name: Complexity Comparison
    - name: Appendix — Naming Convention of Key, Value and Query


---

### Cross-Attention

Let us introduce this concept with an example. Figure
[7](#fig:attention) shows an encoder-decoder network with attention similar to the previous
encoder-decoder RNN (Figure
[6](https://www.tobiasstenzel.com/blog/2023/dl-rnn/#fig:encoder-decoder-rnn)). The core idea of cross-attention is
defining the hidden state of the decoder-RNN as a function of every
hidden state from the encoder-RNN for every time period without
recursion. Here, the result of the attention function is the context vector.
We will use this vector for every output element. In the specific
network in Figure [7](#fig:attention), the context vector is a function of both the
decoder states $$s$$ and the encoder states $$h$$. Further, it is
additionally concatenated with $$s$$ to predict the output layer.

<figure id="fig:attention">
<center><img src="/assets/img/dl-series/2h-attention.png" style="width:50%"></center>
</figure>
<b>Figure 7. Encoder-Decoder with cross-attention.</b> In contrast to the encoder-decoder RNN, the output layer is a function of the concatenation of the hidden states and a time-dependent context vector (black boxes). The main idea is that the context vector $$c_t$$ is a function of all hidden states $$\{h_t\}_{t=1}^{T}$$ instead of only the last one (and the previous state $$s_{t-1}$$). This function is called *attention* (red). The black box represents vector concatenation. $$c_1$$ is initialized with $$h_4$$, $$s_1$$ with arbitrary values, and $$o_1$$ is discarded.
<br>

The following paragraph describes the architecture in more detail. The **encoder** processes the input sequence and produces a set of vectors, each representing the input at different time steps. These vectors are often referred to as encoded hidden states. During **decoding**, for each output token:

- The model computes a set of **attention weights**. These weights are calculated based on the current state of the decoder and each of the encoder's hidden states. This step is where the model decides which parts of the input sequence are relevant for the current output token.
- A weighted sum of the encoder's hidden states is computed using these attention weights. This sum is called the **context vector**, which effectively captures relevant information from the input sequence needed for generating the current output token.
- The context vector, along with the current state of the decoder, is used to **generate the output token**.

The attention function that returns the context vector for output $$y_t$$
wit attention scores for each input is given by the following
expression:

$$\begin{align}c_t = \sum_{i=1}^n \alpha_{t,i} h_{i}\end{align}$$

$$\alpha_{t,i}$$ is a softmax function of another function *score* that
measures how well output $$y_t$$ (represented by $$s_t$$) and input $$x_i$$ (represented by $$h_t$$) are aligned:

$$\begin{align}\alpha_{t,i} = \text{align}(y_t,x_i) = \frac{\exp \text{score}(s_{t-1}, h_i)}{\sum_{j=1}^n\exp\text{score} (s_{t-1}, h_j)}.\end{align}$$

Here, we subtract one from the $$s$$ index compared to the $$y$$ index because $$s_1$$ is used to intialize the decoder hidden states and is not used to generate $$y_1$$.

There are many different scoring functions <d-cite key="weng_attention_2018"></d-cite>. A
common choice is the scaled dot-product score
$$(s_t, h_i)=\frac{s_t^T h_i}{\sqrt{d}}$$, where $$d$$ is the hidden state
dimension of both encoder and decoder states. Here, the alignment score
for one sequence element is given by a relative score of the dot-product
between the respective encoder hidden state and the current decoder
hidden state. We scale down the dot product to prevent vanishing
gradients from a pass to a softmax layer. After training the model, we
can analyze how much each output element depends on, or *attends* to,
each input. We do this by assembling a table with outputs as columns and
the output-specific alignment scores for each input as rows.

Note that, the attention mechanism processes the input twice.
- The encoder states (representing the input sequence) are used as **"keys"** in the attention mechanism. These keys are part of the mechanism that allows the decoder to determine which parts of the input sequence are relevant at each step of the translation.
- The same encoder states are also used as **"values"**. In the attention calculation, after the relevance (attention weights) between the keys and the query is determined, these weights are applied to the values (the same encoder states) to create a weighted representation of the input sequence. This representation is context-sensitive, meaning it changes depending on which part of the input sequence the model is currently focusing on while creating a specific output.

In contrast, the output is only processed once.
- The current state of the decoder (representing the partially translated sequence up to the current point) is used as the **"query"**. The query interacts with the keys (encoder states) to generate attention weights. These weights reflect how much each part of the input sequence should contribute to the current step in the translation process.

Figure [7](#fig:attention), we have removed the bottleneck between encoder hidden layers and decoder hidden layers.
However, there are still two bottelnecks left. These are the relations between the different encoder hidden states and
the different decoder hidden states. For example, $$h_4$$ is still related to $$h1$$ through the intermediate states as is $$s3$$ to $$s1$$. Therefore, encoder-decoder architetures can include a
self-attention mechanism in addition to the previous cross-attention mechanism (Foreshadown the transformer, perhaps use this as last sentence). In this model, the computation of attention weights is inherently sequential due to its recurrent nature. Each decoder step depends on the previous steps, which can be a bottleneck for parallel processing.
Transformer: The self-attention mechanism in Transformers allows for much more efficient parallel processing. Since each position in the sequence can attend to all other positions independently, it facilitates faster training and better handling of long sequences.


Next, we will discuss the attention module in the transformer. This module is applied as self-attention on the inputs, on the outputs and between inputs and outputs.

### The Transformer's Attention Module

The Transformer uses separate vectors for keys, values, and queries (all derived from the input data), whereas in the previous model, the encoder states are used both as keys and values, and the decoder state is used as the query. Using separate representations for keys and queries in the Transformer allows for more nuanced and flexible modeling of relationships within the data. It enables the model to learn different aspects of the data for different purposes: the queries can learn to seek information relevant to the current context, while the keys can learn to highlight the information that should be made available. This separation enhances the model's capacity to capture complex patterns and dependencies in the data.

As we do not use recurrence of single sequence
elements anymore, let us denote the whole sequence of input embeddings
by $$X \in \mathbb{R}^{L \times D^{(x)}}$$. $$L$$ can either be the complete
input length $$T_x$$ or later only a fraction of it. $$D^{(x)}$$ is the
input embedding's length. Let us denote the sequence of output
embeddings by $$Y \in \mathbb{R}^{M \times D^{(y)}}$$. The transformer
uses an extension of the attention mechanism, the multi-head attention,
as its core building block. The first step is that, instead of using the
softmax of the scaled dot-product between encoder states $$h$$ and decoder
states $$s$$ directly as in the last section, it uses the scaled
dot-product with two different input encodings,
$$K=XW^k \in \mathbb{R}^{L \times D_k}$$ and
$$V=XW^v \in \mathbb{R}^{L \times D_v}$$, and an output encoding
$$Q=YW^q \in \mathbb{R}^{M \times D_k}$$, with
$$W^k \in \mathbb{R}^{D^{(x)} \times D_k}, W^q \in \mathbb{R}^{D^{(y)} \times D_k}$$
and $$W^v \in \mathbb{R}^{D^{(x)} \times D_v}$$. Note that source and
target embeddings are projected into a shared or common vector space with consistent dimensions. With that, we can compute the dot product between input and output sequences of different fixes maximal length and obtain $$D_v$$ embeddings of dimension $$M$$. We compute
attention with

$$\begin{align}c(Q,K,V)=\text{Softmax}\big(\frac{QK^T}{\sqrt{n}}\big)V.\end{align}$$


In this equation, we refer to $$(K, V)$$ as key-value pairs, and $$Q$$ as the query (see Appendix for an explanation of the names). By interpreting the dot product as a similarity measure, the context matrix $$c(Q,K,V) \in \mathbb{R}^{M \times D_v}$$ illustrates the similarity between the input and a representation of the input, which is weighted by its similarity to the output (in the context of cross-attention).

Let's summarize the computation in simpler terms: The context matrix comprises $$M$$ rows, one for each output element, and $$D_v$$ columns, corresponding to the dimensions of the value vectors. To delve deeper, we consider each row in the resulting context matrix $$c(Q, K, V)$$. For a specific output element, its row is computed by taking a weighted sum of the value vectors from matrix $$V$$. These weights are determined by the similarity between the query vector for that output element (from matrix $$Q$$) and the key vectors for all elements in the input sequence (from matrix $$K$$). The Softmax function is applied to normalize these weights for each output element.

Importantly, we apply masking to the embeddings for unseen target elements at each time step, ensuring that only relevant input elements contribute to the computation. The context matrix serves as the foundation for subsequent computations in the attention mechanism, allowing the model to determine how much attention to allocate to each part of the input sequence when generating the output.

In the transformer encoder, there is an important module where the queries are also source representations and in the decoder, there is a module where keys and values are also target representations (self-attention).


### Multi-Head Attention

Instead of computing the attention once, the
multi-head approach splits the three input matrices into smaller parts
and then computes the scaled dot-product attention for each part in
parallel. The independent attention outputs are then concatenated and
linearly transformed into the next layer's input dimension. This allows
us to learn from different representations of the current information
simultaneously with high efficiency. In the [CNN post](https://www.tobiasstenzel.com/blog/2023/dl-cnn/), we have already introduced the principle of applying the same operation multiple times with different learned sets of parameters: remember that CNNs contain stacks of multiple filters to provide the model with multiple feature maps, each covering different aspects of the input image. Multi-head attenttion is defined by

$$\begin{align}\text{MultiHead}(X_q, X_k, X_v)= [ \text{head}_1;...;\text{head}_h ] W^o,\end{align}$$

where $$\text{head}_i=$$Attention$$(X_q W^q_i, X_k W^k_i, X_v W^v_i)$$ and $$W_i^q  \in \mathbb{R}^{D^{(y)} \times D_v /H}$$, $$W_i^k \in \mathbb{R}^{D^{(x)} \times D_k / H}$$, $$W_i^v \in \mathbb{R}^{D^{(x)} \times D_v /H}$$
are matrices to map input embeddings of chunk size $$L \times D$$ into
query, key and value matrices. $$W^o \in \mathbb{R}^{D_v \times D}$$ is
the linear transformation in the output dimensions. These four weight
matrices are learned during training. Target self-attention and cross
attention layers compute outputs in $$\mathbb{R}^{M \times D}$$, and
source self-attention calculates outputs in $$\mathbb{R}^{L \times D}$$.

### TODO: Add code snippet and explain how sequences of different length can be passed

```python
import numpy as np

def compute_attention(num_queries, num_keys, depth_k, depth_v):
    np.random.seed(0)  # For reproducibility

    # Initialize matrices
    Q = np.random.rand(num_queries, depth_k)
    K = np.random.rand(num_keys, depth_k)
    V = np.random.rand(num_keys, depth_v)

    # Calculate raw attention scores
    raw_attention_scores = np.matmul(Q, K.T)

    # Scale the attention scores
    attention_scores = raw_attention_scores / np.sqrt(depth_k)

    # Apply softmax to get attention weights
    attention_weights = np.exp(attention_scores) / np.sum(np.exp(attention_scores), axis=-1, keepdims=True)

    # Multiply attention weights with Value (V) to get the final output
    attention_output = np.matmul(attention_weights, V)

    return attention_output

# Define dimensions
num_queries = 1  # Focusing on a single output token
depth_k = 4      # Depth of key vectors
depth_v = 2      # Depth of value vectors

# Example with 3 input tokens
output_3_tokens = compute_attention(num_queries, 3, depth_k, depth_v)
print("Attention output with 3 input tokens:", output_3_tokens)

# Example with 5 input tokens (different sequence length)
output_5_tokens = compute_attention(num_queries, 5, depth_k, depth_v)
print("Attention output with 5 input tokens:", output_5_tokens)

# Detailed comments
# Main takeaway: The i-th element of the attention scores interacts all output tokens with the i-th input token
# (see matrix dimension K (num_inputs x depth_keys))
"""
K = [[k11, k12, k13, k14],
     [k21, k22, k23, k24],
     [k31, k32, k33, k34]]

Q = [q1, q2, q3, q4]

# Raw Attention Scores Matrix (1x3)
raw_attention_scores = [[q1*k11 + q2*k12 + q3*k13 + q4*k14,
                         q1*k21 + q2*k22 + q3*k23 + q4*k24,
                         q1*k31 + q2*k32 + q3*k33 + q4*k34]]

# Detailed comments
# Main takeaway: The i-th input token is multiplied with the function between the i-th input token and all output tokens,
# in multiple dimensions to capture different aspects
"""
V = [[v11, v12],
     [v21, v22],
     [v31, v32]]

attention_weights = [[a1, a2, a3]]

attention_output[0, 0] = a1*v11 + a2*v21 + a3*v31
attention_output[0, 1] = a1*v12 + a2*v22 + a3*v32
"""
```

### Complexity Comparison

### TODO: Add path length

### add advtanges and disadvtanges of differen modules from chtgpt, good ending of deep learning series before culminating in transformers

To conclude this chapter, let's compare the complexities of the three main architectures in deep learning, as shown in Table [2](#fig:comparison), based on the survey by Lin et al. in 2022 <d-cite key="lin_survey_2022"></d-cite>:

<figure id="fig:comparison">
<center><img src="/assets/img/dl-series/complexity-comparison.png" style="width:55%"></center>
</figure>

**Table 2. Comparison of computation complexity between modules.** The "length" refers to the number of elements in the input sequence, and "dim" denotes the embedding depth for each element. It's worth noting that attention mechanisms, as used in Transformers, excel when the input sequence length is much smaller than the embedding depth. This advantage is evident in tasks like machine translation or question-answering, where capturing long-range dependencies is crucial. However, for direct application to image data, which inherently has a larger input length (e.g., CIFAR images with 3072 elements), preprocessing steps are necessary to reduce the input length effectively.

In summary, the Transformer architecture's success lies in its ability to capture complex dependencies, handle sequences of varying lengths, and process them efficiently through its innovative components, making it a fundamental building block in modern deep learning.


## Appendix — Naming Convention of Key, Value and Query

The naming convention of "key," "query," and "value" in the context of attention mechanisms is rooted in their respective roles and functions within the mechanism. These names help clarify how each component contributes to the computation of the context vector in attention.

**Key (K):** The key is essentially a set of representations derived from the input sequence (or encoder states in the case of sequence-to-sequence models like the Transformer). These representations are crucial for determining how much attention should be given to each element in the input sequence when generating an output. Keys are used to match against the queries.

*Reasoning:* Think of the "key" as a guide or a set of pointers that specify which parts of the input sequence are relevant for generating the output. It helps the model identify what to focus on.

**Query (Q):** The query represents what the model is currently trying to generate or pay attention to within the output sequence. In other words, it's a representation of the current target or decoder state.

*Reasoning:* The "query" represents the current "question" or "target" that the model is interested in. It's used to determine how well each element in the input sequence (represented by keys) aligns or matches with the current target.

**Value (V):** The value component contains the information that will be used to produce the context vector. Like keys, values are derived from the input sequence. However, they represent the content or information associated with each element in the input sequence.

*Reasoning:* The "value" carries the actual information that the model will use when generating the output. It's like the "answer" or "content" associated with each part of the input sequence.

To put it all together, the attention mechanism calculates a weighted sum of values based on the similarity between the query and keys. This weighted sum, known as the context vector, is used to determine how much each part of the input sequence should contribute to the current output. The keys help identify which parts are relevant, the query specifies what is being looked for, and the values provide the information to be used.

This naming convention, while abstract, makes the roles and relationships of these components intuitive, facilitating a clearer understanding of how attention mechanisms work in neural networks.

