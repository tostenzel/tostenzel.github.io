---
layout: distill
title: 8. Transformer
date: 2023-03-08
description: Transformer
categories: deep-learning-series deep-learning transformer
giscus_comments: true
related_posts: true
bibliography: references.bib

authors:
  - name: Tobias Stenzel
    url: "https://www.tobiasstenzel.com"
    affiliations:
      name: #

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Transformer
    subsections:
    - name: Attention
    - name: Key, value and query
    - name: Multi-head attention
    - name: Transformer encoder
    - name: Transformer decoder
    - name: The complete transformer architecture
    - name: Complexity comparison

---

## Transformer

In the last section we learned that RNNs have problems with learning
relations between the first parts of an input sequence with the output
and later parts of the input sequence. An architecture without this
structural problem is the transformer. The architecture was published by <d-cite key="vaswani_attention_2017"></d-cite> and applied to machine translation. We will
develop this model step-by-step, starting with its core component,
attention.

### Attention

Let us introduce this concept with an example. Figure
[6](#fig:attention) shows an encoder-decoder network with attention similar to the previous
encoder-decoder RNN (Figure
[5](#fig:encoder-decoder-rnn)). The core idea of attention is
defining the hidden state of the decoder-RNN as a function of every
hidden state from the encoder-RNN for every time period without
recursion. The result of the attention function is the context vector.
We will use this vector for every output element. In the specific
network in Figure [6](#fig:attention), the context vector is a function of both the
decoder states $s$ and the encoder states $h$. Further, it is
additionally concatenated with $s$ to predict the output layer.

<figure id="fig:attention">
<center><img src="/assets/img/dl-series/2h-attention.png" style="width:50%"></center>
<figcaption><b>Figure 6. Encoder-Decoder with attention.</b> In contrast to the
encoder-decoder RNN, the output layer is a function of the concatenation
of the hidden states and a time-dependent context vector (black boxes).
The main idea is that the context vector <span
class="math inline"><em>c</em><sub><em>t</em></sub></span> is a function
of all hidden states <span
class="math inline">{<em>h</em>}<sub><em>t</em> = 1</sub><sup><em>T</em></sup></span>
instead of only the last one (and the previous state <span
class="math inline"><em>s</em><sub><em>t</em> − 1</sub></span>). This
function is called <em>attention</em> (red). The black box represents
vector concatenation. <span
class="math inline"><em>c</em><sub>1</sub></span> is initialized with
<span class="math inline"><em>h</em><sub>4</sub></span>, <span
class="math inline"><em>s</em><sub>1</sub></span> with arbitrary values,
and <span class="math inline"><em>o</em><sub>1</sub></span> is
discarded.</figcaption>
</figure>

The attention function that returns the context vector for output $y_t$
wit attention scores for each input is given by the following
expression:

$$\begin{align}c_t = \sum_{i=1}^n \alpha_{t,i} h_i\end{align}$$

$\alpha_{t,i}$ is a softmax function of another function *score* that
measure how well output $y_t$ and input $x_i$ are aligned through the
encoder state $h_i$:

$$\begin{align}\alpha_{t,i} = \text{align}(y_t,x_i) = \frac{\exp \text{score}(s_{t-1}, h_i)}{\sum_{j=1}^n\exp\text{score} (s_{t-1}, h_j)}.\end{align}$$

There are many different scoring functions <d-cite key="weng_attention_2018"></d-cite>. A
common choice is the scaled dot-product score
$(s_t, h_i)=\frac{s_t^T h_i}{\sqrt{d}}$, where $d$ is the hidden state
dimension of both encoder and decoder states. Here, the alignment score
for one sequence element is given by a relative score of the dot-product
between the respective encoder hidden state and the current decoder
hidden state. We scale down the dot product to prevent vanishing
gradients from a pass to a softmax layer. After training the model, we
can analyze how much each output element depends on, or *attends* to,
each input. We do this by assembling a table with outputs as columns and
the output-specific alignment scores for each input as rows.

Another option for an encoder-decoder with attention is using a
self-attention mechanism to compute the context vector, for example,
with score $(h_j, h_i)=\frac{h_j^T h_i}{\sqrt{d}}$. We can use the
scores, for instance, in machine translation, to model how important the
previous words are for translating the current word in a sentence. Note
that we can execute many of these operations in parallel for the whole
input and output sequences using matrix operations.

Next, we will discuss an expansion of the attention mechanism and how it
is applied to the transformer's encoder and decoder, separately.
Finally, we will look at the complete model, and how it replaces the
positional information from the encoder RNN in a simple way without any
recurrence.

### Key, value and query

As we do not use recurrence of single sequence
elements anymore, let us denote the whole sequence of input embeddings
by $X \in \mathbb{R}^{L \times D^{(x)}}$. $L$ can either be the complete
input length $T_x$ or later only a fraction of it. $D^{(x)}$ is the
input embedding's length. Let us denoted the sequence of output
embeddings by $Y \in \mathbb{R}^{M \times D^{(y)}}$. The transformer
uses an extension of the attention mechanism, the multi-head attention,
as its core building block. The first step is that, instead of using the
softmax of the scaled dot-product between encoder states $h$ and decoder
states $s$ directly as in the last section, it uses the scaled
dot-product with two different input encodings,
$K=XW^k \in \mathbb{R}^{L \times D_k}$ and
$V=XW^v \in \mathbb{R}^{L \times D_v}$, and an output encoding
$Q=YW^q \in \mathbb{R}^{M \times D_k}$, with
$W^k \in \mathbb{R}^{D^{(x)} \times D_k}, W^q \in \mathbb{R}^{D^{(y)} \times D_k}$
and $W^v \in \mathbb{R}^{D^{(x)} \times D_v}$. Note that source and
target embeddings are projected linearly into the same space. We compute
attention with

$$\begin{align}c(Q,K,V)=\text{Softmax}\big(\frac{QK^T}{\sqrt{n}}\big)V.\end{align}$$

We call (K,V) key-value pairs and Q the query. Using the interpretation
of the dot product as a similarity measure, the context matrix shows the
(self-)similarity between the input and a representation of the input
that is weighted by its similarity to the output (so far). $c(Q,K,V)$ is
a matrix because we now compute the attention scores for every target
(query) dimension at once. However, we mask embeddings for unseen target
elements in every period. In the transformer encoder, there is an
important layer where the queries are also source representations and in
the decoder, there is layer where keys and values are also target
representation.

### Multi-head attention

Instead of computing the attention once, the
multi-head approach splits the three input matrices into smaller parts
and then computes the scaled dot-product attention for each part in
parallel. The independent attention outputs are then concatenated and
linearly transformed into the next layer's input dimension. This allows
us to learn from different representations of the current information
simultaneously with high efficiency.

$$\begin{align}\text{MultiHead}(X_q, X_k, X_v)= [ \text{head}_1;...;\text{head}_h ] W^o,\end{align}$$

where $\text{head}_i=$Attention$(X_q W^q_i, X_k W^k_i, X_v W^v_i)$ and $W_i^q  \in \mathbb{R}^{D^{(y)} \times D_v /H}$, $W_i^k \in \mathbb{R}^{D^{(x)} \times D_k / H}$, $W_i^v \in \mathbb{R}^{D^{(x)} \times D_v /H}$
are matrices to map input embeddings of chunk size $L \times D$ into
query, key and value matrices. $W^o \in \mathbb{R}^{D_v \times D}$ is
the linear transformation in the output dimensions. These four weight
matrices are learned during training. Target self-attention and cross
attention layers compute outputs in $\mathbb{R}^{M \times D}$, and
source self-attention calculates outputs in $\mathbb{R}^{L \times D}$.

### Transformer encoder

Figure [7](#fig:transformer-encoder) depicts the encoder network. It
computes an input representation based on the self-attention mechanism
that allows it to locate particular pieces of information from a large
context at all positions.

<figure id="fig:transformer-encoder">
<center><img src="/assets/img/dl-series/2i-transformer-encoder.png" style="width:33%"></center>

<figcaption><b>Figure 7. Transformer encoder.</b> In the original form, the encoder is a
stack of <span class="math inline"><em>N</em> = 6</span> identical
layers but with different parameters. It consists of two similar
components. The first sub-layer starts with a multi-head
<em>self</em>-attention layer (orange) and the second with a
<em>point-wise</em> fully-connected feed forward network (blue).
Point-wise means that the same weights are applied to each input
element. Afterwards, the respective previous input vector is added to
both outputs and the results are normalized by the normalized residual
layers (yellow). Crucially, in the self-attention layer, the queries are
also functions of the input embeddings. Adapted from <d-cite key="vaswani_attention_2017"></d-cite>.</figcaption>
</figure>

### Transformer decoder

Figure [8](#fig:transformer-decoder) shows the decoder network. It is
able to retrieve relevant information from the encoded source
representation to compute feature representations for generating the new
target sequence in autoregressive fashion. The key component is the
multi-head *cross*-attention layer (in contrast to the other multi-head
*self*-attention blocks).

<figure id="fig:transformer-decoder">
<center><img src="/assets/img/dl-series/2j-transformer-decoder.png" style="width:33%"></center>
<figcaption><b>Figure 8. Transformer decoder. </b> In the original form, the encoder is a
stack of <span class="math inline"><em>N</em> = 6</span> identical
layers. It first encodes the output sequence in the masked multi-head
<em>self</em>-attention layer (orange). The masked elements are the
target representations that are not generated so far. Next, it passes
these target representations as queries to the multi-head attention
layer (orange) together with the output input representations as keys
and values. Finally, the results pass through a fully-connected feed
forward network (blue). Each of these three layers is subsequently
transformed by a normalized residual layer (yellow). Adapted from <d-cite key="vaswani_attention_2017"></d-cite>.</figcaption>
</figure>

### The complete transformer architecture

Figure [9](#fig:transformer-complete) shows the complete architecture. It
has the following properties:

-   **Inductive bias for self-similarity:** The self-attention layers
    allow the model to detect reoccuring themes independent of their
    distances from each other. This works well in many applications
    becaue reoccurence is an important pattern in numerous real-world
    domains.

-   **Expressive forward pass:** The transformer interacts all elements
    of the input and output with themselves and each other in relatively
    simple and direct connections. This allows the model to learn many
    algorithms in just a few steps.

-   **Wide and shallow compute graph:** Due to the residual layers and
    the matrix products in the attention layers the compute graph is
    wide and shallow. This makes forward and backward passes fast on
    parallel hardware. Furthermore, together with layer normalization
    and dot product scaling, this mitigates vanishing or exploding
    gradients in backpropagation.

<figure id="fig:transformer-complete">
<center><img src="/assets/img/dl-series/2k-transformer-complete.png" style="width:66%"></center>
<figcaption><b>Figure 9. The complete transformer.</b> In the original form, both the
source and target sequence are passed to embedding layers to produce a
vector of length <span class="math inline"><em>D</em> = 512</span> for
every element. To preserve the ordering information of the inputs, we
add a respective sinusoidal positional encoding vector to every
embedding. To compute the probabilities for each element in the output
space at every position, we pass the decoder output through a linear and
a softmax layer. Adapted from <d-cite key="vaswani_attention_2017"></d-cite>.</figcaption>
</figure>

### Complexity comparison

Let us conclude this chapter by comparing the complexities of the three main architectures in deep learning. Table
[2](#fig:comparison) shows the differences in the number of FLOPS (floating point operations)
for the main units in the tree main architectures that we have discussed
so far. We observe that attention scales better than the other units if
the length of the input sequence is much smaller than the depth of each
elements embeddings. This applies to tasks like machine translation in <d-cite key="vaswani_attention_2017"></d-cite> or question-answering. However, for direct
applications to image data, this property does not hold. For instance,
the length from a CIFAR image equals $32 \cdot 32 \cdot 3 = 3072$.
Therefore, applying a transformer to image data in order to profit from
its advantages requires architectures that reduce the input length
beforehand.


<figure id="fig:comparison">
<center><img src="/assets/img/dl-series/complexity-comparison.png" style="width:55%"></center>
<figcaption><b>Table 2. Comparison of computation complexity between models.</b> length refers
to the number of elements in the input sequence and dim to the
embedding depth for each element. Entries adapted from
<d-cite key="vaswani_transformers_2021"></d-cite>.</figcaption>
</figure>
