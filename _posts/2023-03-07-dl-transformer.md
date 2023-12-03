---
layout: distill
title: 8. Transformer
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
  - name: Transformer
    subsections:
    - name: Attention
    - name: Key, Value and Query
    - name: Multi-Head Attention
    - name: Transformer Encoder
    - name: Transformer Decoder
    - name: The Complete Transformer Architecture
    - name: Complexity Comparison

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
[7](#fig:attention) shows an encoder-decoder network with attention similar to the previous
encoder-decoder RNN (Figure
[6](https://www.tobiasstenzel.com/blog/2023/dl-rnn/#fig:encoder-decoder-rnn)). The core idea of attention is
defining the hidden state of the decoder-RNN as a function of every
hidden state from the encoder-RNN for every time period without
recursion. The result of the attention function is the context vector.
We will use this vector for every output element. In the specific
network in Figure [7](#fig:attention), the context vector is a function of both the
decoder states $$s$$ and the encoder states $$h$$. Further, it is
additionally concatenated with $$s$$ to predict the output layer.

<figure id="fig:attention">
<center><img src="/assets/img/dl-series/2h-attention.png" style="width:50%"></center>
</figure>
<b>Figure 7. Encoder-Decoder with attention.</b> In contrast to the
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
discarded.  
<br>

The attention function that returns the context vector for output $$y_t$$
wit attention scores for each input is given by the following
expression:

$$\begin{align}c_t = \sum_{i=1}^n \alpha_{t,i} h_i\end{align}$$

$$\alpha_{t,i}$$ is a softmax function of another function *score* that
measure how well output $$y_t$$ and input $$x_i$$ are aligned through the
encoder state $$h_i$$:

$$\begin{align}\alpha_{t,i} = \text{align}(y_t,x_i) = \frac{\exp \text{score}(s_{t-1}, h_i)}{\sum_{j=1}^n\exp\text{score} (s_{t-1}, h_j)}.\end{align}$$

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

Another option for an encoder-decoder with attention is using a
self-attention mechanism to compute the context vector, for example,
with score $$(h_j, h_i)=\frac{h_j^T h_i}{\sqrt{d}}$$. We can use the
scores, for instance, in machine translation, to model how important the
previous words are for translating the current word in a sentence. Note
that we can execute many of these operations in parallel for the whole
input and output sequences using matrix operations.

Next, we will discuss an expansion of the attention mechanism and how it
is applied to the transformer's encoder and decoder, separately.
Finally, we will look at the complete model, and how it replaces the
positional information from the encoder RNN in a simple way without any
recurrence.

### Key, Value and Query

As we do not use recurrence of single sequence
elements anymore, let us denote the whole sequence of input embeddings
by $$X \in \mathbb{R}^{L \times D^{(x)}}$$. $$L$$ can either be the complete
input length $$T_x$$ or later only a fraction of it. $$D^{(x)}$$ is the
input embedding's length. Let us denoted the sequence of output
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
target embeddings are projected linearly into the same space. We compute
attention with

$$\begin{align}c(Q,K,V)=\text{Softmax}\big(\frac{QK^T}{\sqrt{n}}\big)V.\end{align}$$

We call (K,V) key-value pairs and Q the query. Using the interpretation
of the dot product as a similarity measure, the context matrix shows the
(self-)similarity between the input and a representation of the input
that is weighted by its similarity to the output (so far). $$c(Q,K,V)$$ is
a matrix because we now compute the attention scores for every target
(query) dimension at once. However, we mask embeddings for unseen target
elements in every period. In the transformer encoder, there is an important module where the queries are also source representations (self-attention) and in the decoder, there is a module where keys and values are also target representations (cross-attention).

### Multi-Head Attention

Instead of computing the attention once, the
multi-head approach splits the three input matrices into smaller parts
and then computes the scaled dot-product attention for each part in
parallel. The independent attention outputs are then concatenated and
linearly transformed into the next layer's input dimension. This allows
us to learn from different representations of the current information
simultaneously with high efficiency.

$$\begin{align}\text{MultiHead}(X_q, X_k, X_v)= [ \text{head}_1;...;\text{head}_h ] W^o,\end{align}$$

where $$\text{head}_i=$$Attention$$(X_q W^q_i, X_k W^k_i, X_v W^v_i)$$ and $$W_i^q  \in \mathbb{R}^{D^{(y)} \times D_v /H}$$, $$W_i^k \in \mathbb{R}^{D^{(x)} \times D_k / H}$$, $$W_i^v \in \mathbb{R}^{D^{(x)} \times D_v /H}$$
are matrices to map input embeddings of chunk size $$L \times D$$ into
query, key and value matrices. $$W^o \in \mathbb{R}^{D_v \times D}$$ is
the linear transformation in the output dimensions. These four weight
matrices are learned during training. Target self-attention and cross
attention layers compute outputs in $$\mathbb{R}^{M \times D}$$, and
source self-attention calculates outputs in $$\mathbb{R}^{L \times D}$$.

### Transformer Encoder

The Transformer architecture, as introduced by Vaswani et al. in 2017 <d-cite key="vaswani_attention_2017"></d-cite>, represents a groundbreaking advancement in deep learning, particularly in the fields of natural language processing and sequence modeling. However, nowadays it is the go-to model for many other areas, especially when a lot of resources are available. The first major part of the Transformer is its encoder, which plays a pivotal role in generating rich input representations using self-attention mechanisms.

<figure id="fig:transformer-encoder">
<center><img src="/assets/img/dl-series/2i-transformer-encoder.png" style="width:40%"></center>
</figure>

**Figure 8. Transformer encoder.** The Transformer Encoder consists of a stack of identical layers, including a multi-head self-attention layer (Red) for capturing contextual information and a position-wise fully-connected feed-forward network for introducing non-linearity. These two components work synergistically to process input sequences effectively and extract meaningful representations, with attention parameters focusing on dependencies and position-wise FNN parameters capturing position-specific patterns. Image adapted from <d-cite key="lin_survey_2022"></d-cite>.

In its original form, the encoder consists of a stack of N = 6 identical layers, each with unique parameters. These layers consist of two critical components:

1. **Multi-head self-attention layer (red):** This layer is the cornerstone of the Transformer's ability to capture contextual information from input sequences. It allows the model to focus on different parts of the input sequence simultaneously. Self-attention is a mechanism where each input element contributes to the representation of every other element, making it robust to capturing long-range dependencies and essential for tasks such as machine translation <d-cite key="vaswani_attention_2017"></d-cite>.

2. **Position-wise fully-connected feed-forward network (blue):** This layer introduces non-linearity into the network, allowing the model to capture complex patterns and relationships within the data.

These two components, self-attention and position-wise FNN, are fundamental in processing input sequences of varying length effectively and extracting meaningful representations. It is crucial to understand the clear distinction between the parameters used in the attention mechanism and those employed in the position-wise Fully-Connected Feed-Forward Network (FNN) in order to understand the modules specific tasks and mechanisms: 

- **Attention parameters:** The attention mechanism's parameters are responsible for capturing dependencies and relationships between elements within the sequence. These parameters are shared across all positions in the sequence, enabling the model to learn to attend to different parts of the input sequence effectively.

- **Position-wise FNN parameters:** In contrast, the position-wise FNN parameters introduce non-linearity and complexity into the model. Specifically, the same position-wise Feed-Forward Network (FNN) is applied to each position within the input sequence independently. This means that the parameters of the FNN are shared across all positions in the sequence. The position-wise FNN typically consists of two linear transformations followed by a non-linear activation function (e.g., ReLU). These transformations are applied uniformly to each element in the sequence, allowing the model to capture position-specific patterns and relationships. The number of FNN parameters is much larger than the number of attention parameters.

These two sets of parameters work in synergy to process input sequences effectively while accommodating varying sequence lengths and capturing both global and position-specific information. After processing through these layers, the Transformer encoder adds the respective previous input vector to both outputs and normalizes the results using normalized residual layers (yellow). Importantly, in the self-attention layer, queries are also functions of the input embeddings, which enhances the model's ability to capture context and dependencies effectively.


### Transformer Decoder

The decoder network, illustrated in Figure [9](#fig:transformer-decoder), is equally crucial in the Transformer architecture, as it enables the generation of sequences autoregressively based on the encoded source representation.

<figure id="fig:transformer-decoder">
<center><img src="/assets/img/dl-series/2j-transformer-decoder.png" style="width:66%"></center>
</figure>

**Figure 9. Transformer decoder.** The Transformer decoder plays a crucial role in autoregressively generating sequences based on encoded source representations. It includes a masked multi-head self-attention layer to capture dependencies within the target sequence, a multi-head cross-attention layer for accessing relevant source information, and a fully-connected feed-forward network to model complex relationships, all followed by normalized residual layers for stability. Image adapted from <d-cite key="lin_survey_2022"></d-cite>.

Similar to the encoder, the decoder comprises N = 6 identical layers. These layers consist of:

1. **Masked Multi-Head Self-Attention Layer (Red):** In the decoder, this layer is masked to prevent attending to future positions in the output sequence. It focuses on capturing dependencies among elements within the target sequence that have already been generated. This masking is crucial for ensuring the autoregressive nature of the decoder, allowing it to generate sequences one element at a time <d-cite key="vaswani_attention_2017"></d-cite>. Specifically, the masked multi-head self-attention layer in the decoder prevents the model from attending to future positions in the (Shifted) Outputs sequence. This mechanism ensures that each position in the (Shifted) Outputs sequence is generated based on the previously generated elements, adhering to the autoregressive nature of sequence generation.

2. **Multi-Head Cross-Attention Layer (Red):** The decoder generates queries from the previously generated target representations and uses them to attend to the encoded source representations. This mechanism ensures that the decoder accesses relevant information from the source to generate the next part of the target sequence.

3. **Fully-Connected Feed-Forward Network (Blue):** Similar to the encoder, this layer introduces non-linearity, enabling the decoder to model complex relationships.

Each of these layers is followed by a normalized residual layer (yellow), contributing to the stability and effectiveness of the Transformer's decoding process.

### The Complete Transformer Architecture

Figure [10](#fig:transformer-complete) presents the complete Transformer architecture:

<figure id="fig:transformer-complete">
<center><img src="/assets/img/dl-series/2k-transformer-complete.png" style="width:100%"></center>
</figure>

**Figure 10. The complete transformer.** In its original form, the Transformer processes both source and target sequences through embedding layers (light blue), creating representations of dimension D = 512 for each element. This embedding strategy is a cornerstone of the Transformer's ability to handle input and output sequences of different lengths effectively. The key to this flexibility lies in the use of position embeddings (purple), which preserve positional information by adding sinusoidal positional encoding vectors to the embeddings. These position embeddings carry information about the position of each element within the sequence, allowing the model to distinguish between different positions and capture positional dependencies. The term "(Shifted) Outputs" emphasizes the autoregressive nature of the decoding process, where each output position depends on previously generated positions. Image adapted from <d-cite key="lin_survey_2022"></d-cite>.

The Transformer possesses several critical properties:

- **Inductive Bias for Self-Similarity:** The self-attention mechanism empowers the model to identify recurring patterns or themes within the data, irrespective of their positions. This inductive bias is invaluable in real-world domains where recurrence is a prevalent pattern <d-cite key="vaswani_attention_2017"></d-cite>.

- **Expressive Forward Pass:** The Transformer establishes direct and intricate connections between all input and output elements, facilitating the learning of complex algorithms in a few steps.

- **Wide and Shallow Compute Graph:** Thanks to residual layers and matrix products in attention layers, the compute graph remains wide and shallow. This promotes fast forward and backward passes on parallel hardware while mitigating issues like vanishing or exploding gradients through techniques such as layer normalization and dot product scaling.


### Complexity Comparison

To conclude this chapter, let's compare the complexities of the three main architectures in deep learning, as shown in Table [2](#fig:comparison), based on the survey by Lin et al. in 2022 <d-cite key="lin_survey_2022"></d-cite>:

<figure id="fig:comparison">
<center><img src="/assets/img/dl-series/complexity-comparison.png" style="width:55%"></center>
</figure>

**Table 2. Comparison of computation complexity between models.** The "length" refers to the number of elements in the input sequence, and "dim" denotes the embedding depth for each element. It's worth noting that attention mechanisms, as used in Transformers, excel when the input sequence length is much smaller than the embedding depth. This advantage is evident in tasks like machine translation or question-answering, where capturing long-range dependencies is crucial. However, for direct application to image data, which inherently has a larger input length (e.g., CIFAR images with 3072 elements), preprocessing steps are necessary to reduce the input length effectively.

In summary, the Transformer architecture's success lies in its ability to capture complex dependencies, handle sequences of varying lengths, and process them efficiently through its innovative components, making it a fundamental building block in modern deep learning.


## Citation

In case you like this series, cite it with:
<pre tabindex="0"><code  class="language-latex">@misc{stenzel2023deeplearning,
  title   = &quot;Deep Learning Series&quot;,
  author  = &quot;Stenzel, Tobias&quot;,
  year    = &quot;2023&quot;,
  url     = &quot;https://www.tobiasstenzel.com/blog/2023/dl-overview/
}
</code></pre>
