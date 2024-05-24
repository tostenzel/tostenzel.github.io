---
layout: distill
title: 9. Transformer Pt. 2 – Self and Cross-Attention Combined
date: 2023-03-09
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
    - name: Transformer Encoder
    - name: Transformer Decoder
    - name: The Complete Transformer Architecture
    - name: Complexity Comparison
    - name: Appendix — Naming Convention of Key, Value and Query


---

### Transformer Encoder

The Transformer architecture, as introduced by Vaswani et al. in 2017 <d-cite key="vaswani_attention_2017"></d-cite>, represents a groundbreaking advancement in deep learning, particularly in the fields of natural language processing and sequence modeling. However, nowadays it is the go-to model for many other areas, especially when a lot of resources are available. The first major part of the Transformer is its encoder, which plays a pivotal role in generating rich input representations using self-attention mechanisms.

<figure id="fig:transformer-encoder">
<center><img src="/assets/img/dl-series/2i-transformer-encoder.png" style="width:40%"></center>
</figure>

**Figure 8. Transformer encoder.** The Transformer Encoder consists of a stack of identical layers, including a multi-head self-attention layer (Red) for capturing contextual information and a position-wise fully-connected feed-forward network for introducing non-linearity. These two components work synergistically to process input sequences effectively and extract meaningful representations, with attention parameters focusing on dependencies and position-wise FNN parameters capturing position-specific patterns. Image source: <d-cite key="lin_survey_2022"></d-cite>.

In its original form, the encoder consists of a stack of N = 6 identical layers, each with unique parameters. These layers consist of two critical components:

1. **Multi-head self-attention layer (red):** This layer is the cornerstone of the Transformer's ability to capture contextual information from input sequences. It allows the model to focus on different parts of the input sequence simultaneously. Self-attention is a mechanism where each input element contributes to the representation of every other element, making it robust to capturing long-range dependencies and essential for tasks such as machine translation <d-cite key="vaswani_attention_2017"></d-cite>.

2. **Position-wise fully-connected feed-forward network (blue):** This layer introduces non-linearity into the network, allowing the model to capture complex patterns and relationships within the data.

These two components, self-attention and position-wise FNN, are fundamental in processing input sequences of varying length effectively and extracting meaningful representations. It is crucial to understand the clear distinction between the parameters used in the attention mechanism and those employed in the position-wise fully-connected feed-forward network (FNN) in order to understand the modules specific tasks, mechanisms and how they can process sequences of varying length: 

- **Attention parameters:** The attention mechanism, which consists of the attention parameters, is responsible for capturing dependencies and relationships between elements within the input sequence. It does this by assigning different attention weights to each element in the sequence based on its relevance to other elements. This allows the model to focus more on important elements and less on irrelevant ones. Importantly, the attention mechanism can adapt to input sequences of different lengths because the attention weights are calculated dynamically for each position in the sequence. Longer sequences may receive different attention distributions compared to shorter sequences.

- **Position-wise FNN parameters:** The position-wise FNN introduces non-linearity and complexity into the model. While the same position-wise FNN is applied to each position within the input sequence independently, the transformations performed by this network can capture position-specific patterns and relationships. This means that even though the same FNN parameters are shared across positions, the content of the positions can lead to different activations and outputs, allowing the model to handle sequences of varying lengths. The majority of the model's parameters are part of this module.

The combination of attention and position-wise FNN parameters enables the transformer to process input sequences of different lengths by dynamically adjusting the attention weights and capturing position-specific information. The attention mechanism provides a mechanism for the model to focus on relevant parts of the sequence, while the position-wise FNN allows for nonlinear transformations that can adapt to different content in the sequence. This flexibility is one of the key strengths of the Transformer architecture and makes it well-suited for a wide range of natural language processing tasks where input sequences may vary in length.


### Transformer Decoder

The decoder network, illustrated in Figure [9](#fig:transformer-decoder), is equally crucial in the Transformer architecture, as it enables the generation of sequences autoregressively based on the encoded source representation.

<figure id="fig:transformer-decoder">
<center><img src="/assets/img/dl-series/2j-transformer-decoder.png" style="width:66%"></center>
</figure>

**Figure 9. Transformer decoder.** The Transformer decoder plays a crucial role in autoregressively generating sequences based on encoded source representations. It includes a masked multi-head self-attention layer to capture dependencies within the target sequence, a multi-head cross-attention layer for accessing relevant source information, and a fully-connected feed-forward network to model complex relationships, all followed by normalized residual layers for stability. Image source <d-cite key="lin_survey_2022"></d-cite>.

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

**Figure 10. The complete transformer.** In its original form, the Transformer processes both source and target sequences through embedding layers (light blue), creating representations of dimension D = 512 for each element. Adding sinusoidal positional encoding vectors to the embeddings pallows the model to distinguish between different positions and capture positional dependencies. The Transformer's ability to handle input and output sequences of different lengths is primarily achieved through its masking mechanism. Padding masks ensure that the model does not consider padded elements when calculating attention scores, and the self-attention mechanism can adapt to variable-length sequences while maintaining coherence. The term "(Shifted) Outputs" emphasizes the autoregressive nature of the decoding process, where each output position depends on previously generated positions. Image source <d-cite key="lin_survey_2022"></d-cite>.

The Transformer possesses several critical properties:

- **Inductive Bias for Self-Similarity:** The self-attention mechanism empowers the model to identify recurring patterns or themes within the data, irrespective of their positions. This inductive bias is invaluable in real-world domains where recurrence is a prevalent pattern <d-cite key="vaswani_attention_2017"></d-cite>.

- **Expressive Forward Pass:** The Transformer establishes direct and intricate connections between all input and output elements, facilitating the learning of complex algorithms in a few steps.

- **Wide and Shallow Compute Graph:** Thanks to residual layers and matrix products in attention layers, the compute graph remains wide and shallow. This promotes fast forward and backward passes on parallel hardware while mitigating issues like vanishing or exploding gradients through techniques such as layer normalization and dot product scaling.


## TODO: The Transformer at Training Time (Machine Translation)


IN numpy code example, say that V really is only to preserve the flexible dimension. attention would essentially also work for only query times keys....

## TODO: The Transformer at Inference Time (Machine Translation)

## TODO: Remove unclear statements about how to handle sequences of different lenghts...


## Citation

In case you like this series, cite it with:
<pre tabindex="0"><code  class="language-latex">@misc{stenzel2023deeplearning,
  title   = &quot;Deep Learning Series&quot;,
  author  = &quot;Stenzel, Tobias&quot;,
  year    = &quot;2023&quot;,
  url     = &quot;https://www.tobiasstenzel.com/blog/2023/dl-overview/
}
</code></pre>
