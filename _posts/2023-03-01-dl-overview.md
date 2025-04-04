---
layout: distill
title: 1. Overview
date: 2023-03-01
description: 🎬 <b><i>Read this post before any other post.</i></b>
#categories: deep-learning
tags: dl-fundamentals highlights motivation target-group credits
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
  - name: Overview
  - name: Highlights
  - name: Target group
  - name: Credits
---

## Overview

I wrote most of the content for the Background section of my Master's Thesis
"Multi-camera Multi-object Tracking with Transformers". During this time, I discovered some new aspects and thought about
how to best introduce some concepts like backpropagation or the transformer. The purpose of this series is to share
this content about the fundamentals of deep learning with you.

This is the structure:
1. [Overview](https://www.tobiasstenzel.com/blog/2023/dl-overview/)
2. [Supervised Learning](https://www.tobiasstenzel.com/blog/2023/dl-supervised-learning/)
3. [Optimization](https://www.tobiasstenzel.com/blog/2023/dl-optimization/)
4. [Backpropgation](https://www.tobiasstenzel.com/blog/2023/dl-backprop/)
5. [Feedforward Neural Networks](https://www.tobiasstenzel.com/blog/2023/dl-fnn/)
6. [Convolutional Neural Networks](https://www.tobiasstenzel.com/blog/2023/dl-cnn/)
7. [Recurrent Neural Networks](https://www.tobiasstenzel.com/blog/2023/dl-rnn/)
8. [Transformer](https://www.tobiasstenzel.com/blog/2023/dl-transformer/)

## Highlights

Some highlights are:

1. Preparing the introduction of the transformer carefully via self-made, detailed figures of [vanilla RNN](https://www.tobiasstenzel.com/blog/2023/dl-rnn/#fig:vanilla-rnn), [encoder-decoder RNN](https://www.tobiasstenzel.com/blog/2023/dl-rnn/#fig:encoder-decoder-rnn) and [encoder-decoder RNN with attention](https://www.tobiasstenzel.com/blog/2023/dl-transformer/#fig:attention/) ✨
2. Explicitly spelling out [what makes the transformer great](https://www.tobiasstenzel.com/blog/2023/dl-transformer/#the-complete-transformer-architecture) ✨
3. Explaining convolution (or rather cross-correlation) with the [right figure and equation](https://www.tobiasstenzel.com/blog/2023/dl-cnn/#cross-correlation) ⚖
4. Explaining backpropagation with a very simple [toy example](https://www.tobiasstenzel.com/blog/2023/dl-backprop/#toy-example) 🪀
5. An explanation of [why we do not "frontpropagate"](https://www.tobiasstenzel.com/blog/2023/dl-backprop/#reverse-accumulation) although the multiplication in the chain rule is commutative 🤯
6. Showing backward passes of important layers. Did you know that the [backward pass of a convolution](https://www.tobiasstenzel.com/blog/2023/dl-cnn/#fig:gradient-cross-correlation) is also a convolution? 🤯

In my view, many deep learning concepts are usually introduced way too quickly or there is not enough reflection about the properties of some architectures. An example of the first observation is showing the pictures of the architecture from the original transformer paper without clarifying the encoder-decoder
beforehand in greater detail. Examples for the second observation are that it is frequently not stated what makes
the transformer great, what the inductive bias of a convolutional layer is, or why we even propagate gradients back. I took
the time to think about and research these things.

## Target group

In this series, the content is very dense, although I go into some detail regarding the main concepts. Ideally, there would be much more pictures, too. As a consequence, these posts are not for beginners. Instead, I recommend this series to three groups of people:

1. as complementary material for students who are taking a deep learning class right now
2. people who want to refresh some already present knowledge
3. DL professionals who can perhaps fill some small gaps.

## Credits

Of course, this series is essentially a recompilation of material from other people.
These are my main references: I recommend the Deep Learning book by Goodfellow <d-cite key="goodfellow_deep_2016"></d-cite>
, the CS231n lecture notes by Fei-Fei Li <d-cite key="li_cs231n_2018"></d-cite>, and Andrej Karpathy's dissertation <d-cite key="karpathy_connecting_2016"></d-cite> Other references are the Wikipedia article about automatic differentiation <d-cite key="noauthor_automatic_nodate"></d-cite>
, these blogposts about convolutional neural nets <d-cite key="kafunah_backpropagation_2016"></d-cite>, recursive neural nets <d-cite key="arat_backpropagation_2019"></d-cite>, and these posts about transformers <d-cite key="weng_attention_2018"></d-cite>, <d-cite key="karpathy_transformer_2022"></d-cite>, <d-cite key="vaswani_transformers_2021"></d-cite>.

I also want to thank Prof. Rainer Gemulla for his excellent lectures, especially his Deep Learning class, at Mannheim University.

## Citation

In case you like this series, cite it with:
<pre tabindex="0"><code  class="language-latex">@misc{stenzel2023deeplearning,
  title   = &quot;Deep Learning Series&quot;,
  author  = &quot;Stenzel, Tobias&quot;,
  year    = &quot;2023&quot;,
  url     = &quot;https://www.tobiasstenzel.com/blog/2023/dl-overview/
}
</code></pre>

