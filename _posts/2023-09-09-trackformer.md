---
layout: distill
title: Trackformer — Multi-Object Tracking with Transformers
date: 2023-03-08
description: A lower-level explanation of Meinhardt et al. (2022)'s paper about applying the transformer to multi-object tracking (MOT)
#categories: deep-learning
#categories: deep-learning
tags: applications transformer MOT multiple-object-tracking
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
  - name: Trackformer
    subsections:
    - name: 1) DETR as Object Detector
    - name: 2) Trackformer as Multi-Object Tracker
    - name: 3) Extra. Deformable DETR as More Efficient Detector



---

This blog post provides a detailed explanation of the transformer-based tracking model Trackformer <d-cite key="meinhardt_trackformer_2022"></d-cite>. Trackformer not only achieved state-of-the art results but has also a comparably simple architecture that favors implementation and maintenance for pracitioners. The coolest thing about this blog post is the inclusion of two self-made images that depict Trackformer’s training process for different scenarios where objects are (re-)entering, leaving, or moving in a camera-recorded, dynamic scene.<br/>

Multi-object tracking (MOT) is an important task in the Computer Vision domain. MOT is closely related to object detection because tracking models utilize the detections in each frame of a video with the aim of assigning the same ID to the same objects across different frames over time. Trackformer uses the transformer-based object detector DETR <d-cite key="carion_end--end_2020"></d-cite>. The primary concept behind DETR is to learn a larger number of transformer-decoder embeddings that generate bounding boxes (by cross-attending to the image features), with a set of these embeddings successfully matched to the actual objects in an image. The fundamental idea behind Trackformer is to feed the output embeddings from successfully matched embeddings to the transformer for the next image in a video sequence and train these embeddings to match the same object again. In this configuration, the intertemporally matched embeddings carry the identity of an object over to the next image.<br/>

We will delve into both models in more detail separately. First, we will explore DETR <d-cite key="carion_end--end_2020"></d-cite>. I will explain the complete model during inference time, then during training time, including the loss function, and subsequently, we will discuss advantages, disadvantages, and results. Second, we will repeat the same process for Trackformer <d-cite key="meinhardt_trackformer_2022"></d-cite>. Finally, we will learn about Deformable DETR <d-cite key="zhu_deformable_2021"></d-cite>, which is a more efficient version of DETR. Deformable DETR employs deformable attention instead of regular attention. Trackformer actually utilizes Deformable DETR for its detections. The Deformable DETR paper is quite technical, so feel free to skip this part!<br/><br/>

<figure id="fig:object-queries">
<center><img src="/assets/img/dl-series/paper-detr.png" style="width:66.6%"></center>
</figure>


### 1) DETR as Object Detector

Previous object detection systems incorporate various manually designed
elements, such as anchor generation, rule-based training target
assignment, and non-maximum suppression (NMS) post-processing <d-cite key="liu_deep_2018"></d-cite>. These components do not constitute fully end-to-end
solutions. In a recent study, <d-cite key="carion_end--end_2020"></d-cite> introduced DETR,
an object detection approach that eliminates the need for such manual
components. DETR represents the first fully end-to-end object detector
and achieved highly competitive results on the COCO 2017 detection
dataset <d-cite key="lin_microsoft_2014"></d-cite>. The two main components are a set-based
global loss that enforces a subset of unique class predictions via
bipartite matching, and a transformer encoder-decoder architecture with
learned embeddings called *object queries*. In contrast to <d-cite key="vaswani_attention_2017"></d-cite>, these queries are learned encodings of
potential objects on the image instead of rule-based encoded
representations of the target sequence. This section explains first the
transformer at inference time, and second the bipartite matching loss
with which we train the model.

### Complete transformer-based DETR architecture

The overall model has
three components as depicted by Figure
[1](#fig:object-queries). The first component is a conventional
CNN backbone that generates a lower-resolution activation map many
channels. The transformation from large embedding length (image height
$$\cdot$$ width) to large embedding depth is crucial for the efficiency of
the attention modules (see
Table [2](https://www.tobiasstenzel.com/blog/2023/dl-transformer/#complexity-comparison). The second component is the transformer
encoder-decoder. In contrast to the original decoder by <d-cite key="vaswani_attention_2017"></d-cite>, DETR decodes $$N$$ objects at once at each
decoder layer instead of predicting the output sequence in an
autoregressive manner by masking later elements. Because the decoder is
permutation-invariant like the encoder, we also require $$N$$ different
decoder input embeddings to generate different results. We achieve this
by feeding learned embeddings with dimension $$d$$ that we call *object
queries* to the decoder. In contrast to that, the original transformer
in <d-cite key="vaswani_attention_2017"></d-cite> takes the decoder outputs from the previous
iteration as decoder inputs. The decoder transforms the $$N$$ object
queries into $$N$$ output embeddings. The third component is a two-headed
prediction network that is shared for all output embeddings. It is
defined as an FFN, i.e. a a 3-layer vanilla neural net with ReLU
activations for the normalized bounding box values and hidden dimension
$$d$$. We normalize the input to the prediction FFNs with a shared
layer-norm. The one head is a linear projection layer to predict the
bounding box values.
The other head is a softmax layer that predicts the
class labels.<br/>

<figure id="fig:object-queries">
<center><img src="/assets/img/dl-series/4b-detr-object-queries.png" style="width:100.0%"></center>
</figure>
<b>Figure 1. High-level DETR architecture with CNN backbone,
transformer-encoder decoder and FFN prediction heads.</b> In contrast to the
autoregressive sequence generation in <d-cite key="vaswani_attention_2017"></d-cite>, the outputs are computed at
once by feeding different learned positional encodings, called
<em>object queries</em>, into the transformer decoder. Image source: <d-cite key="carion_end--end_2020"></d-cite>.<br/><br/>

The transformer allows the model to use self- and cross-attention over
the object queries to include the information about all potential
objects with pair-wise relations in their prediction of only one
potential object, while using the whole image as context.


### Object Detection Set Prediction Loss

During training, we want to
learn predictions that have the right number of no classes and the right
number of classes with the right class label and bounding box. Formally,
we achieve this the following way: From the transformer decoder
prediction head, DETR infers $$N$$ predictions for the tuples of bounding
box coordinates and the object class. $$N$$ has to be at least as large as
the maximal number of objects on one image. The first tuple element are
the bounding box coordinates, denoted by $$b \in [0,1]^4$$. These are four
values: the image-size normalized center coordinates, and the normalized
height and width of the box w.r.t. the input image's borders. Using
center coordinates and normalization help us to deal with images of
different sizes. Examples for the second tuple element, class $$c$$, are
\"person A\", for person detection, or \"car\" for object detection. The
remaining class is the \"no object\" class, denoted by $$\emptyset$$. For
every image, we pad the target class vector $$c$$ of length $$N$$ with
$$\emptyset$$ if the image contains less than $$N$$ objects. To score
predicted tuples
$$\hat{y}=\{\hat{y}_i\}_{i=1}^N=\{(\hat{c}_i,\hat{b}_i)\}_{i=1}^N$$ with
respect to the targets $$y$$, <d-cite key="carion_end--end_2020"></d-cite> apply a loss
function that we apply to a class permutation based on an *optimal*
bipartite matching between the predicted and target tuples. With this
loss function, we jointly maximize the log likelihood of the class
permutation and minimize the bounding box losses.
Figure [2](#fig:bipartite-matching) depicts an example of bipartite
matching between predictions and ground truth for a picture of two
seagulls at a shore during training time. We observe that the matching
procedure selects a unique permutation that directly maps exactly one
prediction to one target. Thus, with that approach we do not have to
handle mappings of many similar predicted bounding boxes and classes to
one target, for instance, to the same seagull.

<br/>

<figure id="fig:bipartite-matching">
<center><img src="/assets/img/dl-series/4a-detr-bipartite-matching.png" style="width:100.0%"></center>
</figure>
<b>Figure 2. DETR training pipeline with bipartite matching loss.</b> The
loss function generates an optimal one-to-one mapping between
predictions and targets according to bounding box and object class
similarity (colors). In this unique permutation of <span
class="math inline"><em>N</em></span> classes, class predictions with no
match among the class targets are regarded as "no object" predictions,
too (green). Image source: <d-cite key="carion_end--end_2020"></d-cite>.

<br/><br/>

The next section describes the loss function required for the bipartite
matching between predicted and target detections that is optimal in
terms of bounding box and class similarity. We find the optimal
permutation of predicted detections
$$\hat{\sigma}\in\mathcal{\mathfrak{S}}_N$$ of $$N$$ elements from:

$$\begin{align}
    \hat{\sigma} = arg\,min_{\sigma\in\mathcal{\mathfrak{S}}_N} \sum_{i}^{N} \text{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)}),
\end{align}$$

where $$\text{L}_{\text{match}}$$ is a pair-wise matching cost between
target $$y_i$$ and prediction $$\hat{y}$$ with index $$\sigma(i)$$. We compute
the assignment efficiently with the Hungarian matching algorithm <d-cite key="kuhn_hungarian_1955"></d-cite> instead of brute force.

We want to assign predictions and targets that are close in terms of
class and bounding box. Thus, the matching considers both the class
score for every target and the similarity of predicted and target box
coordinates. We reward a high class score for the target class and a
small bounding box discrepancy. To this end, let us denote the index function by $$\text{I}$$, the
probability of predicting class $$c_i$$ for detection with permutation
index $$\sigma(i)$$ by $$\hat{p}_{\sigma(i)}(c_i)$$ and the predicted box
by $$\hat{b}_{\sigma(i)}$$. With that, we define the pair-wise matching
loss as

$$\begin{align}
\text{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)}) = -\text{I}_{c_i\neq\emptyset} \hat{p}_{\sigma (i)}(c_i) + \text{I}_{c_i\neq\emptyset} \text{L}_{\text{box}}(b_{i}, \hat{b}_{\sigma(i)}).
\end{align}$$

Here, our objective is to find the best matching for the \"real\"
classes $$c \neq \emptyset$$ by ignoring class predictions that are
directly mapped to \"no class\" by the prediction. However, the model
still has to learn not to predict too many real classes. Given our
optimal matching $$\hat{\sigma}$$, we achieve this by minimizing the
*Hungarian loss*. The function is given by

$$\begin{align}\text{L}_{\text{Hungarian}}(y, \hat{y}) = \sum_{i=1}^N \left[-\log  \hat{p}_{\hat{\sigma}(i)}(c_{i}) + \text{I}_{c_i\neq\emptyset} \text{L}_{\text{box}}\Big(b_{i}, \hat{b}_{\hat{\sigma}}(i)\Big)\right]\end{align}$$

The difference to the matching loss is two-fold. First, we now penalize
wrong assignments to all classes including \"no class\" with the first
class-specific term. We do not include the \"no class\" instances to the
box loss $$\text{L}_{\text{box}}$$ because they are not matched to a
bounding box anyway. Second, we scale the class importance compared to
the bounding boxes by taking the log of predicted class probabilities.
In practice, the class term is further reduced for \"no class\" objects
to take class imbalance into account. The last expression is the box
loss. It is the L1 loss of the bounding box vector $$b_i$$. Note, however,
that the L1 loss penalizes larger boxes.
The following equations shows the expression:

$$\begin{align}
    \text{L}_{\text{box}}(b_{i}, \hat{b}_{\sigma(i)}) =  \lambda_{\text{box}}||b_{i}- \hat{b}_{\sigma(i)}||_1,
\end{align}$$


where $$\lambda_{\text{box}} \in \mathbb{R}$$ is a hyperparameter. We
normalize the loss by the number of objects in each image.

### DETR's drawbacks

In spite of its intriguing design and commendable
performance, DETR encounters certain challenges. Firstly, it requires
significantly more training epochs to reach convergence compared to
existing object detectors. For instance, when evaluated on the COCO
benchmark <d-cite key="lin_microsoft_2014"></d-cite>, DETR necessitates 500 epochs for
convergence, making it approximately 10 to 20 times slower than Faster
R-CNN <d-cite key="ren_faster_2015"></d-cite>. Secondly, DETR exhibits relatively lower
proficiency in detecting small objects <d-cite key="zhu_deformable_2021"></d-cite>.
Contemporary object detectors typically utilize multi-scale features,
employing high-resolution feature maps for the detection of small
objects. However, employing high-resolution feature maps leads to
impractical complexities for DETR. These aforementioned issues primarily
stem from the deficiency of Transformer components in processing image
feature maps. During initialization, the attention modules distribute
nearly uniform attention weights to all pixels in the feature maps. It
requires many training epochs for the attention weights to learn to
concentrate on sparse meaningful locations. Additionally, the
computation of attention weights in the Transformer encoder is quadratic
in relation to the number of pixels. Consequently, processing
high-resolution feature maps becomes highly computationally and memory
intensive.<br/><br/>

<figure id="fig:object-queries">
<center><img src="/assets/img/dl-series/paper-trackformer.png" style="width:66.6%"></center>
</figure>

### 2) Trackformer as Multi-Object Tracker

Trackformer <d-cite key="meinhardt_trackformer_2022"></d-cite> extends DETR to multi-object
tracking. To be precise, it uses the variant Deformable DETR <d-cite key="zhu_deformable_2021"></d-cite>, presumably because the results were better.
Trackformer not only achieved state of the art results for online
tracking but also presented an end-to-end architecture that solves the
three sub-tasks of track initialization (detection), prediction of next
positions, and matching predictions with detections. Thereby, it
bypasses intermediate layers that are usually present in previous
pipeline designs, similar to how DETR facilitated object detection. The
main idea is depicted by
[3](#fig:trackformer). It is to re-use DETR's *decoded* object
queries that have been matched to an actual object in one frame and use
them as additional object queries for the next frame as *autoregressive
track queries*. Accordingly, we dynamically adjust the transformer
decoder sequence length. The static object queries are responsible for
initializing new tracks and the taken over track queries allow tracking
objects across frames. In contrast do DETR, besides the bounding box
quadruple $$b$$ and the object class $$c$$, we additionally predict predict
the track ID across frames in an implicit way from by enumerating the
track queries.

With the described approach, we train the model to not only decode
learned object queries into representations that can detect objects but
also to use the decoded queries again as decoder input to detect the
same object when possible. If we match the decoder output from an object
query in frame $$t-1$$, it is re-used as an additional track query as
decoder input for frame $$t$$. If its output is matched again, we assume
that both detections belong to the same object with ID $$k$$.

<br/><br/>

<figure id="fig:trackformer">
<center><img src="/assets/img/dl-series/4-trackformer.png" style="width:100.0%"></center>
</figure>
<b>Figure 3.Trackformer architecture.</b> Trackformer extends DETR to
tracking on video data by feeding the decoded object queries that are
matched to actual objects as additional <em>autoregressive tracking
queries</em> (colored detection squares) next to the object queries for
the next image into the transformer decoder (dark blue). The decoder
processes the set of <span
class="math inline"><em>N</em><sub>track</sub> + <em>N</em><sub>object</sub></span>
queries to further track or remove existing tracks (light blue) and to
initialize new tracks (purple). Image source: <d-cite key="meinhardt_trackformer_2022"></d-cite>.

### Set prediction loss

We now want to formulate a loss that allows the
model to learn the bipartite matching $$j=\pi (i)$$ between target objects
$$y_i$$ to the set of both object and track query predictions $$\hat{y}_j$$.
For this purpose, let us denote the subset of target track identities at
frame $$t$$ with $$K_t \subset K$$. This is different from DETR as $$K$$
contains all object identities for all images in the video sequence.
These object or track identities can be present in multiple frames,
i.e. they can intersect from frame to frame. Trackformer takes three
steps to associate queries with targets. The last step corresponds to
DETR's method. The steps to obtain the mapping of predicted detections
to target detections $$\hat{\sigma}$$ for one frame are the following:
first, we match $$K_{\text{track}} = K_{t-1} \cap K_t$$ (target objects in
the current frame that were also present in the previous frame) by track
identity. This means, we associate these targets with the output from
the previous query. Second, we match
$$K_{\text{leaving}} = K_{t-1} \setminus K_t$$ (objects leaving the scene
between two frames) with background class $$\emptyset$$. And third, we
match $$K_{\text{init}} = K_{t} \setminus K_{t-1}$$ (objects entering the
scene) with the $$N_{\text{object}}$$ object queries by minimum cost
mapping based on object class and bounding box similarity the same ways
as DETR assigned its targets to object queries.

Output embeddings which were not matched, i.e. 1) proposals with worse
class and bounding box similarity than others or 2) track queries
without corresponding ground truth object, are assigned to background
class $$\emptyset$$.

With this order, we prioritize matching track queries from the last
frame even if object queries from the current frame yield more similar
bounding boxes and classes. This is necessary because, in order to
assign the same object ID to detections in multiple frames, we have to
train the object queries to not only initialize a track after on pass
through the decoder but also to detect the same object after two passes
through the decoder with given the respective interactions from the
image encodings.

The final set prediction loss for one frame is computed over all
$$N=N_{\text{object}}+N_{\text{track}}$$ model outputs. Because
$$K_{t-1} \setminus K_t$$ (objects that left the scene) are not contained
in the current-frame permutation, we write the loss as

$$\begin{align}
\text{L}_{\text{MOT}}(y,\hat{y},\pi)=\sum_{i=1}^N \text{L}_{\text{query}}(y,\hat{y},\pi).
\end{align}$$
Further, we define the loss per query and differentiate two categories.
First, we have the object query loss $$L_0$$ for outputs from unmatched
embeddings. And second, we have the track query loss $$L_1$$ for outputs
from matched embeddings that will be overtaken to the next time period
as track queries. Formally, the query loss is given by

$$\begin{align}\text{L}_{\text{query}}=
    \begin{cases}
        \phantom{.}L_0 &= -\lambda_{\text{cls}} \log \hat{p}_i (\emptyset) \quad \phantom{..................}\text{if } i \notin \pi \\
        \phantom{.}L_1 &= -\lambda_{\text{cls}} \log \hat{p}_i (c_{\pi=i}) + \text{L}_{\text{box}}(b_{\pi=i},\hat{b}_i) \quad \text{if } i \in \pi.
    \end{cases}\end{align}$$

The expression captures two features: first, $$L_1$$ rewards track queries
that find the right bounding box for objects that are still present on
the current frame and it rewards object queries with similar outputs to
new objects. Second, $$L_0$$ not only rewards track queries that predict
the background class if their object has left the scene but also object
queries that predict the background class if their bounding box
prediction is off. The discussed details about track and object queries,
and the matching rules with examples for assigned bounding boxes and
losses are depicted in
[4](#fig:trackformer_query_t0t1).

<br/><br/>

<figure id="fig:trackformer_query_t0t1">
<center><img src="/assets/img/dl-series/4_trackformer_t1t2.png" style="width:100.0%"></center>
</figure>
<b>Figure 4. Training track and object queries.</b> The black boxes are
ground truth detections and the colorful, annotated boxes are
predictions from the respective output embedding. Embeddings that do not
spawn a box predict a class score smaller than the threshold. In t=0,
the most similar boxes (green and red) are matched with the two target
boxes according to matching step 2: <span
class="math inline"><em>K</em><sub>init</sub></span> (symbolized by
"<span class="math inline">/</span>"). For these boxes, we compute <span
class="math inline"><em>L</em><sub>1</sub></span> based on boxes and
class scores, and for the unmatched boxes, we compute <span
class="math inline"><em>L</em><sub>0</sub></span> solely based on the
class scores. Then we update the model parameters accordingly. The
matched output embeddings are taken over as additional input embeddings,
carrying the object IDs from the objects on the previous image. They are
matched with priority according to matching step 1: <span
class="math inline"><em>K</em><sub>track</sub></span> (symbolized by
"<span class="math inline">∩</span>"). Embedding 2’ is matched although
the bounding boxes from output embedding <span
class="math inline"><em>a</em>′</span> is more similar to the target. We
feed embeddings <span class="math inline">1′</span> and <span
class="math inline">2′</span> to the decoder in period <span
class="math inline"><em>t</em> = 2</span> (see Figure 5).

### Track query re-identification

What happens if objects are occluded
or re-enter the scene? To deal with such cases, we keep feeding
previously removed track queries for a *patience window* of
$$T_{track-reid}$$ frames into the decoder. During this window,
predictions from track ids are only considered if a classification score
higher than $$\sigma_{track-reid}$$ is reached.

<figure id="fig:trackformer_query_t2t3">
<center><img src="/assets/img/dl-series/4_trackformer_t3t4.png" style="width:100.0%"></center>
</figure>
<b>Figure 5. Training track queries with re-identification. Make sure to
compare this Figure to Figure 4.</b>
From <span class="math inline"><em>t</em> = 1</span>, we additionally
feed output embeddings <span class="math inline">1′</span> and <span
class="math inline">2′</span>, carrying the object identities for
pedestrian 1 (green) and pedestrian 2 (blue) to the encoder. However,
pedestrian 2 has left the scene between <span
class="math inline"><em>t</em> = 1</span> and <span
class="math inline"><em>t</em> = 2</span> and pedestrian 1 is occluded
by a news pedestrian with ID 3. Here, we depict the case where we have
no ground truth annotation for the occluded pedestrian. Note that the
prediction from embedding <span class="math inline">1′</span> is quite
reasonable given that pedestrian 1 is almost invisible. In contrast,
embedding <span class="math inline">2′</span> keeps predicting a
bounding box in the close to the upper right corner independent of the
image. Since there are no detections with IDs previously matched to 1 or
2, we cannot apply matching step 1: <span
class="math inline"><em>K</em><sub>track</sub></span> (symbolized by
"<span class="math inline">∩</span>") to output embeddings <span
class="math inline">1″</span> and <span class="math inline">2″</span>.
Instead, we match both outputs with the background class according to
step 2 <span class="math inline"><em>K</em><sub>leaving</sub></span>
(symbolized by "<span class="math inline">∖</span>"). Assuming the green
pedestrian would be annotated, however, we would have apply step 1:
<span class="math inline"><em>K</em><sub>track</sub></span> (symbolized
by "<span class="math inline">∩</span>") to embedding <span
class="math inline">1″</span> instead. To track the new pedestrian with
ID 3, we apply matching step 3: <span
class="math inline"><em>K</em><sub>init</sub></span> (symbolized by
"<span class="math inline">/</span>") to embedding <span
class="math inline">3</span> and update the model according to the
respective losses. Since we keep unmatched embeddings with patience, we
take over embeddings <span class="math inline">1″</span> and <span
class="math inline">2″</span> to the next frame in addition to embedding 3. This allows the model to re-identify object 1 (green) in period <span
class="math inline"><em>t</em> = 3</span>.

### Results

Trackformer achieved state-of-the-art performance in
multi-object tracking on MOT17 and MOT20 datasets. It is important to pre-train the model on large tracking datasets, to use track augmentations, and probably also to use pre-trained detection weights.<br/><br/>

<figure id="fig:object-queries">
<center><img src="/assets/img/dl-series/paper-deformable-detr.png" style="width:66.6%"></center>
</figure>

### 3) Extra. Deformable DETR as More Efficient Detector

In order to tackle these challenges, <d-cite key="zhu_deformable_2021"></d-cite> introduces a
deformable attention module as replacement for the conventional
attention module in DETR's transformer model. Drawing inspiration from
deformable convolution <d-cite key="dai_deformable_2017"></d-cite> <d-cite key="zhu_deformable_2021"></d-cite>, the
deformable attention module focuses its attention solely on a limited
set of key sampling points surrounding a reference point. By allocating
a fixed number of keys per query, <d-cite key="zhu_deformable_2021"></d-cite> can alleviate the
problems associated with convergence and feature spatial resolution by
decreasing the transformer's complexity as a function of the image
dimension to a sub-quadratic level. Furthermore, <d-cite key="zhu_deformable_2021"></d-cite>
use this module to aggregate multiple feature maps of different
resolution taken from the CNN backbone in the encoder to \"multi-scale\"
feature maps and for the object queries to aggregate the relevant
information from these maps for the detection predictions. The design is
inspired by the finding that multi-scale feature maps are crucial for
teaching image transformers to effectively represent objects depicted at
strongly distinct scales <d-cite key="lin_feature_2017"></d-cite>.
Figure [6](#fig:deformable-detr) depicts the complete Deformable DETR
architecture.<br/>

<figure id="fig:deformable-detr">
<center><img src="/assets/img/dl-series/4d-deformable-attention.png" style="width:100.0%"></center>
</figure>
<b>Figure 6. High-level Deformable DETR architecture.</b> We extract three
feature maps at different resolution levels from the CNN backbone. In
the encoder, we aggregate the information from all multi-scale feature
maps with deformable self-attention, attending only to a learned sample
of important locations around a learned reference point from every
feature map for each feature map (deformable attention). This results in
three encoder feature maps of the same dimensions. In the decoder, the
learned object queries extract features from queries and values from
themselves with conventional self-attention and from the keys from the
encoder feature maps with deformable cross-attention. Again, for each
object query, the learned reference point and a set of learned offsets
is used to only query a set of keys from the encoder. With that, we
replace the self-attention modules in the encoder and the and
cross-attention modules in the decoder with 4 heads of the deformable
attention module and learn from feature maps at different resolutions.
This makes the transformer’s complexity as a function of pixel number
sub-quadratic and allows the model to better learn objects at strongly
distinct scales. Image source: <d-cite key="zhu_deformable_2021"></d-cite>.<br/><br/>


### Multi-Head Attention Revisited

When provided with a query element
(e.g., a target word in the output sentence) and a set of key elements
(e.g., source words in the input sentence), the multi-head attention
module aggregates the key contents based on attention weights, which
gauge the compatibility of query-key pairs. In order to enable the model
to focus on diverse representation subspaces and positions, the outputs
of various attention heads are combined linearly using adjustable
weights.

Let $$q \in \Omega_q$$ denote an index for the query element represented
by feature $$z_q \in \mathbb{R}^C$$, and let $$k \in \Omega_k$$ denote an
index for the key element represented by feature $$x_k \in \mathbb{R}^C$$
with feature dimension $$C$$ and set of query and key elements $$\Omega_q$$
and $$\Omega_k$$, respectively. We compute the multi-head attention
feature for query index $$q$$ with

$$\begin{align}\text{MultiHeadAttn}(z_q, x) = \sum_{m=1}^M W_m [\sum_{k \in \Omega_k} A_{mqk} \cdot W_m^{T} x_k],\end{align}$$

where $$m$$ is index the attention head and
$$W_m \in \mathbb{R}^{C \times C_v}$$ are learned weights with
$$C_v = C/M$$. We normalize the attention weights
$$A_{mqk} \propto \text{exp}\{\frac{z_q^T U_m^T V_m x_k}{\sqrt{C_v}}\}$$
with learned weights $$U_m, V_m \in \mathbb{R}^{C \times C_v}$$ to
$$\sum_{k \in \Omega_k} A_{mqk}=1$$. In order to clarify distinct spatial
positions, the representation features, denoted as $$z_q$$ and $$x_k$$, are
formed by concatenating or summing the element contents with positional
embeddings.

### Deformable Attention

The main challenge in applying Transformer
attention to image feature maps is that it considers all potential
spatial locations. To overcome this limitation, <d-cite key="zhu_deformable_2021"></d-cite>
propose a deformable attention module. It selectively focuses on a small
number of key sampling points around a reference point, irrespective of
the spatial dimensions of the feature maps. By employing a small number
of keys per query, it can address the issues of convergence and feature
spatial resolution. In contrast to the previous notation, let
$$x \in \mathbb{R}^{C \times H \times W}$$ denote an input feature map and
let $$q$$ denote the index for a query element with feature $$z_q$$ and a
2-d reference point $$p_q$$. We calculate the deformable attention feature
with

$$\begin{align}\text{DeformAttn}(z_q, p_q, x) = \sum_{m=1}^M W_m [\sum_{k \in \Omega_k} A_{mqk} \cdot W_m^{T} x_k (p_q + \Delta p_{mqk})],\end{align}$$

where $$k$$ indexed the sampled keys and $$K   \ll  HW$$ is the total number
of sampled keys. Futher, $$\Delta p_{mqk} \in \mathbb{R}^2$$ denotes the
sampling offset and $$A_{mkq}$$ the attention weights of the
$$k^{\text{th}}$$ sampling point in the $$m^{\text{th}}$$ attention head,
respectively. The attention weights are normalized to own over the
sample keys. We feed query feature $$z_q$$ to a $$3MK$$-channel linear
projection operator. The inital $$2MK$$ channels encode the sampling
offsets $$Delta p_{mkq}$$, and the latter $$MK$$ channels are used as input
to the softmax function to compute the attention weights $$A_mqk$$.

### Multi-scale Deformable Attention

We extend the deformable attention
module to multiple differently-scaled feature maps with a few small
changes. Let $$\{x^l\}_{l=1}^L$$ denote the set of differently scaled
feature maps, where $$x^l \in \mathbb{R}^{C \times H_l \times W_l}$$ and
let $$\hat{p}_q \in [0,1]^2$$ denote the normalized reference points for
each query element element $$q$$. Then, we calculate the multi-scale
deformable attention with

$$\begin{gathered}
\text{MSDeformAttn}(z_q, \hat{p}_q, \{x^l\}_{l=1}^L) = \\
\sum_{m=1}^M W_m [\sum_{l=1}^L \sum_{k \in \Omega_k} A_{mlqk} \cdot W_m^{T} x^l (\phi_l (\hat{p}_q ) + \Delta p_{mlqk})],
\end{gathered}$$

where $$l$$ indexes the input feature and we expand the normalized
attention weights and the reference point by this dimension.
$$\hat{p}_q  \in [0,1]^2$$ are also normalized coordinates with $$(0,0)$$
and $$(1,1)$$ as top-left and bottom-right coordinates, respectively.
Function $$\phi_l(\hat{p}_q )$$ is the inverse of the normalization
function and maps $$\hat{p}_q$$ back to the respective feature map
coordinates. In constrast to the singlescale deformable attention
module, we sample $$LK$$ feature map points instead of $$K$$ points and
interact the different feature maps with each other.

### Results

On the COCO benchmark for object detection, Deformable DETR
outperforms DETR, particularly in detecting small objects, while
requiring only about one-tenth of the training epochs.