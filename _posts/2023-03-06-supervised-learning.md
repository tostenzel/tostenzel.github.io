---
layout: post
title: Part A — Supervised Learning
date: 2023-03-06
description: an example of a blog post with giscus comments
categories: deep-learning-series deep-learning supervised-learning
giscus_comments: true
related_posts: true
toc: true

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Supervised Learning
    subsections:
    - name: Loss function
    - name: Regularization
    - name: Model Validation
    - name: Example — Linear regression
    - name: Example — Neural network regression
    - name: Example — Neural network classification
  - name: Summary

---

## Supervised Learning

Prediction rules based on some specific set of information can be
written as a mapping $$f:X \rightarrow Y$$, where $$X$$ is an input space
and $$Y$$ is an output space. To recognize a dog on a photo, for example,
$$X$$ would be the space of images and $$Y$$ would be a probability interval
$$[0,1]$$ for the presence of a dog. However, it is oftentimes very
difficult to find an explicit function $$f$$ from theoretical
considerations about the problem. For problems where it is easy to find
many examples $$(x,y) \in X \times Y$$, the supervised learning approach
is usually well-suited. In our example, the requirement would be a
dataset that consists of a large number of images which are annotated
with presence or absence of a dog.

### Loss function

To be concrete, our dataset
$$\{(x_1,y_1),...(x_n,y_n)\}$$ includes $$n$$ examples. Our theoretical
assumption about the data is that these examples are drawn from a data
generating distribution $$D$$ with independent and identically distributed
(i.i.d.) random variables, i.e.
$$(x_i, y_i) \overset{\mathrm{iid}}{\sim} D$$. In the supervised learning
context, learning the mapping $$f:X \rightarrow Y$$ means selecting the
function $$f$$ from a set of candidate functions $$\mathcal{F}$$ so that $$f$$
yields the best predictions for $$y$$ given any $$x$$. I.e., we want to find
the best approximation for $$f(Y|X)$$. We achieve this by selecting a
scalar-valued loss function $$L(\hat{y},y)$$ which measures the difference
between prediction $$\hat{y}_i$$ and actual outcome $$y_i$$. In theory, our
objective is to find the function with the lowest expected loss for all
examples from the data generating distribution $$D$$. In practice,
however, we have to rely on our dataset. Therefore, we approximate our
theoretical objective using the assumption that our data is drawn from
an i.i.d. distribution. We search for function $$f^*$$ with the lowest
average loss over the data: 

$$\begin{align}
\label{eq:loss}
f^* \approx \arg \min_{f \in \mathcal{F}} \frac{1}{n} \sum^n_{i=1} L \big  ( \ f(x_i), y_i \big  ).
\end{align}$$

### Regularization

Oftentimes, the i.i.d. assumption is too strong. In
this case, we usually do not achieve the best result for predicting
unseen outcomes with a function that is optimized solely with regards to
our dataset. In short, this function does not necessarily generalize
well. A further issue is how to choose between multiple functions with
the same minimal loss. The approach that addresses both problems at once
is regularization. The idea is to add a regularization penalty to our
objective function from which we obtain our predictor. The penalization
criterion $$R(f)$$ is function complexity. We thus search for the function
that fits the data best *and* has a low complexity:

$$\begin{align}
\label{eq:cost}
f^* \approx \arg \min_{f \in \mathcal{F}} \frac{1}{n} \sum^n_{i=1} L \big  (f(x_i), y_i \big  ) + R(f),
\end{align}$$
where $$R(f)$$ is a scalar-valued function. With regularization, we can
achieve a better generalization and choose between functions that
achieve a similar loss.

Frequently, we have already selected our model but not its parameters.
Hence, we take function $$f$$ as given but we want to learn the best
parameters $$\theta$$ for this function. In this situation, the concept of
loss and regularization directly translates from finding the optimal
function to finding the optimal parameters: 

$$\begin{align}
\label{eq:params}
\theta^* \approx \arg \min_{\theta \in \Theta} \frac{1}{n} \sum^n_{i=1} L \big (f(x_i;\theta), y_i \big  ) + R(\theta),
\end{align}$$

where $$\Theta$$ denotes the parameter space. Common examples for
$$R(\theta)$$ are multiples of vector norms. In this setting, choices like
the model, the loss or the regularization are called hyperparameters.
These are parameters that are set before the actual model parameters are
learned. Hyperparameter choices are oftentimes critical to the quality
of the learned model. There are several other ways to prevent
overfitting the model besides regularization. Examples are choosing
simpler models, stopping the optimization process early, changing or
disabling some model units during training (dropout), and dataset
augmentations.

### Model validation

How do we test whether our model generalizes well
to unseen data? The usual machine learning approach is as follows: at
first, we split our examples in three groups: a large training, and
smaller validation and test sets. Secondly, we learn the model
parameters with the training data. The third step is to compute
evaluation metrics for this model from the unseen test data. In general,
we want to minimize the distance between the prediction and the target
vector. For classification tasks, we can use evaluation metrics based on
the confusion matrix. This enables us to identify single classes which
are more difficult to predict for the model. Optionally, we can repeat
the third step multiple times for different hyperparameter
configurations and compare the generalization errors. Then, we evaluate
the best model once again on the validation data and report its
performance. We will briefly discuss hyperparameter selection in the end
of the optimization section.

### Example — Linear regression

Assume we have a dataset with 100
observations $$(n=100)$$ of two features each $$(m=2, X=\mathbb{R}^2)$$ and
a scalar annotation $$(Y=\mathbb{R})$$. According to the input and output
space, we choose to restrict the candidate class $$\mathcal{F}$$ to the
family of linear functions, i.e.
$$\mathcal{F}=\{w^Tx + b | w \in \mathbb{R}^2, b\in \mathbb{R}$$. With
this choice, we set our hypothesis space from a class of functions to
the set of three parameters $$\theta=\{w_1, w_2, b\}$$ with
$$w=[w_1, w_2]$$. Next, I list two common hyperparameter choices. First,
the squared difference between predicted value and target,
$$L(\hat{y}, y)=(\hat{y} - y)^2$$, is a common loss function. Second, the
L2 norm of the weights multiplied with importance parameter $$\lambda$$ is
a typical regularizer choice, i.e. $$R(w,b)=\lambda(w_1^2 + w_2^2)$$. The
L2 norm counteracts extreme weights and an excessive effect of one
single weight on the prediction $$\hat{y}$$. Taken together, our objective
is

$$\begin{align}
\theta^* =  \arg \min_{w,b} \underbrace{
    \Bigg [ \frac{1}{n} \sum^n_{i=1} (w^T x_i + b - y_i)^2 \Bigg ]}_\text{data fitting}
    + \underbrace{\Bigg [ \lambda (w_1^2 + w_2^2) \Bigg ]}_\text{regularization}.
\end{align}$$

### Example — Neural network regression

Perhaps the relationship between
features $$x_1, x_2$$ and the scalar target $$y$$ is not liner and there are
interactions between the two features. A model class that is well-known
for its theoretical capabilities of approximating continuous functions
are feedforward neural networks (FNN). As a preview, we can extend the
previous example to a specific FNN called neural network regression. It
has the form $$f(x;\theta)= w_2 \text{ tanh} (W_1^T x + b_1) + b_2$$ with
parameters $$\theta=\{W_1, b_2, w_1, b_1\}$$. $$W_1$$ is matrix with
dimension $$H \times 2$$, $$b_1$$ and $$w_2$$ are both vectors of length $$H$$,
and $$b_2$$ is a scalar. $$H$$ is an integer-type hyperparameter.
$$W_1^T x + b_1$$ is often called a hidden layer. Its elements, or
neurons, are outputs of multiplicative interactions between elements
from previous layers, in this case the two inputs $$x_1$$ and $$x_2$$. tanh
denotes the hyperbolic tangent that squashes elements from the
real-valued domain to the interval \[-1,1\]. It is applied element-wise
and introduces non-linearity to the model. The objective is given by:  

$$\begin{align}
    \theta^* =  \arg \min_{W_1, b_2, w_1, b_1} \underbrace{
    \Bigg [ \frac{1}{n} \sum^n_{i=1} \big (w_2 \text{ tanh}  ( W_1^T x_i + b_1 ) + b_2 - y_i \big)^2 \Bigg ]}_\text{data fitting}
    + \underbrace{\Bigg [ \lambda \big (||W_1||^2_2 + ||w_2||_2^2 \big ) \Bigg ]}_\text{regularization}.
\end{align}$$

### Example — Neural network classification

Oftentimes, we do not want
to predict a real number but we want to predict whether an input
corresponds to an output of a specific class $$k$$ of $$K$$ possible
classes. For instance, we may want to predict whether a dog, a, cat, or
a budgie is shown in a picture. To formalize this problem, we encode our
classes as non-negative integers starting at 0. We also encode our
output as a vector of $$|K|$$ zeros, where only the $$k^*$$-th element
representing the actual class encoded as $$k^*$$ is set to one. Using the
previous order, a picture that shows a dog is encoded as $$y=[1, 0, 0]$$.
We can achieve a model that predicts a vector of this form with two
simple adjustments of the neural network regression model. The first
adjustment is replacing vector $$w_2$$ by matrix $$W_2$$ with dimensions
$$K \times H$$. This gives us a real-valued vector $$z$$ with one
real-valued element for every class. The second adjustment is that we
compute class probabilities by applying the softmax function to $$z$$,
i.e. $$\hat{p}_k = e^{z_k} / \sum_{i=1}^K e^{z_i}$$. Handy properties of
the softmax function in the probability context are, first, that every
element is mapped to $$[0,1]$$, and second, that the sum of all elements
is normalized to one. Hence, our class prediction would be the class
that is represented by the largest element in vector $$\hat{p}$$. For
instance, with $$K=3$$ and $$\hat{p}=[0.1, 0.7, 0.2]$$, our prediction
$$\hat{y}$$ equals $$[0, 1, 0]$$. The most common loss function for
classification is the cross-entropy loss. It takes the predicted class
probabilities $$\hat{p}$$ instead of the predicted class vector $$\hat{y}$$.
We denote both options by output $$o$$:  

$$\begin{align}
L(o,y) = L(\hat{p},y) = - \sum_{k=1}^K y_k \log \hat{p}_k = -\log  \hat{p}_{k=k^*}.
\end{align}$$
The first equality is the cross-entropy definition for two distribution,
i.e $$H(q,r) = - \sum_x q(x) \log r(x)$$. The second equality shows that
the loss equals the negative log probability from vector $$p$$ at the
position of the actual class $$k^*$$. It follows from the fact that target
distribution is degenerate. This means that only the true class has
probability one. Additional motivation for choosing the cross-entropy
for classification problems is that minimizing this loss is equivalent
to maximizing the likelihood of observing model parameters $$\theta$$
conditional on predicting the true class label.
  

## Summary

Supervised learning requires a dataset of $$n$$ examples
$$\{(x_1,y_1)$$, \..., $$(x_n,y_n)\}$$, where $$(x_i,y_i) \in X \times Y$$.
$$y_i$$ represents the annotation that we want to predict based on
features $$x_i$$. Finding a mapping $$f:X \rightarrow Y$$ is a two-step
process. First, we have to formalize the problem by choosing

1.  The search space of functions $$\mathcal{F}$$ with
    $$f \in \mathcal{F}$$.

2.  The scalar-valued loss function $$L(\hat{y}, y)$$ that measures the
    difference between the network's predictions
    $$\hat{y} = f_{\theta}(x)$$ and the target $$y$$.

3.  The scalar term $$R(f)$$ that penalizes overly complex functions.

In deep learning without architecture search, the space of functions is
one neural network $$f_\theta$$ with some parameters $$\theta \in \Theta$$.
Putting these parts together, the second step is to find these
parameters from the optimization problem
$$\theta^* = \arg \min_{\theta \in \Theta} C(\theta)$$ with
$$C(\theta) = \frac{1}{n} \sum^n_{i=1} L \big (f_\theta), y_i \big  ) + R(f_{\theta})$$.
We call $$C(\theta)$$ the cost function.
  
  
  

<figure id="fig:supervised-learning">
{% include figure.html path="assets/img/dl-series/supervised-learning.png" class="img-fluid rounded z-depth-1" %}
<figcaption><b>Figure 1: Computational graph for a general supervised learning
approach.</b> Examples <span
class="math inline">{<em>x</em><sub><em>i</em></sub>}<sub><em>i</em> = 1</sub><sup><em>n</em></sup></span>
and parameters <span class="math inline"><em>θ</em></span> are taken by
model f to predict the targets by <span
class="math inline">{<em>ŷ</em><sub><em>i</em></sub>}<sub><em>i</em> = 1</sub><sup><em>n</em></sup></span>.
Data loss L computes the difference between the predictions and the
targets <span
class="math inline">{<em>y</em><sub><em>i</em></sub>}<sub><em>i</em> = 1</sub><sup><em>n</em></sup></span>.
Regularization loss R penalizes extreme parameters. The sum of both
penalties is given by cost C. Oftentimes we use predicted class
probabilities <span class="math inline"><em>p̂</em></span> instead of
(rounded) predictions.</figcaption>
</figure>

