---
layout: distill
title: 3. Optimization
date: 2023-03-03
description:
categories: deep-learning
tags: optimization first-order-methods
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
  - name: Optimization
    subsections:
    - name: First order methods
    - name: Advanced first order methods
    - name: Hyperparamter optimization

---

## Optimization

In the previous section we formalized supervised learning of a
predictive model as solving the optimization problem
$$\theta^* = \arg \min_{\theta \in \Theta} C(\theta)$$, where $$\theta$$
denotes the model parameters and cost function $$C$$ represents the
average loss of all examples plus a regularization penalty. In realistic
settings, there is no closed form solution to this problem. Therefore,
we have to rely on schemes that iteratively proposes new parameters
given the previous choice or the initial guess. We want to choose a
method that proposes new candidates with a high likelihood of reducing
the loss compared to the current parameters, so we can replace them.

### First order methods
In practice, a neural network is a composition
of many small functions that are easy to differentiate. Therefore, we
can compute the gradient $$\nabla_\theta C$$ from our network via the
backpropagation technique. We will discuss this method in detail in the
next section. The gradient of our cost function is a vector of
first-order partial derivatives. It contains the direction of its
fastest increase and its magnitude is the rate of increase in that
direction. The gradient at a minimum of the loss function equals zero.
Therefore, we can use the negative gradient as search direction for
selecting the next proposal $$\theta_{i+1}$$ from current candidate
$$\theta_i$$. This is the basic idea of the gradient descent algorithm.
The method alternates between two steps: 1.) Compute the gradient
$$\nabla_\theta C(\theta)$$. 2.) Update $$\theta$$ by subtracting a small
multiple of the gradient. Many applications use very large datasets. For
instance, the number of training images in ImageNet is about 1 million.
Therefore, it is handy to approximate the loss gradient from a small
minibatch of examples (e.g. 100). With that, we can update $$\theta$$ many
times for every epoch. An epoch is one iteration over the complete
training set. In practice, a large number of approximate updates works
better than a small number of exact updates. This algorithm is called
Stochastic Gradient Descent (SGD). It is summarized in Algorithm 1.

<figure id="fig:supervised-learning">
<center><img src="/assets/img/dl-series/sgd.png" style="width:90%"></center>
</figure>

A crucial parameter for gradient descent algorithms is the learning
rate, or step size, $$\eta$$. If we set it too small, the optimization
requires too many steps. If we set it too large, the algorithm may not
converge or even diverge. As an illustration, consider the following toy
example with convex objective function: Let $$f=x^2$$ with gradient
$$\frac{df}{dx}=2x$$ and initial parameter value $$x = 1$$. With $$\eta<1$$
the algorithm finds $$x^*=0$$. However, with $$\eta=1$$, it oscillates
between $$x=-1$$ and $$x=1$$, and with $$\eta>1$$ it diverges to $$x=\infty$$.
Generally, finding a useful learning rate depends on the objective
function.

### Advanced first order methods

Oftentimes, we can achieve faster
convergence speed with modified formulations of the update direction
$$\Delta \theta$$ from Step 3 in Algorithm 1. The
two main ideas are to use weighted moving averages of all gradients so
far and to compute parameter-specific learning rates. Common variants
are Momentum <d-cite key="sutskever_importance_2013"></d-cite>, Adagrad <d-cite key="duchi_adaptive_2011"></d-cite>, RMSProp <d-cite key="hinton_lecture_2012"></d-cite> and Adam <d-cite key="kingma_adam_2014"></d-cite>. The **Momentum** update is inspired by the physics
notion of momentum. The current update direction is not determined by
the current gradient but also by the previous gradients. The impact of
the previous gradients, however, decays exponentially for every previous
iteration. Let $$\alpha$$ be the decay parameter and
$$g := \nabla_{\theta}C(\theta)$$. We then replace Step 3 by two steps:
First, we compute the \"velocity\" $$v= \alpha v + g$$ (initialized at
zero), and second, we compute the parameter update with
$$\Delta \theta = -\eta v$$. **Adagrad** introduces element-wise learning
rates for the gradient. I.e. we weigh every partial derivative with the
moving average of its squared sum. The motivation is two-fold: first the
shape of the objective function can vary between different dimensions.
Therefore, dimension-specific learning rates may improve the algorithm.
Second, we can achieve a more equal exploration of each dimension by
equipping dimensions with a history of small gradients with larger step
sizes and vice versa. Formally, Adagrad first introduces the
intermediate variable $$r = r + g \odot g$$ where $$\odot$$ denotes
elementwise multiplication. Second, it computes the gradient update with
$$\Delta \theta = -\frac{\epsilon}{\delta + \sqrt{r}} \odot g$$. In the
last expression, $$\delta$$ is a small number to avoid division by zero.
**RMSProp** introduces an additional hyperparameter to weigh Adagrad's
running average, i.e. $$r s= \rho r + (1-\rho)g \odot g$$. Lastly, the
**Adam** update combines gradient momentum and RMSProp's
dimension-specific learning rates.

### Hyperparamter optimization

In the discussion of supervised learning
and optimization, we encountered many configurations that we did not
specify. Examples are the number of hidden units $$H$$ in our neural
network, the regularization strength $$\lambda$$, or the learning rate
$$\eta$$ in gradient descent. These parameters either change the
composition of the model parameters $$\lambda$$ or impact their selection
during training. Therefore, they have to be set before a model is
trained. To this end, we have to select our hyperparameter space
$$\mathcal{H}$$. Due to restrictions of time and computational power, this
choice is often based on experiences with similar models from previous
research. Principled approaches range from model-free methods like
random search to global optimization frameworks like Bayesian
optimization. Model-free methods are simple and do not use the
evaluation history whereas global optimization techniques additionally
consider the uncertain trade-off between exploring new values and
exploiting good values that have already been found. I recommend the
textbook chapter by <d-cite key="feurer_hyperparameter_nodate"></d-cite> for more explanations
and concrete examples.


## Citation

In case you like this series, cite it with:
<pre tabindex="0"><code  class="language-latex">@misc{stenzel2023deeplearning,
  title   = &quot;Deep Learning Series&quot;,
  author  = &quot;Stenzel, Tobias&quot;,
  year    = &quot;2023&quot;,
  url     = &quot;https://www.tobiasstenzel.com/blog/2023/dl-overview/
}
</code></pre>