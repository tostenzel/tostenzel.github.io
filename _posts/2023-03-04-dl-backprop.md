---
layout: distill
title: 4. Backpropagation
date: 2023-03-04
description: 💡 <b><i>Includes a nice detail about why we don't "frontpropagate". ;-) </i></b>
categories: backprop reverse-accumulation compute-graph
giscus_comments: true
related_posts: true

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
  - name: Backpropagation
    subsections:
    - name: Toy example
    - name: Vector-valued intermediate functions
    - name: Reverse accumulation
    - name: Implementation

---

## Backpropagation

We learned that we can find a mapping $$f \in \mathcal{F}$$ that maps
features X to outcome Y consistent with the data by minimizing the cost
function with repeated gradient evaluations using a gradient descent
optimizer.

We compute the gradient with backpropagation. This algorithm allows us
to efficiently compute gradients of functions that 1.) are
scalar-valued, 2.) have many input parameters, and 3.) can be decomposed
into simpler intermediate functions that are differentiable. The
mathematical formula of backpropagation is inspired by the third
property. It is the gradient computation via recursive applications of
the chain rule from calculus. The algorithmic order is inspired by the
first two properties. The idea is to efficiently compute the resulting
derivative starting from the intermediate functions on the parameter
instead of the cost function side.

### Toy example

Recall that we would like to compute the gradient of
cost function $$C(\theta)$$, which takes not only parameters $$\theta$$ but
also multiple examples $$(x_i, y_i)$$ as input. Specifically, our
first-order optimizer requires the gradient of the cost function with
respect to the model parameters $$\nabla_{\theta}C(\theta)$$ in order to
update the parameters $$\theta$$. Let us consider the following example.
Let $$C(\theta_1, \theta_2)=\theta_1 \theta_2 + \tanh (\theta_1)$$ be the
cost function of a neural network with parameters $$\theta_1$$ and
$$\theta_2$$. Figure [1](#fig:toy_graph) shows that we can view this equation as a
computational graph with cost function $$C$$ as root and parameters as
leaf nodes. We introduce intermediate variables to write $$C$$ as a
sequence of the intermediate functions $$z_3=\tanh (\theta_1)$$,
$$z_4=\theta_1 \theta_2$$, $$z_5=z_3+z_4$$, and $$C(\theta)=z_5$$. For ease of
notation, we will later also write $$\theta_1,\theta_2$$ and $$C(\theta)$$
as $$z_1, z_2$$ and $$z_6$$. We can write the gradient of the cost function
with respect to each parameter as a combination of the gradients of its
parent nodes using the chain rule from calculus and the fact that
multiple occurrences of a term add up in its derivative. This is shown
in Equation [1](#eq:dtheta_1) and
[2](#eq:dtheta_2):

$$\begin{align}
\frac{\partial C}{\partial \theta_1}=\frac{\partial C}{\partial z_5} \bigg( \frac{\partial z_5}{\partial z_3} \frac{\partial z_3}{\partial \theta_1} + \frac{\partial z_5}{\partial z_4} \frac{\partial z_4}{\partial \theta_1} \bigg) = 1 - \tanh ( \theta_1)^2 + \theta_2,
\label{eq:dtheta_1}\end{align}$$

$$\begin{align}
\label{eq:dtheta_2}
\quad \frac{\partial C}{\partial \theta_2}=\frac{\partial C}{\partial z_5}\frac{\partial z_5}{\partial z_4}\frac{\partial z_4}{\partial \theta_2}   =\theta_1.\end{align}$$

<figure id="fig:toy_graph">
<center><img src="/assets/img/dl-series/compute-graph.png" style="width:75%"></center>
<figcaption><b>Figure 1. Computational graph for a toy example of a neural network’s
forward pass without vector-valued intermediate functions, data,
regularization and loss.</b> The function given by the graph is <span
class="math inline"><em>C</em>(<em>θ</em><sub>1</sub>,<em>θ</em><sub>2</sub>) = <em>θ</em><sub>1</sub><em>θ</em><sub>2</sub> + tanh (<em>θ</em><sub>1</sub>)</span>.
Intermediate functions that are relevant for deriving the gradient with
the chain rule are denoted by <span
class="math inline"><em>z</em><sub><em>i</em></sub></span> with <span
class="math inline"><em>i</em> ∈ {1, ...., 6}</span>. The edges along
which the intermediate evaluations move forth during the forward pass
and along which the gradients move back during the backward pass are
denoted by <span
class="math inline"><em>e</em><sub><em>i</em>, <em>j</em></sub></span>
with <span
class="math inline"><em>i</em>, <em>j</em> ∈ {1, ..., 6}</span>. We can
view two edges leaving one node as a a shortcut depiction for applying
the <span class="math inline"><code>duplicate</code></span> or fork
operation along both edges.</figcaption>
</figure>

### Vector-valued intermediate functions
The toy example deviates from
a realistic neural network application in a few aspects. One of these
aspects is that we usually consider vectors of large parameter groups.
Another aspect is that the intermediate functions that transform the
parameter vector $$\theta$$ step-by-step are usually vector-valued (unlike
the final cost function). Apart of confluence operators like \"+\" and
\"\*\" that usually combine different parameter groups, the final
gradients can be written as a sequence of \"local\" gradient
computations. Due to vector-valued functions and the number of
parameters, these computations are written as matrix multiplications. We
look at this extension more formally. To this end, let $$z_0$$ be the
input vector which we transform through a series of functions be
$$z_i = f_i(z_{i-1})$$ where $$i=1,...,k$$ and only the last $$z_i$$ can be
scalar. Assuming that the functions $$f_i$$ are once differentiable, we
can compute the Jacobian matrix $$\frac{\partial z_i}{\partial z_{i-1}}$$
for all intermediate functions. This will give us the values of the
first derivative of every output dimension of $$z_{i}$$ depending on each
single input dimension of $$z_{i-1}$$. From the multivariable chain rule,
we obtain the result that the gradient of our final function with
respect to input vector equals the product of all intermediate
Jacobians:
$$\frac{\partial z_k}{\partial z_{0}} = \prod_{i=1}^k \frac{\partial z_i}{\partial z_{i-1}}$$.

### Reverse accumulation
We can compute
$$\frac{\partial C}{\partial \theta}$$ in two different ways. In this
section we learn that one approach is much more efficient. The reason is
the the structure of our problem: we minimize a scalar-valued function
with a large number of parameters. One approach is to compute the
gradient from parameters to output:
$$\frac{\partial C}{\partial \theta} = \frac{\partial z_k}{\partial z_{k-1}} \cdot ... \cdot \frac{\partial z_2}{\partial z_1}$$.
This is called forward accumulation. The other approach, reverse
accumulation, is to compute the gradients from output to parameters:
$$\frac{\partial C}{\partial \theta} = \frac{\partial z_2}{\partial z_1}\cdot   ... \cdot \frac{\partial z_k}{\partial z_{k-1}}$$.
The first approach is less efficient but more intuitive. It is more
intuitive because the derivatives can be computed in sync with the
evaluation steps. This makes it easier to think about how confluence
operations like \"$$+$$\", \"$$*$$\", the fork operation `duplicate`, or
filter operations like `max` or `average` transform the gradients. We
will later see that it is crucial to use these operations in a careful
way in the architecture design of neural networks because they can have
large effects on the model performance. In reverse accumulation, we fix
the dependent variable to be differentiated and compute the derivative
with respect to each intermediate function recursively. Table
[1](#tab:reverse) shows
how we can calculate the two gradients of our toy example step-by-step
with this method. There are two important things to note. First,
computing the backward operations corresponding to the forward
operations is prone to error. This is one reason why this should be done
automatically by a graph-based computer program. Second, we can re-use
our intermediate computations $$\bar{z}_3$$ to $$\bar{z}_5$$ for both
gradients based only on one evaluation of the cost function
$$C(\theta) = z_6$$. In realistic neural network applications, we are able
to re-use a large number of intermediate results based on only one
forward evaluation for an even larger number of input parameters. In
contrast, with forward-mode accumulation, computing the gradient
requires to evaluate each intermediate function with the whole parameter
vector. Although re-using the intermediate gradients requires more
storage, the reverse-mode accumulation strategy is much more efficient
for functions like neural networks where the number of output values is
much smaller than the number of input values.

<figure id="tab:reverse">
<center><img src="/assets/img/dl-series/table-backprop.png" style="width:80%"></center>
<figcaption><b>Table 1. Gradient computations in reverse accumulation mode for toy
example <span
class="math inline"><em>C</em>(<em>θ</em><sub>1</sub>,<em>θ</em><sub>2</sub>) = <em>θ</em><sub>1</sub><em>θ</em><sub>2</sub> + tanh <em>θ</em><sub>1</sub></span>.</b>
The left column shows the evaluation and the right column depicts the
derivation steps. <span class="math inline"><em>z̄</em></span> denotes
the derivative of the cost function with respect to an intermediate
expression, i.e. <span class="math inline">$$\frac{\partial C}{\partial
z}$$</span>. In the backward pass, we first evaluate the scalar-cost
function and use its derivative with respect to <span
class="math inline"><em>z</em><sub>5</sub></span> at the top of the
graph. Here we assume <span
class="math inline"><em>z̄</em><sub>5</sub></span> equals <span
class="math inline">1</span>. In the next step, we combine the chained
gradients of the intermediate expressions according to the respective
backward functions of the <code>duplicate</code> and the confluence
operations "<span class="math inline">+</span>" and "<span
class="math inline">*</span>" from top to bottom. Addition distributes
the upstream gradient down to all its inputs. Multiplication passes the
upstream gradient multiplied with the <em>other</em> input back. I.e.,
<span class="math inline">$$\frac{\partial (z_3 + z_4)}{\partial z_{4}} =
1, \frac{\partial (z_1 * z_2)}{\partial z_{2}} = z_1$$</span>, and the
backward function of <span
class="math inline">(<em>z</em><sub>1, 1</sub>,<em>z</em><sub>1, 3</sub>) = <code>duplicate</code>(<em>z</em><sub>1</sub>)</span>
equals <span class="math inline">$$\frac{\partial z_3}{\partial z_{1}} +
\frac{\partial z_4}{\partial z_{1}}$$</span>, where <span
class="math inline"><em>z</em><sub>1, 1</sub></span> and <span
class="math inline"><em>z</em><sub>1, 3</sub></span> denote abbreviated
nodes from the implicit duplication operation in the first and third
evaluation step.</figcaption>
</figure>

### Implementation
In our toy example, we have already discovered that
it is easier to think of the evaluation and differentiation of a neural
*network* in terms of a directed graph of operations instead of a linear
sequence of function applications. In this graph, the nodes stand for
differentiable operations that take a number of vectors from its
incoming edges, transforms and potentially interrelates them, and sends
the result to the next nodes along its outgoing edges. The graph
abstraction translates to most implementations of the backpropagation
algorithm. A *Graph* object keeps the connections (e.g. `duplicate` or
$$+$$ ) between the nodes and the collection of operations (e.g. $$\tanh$$).
Both Node and Graph objects implement a `forward()` and a `backward()`
method. The Graph's `forward()` calls the Nodes' `forward()` methods in
their topological order. With that, every Node computes its operation on
its input, and the Graph sends it to the next Node. The Graph's
`backward()` iterates over the nodes in reverse topological order and
calls their `backward()` methods. In the backward pass, each Node is
given the gradient of the cost function with respect to its output and
it returns the \"chained\" gradients with respect to all its inputs.
\"Chained\" means that the Node multiplies the Jacobian that it received
from its parent nodes with its own local Jacobian. The Graph sends the
resulting product to the Nodes' children and the process repeats until
the recursion ends at the last computation which includes the data and
the current parameter values. At last, the optimizer updates the
parameter vector based on the final gradient (Step 3 in Algorithm [1](https://www.tobiasstenzel.com/blog/2023/dl-optimization/#first-order-methods). Note
that the implemented compute graph for Figure
[1](#fig:toy_graph) would in practice be larger and rudimentary because we can decompose
many operations, like divide or subtract in tanh, into more basic
operations.


## Citation

In case you like this series, cite it with:
<pre tabindex="0"><code  class="language-latex">@misc{stenzel2023deeplearning,
  title   = &quot;Deep Learning Series&quot;,
  author  = &quot;Stenzel, Tobias&quot;,
  year    = &quot;2023&quot;,
  url     = &quot;https://www.tobiasstenzel.com/blog/2023/dl-overview/
}
</code></pre>