---
layout: distill
title: 1. Overview
date: 2026-10-17
description: 🧭 <b><i>Start here. Why these notes began as a hand-drawn A2 atlas, and how to read the series.</i></b>
#categories: data-structures-algorithms
tags: dsa-fundamentals motivation big-picture knowledge-graph target-group credits
giscus_comments: true
related_posts: true

authors:
  - name: Tobias Stenzel
    url: "https://www.tobiasstenzel.com"
    affiliations:
      name: N/A

toc:
  - name: Overview
  - name: Why an atlas?
  - name: The series
  - name: The big picture
    subsections:
    - name: The spine
    - name: What you are actually learning
    - name: The master cost table
    - name: Complexity tiers
  - name: Target group
  - name: Credits
  - name: Citation
---

## Overview

This series is a tour of the data structures and algorithms (DSA) that come up again and again in
interviews and in day-to-day engineering. It grew out of a personal project: I wanted a single
reference that made the *connections* between concepts the star of the show, not an afterthought.

The notes did not start life as blog posts. They started as a hand-drawn **DIN A2 "atlas"** — big
sheets of paper, one master map plus several zoom-in sheets, with the whole toolbox laid out at once
so I could see how every piece related to every other piece. This series linearizes that atlas into
posts you can read in order, while deliberately keeping the connection web intact.

## Why an atlas?

Deep learning fits a blog series perfectly. DSA fights it. The reason is the *shape of the knowledge*,
and it is worth making explicit because it is exactly what the A2 format was built to exploit.

**Deep learning is a stack; DSA is a map.**

<center><img src="/assets/img/dsa-series/knowledge_graph_contrast.png" style="width:100%"></center>
<p class="caption"><i>The two knowledge graphs have different shapes. Left: deep learning is a near-linear
dependency chain &mdash; you cannot grasp backprop without the computational graph, or the transformer
without RNN + attention. Right: DSA is a web of a few primitives recombined into many tools, where the
edges carry the knowledge.</i></p>

Look at my [Deep Learning series](https://www.tobiasstenzel.com/blog/2023/dl-overview/): supervised
learning leads to optimization leads to backprop leads to the feedforward net, and only then do CNNs,
RNNs, and the transformer branch off. It is a **dependency chain** with one canonical reading order
and a small fan-out at the very end. A linear medium &mdash; a blog, a book, a lecture &mdash; matches
that shape *losslessly*: the chapter order simply **is** the dependency order.

DSA is the opposite. A handful of primitives &mdash; recursion, a stack, a queue, a heap, sorting,
union-find &mdash; **recombine** into dozens of tools. A heap powers Dijkstra, Prim, heapsort, and
top-$$K$$. Recursion powers backtracking, mergesort, and tree and graph DFS. The graph is a **web**
with high fan-in and fan-out, and no single correct traversal.

Five differences follow from that, and each one is a reason the atlas format earns its space:

- **Topology &mdash; a chain vs. recombination.** Deep learning is a ladder you climb once. DSA is a
  small set of parts bolted together many ways. A ladder wants a sequence; a lattice wants a surface.
- **The knowledge is comparative, not cumulative.** In deep learning, understanding *accrues*: each
  post adds a layer you then treat as settled. In DSA the payload is the *comparison between siblings*
  &mdash; array vs. linked list, BFS vs. DFS, hash map vs. balanced tree, quicksort vs. mergesort vs.
  heapsort. A heap in isolation is nearly useless; "heap *vs.* sorted array *vs.* BST for repeatedly
  pulling the minimum" is the actual knowledge. Comparison is inherently two-dimensional.
- **No canonical reading order.** Deep learning has prerequisites that force an order. DSA has only
  weak ordering over a lot of lateral structure. Any linear medium must *pick* an order, and in doing
  so it demotes every lateral edge to a "see the chapter on X" reference that the reader has to
  re-stitch in their head. Serializing a web is lossy; serializing a chain is not.
- **One shared coordinate system.** Every DSA tool lands on the *same axes*: Big-O for access, search,
  insert, delete, and space, plus the time&ndash;space&ndash;ordering trade. The whole field embeds
  into one cost table, which is a two-dimensional object that wants a two-dimensional surface.
- **Leaky trade-offs vs. clean abstraction.** Deep learning has clean boundaries &mdash; once you
  understand a layer you black-box it, which is what makes it chunkable into sequential posts. DSA
  resists that, because the leaks are the point: you pick quicksort over mergesort for cache behavior
  and in-place operation, an implementation detail that crosses the interface. Right choices require
  reasoning *across* boundaries, so you want both sides visible at once.

An A2 sheet is large enough to show several **orthogonal projections of the same concept, side by
side**: a cost table, a dependency web, the "why," and the trade-off, all in one glance. The eye
integrates them for free instead of your working memory juggling three representations across three
pages. That is the whole bet.

**The honest tension:** turning the atlas back into a linear series re-imposes the very constraint the
A2 was escaping. So this series keeps the web as first-class content &mdash; the connection diagrams,
the shared cost tables, and the "this links to X because…" notes are not decoration. They are the
2D edges, flattened but preserved. Read them as load-bearing.

## The series

The atlas had a master "Big Picture" sheet plus detail sheets. Here is the reading order:

1. [Overview](https://www.tobiasstenzel.com/blog/2026/dsa-overview/) &mdash; you are here.
2. [Foundations](https://www.tobiasstenzel.com/blog/2026/dsa-foundations/) &mdash; how to reason: Big-O, recursion, invariants, correctness.
3. [Linear Structures](https://www.tobiasstenzel.com/blog/2026/dsa-linear/) &mdash; dynamic array, hash map/set, linked list, stack, queue/deque.
4. [Array &amp; String Techniques](https://www.tobiasstenzel.com/blog/2026/dsa-techniques/) &mdash; two pointers, sliding window, prefix sums, merge intervals, bit tricks.
5. [Hierarchical Structures](https://www.tobiasstenzel.com/blog/2026/dsa-hierarchical/) &mdash; trees/BST/balanced, heaps, tries, union-find.
6. [Recursion &amp; Backtracking](https://www.tobiasstenzel.com/blog/2026/dsa-recursion-backtracking/) &mdash; the recursion model and the backtracking template.
7. [Sorting &amp; Searching](https://www.tobiasstenzel.com/blog/2026/dsa-sorting-searching/) &mdash; sorts, binary search, binary search on the answer.
8. [Graphs](https://www.tobiasstenzel.com/blog/2026/dsa-graphs/) &mdash; representation, BFS/DFS, topological sort, shortest paths, MST. The capstone that reuses everything.

## The big picture

### The spine

One idea threads every post:

> **Almost all of DSA is "store a collection so you can answer a kind of question fast."**
> You pick a **data structure** by which operations must be cheap, and an **algorithm** by what
> question you ask. Everything is a trade between *time*, *space*, and *what stays ordered*.

### What you are actually learning

Every program keeps a collection of data and asks questions of it: *is this here? what is the largest?
who connects to whom?* The slow way is to scan everything each time. This field is the set of tricks
for answering the question you ask most often without that scan. Each row below is one trick: a need
you keep hitting, the slow first attempt, and the tool that drops it to a faster complexity class.

| What you keep needing to do | What you would write first | The upgrade you learn | The leap |
| --- | --- | --- | --- |
| Check if an item is in a collection | scan a list, $$O(n)$$ | hashing (`set`/`dict`) | $$O(n) \to O(1)$$ |
| Keep a changing set of keys sorted | re-sort after each change | sorting + binary search | search $$O(n) \to O(\log n)$$ |
| Pull the smallest item, over and over | re-scan for the max each time | heap | $$O(n) \to O(\log n)$$ per op |
| Tell if two items are in the same group | re-scan the whole group each time | union-find | per op $$O(n) \to$$ near-$$O(1)$$ |
| Find a path between two nodes | ad-hoc loops that revisit or loop forever | graph traversal (BFS/DFS) | may loop $$\to O(V+E)$$ |
| Re-answer a query over a sliding range | recompute the range each time | sliding window / prefix sums | $$O(n) \to O(1)$$ per query |
| Enumerate all combinations of choices | deeply nested loops | recursion / backtracking | nested loops $$\to$$ one prunable search |

### The master cost table

The whole toolbox on one set of axes. Costs are typical-case; "avg" means average case (hashing is
$$O(1)$$ in practice and only degrades on pathological keys).

| Structure | Python | Access | Search | Insert | Delete | Space |
| --- | --- | --- | --- | --- | --- | --- |
| Dynamic array | `list` | $$O(1)$$ | $$O(n)$$ | $$O(1)$$* end / $$O(n)$$ front | $$O(n)$$ | $$O(n)$$ |
| Deque (both ends) | `collections.deque` | $$O(n)$$ | $$O(n)$$ | $$O(1)$$ both ends | $$O(1)$$ ends | $$O(n)$$ |
| Hash map / set | `dict` / `set` | — | $$O(1)$$ avg | $$O(1)$$ avg | $$O(1)$$ avg | $$O(n)$$ |
| BST (balanced) | `sortedcontainers` | $$O(\log n)$$ | $$O(\log n)$$ | $$O(\log n)$$ | $$O(\log n)$$ | $$O(n)$$ |
| Binary heap | `heapq` | peek $$O(1)$$ | $$O(n)$$ | push $$O(\log n)$$ | pop-min $$O(\log n)$$ | $$O(n)$$ |
| Trie (prefix tree) | nested `dict`s | — | $$O(L)$$ | $$O(L)$$ | $$O(L)$$ | $$O(\Sigma L)$$ |
| Union-Find | `list` of parents | — | find $$O(\alpha n)$$ | union $$O(\alpha n)$$ | — | $$O(n)$$ |

(* append is $$O(1)$$ *amortized* &mdash; rare doublings averaged out. $$L$$ = key length,
$$\Sigma L$$ = total characters stored, $$\alpha$$ = inverse Ackermann, effectively constant.)

### Complexity tiers

A gut-feel scale for "is this fast enough?", reused on every post:

| Tier | Big-O | Read as |
| --- | --- | --- |
| 🟢 | $$O(1)$$ | instant: hash lookup |
| 🟢 | $$O(\log n)$$ | stable: heap op, binary search |
| 🟡 | $$O(n)$$ | fair: one pass |
| 🟡 | $$O(n \log n)$$ | standard: the price of sorting |
| 🔴 | $$O(n^2)$$ | dangerous: usually a missing hash map/set |
| ⚫ | $$O(2^n)$$, $$O(n!)$$ | brute force: prune via backtracking |

## Target group

The content is dense. I go into detail on the main ideas, but these posts are not a gentle first
introduction. I recommend the series to three groups:

1. students taking an algorithms class right now, as complementary material;
2. people refreshing knowledge they already have;
3. engineers who want to connect scattered pieces into one mental model.

## Credits

This series is a recompilation and re-framing of standard material. My main references are MIT's
6.006 *Introduction to Algorithms* lecture notes, the NeetCode problem set and its "Blind 75", and
CLRS (*Introduction to Algorithms*) as the canonical text. The figures are generated with matplotlib
so they can be regenerated and corrected rather than hand-pasted.

## Citation

If you find this series useful, cite it with:
<pre tabindex="0"><code class="language-latex">@misc{stenzel2026dsa,
  title   = &quot;DSA Atlas Series&quot;,
  author  = &quot;Stenzel, Tobias&quot;,
  year    = &quot;2026&quot;,
  url     = &quot;https://www.tobiasstenzel.com/blog/2026/dsa-overview/&quot;
}
</code></pre>
