---
layout: distill
title: 4. Array & String Techniques
date: 2026-10-20
description: "🪟 <b><i>The moves that turn O(n²) into O(n): two pointers, sliding window, prefix sums, intervals, bit tricks.</i></b>"
tags: dsa-fundamentals two-pointers sliding-window prefix-sums intervals bit-tricks
giscus_comments: true
related_posts: true

authors:
  - name: Tobias Stenzel
    url: "https://www.tobiasstenzel.com"
    affiliations:
      name: N/A

toc:
  - name: "A. Two pointers"
  - name: "B. Sliding window"
  - name: "C. Prefix / suffix sums"
  - name: "D. Merge intervals"
  - name: "E. Bit tricks"
  - name: "🔗 Connections"
---

 

**The unifying idea.** Every technique here replaces re-examining the whole array on every step with
maintaining a *claim about the current state* — an invariant — and updating it in $$O(1)$$ per step. Name
the invariant and you can both trust the algorithm and often derive the code from it.

---

## A. Two pointers

> **Why** · $$O(n)$$ with $$O(1)$$ space for problems with a two-ended structure.
> **Where** · "find a pair," "compare ends," "shrink from both sides" — the upcoming pattern-triggers post trigger row.

**Trigger:** two quantities are at either end of a sorted array or string, and moving one end is always
the right choice.

**Invariant:** *if an answer exists, it lies within the index range* `[left, right]` — so advancing
the pointer that cannot improve the answer never misses a solution.

**Example problem:** given an array of wall heights, find the two walls that enclose the maximum water area.

**Algorithm:**
```python
heights = [1, 8, 3, 7]
left, right = 0, len(heights) - 1
best = 0
while left < right:
    area = min(heights[left], heights[right]) * (right - left)
    best = max(best, area)
    if heights[left] <= heights[right]:  # shorter wall limits area — move it
        left += 1
    else:
        right -= 1
# The formula and the if-condition change per problem; the while-loop structure stays.
```

**Figure 1 · Two pointers converging: move the shorter wall.**
<center><img src="/assets/img/dsa-series/two_pointers.png" style="width:80%"></center>
*The shorter wall limits the area — moving it is the only step that could possibly find something better, so the invariant is preserved.*

**Mechanism trace — Container With Most Water, heights = `[1, 8, 3, 7]`:**

1. left = 1, right = 7: area = 1 × 3 = **3**. Left shorter → move left inward.
2. left = 8, right = 7: area = 7 × 2 = **14**. Right shorter → move right inward.
3. left = 8, right = 3: area = 3 × 1 = **3**. Converged. Best = **14**.

Each step is $$O(1)$$; at most $$n$$ steps → **O(n) total.**

**Variant — fast/slow pointer (Floyd's cycle):** fast moves 2 steps, slow moves 1; they meet if and only
if a cycle exists. Lives on [Graphs](https://www.tobiasstenzel.com/blog/2026/dsa-graphs/).

**Costs:** $$O(n)$$ time · $$O(1)$$ space.

---

## B. Sliding window

> **Why** · $$O(n)$$ for any "longest/shortest subarray satisfying a constraint" problem.
> **Where** · substrings, subarrays, rate-limiting windows in streaming data (the upcoming DSA-at-work post).

**Trigger:** "longest/shortest subarray or substring satisfying a constraint."

**Invariant:** *the window `[left, right]` always satisfies the constraint*. On each step: expand `right` (add
one element); if the constraint breaks, shrink `left` until it holds again.

**Example problem:** given a string, find the longest substring with no repeated character.

**Algorithm:**
```python
seen = {}   # tracks last index of each character
left = 0
best = 0
for right, char in enumerate(a):
    if char in seen and seen[char] >= left:  # char is already in the window
        left = seen[char] + 1               # shrink: move left past the duplicate
    seen[char] = right
    best = max(best, right - left + 1)       # window [left..right] has no repeats
```

**Figure 2 · Sliding window expanding and shrinking.**
<center><img src="/assets/img/dsa-series/sliding_window.png" style="width:80%"></center>
*Each row is one state. The window (blue) expands right until a repeat forces a left shrink, then
expands again. Each element enters and leaves at most once.*

**Mechanism trace — longest substring without repeating characters, `s = "abcbad"`:**

| right | char | action | window | len |
| --- | --- | --- | --- | --- |
| 0 | `a` | expand | `"a"` | 1 |
| 1 | `b` | expand | `"ab"` | 2 |
| 2 | `c` | expand | `"abc"` | 3 |
| 3 | `b` | repeat → shrink left to 2 | `"cb"` | 2 |
| 4 | `a` | expand | `"cba"` | 3 |
| 5 | `d` | expand | `"cbad"` | **4** ← answer |

**Costs:** $$O(n)$$ — each element enters and leaves the window at most once.

---

## C. Prefix / suffix sums

> **Why** · $$O(1)$$ per range query after an $$O(n)$$ build — the same idea powers the "product except self" trick.
> **Where** · repeated range aggregates, cumulative statistics, "product of all except self."

**Trigger:** "answer a range sum/product query many times" or "need aggregate of all elements except
position $$i$$."

**Invariant:** `prefix[i]` = aggregate of `a[0..i-1]`. Range query `[start, end]` =
`prefix[end+1] - prefix[start]` — one subtraction, no scan.

**Example problem:** given an array, answer many "sum of elements from index `start` to `end`" queries in O(1) each.

**Algorithm:**
```python
a = [3, 1, 4, 1, 5, 9]
n = len(a)

# Build prefix array — O(n)
prefix = [0] * (n + 1)
for i in range(n):
    prefix[i + 1] = prefix[i] + a[i]   # prefix[i] = sum of a[0..i-1]

# Answer any range query — O(1)
def range_sum(start, end):
    return prefix[end + 1] - prefix[start]   # sum of a[start..end]

range_sum(2, 5)   # → 19  (= 4+1+5+9)
```

**Figure 3 · Prefix array and O(1) range query.**
<center><img src="/assets/img/dsa-series/prefix_sum.png" style="width:80%"></center>
*Top row: original values (orange = queried range). Bottom row: prefix sums. The query resolves to one
subtraction between two highlighted prefix cells.*

**Mechanism trace — `a = [3, 1, 4, 1, 5, 9]`, query sum of `a[2..5]`:**

Build the prefix array once, then answer in O(1):

| i | a[i] | prefix[i+1] = prefix[i] + a[i] |
| --- | --- | --- |
| 0 | 3 | 3 |
| 1 | 1 | 4 |
| 2 | 4 | 8 |
| 3 | 1 | 9 |
| 4 | 5 | 14 |
| 5 | 9 | 23 |

`range_sum(2, 5) = prefix[6] - prefix[2] = 23 - 4 = `**19**` = 4+1+5+9` ✓

**Variant — Product of Array Except Self.** Run prefix *products* left-to-right and suffix products
right-to-left; `result[i] = prefix[i] × suffix[i]`. O(n), no division needed.

**Costs:** $$O(n)$$ build — one pass to fill the prefix array · $$O(1)$$ per query — one subtraction.

---

## D. Merge intervals

> **Why** · $$O(n \log n)$$ to merge all overlapping spans in one sorted sweep.
> **Where** · calendar/booking systems, sessionization, rate-limit windows (the upcoming DSA-at-work post).

**Trigger:** "merge overlapping ranges," "find gaps," "detect conflicts in time spans."

**Invariant:** *sort by start; the current interval overlaps the last merged one if and only if
`curr.start ≤ last.end`*. If it does, extend `last.end`; otherwise start a new merged span.

**Example problem:** given a list of time intervals, merge all overlapping ones.

**Algorithm:**
```python
intervals = [[1, 3], [2, 6], [8, 10]]
intervals.sort(key=lambda interval: interval[0])   # sort by start time
merged = [intervals[0]]
for start, end in intervals[1:]:
    if start <= merged[-1][1]:                     # overlaps last merged span
        merged[-1][1] = max(merged[-1][1], end)    # extend it
    else:
        merged.append([start, end])                # disjoint: start a new span
# merged → [[1, 6], [8, 10]]
```

**Figure 4 · Merge intervals: before and after.**
<center><img src="/assets/img/dsa-series/merge_intervals.png" style="width:80%"></center>
*Top: original intervals, orange pair overlaps. Bottom: merged result — overlapping spans combined,
disjoint spans kept separate.*

**Mechanism trace — `[[1,3],[2,6],[8,10]]`:**

1. `intervals.sort(key=lambda x: x[0])` → `[[1,3],[2,6],[8,10]]`.
2. Start with `[1,3]`. Next `[2,6]`: $$2 \leq 3$$ → overlap → extend to `[1,6]`.
3. Next `[8,10]`: $$8 > 6$$ → no overlap → append. Result: `[[1,6],[8,10]]`.

**Costs:** $$O(n \log n)$$ (sort dominates) · $$O(n)$$ space.

---

## E. Bit tricks

> **Why** · $$O(1)$$ per operation for problems involving counting bits, pairing, or set membership.
> **Where** · duplicate detection, bitmasked state, performance-critical loops.

**Trigger:** "find the element without a pair," "count set bits," "check if a number is a power of two."

**Bit basics (everything below follows from these):**

- A number is stored as binary digits. Each position $$i$$ (from right, starting at 0) has weight $$2^i$$.
  So $$6 = 110_2 = 1{\cdot}2^2 + 1{\cdot}2^1 + 0{\cdot}2^0 = 4+2+0$$ and
  $$5 = 101_2 = 1{\cdot}2^2 + 0{\cdot}2^1 + 1{\cdot}2^0 = 4+0+1$$.
- **AND (`&`):** bit is 1 only if *both* inputs are 1. `0b110 & 0b101 = 0b100` (=4).
- **OR (`|`):** bit is 1 if *either* input is 1. `0b110 | 0b101 = 0b111` (=7).
- **XOR (`^`):** bit is 1 if inputs *differ*. `0b110 ^ 0b101 = 0b011` (=3). Same bits cancel: `x ^ x = 0`.
- **`-x` in binary (two's complement):** to negate a number, flip all bits then add 1.
  Step by step for $$x = 6$$:
  `0110` (start) → `1001` (flip all bits) → `1010` (add 1) = $$-6$$.
  Compare: $$x = \mathtt{0110}$$, $$-x = \mathtt{1010}$$ — bit 1 (value 2) is set in *both*.
  The lowest set bit always survives negation. That is why `x & (-x)` isolates it.

**Trick 1 — XOR: identical values cancel.** Follows directly from the XOR definition above: `x ^ x = 0`
and `x ^ 0 = x`, so every pair cancels and only the unique value survives.

```python
nums = [2, 3, 2, 5, 5]   # every value appears twice except 3
result = 0
for v in nums:
    result ^= v   # pairs cancel; singleton survives
# result → 3
```
Trace: `0^2=2 → 2^3=1 → 1^2=3 → 3^5=6 → 6^5=3` ✓

**Cost:** $$O(n)$$ time · $$O(1)$$ space.

---

**Trick 2 — `x & (x-1)` clears the lowest set bit.** Subtracting 1 from `x` flips the lowest set bit
and all bits below it (borrow chain in binary); AND with the original keeps only the bits above it.

**Figure 5 · `x & (x−1)`: how subtracting 1 flips the lowest set bit.**
<center><img src="/assets/img/dsa-series/bit_tricks.png" style="width:80%"></center>
*Three rows: x, x−1, and x&(x−1). The orange bit (lowest set bit) and everything below it flip in x−1; AND with x clears them.*

```python
x = 0b10110100   # 180: has 4 set bits
count = 0
while x:
    x &= x - 1   # clear one set bit per iteration
    count += 1   # count = 4 after the loop
# Also: x & (x-1) == 0 iff x is a power of two (only one set bit)
```

**Cost:** $$O(k)$$ where $$k$$ = number of set bits — faster than scanning all bits.

---

**Trick 3 — `x & (-x)` isolates the lowest set bit.** Negating `x` (flip all bits, add 1) preserves
the lowest set bit while flipping everything above it. AND keeps only the bit set in both.

**Figure 6 · `x & (−x)`: the lowest set bit survives negation.**
<center><img src="/assets/img/dsa-series/bit_and_neg.png" style="width:80%"></center>
*x = 6 (0110) and −x = 10 (1010) share only bit 1; AND isolates it — result = 2 (0010).*

```python
x = 6             # binary: 0110 — lowest set bit is bit 1 (value 2)
lsb = x & (-x)    # → 2  (binary: 0010)
```
Used in Fenwick trees ([Hierarchical Structures](https://www.tobiasstenzel.com/blog/2026/dsa-hierarchical/)) for index arithmetic.

**Cost:** $$O(1)$$ — single arithmetic operation.

---

## 🔗 Connections

- Linear structures these techniques operate on: [Linear Structures](https://www.tobiasstenzel.com/blog/2026/dsa-linear/).
- Invariant concept (init / maintain / terminate): [Sorting & Searching](https://www.tobiasstenzel.com/blog/2026/dsa-sorting-searching/).
- Technique recognition triggers and work applications: the upcoming pattern-triggers post / the upcoming DSA-at-work post.
- Floyd's cycle (fast/slow pointer variant of two pointers): [Graphs](https://www.tobiasstenzel.com/blog/2026/dsa-graphs/).
- Fenwick tree (uses bit trick `x & (-x)`): [Hierarchical Structures](https://www.tobiasstenzel.com/blog/2026/dsa-hierarchical/).
