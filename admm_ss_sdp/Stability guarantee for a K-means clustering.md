# Stability guarantee for a K-means clustering

## Description

This software verifies if your clustering $C$ is approximately correct. 

The software is based on [this paper](https://sites.stat.washington.edu/mmp/Papers/sdp-kmeans-nips18.pdf) and the [original code](https://github.com/mathcg/admm_ss_sdp/) written by Gang Cheng.

## How it works

1. **Enter the data Data and a clustering** $C$. [Example] [Link to format]
2. Optionally, choose solver parameters. An optimization problem will be run in the background. If the problem times out,  you can try to change the parameters. We do not recommend this, because the chance of getting a benefit from it is little.
3. **Click Run Sublevel Set (SS) algorithm.** An optimization problem is set up and solved.
4. **Get the answer.**

✅ Guaranteed $\varepsilon=\ldots$

$\varepsilon$ is the *Optimality Interval (OI)* (or *bound*, or error *margin*). The smaller, the better. Note that the OI is not a Confidence Interval (CI); because it is deterministically 100% guaranteed.

❓Not guaranteed ($\varepsilon=\ldots,\,p_{min}=\ldots$)

This means that your clustering $C$ is not stable enough to obtain a guaranteed. This can be because

- The data Data is not clusterable (which means that the clusters are not distinct enough, and another way of clustering the data may be just as good)
- $C$ is a local minimum and some other global minimum exists.
- Data is clusterable and $C$ is stable, but the algorithm may fail to guarantee borderline cases.

### Guaranteed clusterings examples

### Not guaranteed: examples

A few things you can do if your clustering is not guaranteed:

- Is $\varepsilon$ close to $p_{min}$? If $\varepsilon$ exceeds $p_{min}$, a clustering cannot be guaranteed. But even so, a small $\varepsilon$ heuristically indicates more stability.
- We have used the value of $\varepsilon$, heuristically, to select the number of clusters [Image here]
- If it makes sense for your application, you can *remove the outliers* from your Data and try again. This is often very effective in improving the OI, and often results in obtaining a guaranteed $$\varepsilon$$.

## What does $\varepsilon$ actually mean?

Remember that a clustering is evaluated by its K-means cost $Cost(\mathcal{C})=\sum_{k=1}^K\sum_{i\in {\rm cluster} k} \|x_i-\mu_k\|^2$.

**What we know:** Data $\mathcal{D}$ , clustering $\mathcal{C}$, and its $Cost(C)$

**What we want to know (first version):** “Can there be another $C’$ so that $Cost(C’)\leq Cost(C)$? “

The answer, if we could know it, would not be very informative, because if we reassign a single point to a different cluster, the change in cost will be very small.

**What we want to know (better version):** “Can there be **another $C’$**, **very different from $C$**, so that $Cost(C’)\leq Cost(C)$? “

This is what our  **SS** algorithm finds. When it returns a Guaranteed $\varepsilon$, then we know that any clustering $C’$ that has $Cost(C’)\leq Cost(C)$, must be $\varepsilon$-close to $C$. 

$\varepsilon$ is a difference between two clusterings $C,\,C’$, measured by the *fraction of the* $n$ *points* that must change cluster assignment to turn $C’$ into $C$. For example, if $n=200$ points, and $\varepsilon=0.05$, it means that any clustering $C’$ that is as good as $C$ or better must differ from $C$ in at most 10 points; and if $\varepsilon=10^{-4}$ and $n=200$, it means that no clustering can be better than $C$.
