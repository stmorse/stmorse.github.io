---
layout: post
title: "New site"
categories: journal
tags: [documentation,sample]
---

I am now running my personal site using Jekyll, hosted by Github Pages.  I modded a theme I found [here](https://github.com/lenpaul/lagrange).  Feel free to fork my version or the original.

This post is mainly a test page for different theme/format elements.

A code block looks like this:

```python
import numpy as np
u = np.random.rand()
if u < 0.1:
  print 'Dang'
```

and inline code looks like `this`.

We have *italics*, **bold**, and ***italicized bold*** text.

Some lists are

1. Who's on first?
2. Yes.
3. That's what I asked you.

and a blockquote is

> A man, a plan, Bob Loblaw.

Testing some math formatting inline $$\sum_i x_i^2$$, some block...

$$\Theta_{ij} = \sum_{i,j} \int_0^t f(i,j,t) dt$$

some alignment:

$$
\begin{align*}
\lambda &= \tau + \int_0^t h(s) ds \\
 &= \tau + \sum_i h(t_i)
\end{align*}
$$

and without the `$$`,

\begin{equation}
A \sim \mathcal{N}(0, \frac{1}{\tau})
\label{eq:test}
\end{equation}

which I can reference with Eq. \eqref{eq:test}.  And some collapsible code:

<collapsible>
  <summary><span class="icon">></span>Code</summary>
  <p>

  ```python
  for i in range(5):
    print('See me now?')
  ```

  </p>
</collapsible>

Here's a margin note test.  Should appear over in the margin over there, something paranthetical yet interesting. 

<div class="marginnote">I have a discovered a truly remarkable proof of this theorem which won't fit in this Tufte-style margin note.</div>

And the headers are

# Header 1

## Header 2

### Header 3

#### Header 4

##### Header 5

###### Header 6

That'll probably do it for now.

### Questions?

This theme is completely free and open source software. You may use it however you want, as it is distributed under the [MIT License](http://choosealicense.com/licenses/mit/). 
