---
layout: post
title: "Making crossword puzzles with integer programming"
categories: journal
tags: [projects, crossword, puzzles, optimization, mathematics]
---

I love solving crossword puzzles.  The Will Shortz-edited [NYTimes puzzle](http://nytimes.com/crosswords) has set the bar for many years, although there is now a growing [indie crossword scene](https://fivethirtyeight.com/features/indie-crossword-puzzlers-are-shaking-up-a-very-square-world/).  I've tried making my own by hand, and failed, an embarrassing number of times.  I always wondered: how the heck do people do it?  Do they agonize over a piece of grid paper until the eraser rubs right through the page, which was my method of choice?

Of course they don't.  I discovered that the majority of crossword constructors use software, and the challenge isn't putting the grid together, but feeding the computer a solid, and fresh set of words.  Upon further investigation, it appears the industry standard software is a Windows-only app called [Crossword Compiler](http://www.crossword-compiler.com) that starts at about 50 bucks.  ~~Surprisingly though, I could not find a freeware (much less open source) version of this functionality, even at a lower quality.~~  

**EDIT:** There is a [free and open-source crossword compiler](https://www.quinapalus.com/qxw.html) that a kind reader alerted me to after my initial post.  I haven't tested it personally yet, but it appears to be Windows and Linux friendly, possibly OS X friendly, with user interface, written in C.  It also appears to use constraint satisfaction, which I discuss below.   (If you know of any other free or open source options, please [contact me](http://twitter.com/thestevemo).)  I still think the IP formulation deserves some attention though, as I will attempt to justify below.


## IP formulation of the crossword generation problem 

It seems like most of the crossword generation research out there formulates the problem as a **[constraint satisfaction problem](http://aima.cs.berkeley.edu/2nd-ed/newchap05.pdf)**.  Unfortunately, this field is not in my wheelhouse, so I chose a different formulation: integer optimization aka integer programming (IP).  (There is actually a lot of parallels between IP and CSP, and possibly something like a one-to-one correspondence, but that is beyond the scope of my brain and this blog post at the moment.)  To my limited understanding though, CSP generates *feasible* solutions, while IP generates *optimal* solutions.  So in theory, where CSP gives you several possible crosswords and you pick the one you want, an IP generated crossword would give you the best *possible* crossword (all relative to your scored word list).  Another benefit is the ability to plug the problem into a form already amenable to several free, open source, and heavy-duty solvers that are built to handle IPs.

So let's formulate crossword puzzle generation as an IP, and let's start by defining our decision variables.  There seem to be two approaches possible: (1) assigning words to slots, or (2) assigning letters to cells.  In the letter-to-cell approach, we make sure each cell in the grid has a single unique letter assignment, and the hard part is then making sure each slot in the puzzle ("23 down", "5 across") actually spells a word in the vocabulary.

I'll take the words-to-slots approach instead.  This method makes sure each slot has a single word assigned, and the hard part is making sure the intersecting letters for across and down slots are the same.  But let's start with the easy stuff.

Let's define our available vocabulary as $$W=\{w_k\}$$, so $$w_3$$ might be "epic" for example. The slots in the puzzle we'll denote $$S=\{s_m\}$$, so $$s_{10}$$ might represent "12 across."  Now let's define our decision variables as $$z_{k,m}\in\{0,1\}$$, so that $$z_{3,10}=1$$ means that "epic" is in the "12 across" slot.

As a preprocessing step, we can go through and set $$z_{k,m}=0$$ wherever $$\|W_k\| \neq \|S_m\|$$, i.e. wherever the size of the word isn't equal to the size of the slot --- we can't assign a word to a slot if it doesn't even fit.  We can then define our first two sets of constraints in the usual IP way, as

$$
\begin{align*}
\sum_k z_{k,m} &= 1 \ \ \forall k \\
\sum_m z_{k,m} &\leq 1 \ \ \forall m
\end{align*}
$$

The first constraint ensures every slot has exactly one word assigned to it.  The second constraint ensures every word is assigned to at most one slot (crosswords don't like a word to occur more than once).

Now things get tricky.  We now need to ensure that each letter in the assigned word for an *across* slot matches the corresponding letters in the intersecting *down* slots.  This is a simple idea that is just a pain to translate into an IP formulation.  Let's introduce the following notation: $$G=\{g_{ij}\}$$ for the set of all open cells in the grid, indexed by row $$i$$ and column $$j$$; also $$r(g)$$ and $$c(g)$$ to denote the slot index of the row and column, respectively, corresponding to grid $$g$$.  Also, let $$P_{\ell, r(g)}$$ be the set of indices to all possible words with letter $$\ell$$ in grid $$g$$ for the corresponding row/across slot $$r(g)$$, and $$P_{\ell, c(g)}$$ be the corresponding set for down slots.   Now our last constraint is:

$$
\sum_{k\in P_{\ell, r(g)}} z_{k, r(g)} = \sum_{k \in P_{\ell, c(g)}} z_{k,c(g)} \ \ \forall g, \ \forall \ell
$$

That's nasty notation.  But basically this says, take the set of all possible words with letter $$\ell$$ in cell $$g$$ in the across slot $$r(g)$$, and the same set for the down slot $$c(g)$$.  Then, make sure that whatever assignment we gave for the across slot matches the down slot.  Finally, ensure this matching is true for all grid squares and all letters.

Last but not least, let's express our objective function as

$$
\text{max}_z \ \ \sum_k \sum_m c_k z_{k,m}
$$

where $$c_k$$ is some score associated with each word.  For example if we had some trendy, never-before-used phrase that we wanted to make sure made it into our next puzzle, we'd give it $$c_k = 100$$, and some awful crosswordese like "oleo" we'd give $$c_k = 1$$.  Etc.  And that's it!

## Advantages of the IP approach

This formulation allows us to very easily incorporate two vital parts of crossword construction:  specifying word-to-slot assignments in advance, and/or specifying that certain words make it into the puzzle somewhere.

The word-to-slot assignment is easy: just set $$z_{k,m}=1$$ as an additional constraint, where $$k$$ is the word you need and $$m$$ is the desired slot.  Voila.

The "make sure this word is in the puzzle anywhere" is also straightforward, and can be done two non-equivalent ways.  The "soft" way would be to just increase the value $$c_k$$ of the desired word sky-high, so the optimizer *really* wants to include it in the puzzle.  The "hard" way would be to actually include another constraint, $$\sum_m z_{k,m} = 1$$ for the desired $$k$$, so that exactly *one* of the decision variables corresponding to that word is turned on.


## Implementation in Python

To implement this in Python, I tried to stay away from any unnecessary dependencies, so I used pure base Python and the `pulp` package for the integer optimization.  All the code is in a [Github repo over here](http://github.com/stmorse/ip-crossword) so I won't bore you with the coding details, but I basically wrote a `Grid` class to handle all the messy grid manipulation, a couple dicts with preprocessing information, and a code chunk with the IP formulation as just described.

However, although it works, it can only manage grids that are $$3\times 3$$ and $$4\times 4$$ (or even $$5\times 5$$ if you put black squares in the right spots) but not much bigger.  I think this is due to a couple non-formulation related problems that I'd like to fix:

1. Improve the grid handling --- currently very hacky, using base Python, lots of list comprehensions and loops ... blech.
2. Use something better than the default solver for `PuLP`.
3. Related: a better solver might be able to give insight if there are pre-processing steps that speed things up.
4. Improve the word list and add points per word. Since most of a crossword maker's struggle (I'm told) is getting a good, well-sorted word list. Currently I'm using a ``ospd.txt file with a few thousand words, most of which are crummy/archaic, and assigning each word a random score.

Another thing the code doesn't implement is creating the grid itself.  My code takes a predefined grid as input, but ideally, we just feed the optimization a word list, and it figures out where to put the grid squares on its own.  This introduces a new slew of constraints: for example, crossword grids are symmetric, they must have either rotational or reflectional symmetry.

Anyway, the code as-is can be run like this:
```python
from ipxword import Grid, IPXWordGenerator
G = Grid(3, blacksq=[(0,0)])
ipx = IPXWordGenerator(G, numk=500)
ipx.build()
```
which creates a $$3\times 3$$ grid with a single black square in the top left corner, and then attempts to fill it with 500 words sampled randomly from the `ospd.txt` file (with random word scores).  This will output something like
```
Assignments: (index, (slot), word)
(64, (1, 'down'), 'eau')
(165, (2, 'down'), 'hyp')
(258, (0, 'down'), 'by')
(279, (2, 'across'), 'yup')
(327, (0, 'across'), 'eh')
(473, (1, 'across'), 'bay')
```
(like I said, the word list is pretty awful), which is the following little toy puzzle:
```
# E H
B A Y
Y U P
```
Yup!

That's all for now.  If you have comments, questions feel free to message me on [Twitter](http://twitter.com/thestevemo).  Thanks for reading!



