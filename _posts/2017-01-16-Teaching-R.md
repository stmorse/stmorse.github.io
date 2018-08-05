---
layout: post
title: Teaching R in a beginner data science class
date: 2017-01-16
tags: [R, documentation, teaching]
---


I recently got talked into helping with a student-led course on software tools for optimization and analytics, put together by colleagues in the [OR Center](http://orc.mit.edu) at MIT.  It was a great experience, and was a confidence boost for the actual teaching I'll be doing in the fall.  

However, the path from "hey let's switch up the course this year" to "good morning class" was tricky and full of interesting revelations for me.  So I wanted to sum up some observations, and shamelessly link to my:

<center><b><a href="https://stmorse.github.io/intro-tidyverse/master.html">COURSE NOTES</a></b></center>

(which are awesome).  

Shout out to [Phil Chodrow](http://philchodrow.github.io) who is hosting the **[official page site](http://philchodrow.github.io/cos_2017)**.  This has links to the rest of the sessions which covered *machine learning, optimization, and deep learning.*


## 1. Data wrangling and visualization should be covered together.

Last year the course had one day for "wrangling" and the next day for "visualization."  I took this as a student.  The wrangling session focused on things like manipulating a large, unwieldy dataset into useful summary tables, aggregating information, a little bit of cleaning.  The visualization session took these results and plotted them: lines, bars, histograms.

This allowed us to deep-dive into each topic, but as a student sometimes felt slow and clunky.  We learned all about wrangling, but the only way we could look at our results was through a boring R summary table printout.  Then we finally got to visualization, which caused us to raise new questions that required new wrangling tools that we weren't going to learn.

So this year we merged the wrangling and visualization into one session, but kept everything more basic (with an "advanced" day the next week).  This allowed you to fiddle with the data, and then look at what you did with a plot.  Then fiddle some more, and look at it again.  It was hard to build both things up at once in a well-paced and thorough way, but regardless of our particular session's success, I think it is the better way to teach intro to a data analytics software tools course.

## 2. The Tidyverse is hard to self-teach.

The other big debate was whether we would teach R or Python.  We settled on R because: (1) RStudio is more beginner friendly than a Jupyter notebook, (2) R is easier out-of-the-box for data analytics and wrangling (compare the amount of overhead in Python to fit a linear model and get summary statistics with the same task in R), and (3) last year's materials were in R.

The sub-debate then was: should we teach the Doctrine of the Tidyverse, or stick to base R?  I found the idiomatic, intuitive approach to data handling that the `tidyr` and `dplyr` packages provide to be very enticing, as a student, so I began down that path when I started preparing these materials.  

But more importantly, I realized, is that learning good practices in the Tidyverse is harder to self-teach than base R, or other languages.  The whole idea of "long" vs. "wide" data is very confusing, and really the whole idea of long chains of vectorized, verb-like data manipulations that you see in the tidyverse is a complete Twilight zone episode the first time you see it.  If I hadn't been shown the Tidyverse in a course, I'm not sure I would have been comfortable using it to the extent that I now do.

So unlike base R, or Python, or many other languages that you can slowly, painfully pick up from the docs, stack overflow forums, and trial and error, learning the Tidyverse kinda requires a real person to teach you.

## 3. The "get a bigger hammer" philosophy doesn't work in the Tidyverse.

A corollary to the previous point is that you can't often brute force a solution in the tidyverse like you can in other syntaxes.  The Tidyverse demands a fairly specific way of handling and plotting data.  For example, if you have a "wide" table in Python, you can do some nasty for-loop and get a plot going.  By contrast, `ggplot` won't even speak to you unless you gather that data into long format and correctly set your aesthetic mapping.

This is extremely frustrating, especially for a beginner.  You can see it as a feature, not a bug, in that it forces you to rub your chin a little before you start coding.  But man, I just get tired of thinking tidy and wish I could just bang out a sloppy solution to at least see what's going on before I beautify my code.

Either way, it reinforces the point that if you want to learn the Tidyverse, it's best to have a teacher.

## 4. Long vs. Wide (i.e. what "tidy data" means) is the inflection point for deeper understanding.

The most stubborn sticking point for me, when learning the `tidyr` to `ggplot` symbiotic relationship, was that "wide" vs "long" data thing.  You know, the definition of "tidy data."  At some point, things clicked.

But I noticed that with students in the course, and talking to others before and after, this seems to be the sticking point.  Even if they understand it, it's such an odd/new concept that it just takes a while to sink in.  Unfortunately, since you have to have this pretty well mastered to make anything interesting happen in `ggplot`, a lot of people get off the Tidy Train before they even start moving.

## 5. I missed Python.

Long story short, at the end of making these course notes and immersing myself in the Hadleyverse for a few days, I just ... missed coding in Python.  

```python
import time,random
while 1:
  locals().setdefault('i',60)
  print(' '*i+'<>'+' '*(80-i-1)+'|')
  time.sleep(.1)
  i += random.randint(-2,2)
```

(Thanks to [Arun Rocks](http://arunrocks.com/python-one-liner-games/) for that one-liner spaceship game.)
