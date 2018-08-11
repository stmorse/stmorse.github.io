---
layout: page
title: Research
---

My current research interests are in complex systems, machine learning, and optimization.  I am currently an instructor in the [West Point Math Department](https://www.usma.edu/math/SitePages/Math.aspx), and collaborate with friends in [Draper Laboratory](http://www.draper.com), [HuMNet Lab](http://humnetlab.mit.edu), and the [MIT Operations Research Center](https://orc.mit.edu).

<hr>

### Hawkes process

The class of self-exciting temporal point processes called the <i>Hawkes process</i> models the probability of an arrival through an intensity function with additive "influence" from previous arrivals.  This is a highly flexible model, with applications in finance, computational neuroscience, seismology, social networks, product adoption, and others.  I worked on a parameter estimation method for a multivariate form of the Hawkes process using (MAP) EM, and am currently working on finishing a project to apply this methodology to predicting user purchasing behavior using credit card history.

<a href="https://github.com/stmorse/hawkes">Repo</a>&nbsp;&nbsp;&#8226;&nbsp;
<a href="https://stmorse.github.io/journal/Hawkes-python.html">Blog</a>&nbsp;&nbsp;&#8226;&nbsp;
<a href="{{ site.baseurl }}/docs/6-867-final-writeup.pdf">Report</a>&nbsp;&nbsp;&#8226;&nbsp;
<a href="{{ site.baseurl }}/docs/JMM18_slides.pdf">Slides (JMM '18)</a>

In my <a href="{{ site.baseurl }}/docs/orc-thesis.pdf">masters thesis</a>, I expand on this and the following topic.  E.g. we can extend techniques from percolation theory to model the idea of "persistence" in cascading patterns, and we apply the method to the Hillary Clinton emails and find hidden influencers. 

<hr>

### Information spread and influence structure

<img align="left" width="30%" src="{{ site.baseurl }}/images/persistent.png" alt="persistent cascades">

A challenge in large-scale passive-collection communication datasets is finding meaningful structure.   We know A called B, but not why, whether it was an accident, or whether this influences B to call C.  One method to extract meaningful structure is to look for recurring patterns: imagine if we see A call B and C, who call D and E, and then the same pattern (or something very similar) occurs again and again over a long period of time.  Using methods of inexact tree matching and hierarchical clustering, we define, find, and analyze these group conversations (termed "persistent cascades") and show they reveal new roles in network topology and spreading dynamics.  

<a href="https://github.com/stmorse/cascades">Repo</a>&nbsp;&nbsp;&#8226;&nbsp;
<a href="{{ site.baseurl }}/docs/BigD348.pdf">Paper (IEEE BD '16)</a>&nbsp;&nbsp;&#8226;&nbsp;
<a href="{{ site.baseurl }}/docs/persistent-cascades-ieee.pdf">Slides</a> 

In my <a href="{{ site.baseurl }}/docs/orc-thesis.pdf">masters thesis</a>, I expand on this and the previous topic.  E.g. we can extend techniques from percolation theory to model the idea of "persistence" in cascading patterns, and we apply the method to the Hillary Clinton emails and find hidden influencers. 

<hr>

### Trace diagrams

<img align="left" width="30%" src="{{ site.baseurl }}/images/diagrams.png" alt="trace diagrams">

My undergraduate work was with so-called *trace diagrams* (also *birdtracks*, *spin networks*, *tensor diagrams*, ...), which are structured graphs representing multilinear functions.  They are essentially a notational invention which provide a simple, graphical, intuitive way of performing otherwise complicated tensor or matrix calculations.  Versions of this notation show up in physics, and the current incarnation has gained some traction (<a href="http://arxiv.org/pdf/1102.0316.pdf">normal factor</a> <a href="http://arxiv.org/pdf/1004.3833.pdf">graphs</a>, generalization to commutative <a href="http://dl.acm.org/citation.cfm?id=1596553">monads</a>), but it is still a niche area of research.  The paper below defines them in a rigorous way and shows the power of the notation with several extremely short proofs of classic results in linear algebra.

<a href="{{ site.baseurl }}/docs/tracediagrams.pdf">Paper (<i>Involve</i>)</a>&nbsp;&nbsp;&#8226;&nbsp;
<a href="{{ site.baseurl }}/docs/mainthesis.pdf">Undergrad thesis</a>
