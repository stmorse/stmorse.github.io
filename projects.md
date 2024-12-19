---
layout: page
title: Projects
---

This is, in no particular order, some non-academic, mostly non-serious projects I've worked on.  Head over here for [actual research](https://stmorse.github.io/research.html).

<hr>

<div class="row">
    <div class="four columns">
        <img style="padding: 10px; float: center;" width="100%" src="{{ site.baseurl }}/images/xwb_logo.png" alt="Crossword Buddy">
    </div>
    <div class="eight columns">
        <p>
        I noticed inexperienced crossword solvers' only help options were full or partial answer reveals, and thought it'd be nice to just get a new or alternative clue that helps you think about the answer in a different way.  Crossword Buddy does that --- and it uses an LLM to generate the "buddy clues", automated, so it was a fun project.  Active project as of Dec 2024.
        </p><br/>
        <a href="https://stmorse.github.io/xwbuddy">Live website</a>
    </div>
</div>

<div class="row">
    <div class="four columns">
        <img style="padding: 10px; float: center;" width="100%" src="{{ site.baseurl }}/images/life_js.png" alt="Conway Game of Life">
    </div>
    <div class="eight columns">
        <p>
        Implementation of Conway's Game of Life in pure Javascript with div's and whatnot.  Comes with some presets and different grids.  I'd like to try a hexagonal grid or different rulesets.
        </p><br/>
        <a href="https://stmorse.github.io/journal/game-of-life-javascript.html">Blog post</a>, 
        <a href="https://stmorse.github.io/life-js">Live game</a>,
        <a href="https://github.com/stmorse/life-js">Repo</a>
    </div>
</div>

<div class="row">
    <div class="four columns">
        <img style="padding: 10px; float: center;" width="100%" src="{{ site.baseurl }}/images/xymo.png" alt="XYMO - Text Adventure Game">
    </div>
    <div class="eight columns">
        <p>
        Took a foray into text adventure games, but in a web app.  I like my gimmick here: you "accidentally" found a secret NASA terminal portal that is remotely controlling a probe on a secret alien planet.  I've only built 2-3 rooms of it though, it's fun but time consuming.
        </p><br/>
        <a href="https://stmorse.github.io/xymo">Live game</a>, 
        <a href="https://github.com/stmorse/xymo">Repo</a>
    </div>
</div>

<div class="row">
    <div class="four columns">
        <img style="padding: 10px; float: center;" width="100%" src="{{ site.baseurl }}/images/scheduler.png" alt="Scheduler App">
    </div>
    <div class="eight columns">
        <p>
        One of my jobs involved approving a monthly special duty schedule, and the process to create it was very manual ... seemed like a perfect application of integer optimization, so I (attempted) a web app.  It was working (really it was!) but now it is not.  Uses <a href="https://github.com/jvail/glpk.js/">GLPK</a> for the in-browser optimizer.  I need to troubleshoot this, it's probably a small tweak that inadvertently broke something.
        </p><br/>
        <a href="https://stmorse.github.io/scheduler">Live site</a>, 
        <a href="https://github.com/stmorse/scheduler">Repo</a>
    </div>
</div>

<div class="row">
    <div class="four columns">
        <img style="padding: 10px; float: center;" width="100%" src="{{ site.baseurl }}/images/ip_crossword.png" alt="Scheduler App">
    </div>
    <div class="eight columns">
        <p>
        I'm into crosswords (see Crossword Buddy), and although I knew crossword construction is done with constraint satisfaction programming (CSP), it seemed interesting to try applying actual optimization.  An optimal crossword!  It is terribly slow and relies too heavily on a constructor's weighting of the word list, but was an interesting project.
        </p><br/>
        <a href="https://stmorse.github.io/journal/IP-Crossword-puzzles.html">Blog</a>, 
        <a href="https://github.com/stmorse/IP-crossword">Repo</a>
    </div>
</div>
