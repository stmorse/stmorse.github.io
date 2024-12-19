---
layout: page
title: Projects
---

This is, in no particular order, some non-academic, mostly non-serious projects I've worked on. Some are completed, some are in-progress, but none are completely abandoned (in spirit)!  Head over here for [actual research](https://stmorse.github.io/research.html). 

<hr>

<div class="row">
    <div class="four columns">
    <a href="https://stmorse.github.io/xwbuddy" target="_blank">
        <img style="padding: 10px; float: center;" width="100%" src="{{ site.baseurl }}/images/xwb_logo.png" alt="Crossword Buddy">
    </a>
    </div>
    <div class="eight columns">
        <p>
        <b>Crossword Buddy.</b> I noticed inexperienced crossword solvers' only help options were full or partial answer reveals, and thought it'd be nice to just get a new or alternative clue that helps you think about the answer in a different way.  Crossword Buddy does that --- and it uses an LLM to generate the "buddy clues", automated, so it was a fun project.  Active project as of Dec 2024.
        <br/>
        <a href="https://stmorse.github.io/xwbuddy">Live website</a></p>
    </div>
</div>

<div class="row">
    <div class="four columns">
    <a href="https://github.com/stmorse/footballdrop" target="_blank">
        <img style="padding: 10px; float: center;" width="100%" src="{{ site.baseurl }}/images/lucky_unlucky_9.png" alt="Fantasy Football">
    </a>
    </div>
    <div class="eight columns">
        <p>
        <b>Fantasy Football API.</b> I've done a number of posts on using the ESPN Fantasy Football API which is public but undocumented, specifically making cool plots in Python you can send to your league and be ignored, or woo yourself into a false sense of security about your team.
        <br/>
        <a href="https://github.com/stmorse/footballdrop">Repo (need to update)</a>,
        <a href="https://stmorse.github.io/journal/espn-fantasy-v3.html">Main blog post</a></p>
    </div>
</div>

<div class="row">
    <div class="four columns">
    <a href="https://stmorse.github.io/life-js" target="_blank">
        <img style="padding: 10px; float: center;" width="100%" src="{{ site.baseurl }}/images/life_js.png" alt="Conway Game of Life">
    </a>
    </div>
    <div class="eight columns">
        <p>
        <b>Game of Life.</b>  Implementation of Conway's Game of Life in pure Javascript with div's and whatnot.  Comes with some presets and different grids.  I'd like to try a hexagonal grid or different rulesets.
        <br/>
        <a href="https://stmorse.github.io/journal/game-of-life-javascript.html">Blog post</a>, 
        <a href="https://stmorse.github.io/life-js">Live game</a>,
        <a href="https://github.com/stmorse/life-js">Repo</a></p>
    </div>
</div>

<div class="row">
    <div class="four columns">
        <a href="https://stmorse.github.io/xymo" target="_blank">
        <img style="padding: 10px; float: center;" width="100%" src="{{ site.baseurl }}/images/xymo.png" alt="XYMO - Text Adventure Game">
        </a>
    </div>
    <div class="eight columns">
        <p>
        <b>XYMO.</b> Took a foray into text adventure games, but in a web app.  I like my gimmick here: you "accidentally" found a secret NASA terminal portal that is remotely controlling a probe on a secret alien planet.  I've only built 2-3 rooms of it though, it's fun but time consuming.
        <br/>
        <a href="https://stmorse.github.io/xymo">Live game</a>, 
        <a href="https://github.com/stmorse/xymo">Repo</a></p>
    </div>
</div>

<div class="row">
    <div class="four columns">
        <a href="https://stmorse.github.io/scheduler" target="_blank">
        <img style="padding: 10px; float: center;" width="100%" src="{{ site.baseurl }}/images/scheduler.png" alt="Scheduler App">
        </a>
    </div>
    <div class="eight columns">
        <p>
        <b>Integer Optimization Scheduler App.</b> One of my jobs involved approving a monthly special duty schedule, and the process to create it was very manual ... seemed like a perfect application of integer optimization, so I (attempted) a web app.  It was working (really it was!) but now it is not.  Uses <a href="https://github.com/jvail/glpk.js/">GLPK</a> for the in-browser optimizer.  I need to troubleshoot this, it's probably a small tweak that inadvertently broke something.
        <br/>
        <a href="https://stmorse.github.io/scheduler">Live site</a>, 
        <a href="https://github.com/stmorse/scheduler">Repo</a></p>
    </div>
</div>

<div class="row">
    <div class="four columns">
        <a href="https://stmorse.github.io/journal/IP-Crossword-puzzles.html" target="_blank">
        <img style="padding: 10px; float: center;" width="100%" src="{{ site.baseurl }}/images/ip_crossword.png" alt="Scheduler App">
        </a>
    </div>
    <div class="eight columns">
        <p>
        <b>IP Crossword.</b> I'm into crosswords (see Crossword Buddy), and although I knew crossword construction is done with constraint satisfaction programming (CSP), it seemed interesting to try applying actual optimization.  An optimal crossword!  It is terribly slow and relies too heavily on a constructor's weighting of the word list, but was an interesting project.
        <br/>
        <a href="https://stmorse.github.io/journal/IP-Crossword-puzzles.html">Blog</a>, 
        <a href="https://github.com/stmorse/IP-crossword">Repo</a></p>
    </div>
</div>

