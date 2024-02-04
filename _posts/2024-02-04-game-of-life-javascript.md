---
layout: post
title: "Conway's Game of Life - Javascript"
categories: journal
date: 2024-02-04
tags: ['javascript', 'games']
---

In 1970, British mathematician John Conway invented the [Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life).  (Not to be confused with the 1860 boardgame [Life](https://en.wikipedia.org/wiki/The_Game_of_Life).)  A zero-player game based on a simple set of rules applied to a grid, the Game of Life creates emergent behavior of *cellular automaton*.  Conway is a heavy hitter, with a lot of serious and foundational work in mathematics, things named after him, etc, so I assume this idea was just a lark --- but it has become a classic in the study of emergent systems behavior, cellular automata, and a staple of computer science and coding, as a simple idea that tests basic things like grid design, looping animation, and offers natural ways to incorporate UI and other customizations.

There's [many](https://www.geeksforgeeks.org/program-for-conways-game-of-life/), [many](https://realpython.com/conway-game-of-life-python/), [many](https://spicyyoghurt.com/tutorials/javascript/conways-game-of-life-canvas) great tutorials on the topic, but I tinkered with this recently and figured I'd share my approach.  Specifically, I noticed most Javascript implementations and tutorials use the [HTML5 Canvas](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API) but I'll offer here a DOM approach.  I'm aiming for an audience of beginner (but not novice) level familiarity with HTML, CSS, and JS. 

**The finished project is live and you can [play with it here](https://stmorse.github.io/life-js/), or check out the [source on Github](https://github.com/stmorse/life-js).**


## The Game of Life

The game occurs on a grid of cells which are either "alive" or "dead" and evolve turn-by-turn according to the following rules:

1. Any cell adjacent to 3 alive neighbors becomes alive (if it wasn't already).
2. Any cell with more than 3 neighbors dies due to overpopulation.
3. Any cell with fewer than 2 neighbors dies due to underpopulation.

Some notes (important when we start coding this up):

- These rules together imply any cell with exactly 2 neighbors remains as-is, whether alive or dead.  
- These conditions are evaluated simultaneously, then applied simultaneously, in each turn.  
- "Neighbor" refers to all 8 neighbors of a cell, including diagonals.  
- The Game of Life was originally designed for a 2-dimensional square grid, but you could certainly modify this to a hexagonal grid, more dimensions, or for that matter make countless other modifications to the rules or environment.

These simple rules lead to surprisingly interesting behavior that can seem lifelike, and there are many well-known patterns (or "lifeforms") of interest.  For example, some small still-life:

<div style="display: inline-flex; text-align: center">
<img align="center" width="40%" src="{{ site.github.url }}/images/2024/conway/block.png" alt="Block">
<img align="center" width="40%" src="{{ site.github.url }}/images/2024/conway/boat.png" alt="Boat">
</div>

And some small moving ones:

<div style="display: inline-flex; text-align: center">
<img align="center" width="30%" src="{{ site.github.url }}/images/2024/conway/blinker.gif" alt="Spinner">
<img align="center" width="30%" src="{{ site.github.url }}/images/2024/conway/toad.gif" alt="Toad">
<img align="center" width="30%" src="{{ site.github.url }}/images/2024/conway/LWSS.gif" alt="Spaceship">
</div>

It's worthwhile to pause and examine the behavior of these to understand, cell by cell, why they behave the way they do.

Then there's some bigger ones, like Gosper's glider gun.  Interesting story (h/t [Sagnik Battachaya](https://sagnikb.github.io/blogposts/Game-of-Life/)) -- the glider gun's existence implies that initial patterns with a finite number of cells can eventually lead to configurations with an infinite number of cells. John Conway conjectured that such a thing was impossible but Bill Gosper discovered the Gosper Glider Gun in 1970, winning $50 from Conway.

<img align="center" width="90%" src="{{ site.github.url }}/images/2024/conway/Gospers_glider_gun.gif" alt="Glider gun">

Anyway, this is obviously a deep rabbit hole of experimentation, and there's [lots of resources](http://www.ericweisstein.com/encyclopedias/life/) if you want to keep going deeper, but let's get on to our purpose which is to code it up so we can tinker with lifeforms ourselves.


## Design choices

One of our first choices is whether to use a Canvas or DOM (Document Object Model) approach.  If you're unfamiliar with this distinction, [this comparison](https://www.kirupa.com/html5/dom_vs_canvas.htm) is a good overview.  In short, the Canvas allows us to directly and efficiently manipulate shapes on the screen, with fast performance but at the cost of coding ease -- shape locations, event listening, etc. all need to be handled explicitly by us.  Canvas is undeniably faster -- I think [Google Docs switched to Canvas](https://thenewstack.io/google-docs-switches-to-canvas-rendering-sidelining-the-dom/) a few years ago -- and so a lot of Game of Life implementations use it, aiming for gigantic grids.  

Using the DOM, whether that's a bunch of `<svg>` elements or a bunch of `<div>`s, requires us to direct the browser to arrange objects the way we want, at the cost of control and performance but with the opportunity to benefit from coding ease (think CSS and event handlers).  Since my goal is to make a medium-size Game of Life grid, with at *most* a thousand or so objects, and to be able to have fun with adding UI features, I thought it'd be nice to implement using the DOM and CSS.  In fact, since we're just dealing with a bunch of squares, I'll just use strictly `div`s and no SVG at all.

**Other notes:**

This project is very small and we could do all in one `index.html` file, but I'll split out CSS to a `style.css` file and the JS to a `game.js` file, all in one directory.

I will be using jQuery!  No judgment!  Of course this could be made [without jQuery](https://youmightnotneedjquery.com) but it certainly makes life easier with this amount of DOM manipulation and working in straight JS.

Lastly, I'm attempting to keep this as unadorned as possible so I won't use classes or really any frameworks at all for the code.  The model will be an array of arrays, all functions will be in the global namespace, etc.

Now onto the code.


## The grid

To code up a medium-size square grid, we'll use a `<div>` in the body of our html to contain all the grid elements, use JS to populate this with a bunch of cells (also `<div>`s), and then use [CSS Grid](https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_grid_layout) to structure everything.

So in `index.html` we'll put this in `<body>`:
```html
<div id="grid"></div>
```
and then setup the grid with some CSS:
```css
:root {
    --gridwidth: 20;
    --gridheight: 10;
    --cellsize: 50px;
}

#grid {
    text-align: center;
    display: grid;
    grid-template-rows: repeat(var(--gridheight), var(--cellsize));
    grid-template-columns: repeat(var(--gridwidth), var(--cellsize));
    justify-content: center; /* centers cells in grid div */
}

.cell {
    background-color: white;
    position: relative;
    border: 0.5px solid lightgray;
    width: calc(var(--cellsize) - 1);
    height: calc(var(--cellsize) - 1);
}

.active {
    background-color: blue;
}
```
Some key things to highlight:
- We're using `:root` to set the grid variables.  This allows us to use them in other CSS formatting and we can set them from our script later.
- The `display: grid` establishes the `#grid` div with the CSS Grid framework.  The `grid-template-` styling forces the rows and columns to a certain number and size.
- We'll add the class `active` to a cell when it's alive, with the base class representing dead.

Finally, in `game.js` we'll have the following code:
```javascript
// grab these variables from :root
let gridheight = document.documentElement.style.getProperty('--gridheight');
let gridwidth = document.documentElement.style.getProperty('--gridwidth');

// initialize a state array to zeros
// this will represent the grid of cells -- 0 for dead, 1 for alive
let state = Array(gridheight).fill(0).map(x => Array(gridwidth).fill(0));

// set a few cells to alive
state[2][2] = 1;
state[2][3] = 1;
state[2][4] = 1;

function initGrid() {
    // grab the grid div
    let $grid = $('#grid);

    for (let i=0; i<gridheight; i++) { // row
        for (let j=0; j<gridwidth; j++) { // col
            // define this cell
            let $cell = $('<div>').addClass('cell');

            // anywhere we turned on state, set to active
            if (state[i][j] == 1) {
                $cell.addClass('active');
            } 

            // append this cell to grid
            $grid.append($cell);
        }
    }
}    
```

Now let's cover the basics of the main game loop.


## The game loop

Now we'll setup a loop to update the model (`state`) and visualization.  Basically, we'll create a function called `step()` that does one update to the grid, then call it iteratively.

```javascript
function step() {
    // create temp state to store active/non for next step
    let temp = Array(gridheight).fill(0).map(x => Array(gridwidth).fill(0));

    // check surrounding
    for (let i=0; i<gridheight; i++) {
        for (let j=0; j<gridwidth; j++) {
            // count the surrounding population
            let numAlive = 0;
            for (let m=-1; m<=1; m++) {
                for (let n=-1; n<=1; n++) {
                    numAlive += isAlive(i+m, j+n);
                }
            }
            numAlive -= isAlive(i,j); // don't count center

            // game logic
            if (numAlive == 2) { // do nothing
                temp[i][j] = state[i][j];
            } else if (numAlive == 3) { // make alive
                temp[i][j] = 1;
            } else { // make dead
                temp[i][j] = 0;
            }
        }
    }

    // apply new state to state
    for (let i=0; i<gridheight; i++) {
        for (let j=0; j<gridwidth; j++) {
            state[i][j] = temp[i][j];

            // apply to viz
            let $c = $(`#cell_${i}_${j}`);
            if (state[i][j] == 1) {
                $c.addClass('active');
            } else {
                $c.removeClass('active');
            }
        }
    }

    // update state counter
    numSteps++;
    $('#stepcounter').html(`Step: ${numSteps}`);

    // take another step
    setTimeout( () => {
        step();
    }, 100);
}
```

All that's left is to call these two functions:

```javascript
// wait to run until DOM is ready
$(document).ready(function() {
    initGrid();
    step();
})
```

And we're off!

## Bells and whistles

There are seemingly endless ways we can take this basic structure and jazz it up.  For a few basic UI ideas:

 - **Add buttons to start/stop the loop.**  This is straightforward with our setup.  Add the buttons in the html as divs:
 ```html
 <div class="button" id="btn-run">Run</div>
 <div class="button" id="btn-pause">Pause</div>
 ```
 then style it with css:
 ```css
 .button {
    width: 100px;
    text-align: center;
    margin: 2px;
    padding: 2px;
    border: 1px solid black;
    border-radius: 10px;
}
```
and give it functionality on click with JS:
```javascript
// maintain a global to track if we're running or not
let isRunning = false;

// RUN button
$('#btn_run').click(function(e) {
    if (isRunning) {
        return;  // if we're already running, do nothing
    }

    // need to set this outside the step(), which is called internally
    isRunning = true;
    step();
});

// PAUSE button
$('#btn_pause').click(function(e) {
    isRunning = false;
});
```

- **Make a random initialization option.**  Add another button like before, but make its functionality something like:
```javascript
function initGrid() {
    ...
    // add this line after you define $cell
    // this allows us to select a specific cell later
    $cell.attr('id', `cell_${i}_${j}`);
    ...
}

function random(p) {
    // p needs to be a number between 0 and 1, 
    // to set the amount of randomness
    
    // stop the game
    isRunning = false;

    // clear all cells
    state = Array(gridheight).fill(0).map(x => Array(gridwidth).fill(0));
    $('.cell').removeClass('active');

    // make proportion `p` of cells active
    for (let i=0; i<gridheight; i++) {
        for (let j=0; j<gridwidth; j++) {
            if (Math.random() < p) {
                $(`#cell_${i}_${j}`).addClass('active');
                state[i][j] = 1;
            }
        }
    }
}
```

- Other ideas: presets, highlighting on hover, ... I implemented a few of these in the [full code](https://github.com/stmorse/life-js) over on Github.


### More interesting mods

More interesting than UI/UX fiddling, I think, is modifying the structure of the game.  Here's a version of the game on a [hexagonal grid](https://arunarjunakani.github.io/HexagonalGameOfLife/), which is a slightly more daunting coding challenge.  There's tons of these [different variations](https://cs.stanford.edu/people/eroberts/courses/soco/projects/2008-09/modeling-natural-systems/gameOfLife2.html) on the rules, like different colors of cells, 3-dimensions, or constrained grids.

That's all for now!  Hope you've enjoyed, let me know your thoughts.