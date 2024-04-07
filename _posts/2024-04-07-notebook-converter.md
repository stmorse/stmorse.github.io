---
layout: post
title: "A Simple Handmade Jupyter Notebook to Markdown converter"
categories: journal
date: 2024-04-07
tags: ['python', 'jupyter', 'projects']
---

I like to write short tutorials on applied math concepts, with a large dose of tinkering, in [Jupyter/IPython notebooks](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html).  These are mostly to help re-educate future me, but are also hopefully helpful to other random people on the internet.  Instead of just sharing the actual notebook on my Github (which would probably actually be more helpful to some...), I reformat them as blog posts and put them on this blog.

There are prefab libraries out there that already do this -- the main one is [`nbconvert`](https://nbconvert.readthedocs.io/en/latest/) -- but they are, in my experience, way more horsepower than I actually need, a little finnicky, require lots of dependencies, and are not necessarily easily customizable. Okay they're great actually, but I like to roll my own and so I wrote my own converter that I'll share the bones of here.  Most of this is fairly straightforward, but the image conversion I thought was particularly clever and worth sharing.


## Basics

The first amazing thing to note is that a Jupyter notebook is just a big [JSON](https://en.wikipedia.org/wiki/JSON) file.  Try opening a notebook as raw text and you'll see something like:

```
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [ ... ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import scipy.stats\n",
    "from scipy.spatial import distance_matrix"
   ]
  }, ...
```

In the `cells` entry, is a list of the cells of the notebook, coded as `markdown` or `code`, with the `source` entry giving the text.  When you run the notebook, it interprets these cells and for code puts the output in `outputs`.  Simple as that.

So, at a high level, all we have to do to convert the notebook is to iteratively append the markdown (directly) and the code (with some formatting) from this JSON into a markdown file.  We'll hit a wrinkle with the **image** outputs (i.e. plots), but more on this later.

To start, we'll create a simple Python script named `converter.py` with the starting structure:

```python
import json    # to parse the notebook's JSON
import os      # to deal with the filesystem
import sys     # to grab command-line arguments
import re      # for some REGEX
import base64  # to deal with images

def main(fname):
    # fname is the filename of our notebook
    f = open(fname)
    book = json.load(f)

    # grab the name of the notebook to use as the name of the md file
    name = fname[:-6]   # -6 cuts off the .ipynb extension
    md_file_name = name + '.md'

    # make a subdirectory for this markdown file and embedded images
    os.mkdir(name)

    # `text` will be a huge string with the entire markdown of our file
    text = ''

    # TODO: build the markdown header and append to `text`
    
    # iterate through `cells`
    for cell in book['cells']:
        # TODO: convert code/markdown input and output to `text`

    with open(name + '/' + md_file_name, 'w') as f: 
        f.write(text)

# standard boilerplate to run `main` when `converter.py` is run from 
# the command line
if __name__ == "__main__":
    # sys.argv[1] will give us the first argument passed to converter.py
    # note: argv[0] is "converter.py" itself
    main(sys.argv[1])
```

Now all we have to do is fill in the TODO's above for the header, code sections, and markdown sections, which we'll do now.


## Making the header

This is the easiest part.  My blog uses Jekyll and so needs a short header with pre-set labels to define the layout, title, tags, etc for the post, which Jekyll then compiles into the static HTML for the actual post.

The header looks something like this:

```
---
layout: post
title: "Fourier Transforms (with Python examples)"
categories: journal
date: 2024-04-06
tags: ['python', 'mathematics']
---
```

We can use Python's [f-strings](https://realpython.com/python-f-strings/) and some educated guesses at what the different entries will be, to do a pretty good first swipe at this, knowing we'll inevitably want to go in and manually edit afterward.

```python
# (in main)

# it's important to avoid extra spaces (or even tabs!) 
# in this as they'll show up in the md
post_header = '---\n\
layout: post\n\
title: "{title}"\n\
categories: journal\n\
date: {date}\n\
tags: {tags}\n\
---'

post_title = name
post_date = '2024-xx-xx'
post_tags = ['python', 'mathematics']

text += post_header.format(title=post_title, date=post_date, tags=post_tags)
text += '\n\n'
```

That's it!  You could probably hang some bells and whistles on this by using today's date (`datetime`), or some fancy word frequency count for the tags ... or just adjust manually later.


## Adding the markdown

Ok recall in our `main` function we're looping through the list of `cells` in the notebook's JSON.  Inside that loop, we'll add an `if/elif` block to select on whether we've got code or markdown.

```python
# (in main)
for cell in book['cells']:
    # code block
    if cell['cell_type'] == 'code':
        # TODO

    elif cell['cell_type'] == 'markdown':
        # TODO
```

That markdown block we'll handle now.  Really all we *have* to do is something like this:

```python
# get content of markdown
temp = ''.join(e for e in cell['source'])

# add it to the `text` master
text += '\n'
text += temp
text += '\n'
```

But for my blog, I have two additional things to deal with.  First, for whatever reason I decided to set Mathjax to require two `$`s, whereas Jupyter just wants one, so I have to add in this insane Regex maneuver to add in the extra dollar signs:

```python
# replace $...$ with $$...$$ for my jekyll build :|
# REGEX explainer of r'([^\$])(\$[^\$]+\$)([^\$])'
# grabs 3 groups: preceding char, $...$, trail char. ignores $$
temp = re.sub(
    r'([^\$])(\$[^\$]+\$)([^\$])', 
    lambda mo: mo.group(1) + '$' + mo.group(2) + '$' + mo.group(3), 
    temp
)
```

Another little quirk I discovered is that Mathjax doesn't like the `|` symbol, so I convert that to the explicit `\vert`, more Regex:

```python
temp = re.sub(
    r'([\|]+)(.)',
    lambda mo: r'\vert'*len(mo.group(1)) + (' ' if mo.group(2) == ' ' else ' ' + mo.group(2)),
    temp
)
```

which is actually a bit buggy I've found, but mostly works -- anyway you add these Regex maneuvers in and then go through the `text += temp` stuff.


## Decoding the code

For the last bit, we'll convert the code.  I saved this for last because we've also got to handle images, which are tricky.

For the inputs, it's easy enough, you can just do something like:

```python
text += '\n```python\n'
text += ''.join(e for e in cell['source'])
text += '\n```\n\n'
```

For the outputs, first note there could be multiple (for example text and image), so we'll do a loop and handle both.  I use a `try/except` because sometimes the output will have an empty or near-empty entry and this seemed the easiest way to handle that.

```python
for o in cell['outputs']:
    try:
        # handle code outputs
        if 'text/plain' in o['data']:
            text += '\n```\n'
            text += '\n'.join(e for e in o['data']['text/plain'])
            text += '\n```\n'
        
        # handle images
        if 'image/png' in o['data']:
            # TODO
    except KeyError:
        pass
```

Finally, for images, let's first note that images are actually stored in the JSON as a giant string!  Take a look at an example:

```
"outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAtEAAAHSCAYAAAAqtZc0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4 ...
```

It turns out, this is a byte-string representing the vectorized image.  Through a little StackOverflowing, I found you can convert this to an image file (like a PNG) like this:

```python
# grab raw image byte string
s = o['data']['image/png']

# save image
ifname = f'image{img_k}.png'
with open(name + '/' + ifname, 'wb') as f:
    # encode converts string->bytes, decode converts to img
    f.write(base64.decodebytes(s.encode('latin-1')))
    img_k += 1
```

where `img_k` is just a counter I defined earlier in the function to keep running track of which image we're on.  So this saves the file to our working direction.  We now just need to put a little HTML in our markdown to display the image.

```python
# add image include to markdown text
img_embed_stem = r'{{ site.github.url }}/images/2024/' + name + '/'
img_format = '<img align="center" width="90%" \
src="{stem}{filename}" alt="{alttext}">'

text += '\n'
text += img_format.format(stem=img_embed_stem, filename=ifname, alttext=ifname)
text += '\n'
```

And that's it!

## Putting it in action

To run this, just make sure the `converter.py` script is in the same folder as your notebook (or modify the filepath accordingly) and do

```
~/yourfolder $ python3 converter.py yournotebook.ipynb
```

And it will create a folder called `yournotebook` with a `yournotebook.md` file and (if applicable) a bunch of `.png` files.

You'll probably have to do some editing of the markdown before it's ready to publish, but I'd say this script gets you 90 percent of the way there.  And, I dunno, I just like knowing exactly what the script is doing.

Hope this was helpful to you!

Here's the full code just for good measure:

```python
import json
import re
import base64
import os
import sys

def main(fname):
    f = open(fname)
    book = json.load(f)

    name = fname[:-6]

    post_title = name
    post_date = '2023-xx-xx'
    post_tags = ['python', 'mathematics']

    post_header = '---\n\
layout: post\n\
title: "{title}"\n\
categories: journal\n\
date: {date}\n\
tags: {tags}\n\
---'

    md_file_name = name + '.md'

    img_embed_stem = r'{{ site.github.url }}/images/2024/' + name + '/'
    img_format = '<img align="center" width="90%" \
    src="{stem}{filename}" alt="{alttext}">'

    # make subdirectory 
    os.mkdir(name)

    text = ''
    img_k = 0

    text += post_header.format(title=post_title, date=post_date, tags=post_tags)
    text += '\n\n'
    for cell in book['cells']:
        # code block
        if cell['cell_type'] == 'code':
            text += '\n```python\n'
            text += ''.join(e for e in cell['source'])
            text += '\n```\n\n'

            # handle text or image outputs
            for o in cell['outputs']:
                try:
                    # handle code outputs
                    if 'text/plain' in o['data']:
                        text += '\n```\n'
                        text += '\n'.join(e for e in o['data']['text/plain'])
                        text += '\n```\n'
                    
                    # handle images
                    if 'image/png' in o['data']:
                        # grab raw image byte string
                        s = o['data']['image/png']

                        # save image
                        ifname = f'image{img_k}.png'
                        with open(name + '/' + ifname, 'wb') as f:
                            # encode converts string->bytes, decode converts to img
                            f.write(base64.decodebytes(s.encode('latin-1')))
                            img_k += 1

                        # add image include to markdown text
                        text += '\n'
                        text += img_format.format(stem=img_embed_stem, filename=ifname, alttext=ifname)
                        text += '\n'
                except KeyError:
                    pass

        # markdown block
        elif cell['cell_type'] == 'markdown':
            # get content of markdown
            temp = ''.join(e for e in cell['source'])

            # replace $...$ with $$...$$ for my jekyll build :|
            # REGEX explainer of r'([^\$])(\$[^\$]+\$)([^\$])'
            # grabs 3 groups: preceding char, $...$, trail char. ignores $$
            temp = re.sub(
                r'([^\$])(\$[^\$]+\$)([^\$])', 
                lambda mo: mo.group(1) + '$' + mo.group(2) + '$' + mo.group(3), 
                temp
            )

            # TODO: this is buggy
            # my jekyll build also doesn't like |
            # replace with \vert
            temp = re.sub(
                r'([\|]+)(.)',
                lambda mo: r'\vert'*len(mo.group(1)) + (' ' if mo.group(2) == ' ' else ' ' + mo.group(2)),
                temp
            )

            text += '\n'
            text += temp
            text += '\n'

    with open(name + '/' + md_file_name, 'w') as f:
        f.write(text)

if __name__ == "__main__":
    main(sys.argv[1])
```