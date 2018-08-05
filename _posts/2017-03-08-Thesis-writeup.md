---
layout: post
title: "Custom LaTeXTools builder for a thesis"
categories: journal
tags: [projects, latex, thesis]
---

In about two months, my thesis will be (Lord willing) in final drafts, and I will be getting ready for the move to New York and teaching cadets.  However, in order to fulfill this prophesy I need to first ... write my thesis.  This post will walk-through my setup --- in short, I am using LaTeX, with chapter-specific bibliographies organized with Mendeley, and writing and compiling in Sublime Text with the LaTeXTools plugin and a custom builder.  If you're not already using LaTeX, this system is a tough sell --- but if you are, I think this post might be immensely helpful.  It definitely would have been for me.

There are a lot of these "how to organize your thesis in LaTeX" tutorials [out there](https://www.sharelatex.com/blog/2013/08/02/thesis-series-pt1.html) and a ton of [templates](https://www.overleaf.com/gallery/tagged/thesis#.WMCzHxjMyL8) too (here's a [really nice one](https://www.ctan.org/tex-archive/macros/latex/contrib/classicthesis/?lang=en)), but in my experience none that cover both the structure and the build environment.  Hopefully this is helpful to someone.


## File structure

The time-tested structure for a large document like this is to separate out the chapters into separate files: one `main.tex` master file and multiple `chapterX.tex` files.  To merge everything together, you have your choice between native LaTeX commands like `\input` or `\include`, and packaged systems like `subfiles` or `standalone`.  I found `\include` had enough functionality for what I needed, so let's check it out.

Your file system will look something like this:
```
thesis/
  main.tex
  images/
    img1.pdf
    ...
  chapters/
    chapter1.tex
    chapter2.tex
    ...
```

The `main.tex` should look something like:
```
\documentclass[12pt]{report}
\usepackage[utf8]{inputenc}
\usepackage[a4paper,margin=1in,headheight=15pt]{geometry}
\usepackage{chapterbib}

\graphicspath{ {images/} {../images/} }

\title{Yo Title}
\author{I. Cube}
\date{$\pi$, 2017}

\begin{document}

\maketitle

\chapter*{Abstract}
Abstract goes here...

\chapter*{Acknowledgements}
I want to thank...

\tableofcontents

\include{chapters/chapter1}
\include{chapters/chapter2}
\include{chapters/chapter3}

\end{document}
```
(and if you have a ton of chapters, you could use a `\foreach` loop.)

Most of this is standard boilerplate, but note that whatever preamble information we have in `main.tex` will apply to all chapter files.  If this isn't going to work for you, and you need separate preambles, you should consider perhaps the `standalone` method instead.  Note also we are using the package `chapterbib` to get chapter-specific bibliographies, which we'll cover in the next section.  We use the `\graphicspath{}` to specify the universal prefix for any `\includegraphics` commands in the parent file (`{images/}`) and any subdirectory files like our chapters (`{../images/}`).  

We use `\include` statements to plug in all the chapters: note that `\include` is basically just `\input` with a `\clearpage` and `\newpage` at the beginning and end.  Also, we can add an `\includeonly` statement in our preamble such as
```
\includeonly{chapters/chapter1,chapters/chapter2}  % no space!
```
to only compile some of the chapters.  This lets you work on one part of the thesis at a time.

Now each chapter looks like:
```
Text of the chapter

\bibliographystyle{acm}
\bibliography{/Users/you/path/to/library.bib}
```
That's it!  We are using an absolute path to some master `library.bib` file, so that LaTeXTools can do autofill but it doesn't confuse the compiler which is working from the parent directory, but more on that later.


### Compiling (the hard way)

Because we want to have a chapter-specific bibliography at the end of every chapter, we needed that `\usepackage{chapterbib}` include and to have the bibliography include at the end of every subfile.  We **also** need to do a little extra work in the compilation process, as described in the [`chapterbib` docs](http://texdoc.net/texmf-dist/doc/latex/cite/chapterbib.pdf).  To compile the thesis, we need to:

1. Run `pdflatex main.tex`, which will create an `.aux` file for each included chapter.
2. Run `bibtex chapters/chapterX` for each included chapter.  (This is really run on the `.aux` file, but you don't have to include the file extension.)
3. Run `pdflatex main.tex` again (to fix the bib references).
4. Run `pdflatex main.tex` again (to fix the fig references).

We can do this all from the command line, assuming we have the LaTeX binaries in our PATH.  But this is a lot of work.  Let me mention a couple of other tools that can simplify this whole process, including compilation.


## Some useful tools

### Sublime Text/LaTeXTools

Writing LaTeX can be a pain.  I used to write everything by hand in a very bare-bones text editor, but as my output requirements grew, and I had typed something like the `\begin{figure} \centering \includegraphics[width=.....` boilerplate for the 100000-th time, I started looking for options.

[Sublime Text](https://www.sublimetext.com) is a widely used (and beautiful) text editor for OS X and Windows with a huge ecosystem of plugins.  It is free to evaluate, and starts asking for payment after continued use.  One basic feature we can use (that is common to many text editors) is Projects.  We can open a new window with all our files for our thesis, and do Projects -> Save Project As...  We can now set settings specific to our project, load all the files together at once, etc.  We can also specify a custom build pattern to remove the awkward by-hand stuff we were doing, which we'll cover at the end.

One of the most essential plugins (maybe *only* essential?) for Sublime Text is [LaTeXTools](https://github.com/SublimeText/LaTeXTools), which enhances Sublime Text for LaTeX editing and compilation with all kinds of thing like previews, autofills, etc.  For example, the "Snippets" feature turns things like the `\begin{figure}` boilerplate into a couple keystrokes.

It is pretty straightforward to setup a default build environment so that instead of compiling from the command line, you can compile with `Cmd+B` within Sublime Text (this mimics functionality in other IDEs like TexnicCenter).  However, LaTeXTools also allows easy customization of build environments, which we will take advantage of later.  


### Mendeley

Managing a bibliography is a pain.  `.bib` files are cryptic, as your number of references grow it is near impossible to keep everything organized, ...  [Mendeley](https://www.mendeley.com) to the rescue.  This web/desktop app provides an interface to organize all your PDF papers, sync them with an online backup repository, and automatically create a synced `.bib` formatted file.  It actually automatically generates all the relevant bibliography information from the PDF, with typically only minor (or zero) tweaks needed.  You download a few papers, re-sync your Mendeley library, and they are immediately available for citation.  (If you use [Overleaf](https://www.overleaf.com), you can now also access your Mendeley library in Overleaf projects.)

This is all nice, and sounds very fancy, but I wasn't completely sold on it until I realized the integration with the LaTeXTools plugin for Sublime Text.

In particular, LaTeXTools will autofill citations from a `.bib` files that is referenced in the `.tex` file.  Mind. Blown.  So here's the workflow:

1. Chugging along writing your thesis in Sublime ...
2. Download new papers containing relevant research.
2. Ensure their metadata is correct in Mendeley and re-sync with your master `library.bib` file.
3. Return to Sublime, type `\cite{` and a popup menu will appear with the new papers in the list.

I know, right?

Now let's put it all together.


## Compiling

We have a sensible file structure, a LaTeX setup that allows chapter-specific bibliographies, and a user-friendly editing environment.  We just need to crack the problem of the painful build sequence.  

We can create a custom builder within LaTeXTools, and then assign this as the default builder for our thesis Sublime Project.  Here's how:

First, write the custom builder.  Builders are written in Python, and are all pretty straightforward extensions of a baseclass `pdfBuilder`.  You can peek around at the included builders in the `Sublime Text 3/Packages/LaTeXTools/builders/` folder (which you can easily navigate to by going to Preferences -> Browse Packages...), or I recommend checking out [this wiki](https://github.com/SublimeText/LaTeXTools/wiki/Custom-Builders) on custom builders.

Create a new folder in the `Sublime Text 3/Packages/User/` folder called `LaTeXTools-Builders` (or something similar, it doesn't matter), and create a new file called `thesisBuilder.py` with the following code:
```python
from pdfBuilder import PdfBuilder
import os

# here we define the commands to be used
# commands are passed to subprocess.Popen which prefers a list of
# arguments to a string
PDFLATEX = ["pdflatex", "-interaction=nonstopmode", "-synctex=1"]
BIBTEX = ["bibtex"]

class thesisBuilder(PdfBuilder):
    def __init__(self, *args):
        super(thesisBuilder, self).__init__(*args)

        # now we do the initialization for this builder
        self.name = "thesisBuilder"

    def commands(self):
        self.display("\n\nthesisBuilder: ")

        # first run of pdflatex
        # this tells LaTeXTools to run:
        #  pdflatex -interaction=nonstopmode -synctex=1 tex_root
        # note that we append the base_name of the file to the command here
        yield(PDFLATEX + [self.base_name], "Running pdflatex...")

        # LaTeXTools has run pdflatex and returned control to the builder
        # here we just add text saying the step is done, to give some feedback
        self.display("done.\n")

        # now run bibtex
        self.display("Running bibtex...\n")
        for file in os.listdir(self.tex_dir + "/chapters"):
            if file.endswith(".aux"):
                yield(BIBTEX + ["chapters/" + file.rstrip(".aux")], 
                      "  (%s)\n" % file)

        self.display("done.\n")

        # second run of pdflatex
        yield(
            PDFLATEX + [self.base_name],
            "Running pdflatex again..."
        )

        self.display("done.\n")

        # third run of pdflatex
        yield(
            PDFLATEX + [self.base_name],
            "Running pdflatex for the last time..."
        )

        self.display("done.\n")
```

This is a scene-for-scene remake of the example in the wiki, with the exception of the `bibtex` section iterating over all files in the `/chapters` sub-directory which end with `.aux`.  This will ensure we only create bibliography files for those chapters that were compiled with the `\includeonly` command.

Now, with your project open, go to Project -> Edit Project and add to file so it looks like this:
```
{
	... any other folder / other settings ...

	"settings" : {
        "TEXroot": "main.tex",
        "tex_file_exts": [".tex"],
        "builder": "thesisBuilder",
        "builder_path": "User/LaTeXTools-Builders",
        "builder_settings": {
            "options": "--shell-escape"
        }
    }
}
```

This points LaTeXTools to the `thesisBuilder.py` file you just made.

You can now go to your `main.tex` file, press `Command+B`, and get some output like this:
```
[Compiling /Users/you/Documents/THESIS/main.tex]

thesisBuilder: Running pdflatex...done.
Running bibtex...
  (chapter1.aux)
  (chapter2.aux)
  (chapter3.aux)
done.
Running pdflatex again...done.
Running pdflatex for the last time...done.

No errors.

[Done!]
```
