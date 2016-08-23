% Mechpy Documentation
% **Neal Gordon**
% Aug 22, 2016

create some darn documentation and demonstrations of mechpy


<!-- Table of contents: Run pandoc with --toc option -->



# Introduction
<div id="sec:intro"></div>

Mechpy is a python library for mechanical engineers


# Headings, Labels, and References
<div id="sec:hlr"></div>

For simple documents, chapters are not necessary, so only 7= are necessary to create a header for the section.


```
chapter	         ========= Heading ========= (9 =)
section	         ======= Heading ======= (7 =)
subsection	     ===== Heading ===== (5 =)
subsubsection	 === Heading === (3 =)
paragraph	     __Heading.__ (2 _)
abstract	     __Abstract.__ Running text...
appendix	     ======= Appendix: heading ======= (7 =)
appendix	     ===== Appendix: heading ===== (5 =)
```

```python
# a comment in my python code
def f(x):
    return 1 + x
```

## Comments and Footnotes
<div id="sec:com"></div>

Creating comments in the text is a handy way to supplement with optional information, one way is to use an inline comment such as [hpl 1: here I will make some
remarks to the text]. Another way to add content is to add a footnote [^footnote] is also possible.

[^footnote]: The syntax for footnotes is borrowed from Extended Markdown.



## Handy Links
<div id="sec:links"></div>

main site <https://github.com/hplgit/doconce>

[tutorial](http://hplgit.github.io/doconce/doc/pub/tutorial/tutorial.html)  


# Sundries

In windows, create a batch file (*.bat) to run a jupyter notebook server in the current directory

```
:: Use to launch jupyter notebooks

:: change console to the current working directory
Pushd "%~dp0"

:: launch jupyter notebook
jupyter notebook

:: write html output
jupyter nbconvert --to html mechpy.ipynb

pause

```

