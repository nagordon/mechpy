#!/bin/bash

# create a variable of the doconce file
dofile='mechpy.do.txt'

#create the github markdown from the doconce_notes.do.txt file
doconce format pandoc $dofile --github_md

# create ipython notebook file
#doconce format ipynb $dofile

# create a styled html file
#doconce format html $dofile --html_style=tactile-black

doconce format html $dofile --html_style=bootswatch_journal
