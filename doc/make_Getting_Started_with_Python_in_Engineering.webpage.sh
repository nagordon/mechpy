#!/bin/bash

## activate conda env for python 2.7
source activate py27

## create a variable of the doconce file
dofile='Getting_Started_with_Python_in_Engineering'

## create the github markdown from the doconce_notes.do.txt file
#doconce format pandoc $dofile --github_md

## create ipython notebook file
#doconce format ipynb $dofile

## create a styled html file
#doconce format html $dofile --html_style=tactile-black
doconce format html $dofile --html_style=bootswatch_journal
mv $dofile.html $dofile.webpage.html

## creaet html files
#doconce slides_html $dofile reveal --html_slide_theme=beige

#doconce format html $dofile --pygments_html_style=autumn --keep_pygments_html_bg SLIDE_TYPE=deck SLIDE_THEME=swiss
#doconce slides_html $dofile deck --html_slide_theme=swiss

#jupyter nbconvert "Getting_Started_with_Python_in_Engineering.ipynb" --to slides --post serve

## create reveal html slides
#doconce format html $dofile --pygments_html_style=default --keep_pygments_html_bg SLIDE_TYPE=reveal SLIDE_THEME=simple
#doconce slides_html $dofile reveal --html_slide_theme=simple
#mv $dofile.html $dofile.revealslides.html

doconce format html $dofile --pygments_html_style=default --keep_pygments_html_bg SLIDE_TYPE=deck SLIDE_THEME=swiss
doconce slides_html $dofile deck --html_slide_theme=swiss
mv $dofile.html $dofile.deckslides.html
