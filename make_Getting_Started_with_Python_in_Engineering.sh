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
doconce replace "http://netdna.bootstrapcdn.com/bootswatch/3.1.1/journal/bootstrap.min.css" "bootstrap.css" $dofile.html
doconce replace "http://ajax.googleapis.com/ajax/libs/jquery/1.10.2/jquery.min.js"          "jquery.min.js" $dofile.html
doconce replace "http://netdna.bootstrapcdn.com/bootstrap/3.0.0/js/bootstrap.min.js"        "bootstrap.js" $dofile.html
mv $dofile.html $dofile.webpage.html

## creaet html files
#doconce slides_html $dofile reveal --html_slide_theme=beige

#doconce format html $dofile --pygments_html_style=autumn --keep_pygments_html_bg SLIDE_TYPE=deck SLIDE_THEME=swiss
#doconce slides_html $dofile deck --html_slide_theme=swiss

### make a pdf from markdown
#dofile='Getting_Started_with_Python_in_Engineering'
#doconce format pdflatex $dofile --latex_code_style=pyg --latex_title_layout=std
#pandoc $dofile.md --latex-engine=xelatex -o $dofile.pdf

####create tex file
doconce format pdflatex $dofile --latex_code_style=vrb --latex_title_layout=std  --latex_section_headings=blue --latex_colored_table_rows=blue --no_abort --device=screen --latex_preamble=customization.tex
pdflatex -shell-escape -interaction=batchmode $dofile.tex

#jupyter nbconvert "Getting_Started_with_Python_in_Engineering.ipynb" --to slides --post serve

## create reveal html slides
#doconce format html $dofile --pygments_html_style=default --keep_pygments_html_bg SLIDE_TYPE=reveal SLIDE_THEME=simple
#doconce slides_html $dofile reveal --html_slide_theme=simple
#mv $dofile.html $dofile.revealslides.html

doconce format html $dofile --pygments_html_style=default --keep_pygments_html_bg SLIDE_TYPE=deck SLIDE_THEME=swiss
doconce slides_html $dofile deck --html_slide_theme=swiss
mv $dofile.html $dofile.deckslides.html

# removes the trash directory
rm -R Trash

# remove all the files that are generated during doconce make
rm *.toc *.log *.aux *.out *.idx *.bbl *.blg *.gz
