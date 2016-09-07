#!/bin/bash

## activate conda env for python 2.7
source activate py27

## create a variable of the doconce file
#dofile='mechpy.do.txt'
dofile='mechpy'

## spellcheck all files
#doconce spellcheck *.do.txt

## create the github markdown from the doconce_notes.do.txt file
#doconce format pandoc $dofile --github_md

## create ipython notebook file
#doconce format ipynb $dofile

## create a styled html file
#doconce format html $dofile --html_style=tactile-black
#doconce format html $dofile --html_style=bootswatch_journal

## create markdown slides
#doconce format pandoc $dofile --github_md
#doconce slides_markdown $dofile remark --slide_theme=light

## creaet html files for html slides
#doconce slides_html $dofile reveal --html_slide_theme=beige

## create reveal html slides
#doconce format html $dofile --pygments_html_style=autumn --keep_pygments_html_bg SLIDE_TYPE=reveal SLIDE_THEME=simple
#doconce slides_html $dofile reveal --html_slide_theme=simple

### sphinx
## initialize
#doconce sphinx_dir version=0.1 dirname=sphinx theme=bootstrap $
## create sphinx documentation
#doconce format sphinx $dofile
#python automate_sphinx.py
#firefox sphinx/_build/html/index/html

## updating github pages
doconce format html $dofile --html_style=bootswatch_journal
python ipynb_to_html.py
mv mechpy.html web/index.html
ghp-import web
rm web/*.html
git add --all
git commit -m 'updated doc webpage'
git push --all origin


