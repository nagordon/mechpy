#!/bin/bash

## activate conda env for python 2.7
source activate py27

## spellcheck all files
#doconce spellcheck *.do.txt

## Convert ipynb to doconce
#doconce ipynb2doconce notebook.ipynb

## diff files
#doconce diff file1.txt file2.txt

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
#mv *.html web

cp mechpy.do.txt index.do.txt

## create a variable of the doconce file
#dofile='mechpy.do.txt'
doconce format html index --html_style=bootswatch_journal
doconce replace "http://netdna.bootstrapcdn.com/bootswatch/3.1.1/journal/bootstrap.min.css" "bootstrap.css" index.html
doconce replace "http://ajax.googleapis.com/ajax/libs/jquery/1.10.2/jquery.min.js"          "jquery.min.js" index.html
doconce replace "http://netdna.bootstrapcdn.com/bootstrap/3.0.0/js/bootstrap.min.js"        "bootstrap.js" index.html
rm index.do.txt

for i in ../tutorials/*.ipynb; do
    jupyter nbconvert --to html $i
done

mv ../tutorials/*.html .

cd ..

ghp-import doc -m "updated doc webpage on gh-pages branch" #-p    ##-p is a push

rm doc/*.html

### add all master branch files
git add --all
git commit -m 'auto add changes to master branch and updated documentation'

git push --all origin
