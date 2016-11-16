
:: activate conda env for python 2.7
call activate py27

:: create a variable of the doconce file
SET dofile="Getting_Started_with_Python_in_Engineering"

:: create ipython notebook file
:: doconce format ipynb %dofile%

:: create a styled html file
doconce format html %dofile% --html_style=bootswatch_journal
ren %dofile%.html %dofile%.webpage.html

:: create reveal html slides
doconce format html %dofile% --pygments_html_style=default --keep_pygments_html_bg SLIDE_TYPE=deck SLIDE_THEME=swiss
doconce slides_html %dofile% deck --html_slide_theme=swiss
ren %dofile%.html %dofile%.deckslides.html

pause