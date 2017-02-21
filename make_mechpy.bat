:: activate conda env for python 2.7
call activate py27

:: create a variable of the doconce file
SET dofile=mechpy

:: create a styled html file
doconce format html %dofile% --html_style=bootswatch_journal
ren %dofile%.html %dofile%.webpage.html

pause