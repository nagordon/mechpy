activate py27

SET dofile=mechpy

doconce format ipynb %dofile%

jupyter nbconvert --to html %dofile%