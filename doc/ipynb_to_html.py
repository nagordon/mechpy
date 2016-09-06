# -*- coding: utf-8 -*-
"""
convert ipython notebooks to html file
"""

from glob import glob

for ipynb in glob('web/*.ipynb'):
    get_ipython().system('jupyter nbconvert --to html ' + ipynb)
    #os.system('jupyter nbconvert --to html '+ ipynb)

print('copy and paste this code into mechpy.do.txt to update website with examples\n')
for html in glob('web/*.html'):
    print('URL:"https://nagordon.github.io/mechpy/'+html+'"')


