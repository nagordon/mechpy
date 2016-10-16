# -*- coding: utf-8 -*-
"""
convert ipython notebooks to html file and move them to web


old code, now replace with bash loop

for ipynb in glob('../mechpy/*.ipynb'):
    #get_ipython().system('jupyter nbconvert --to html ' + ipynb)
    #fb = os.path.splitext( os.path.basename(ipynb) )[0]
    #fbe = os.path.join( 'web',fb+'.html')
    subprocess.call('jupyter nbconvert --to html '+ipynb, shell=True)


"""

from glob import glob
import os, shutil, subprocess

os.chdir('web')

print('\n')
for html in glob('*.html'):    
    print('URL:"https://nagordon.github.io/mechpy/'+html+'"  <linebreak>')
    #shutil.move(html, os.path.join( 'web',html)    )
print('\n')

os.chdir('..')
