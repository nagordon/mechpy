
.. raw:: html

  <style type="text/css">
    .dropdown-menu {
      height: auto;
      max-height: 400px;
      overflow-x: hidden;
  }
  </style>

.. Automatically generated Sphinx-extended reStructuredText file from DocOnce source
   (https://github.com/hplgit/doconce/)

.. Document title:

mechpy Documentation
%%%%%%%%%%%%%%%%%%%%

:Authors: Neal Gordon (nealagordon at gmail.com)
:Date: Aug 26, 2016

.. !split

.. _chp:intro:

Introduction
%%%%%%%%%%%%

Mechpy was created for few reasons.
 * To provide the practicing engineer with applications or boilerplate code to quickly replicate and solve real-world systems common in mechanical engineering

 * To give the engineering student a code base from which to supplement learning through hand-calculations and an easy way to check work.

There are many different tools available to engineers. Hand-calculations, spreadsheets, and code are all great ways to perform calculations or visualize data or concepts. MATLAB is the defacto tool to solve many engineering calculations, but is just too expensive to be a practical tool for many of us. Octave, Scilab, or Freelab are great alternatives, but is limited in scope to calculation. I began using python for calculations and visualizations and have found it to be a very powerful tool with many existing modules for mathematics and plotting, in addition to the thousands of other libraries for general computing.

.. !split

.. _chp:modules:

mechpy Modules
%%%%%%%%%%%%%%

abaqus and abaqus_report
========================
API code to automate running of finite element models and reporting using python-pptx. Abaqus uses python as its native macro/scripting language, which makes it easy to develop powerful tools that utilize the robustness and simplicity of the python programming language.

This program was designed to generate a dynamic powerpoint presentation based on a user defined odb file. The ``abaqus.py`` requires and abaqus license and can be run from the abaqus python console within Abaqus CAE. The ``abaqus_report.py`` has many other modules that are not allowed in Abaqus python. PythonReportTool calls functions located in the AbaqusReportTool.py and can be run in any python console or windows terminal and can be executed by opening a windows command prompt by.

catia
=====
API code to automate cad in catia

composites
==========
specialized code to analyze composite plates using classical laminated plate theory

design
======
shear-bending diagrams, assisting hand calculations for stress

fem
===
example code for computing finite element models for analysis and education

math
====
various math tools for processing engineering systems

statics
=======
code for assisting and checking hand calculations

units
=====
unit conversion

.. !split

.. _chp:tut:

Tutorials
%%%%%%%%%

Importing Python Modules
========================

When using python, the very first code you will need is to import modules. An example of importing the essential scientific computing libraries with some settings is shown here.

.. code-block:: python

    # setup
    import numpy as np
    import sympy as sp
    import scipy
    from pprint import pprint
    sp.init_printing(use_latex='mathjax')
    
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (12, 8)  # (width, height)
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 16
    from matplotlib import patches
    
    get_ipython().magic('matplotlib')  # seperate window
    get_ipython().magic('matplotlib inline') # inline plotting

Converting an ipython notebook to an html document
==================================================
Ipython notebooks can eaily be converted to an html file with the following python code

.. code-block:: python

    get_ipython().system('jupyter nbconvert --to html mechpy.ipynb')

.. !split

.. _chp:ref:

References
%%%%%%%%%%
Hibbler - Statics
Hibbler - Mechanics of Materials
Collins et al - Mechanical Design of Machine Elements and Machines
Flabel - Practical Stress Analysis for Design Engineers
Peery - Aircraft Structures
Niu - Airframe Stress Analysis and Sizing
`Numerical Python - A Practical Techniques Approach for Industry <http://www.apress.com/9781484205549>`__ with `source code <http://www.apress.com/downloadable/download/sample/sample_id/1732/>`__
`Elementary Mechanics Using Python <http://www.springer.com/us/book/9783319195957#aboutBook>`__
A Primer on Scientific Programming With Python

Mechanical Design of Machine Elements and Machines by Collins, Jack A., Busby, Henry R., Staab, George H. (2009)

.. !split

.. _chp:sundries:

Sundries
%%%%%%%%
Running jupyter notebook in windows
===================================
In windows, create a batch file (*.bat) to run a jupyter notebook server in the current directory

.. code-block:: batch

    :: Use to launch jupyter notebooks
    
    :: change console to the current working directory
    Pushd "%~dp0"
    
    :: launch jupyter notebook
    jupyter notebook
    
    :: write html output
    jupyter nbconvert --to html mechpy.ipynb
    
    pause
    

Links

=====  Python Engineering Libraries
-----------------------------------

Units
~~~~~

`pint units <http://pint.readthedocs.io/en/0.7.2/>`__    

| `Unum units <https://pypi.python.org/pypi/Unum>`__ 
| ``scipy units``   
| 

Dynamics and Control Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`pyndamics <https://github.com/bblais/pyndamics>`__  with `example <http://nbviewer.ipython.org/gist/bblais/7321928>`__
`<http://www.siue.edu/~rkrauss/python_intro.html  >`_
`<https://www.cds.caltech.edu/~murray/wiki/Control_Systems_Library_for_Python  >`_
`<http://www.vibrationdata.com/python/  >`_
`<http://www.gribblelab.org/compneuro/index.html  >`_
`<http://scipy.github.io/old-wiki/pages/Cookbook/CoupledSpringMassSystem  >`_
`<https://ics.wofford-ecs.org/  >`_
`<http://www.ni.gsu.edu/~rclewley/PyDSTool/FrontPage.html  >`_
`python control <https://github.com/python-control/python-control>`__
`pydy <http://www.pydy.org/>`__ with `examples <http://nbviewer.jupyter.org/github/pydy/pydy-tutorial-human-standing/tree/online-read/notebooks/>`__  and `here <https://github.com/pydy/scipy-2013-mechanics>`__
`double pendulumn <http://matplotlib.org/examples/animation/double_pendulum_animated.html>`__

Fluids/Aero
~~~~~~~~~~~

`aeropy <http://aeropy.readthedocs.io/en/latest/>`__
"NACA airfoils:"https://github.com/dgorissen/naca"
`aeropython <https://github.com/barbagroup/AeroPython>`__  or `aeropython <http://lorenabarba.com/blog/announcing-aeropython/>`__
`pyaero <http://pyaero.sourceforge.net/>`__

Mechanics/Composites/Materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`<https://github.com/nagordon/ME701  >`_
`<https://github.com/nagordon/mcgill_mech_530_selimb  >`_
`<https://github.com/elainecraigie/MechanicsOfCompositesProject_ECraigie  >`_
`pyply composites <https://github.com/Rlee13/pyPLY  >`__
`sympy classical mechanics <http://docs.sympy.org/latest/modules/physics/mechanics/index.html>`__
`pygear <http://sourceforge.net/projects/pygear/>`__

FEM / Math
~~~~~~~~~~

`pynastran <https://github.com/SteveDoyle2/pynastran/wiki/GUI>`__
`grid solvers <http://pyamg.org/>`__ with "example""https://code.google.com/p/pyamg/wiki/Examples`
`ode solver <http://hplgit.github.io/odespy/doc/web/index.html  >`__
`<http://arachnoid.com/IPython/differential_equations.html>`_
`DiffyQ <http://www.usna.edu/Users/math/wdj/_files/documents/teach/sm212/DiffyQ/des-book-2009-11-24.pdf>`__
`pycalculix <http://justinablack.com/pycalculix/>`__
`FEniCS tutorial <http://fenicsproject.org/documentation/tutorial/>`__
`SfePy-Simple Finite Elements in Python <http://sfepy.org/doc-devel/index.html>`__
`PyODE <http://pyode.sourceforge.net/tutorials/tutorial2.html>`__

Plotting and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~

http://hplgit.github.io/bumpy/doc/pub/._bumpy010.html#app:resources
http://central.scipy.org/item/84/1/simple-interactive-matplotlib-plots
https://github.com/rougier/matplotlib-tutorial
http://www.petercollingridge.co.uk/pygame-physics-simulation
`Pymunk visualization <http://www.pymunk.org/en/latest/readme.html>`__ and `<http://chipmunk-physics.net/>`_
`<http://vpython.org/>`_

General Numeric Python
~~~~~~~~~~~~~~~~~~~~~~

`<https://wiki.python.org/moin/NumericAndScientific>`_
https://wiki.python.org/moin/NumericAndScientific

**Scipy.**
`<http://scipy-cookbook.readthedocs.io/index.html>`_
http://www.davekuhlman.org/scipy_guide_01.html
http://www.scipy-lectures.org/index.html
https://github.com/rojassergio/Learning-Scipy
`scipy <http://docs.scipy.org/doc/scipy/reference/tutorial/>`__

**Numpy.**
https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html
https://github.com/rougier/numpy-100
https://github.com/numpy/numpy/wiki/Numerical-software-on-Windows
https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html
http://scipy.github.io/old-wiki/pages/NumPy_for_Matlab_Users.html

**Sympy.**
http://www.sympy.org/en/features.html
`sympy <http://docs.sympy.org/dev/tutorial/intro.html>`__
http://scipy-lectures.github.io/advanced/sympy.html
http://docs.sympy.org/dev/tutorial/calculus.html
http://www.scipy-lectures.org/advanced/sympy.html
http://docs.sympy.org/dev/modules/physics/mechanics/

General Engineering Software
----------------------------
`<https://github.com/MADEAPPS/newton-dynamics/>`_
`<https://chipmunk-physics.net/>`_
`<http://bulletphysics.org/wordpress/>`_
`<http://physion.net/>`_
`<http://www.algodoo.com/>`_
`<http://box2d.org/>`_
`<http://www.ode.org/>`_
`Sage <http://wiki.sagemath.org/quickref  >`__
`<http://www.gregorybard.com/SAGE.html>`_
`<http://www.people.vcu.edu/~clarson/bard-sage-for-undergraduates-2014.pdf>`_
`<http://vibrationdata.com/software.htm>`_
`libre mechanics <http://www.libremechanics.com/>`__
`materia abaqus plugin <http://sourceforge.net/projects/materia/?source=directory>`__
Method <http://fenicsproject.org/book/index.html#book>`__
http://download.gna.org/getfem/html/homepage/python/pygf.html
https://wiki.scilab.org/Finite%20Elements%20in%20Scilab

Python Engineering Tutorials and Classes
----------------------------------------
`<https://github.com/jrjohansson/scientific-python-lectures>`_
`<https://github.com/numerical-mooc/numerical-mooc  >`_
`Python Numerical MOOC <https://github.com/numerical-mooc/numerical-mooc>`__ and the `course <http://openedx.seas.gwu.edu/courses/GW/MAE6286/2014_fall/about>`__
`Cornell <http://pages.physics.cornell.edu/~sethna/StatMech/ComputerExercises/PythonSoftware/>`__
`Python numerical methods mooc <http://openedx.seas.gwu.edu/courses/GW/MAE6286/2014_fall/about>`__
http://www-personal.umich.edu/~mejn/computational-physics/

Random Links
------------

PyBullet
https://github.com/pybox2d/pybox2d
http://docs.sympy.org/latest/modules/physics/mechanics/index.html
https://github.com/cdsousa/sympybotics
https://pypi.python.org/pypi/Hamilton
https://pypi.python.org/pypi/arboris
https://pypi.python.org/pypi/PyODE
https://pypi.python.org/pypi/odeViz
https://pypi.python.org/pypi/ARS
https://pypi.python.org/pypi/pymunk
http://pyamg.org/
http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
http://nbviewer.jupyter.org/gist/bblais/7321928
http://openalea.gforge.inria.fr/doc/vplants/mechanics/doc/_build/html/user/membrane/sphere%20iso/index.html

Microsoft Excel
---------------
Because it's there and everyone uses it
`Automate the boring stuff, python excel scripting <https://automatetheboringstuff.com/chapter12/>`__
`Use python intead of VBA with xlwings <http://xlwings.org/>`__
`openpyxl <https://openpyxl.readthedocs.io/en/default/>`__
`or just develope directly with Windows COM <http://shop.oreilly.com/product/9781565926219.do>`__
