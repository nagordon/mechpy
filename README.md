mechpy - a mechanical engineer's python toolbox

Tutorials - see the [nbviewer for mechpy](http://nbviewer.jupyter.org/github/nagordon/mechpy/blob/master/mechpy.ipynb)

Mechpy was created for two reasons. 
 * To provide the practicing engineer with applications or boilerplate code to quickly replicate and solve real-world systems common in mechanical engineering
 * To give the engineering student a code baes from which to suppliment learning through hand-calculations and an easy way to check work.

There are many different tools available to engineers. Hand-calcsulations, spreadsheets, and code are all great ways to perform calculations or visualize data or concepts. MATLAB is the defacto tool to solve many engineering calulations, but is just too expensive to be a practical tool for many of us. Octave, Scilab, or Freelab are great alternatives, but is limited in scope to calculation. I began using python for calculations and visualzations and have found it to be a very powerful tool with many existing modules for mathematics and plotting, in addition to the thousands of other libraries for general computing.

TODO
* clean up stupid links 
* check out FEM - Numerical Python pg 690/1311
* add failure criteria to composites and mutli-material
* add composite plate code to composites
* convert matlab code for plotting test data

- - - -

## Modules

### abaqus and abaqus_report
API code to automate running of finite element models and reporting using python-pptx 

### catia
API code to automate cad in catia 

### composites  
specialized code to analyze composite plates using classical laminated plate theory

### design  
shear-bending diagrams, assisting hand calculations for stress

### dynamics  

### fem  
example code for computing finite element models for analysis and education

### math  
various math tools for processing engineering systems

### statics  
code for assisting and checking hand calculations

### units  
unit conversion

- - - -

## References
Hibbler - Statics  
Hibbler - Mechanics of Materials  
Collins et al - Mechanical Design of Machine Elements and Machines
Flabel - Practical Stress Analysis for Design Engineers
Peery - Aircraft Structures
Niu - Airframe Stress Analysis and Sizing
[Numerical Python - A Practical Techniques Approach for Industry](http://www.apress.com/9781484205549) with [source code](http://www.apress.com/downloadable/download/sample/sample_id/1732/)  
[Elementary Mechanics Using Python](http://www.springer.com/us/book/9783319195957#aboutBook)  
A Primer on Scientific Programming With Python  

- - - - 
## Python Engineering Libraries
Check out the dependencies of mechpy for specific examples and documentation
 * [Scipy](), [scipy cookbook](http://scipy-cookbook.readthedocs.io/index.html   )
   * numpy
   * sympy
   * matplotlib
   * pandas
 * [pyndamics](https://github.com/bblais/pyndamics  ) with [example](http://nbviewer.ipython.org/gist/bblais/7321928)
 * [Unum units](https://pypi.python.org/pypi/Unum)
 * [pint units](http://pint.readthedocs.io/en/0.7.2/  )
 * [pynastran](https://github.com/SteveDoyle2/pynastran/wiki/GUI)
 * [aeropython](https://github.com/barbagroup/AeroPython)
 * [control-systems](https://github.com/python-control/python-control)
 * [grid solvers](http://pyamg.org/) with [example)[https://code.google.com/p/pyamg/wiki/Examples]
 * [python dynamics](http://www.pydy.org/),(https://pypi.python.org/pypi/pydy/), [examples](#http://nbviewer.jupyter.org/github/pydy/pydy-tutorial-human-standing/tree/online-read/notebooks/)
 * [sympy classical mechanics](http://docs.sympy.org/latest/modules/physics/mechanics/index.html)

### Dynamics  
http://www.siue.edu/~rkrauss/python_intro.html  
https://www.cds.caltech.edu/~murray/wiki/Control_Systems_Library_for_Python  
http://www.vibrationdata.com/python/   
http://www.gribblelab.org/compneuro/index.html  
http://scipy.github.io/old-wiki/pages/Cookbook/CoupledSpringMassSystem  
https://ics.wofford-ecs.org/  
http://www.ni.gsu.edu/~rclewley/PyDSTool/FrontPage.html

### Aero
http://aeropy.readthedocs.io/en/latest/   
http://www.pdas.com/contents15.html  
http://www.pdas.com/aeroinfo.html  
http://www.pdas.com/aerosoft.html  

### Materials
https://github.com/nagordon/ME701  
https://github.com/nagordon/mcgill_mech_530_selimb  
https://github.com/elainecraigie/MechanicsOfCompositesProject_ECraigie  

### Physics
https://github.com/MADEAPPS/newton-dynamics/  
https://chipmunk-physics.net/  
http://bulletphysics.org/wordpress/
http://physion.net/  
http://www.algodoo.com/  
http://box2d.org/
http://www.ode.org/  
http://vpython.org/

### Microsoft Excel
Great refernce for general python scripting. https://automatetheboringstuff.com/chapter12/  
[Use python intead of VBA with xlwings](http://xlwings.org/)  
[openpyxl](https://openpyxl.readthedocs.io/en/default/)  
[xlwings](http://xlwings.org/)  
[or just develope directly with Windows COM](http://shop.oreilly.com/product/9781565926219.do)  

### General Numeric Python
https://wiki.python.org/moin/NumericAndScientific   
[ode solver](http://hplgit.github.io/odespy/doc/web/index.html  )    
http://matplotlib.org/examples/animation/double_pendulum_animated.html  
https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html  
http://www.davekuhlman.org/scipy_guide_01.html  
http://hplgit.github.io/bumpy/doc/pub/._bumpy010.html#app:resources  
https://github.com/hplgit/bumpy/blob/master/doc/src/fig-bumpy/bumpy_road_fig.py  
[Sage](http://wiki.sagemath.org/quickref  )  
http://www.gregorybard.com/SAGE.html  
http://www.people.vcu.edu/~clarson/bard-sage-for-undergraduates-2014.pdf  
http://www.sympy.org/en/features.html  
http://scipy-lectures.github.io/advanced/sympy.html  
http://docs.sympy.org/dev/tutorial/calculus.html  
http://arachnoid.com/IPython/differential_equations.html  
http://www.usna.edu/Users/math/wdj/_files/documents/teach/sm212/DiffyQ/des-book-2009-11-24.pdf   
http://www.scipy-lectures.org/advanced/sympy.html  
https://wiki.python.org/moin/NumericAndScientific 
[Cornell](http://pages.physics.cornell.edu/~sethna/StatMech/ComputerExercises/PythonSoftware/)
http://vibrationdata.com/software.htm  
http://central.scipy.org/item/84/1/simple-interactive-matplotlib-plots  
https://github.com/rougier/matplotlib-tutorial  
http://www.scipy-lectures.org/index.html  
https://github.com/rougier/numpy-100  
https://github.com/rojassergio/Learning-Scipy  
[Python numerical methods mooc](http://openedx.seas.gwu.edu/courses/GW/MAE6286/2014_fall/about)  
http://www.petercollingridge.co.uk/pygame-physics-simulation  
http://www-personal.umich.edu/~mejn/computational-physics/  
[scipy](http://docs.scipy.org/doc/scipy/reference/tutorial/)  
[sympy](http://docs.sympy.org/dev/tutorial/intro.html)  
[python fea](http://justinablack.com/pycalculix/)  
[aero python](http://lorenabarba.com/blog/announcing-aeropython/)  
[libre mechanics](http://www.libremechanics.com/)  
[pygear](http://sourceforge.net/projects/pygear/)  
[materia abaqus plugin](http://sourceforge.net/projects/materia/?source=directory)  
[pyaero](http://pyaero.sourceforge.net/)  
[kinematics](http://matplotlib.org/examples/animation/double_pendulum_animated.html)  
Pymunk   http://www.pymunk.org/en/latest/readme.html     http://chipmunk-physics.net/  
PyBullet
PyBox2D   https://github.com/pybox2d/pybox2d
PyODE     http://pyode.sourceforge.net/tutorials/tutorial2.html
http://docs.sympy.org/latest/modules/physics/mechanics/index.html
https://github.com/cdsousa/sympybotics
https://pypi.python.org/pypi/Hamilton
https://pypi.python.org/pypi/arboris
https://pypi.python.org/pypi/PyODE
https://pypi.python.org/pypi/odeViz
https://pypi.python.org/pypi/ARS
https://pypi.python.org/pypi/pymunk
http://scipy.github.io/old-wiki/pages/NumPy_for_Matlab_Users.html
https://docs.scipy.org/doc/numpy-dev/user/numpy-for-matlab-users.html
[Python Numerical MOOC](https://github.com/numerical-mooc/numerical-mooc) and the [course](http://openedx.seas.gwu.edu/courses/GW/MAE6286/2014_fall/about)  
https://wiki.scilab.org/Finite%20Elements%20in%20Scilab
http://pyamg.org/
http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DataFiltering.ipynb
http://nbviewer.jupyter.org/gist/bblais/7321928
https://github.com/numpy/numpy/wiki/Numerical-software-on-Windows
http://openalea.gforge.inria.fr/doc/vplants/mechanics/doc/_build/html/user/membrane/sphere%20iso/index.html
https://github.com/pydy/scipy-2013-mechanics
http://docs.sympy.org/dev/modules/physics/mechanics/
http://download.gna.org/getfem/html/homepage/python/pygf.html
>[Sparse matrices (scipy.sparse)](http://docs.scipy.org/doc/scipy-0.14.0/reference/sparse.html)
>[Sparse Matrices in SciPy](https://scipy-lectures.github.io/advanced/scipy_sparse/)
>[The FEniCS Book: Automated Solution of Differential Equations by the Finite Element Method](http://fenicsproject.org/book/index.html#book)
>[FEniCS tutorial (Python)](http://fenicsproject.org/documentation/tutorial/)
>[SfePy: Simple Finite Elements in Python](http://sfepy.org/doc-devel/index.html)




## Abaqus
```
# automated Abaqus reporting 
Abaqus uses python as its native macro/scripting language, which makes it easy to develop powerful tools that utilize the robustness and simplicity of the python programming language.
 - - - -
This program was designed to generate a dynamic powerpoint presentation based on a user defined odb file. The ```abaqus.py``` requires and abaqus license and can be run from the abaqus python console within Abaqus CAE. The ```abaqus_report.py``` has many other modules that are not allowed in Abaqus python. PythonReportTool calls functions located in the AbaqusReportTool.py and can be run in any python console or windows terminal and can be executed by opening a windows command prompt by.

### Advanced operation
In the python console in Abaqus ```>>>``` , commands can be typed to use parts of *AbaqusReportTool*. To execute specific functions in *AbaqusReportTool*, type:
```python
>>>import os
>>>os.chdir('C:/Users/User1/Directory_with_py_files')
>>>import AbaqusReportTool
>>>odbname='bottle_test/bottle_test.odb'  # odb file cannot have spaces in the name or special characters, just letters, numbers hyphens and underscores!
>>>myOdbTest = AbaqusReportTool.AbaqusReportTool(odbname)
```
When myOdbTest is loaded, a check is made to determine if the database needs updated, and does so if it does. Also, *>>>* indicates we are in the python console. and will be dropped henceforth

#### A few examples of how to run individual functions
```python
# to see all the functions available
dir(myOdbTest)
#['BE_makeReportData', 'BT_makeReportData', 'CBE_makeReportData', 'HS_makeReportData', 'TL_makeReportData', 'VP_edit', '__class__', '__delattr__', '__dict__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', 'avi_save', 'clearFiles', 'clearFilesExcept', 'csv_field', 'csv_history', 'csysCyn', 'getMaxField', 'myAssembly', 'myOdb', 'myOdbDat', 'myOdbDat2', 'myOdbVis', 'myOdbs', 'myView', 'myVp', 'png_save', 'png_undeformed', 'replaceElementSets', 'replaceNodeSets', 'rpt_field', 'txt_field_summary', 'txt_history_summary', 'txt_odb_summary', 'xplot_field', 'xplot_history', 'xyList_field', 'xyplot_history']

# to change the viewport to default values of the function, s=0, f=0, o='', c='', csys = '', setsKeyword = '',v='Iso1',myLegend = ON, myState=OFF, myMaxValue = 0, myMinValue = 0)
myOdbTest.VP_edit()

# this code will change the current viewport to step 0,frame 0, object Stress, Component S11, coordinate system default, elementsets all, view Iso1, Legend on, State on, maxvalue auto minvalue auto 
myOdbTest.VP_edit(s=0, f=0, o='S', c='S11', csys = '', setsKeyword = '',v='Iso1', myLegend = ON, myState=OFF, myMaxValue = 0, myMinValue = 0)

# to clear all files from the directory except odb and pptx where the odb file resides
myOdbTest.clearFilesExcept(['odb','pptx'])

# to create a text file that describes all the fields in the odb file
myOdbTest.txt_field_summary()

# saves a png of the current viewport
myOdbTest.png_save()
```

##### Accessing History Region info using the bottle_test example
```python
myOdbTest.myOdbDat.steps.keys() 
# ['pressure']
myStep = myOdbTest.myOdbDat.steps['pressure']
myStep.historyRegions.keys() 
# ['Node CATGSD_F_SHAMPOO_BOTTLE_END-1.58', 'Node CATGSD_F_SHAMPOO_BOTTLE_END-1.574']
myRegion=myStep.historyRegions['Node CATGSD_F_SHAMPOO_BOTTLE_END-1.58']
myRegion.historyOutputs.keys()
# ['RF1', 'RF2', 'RF3', 'U1', 'U2', 'U3']
myRegion.historyOutputs['RF1'].description
# 'Reaction force'
# returns a nested tuple of time and the RF1 data
myDat = myRegion.historyOutputs['RF1'].data
# inverts the tuple and unpacks myDat into time and myRF1
time, myRF1  = zip(*myDat)
```

##### Extracting Field info
```python
myFieldval = myStep.frames[-1].fieldOutputs['U'].values
myU1 = [myFieldval[k].data[0] for k in range(len(myFieldval))]
```

##### Closing the database. even though the database is read-only, it is good practice to close the database
```python
myOdbTest.myOdbDat.close()
```

- - - -
## Installation
To install a minimal python, I recommend [miniconda](http://conda.pydata.org/miniconda.html) as a package maanger. Once miniconda or  is installed, install these packages by running these commands in a windows console and following the prompts  

```batch
conda install numpy  

pip install scipy matplotlib
# or
conda install matplotlib=1.4.3

pip install pandas
# or
conda install pandas

pip install python-pptx  
pip install moviepy  
```

For developers, I'd recommend a more complete python installation that includes an IDE (Integrated Development Environment). 

 - [Anaconda](http://continuum.io/downloads) for Linux and Windows
 - [winpython](http://winpython.sourceforge.net/) for Windows
 - [python(x,y)](https://code.google.com/p/pythonxy/) for Windows

- - - -

## Basic Checklist for generating an Abaqus simulation  
1. Parts
2. Materials
3. Sections
4. Section Assingments
5. Assembly(instances)
6. Steps (after Initial)
7. BCs (Boundary Conditions)
8. Loads
9. Mesh
10. Jobs

## launch an abaqus program  
```bash  
abaqus  
abq6132se  
abaqus doc  
abaqus pde      # launches only PDE (python development environment) for abaqus  
abaqus cae -pde # launches PDE and abaqus cae  
abaqus cae      # launches abaqus cae  
abaqus viewer   # launhces abaqus viewer  
```

## execute an existing script or open a file
```bash
abaqus python odb_to_txt.py test1.odb       # runs a abaqus python script with input 'test1.odb'
abaqus cae script=myscript.py               # launches cae and runs script
abaqus cae database=filename.cae            # opens an odb file in cae
abaqus viewer script=myscript.py            # launches viewer and executes script
abaqus viewer database=cantilever           # opens a odb file in the viewer
abaqus cae noGUI=myscript.py                # launches cae and runs script
abaqus viewer noGUI=myscript.py             # launches viewer and executes script
abaqus job=my_sample_job interactive        # executes *.inp file
abaqus job=my_sample_job cpus=8 interactive # executes *.inp file and utilizes 8 cpu cores
```

# executing an abaqus python script passing in variables
```bash
abaqus cae noGUI=myscript.py -- variable1 variable2
```

## fetch demos
```bash
abaqus fetch job=beamExample   # fetches the beamExample.py file
abaqus fetch job=gsi_frame_caemodel.py     
abaqus cae script=gsi_frame_caemodel.py
```

## Some of the abaqus file types  
### files generated by created and analyzing a model  
* **rpy** -  replay file , all modeling operation commands executed during a session, are saved in this file
* **cae**  - model database, contains models and analysis jobs
* **jnl** - journal python file, all commands necessary to recreate the most current save model database
* **rec** - recover file contains commands to replicate the version of the model database in memory  
### files created when a job is submitted  
* **inp** - abaqus config file that is created when a job is submitted. contains all the information necessary to run a analysis
* **odb** - output database containing results from completed simulation. Use the step module output request manager to choose variables are saved at what rate. viewed in the visualization module 
* **lck** - file that protects the odb file when it is current being written to
* **res** - used to continue an anaylsis that stopped before it was complete. use the step module to specify which analysis steps should write restart information 
* **dat** - printed output from the anaylsis input file processor, as well as printed output of selected results written during the analysis. 
* **msg**  - contains diagnostic or informative messages about the progress of the solution. Control the output in the step module
* **fil**  - contains selected results from the analysis in a neutral format. that can we read/written for third-party post processing
* **rpt* - text file generated from probe data for user specific user requested information
* **ods** - scratch output database file
* **mdb** - database of the model  
* **sim** - binary file
* **sta** - summary of job info
* **lck** - a file that indicates that the odb file is open and cannot be edited if opened by another user

## Outputs
* defaults set in the environment file
*  typical outputs
 * E - strain
 * PE - plastic strain
 * U - displacements and rotations
 * RF - reaction forces
 * CF - concentrated (applied) forces and moments
 * CSTRESS - contact stresses
 * CDISP - contact displacements
 * see Abaqus Scripting User's Guide 5.2 for more info accessing the database 
 

Abaqus Scripting User's Guide - 5.2.2
## field output  
* typically requested for the entire model, often only at selected elements
* used for creating contour, deformed shape, and symbol plots
* Can be used as a source for x-y plots, but history is better
* intended for infrequent requests for large portions of model  data, such as 3d mesh contour plots
* can specify output frequency or time intervals

```
import odbAccess
session.odbs[name].steps[name].frames[i].fieldOutputs[name]
# or 
sdat=session.odbs['customBeamExample/customBeamExample.odb'].steps['beamLoad'].frames[-1].fieldOutputs['S']
integrationPointData = sdat.getSubset(position=INTEGRATION_POINT)
invariantsData = sdat.validInvariants
```


Abaqus Scripting User's Guide - 9.3.2  
Field output is intended for infrequent requests for a large portion of the model and can be used to generate contour plots, animations, symbol plots, and displaced shape plots in the Visualization module of Abaqus/CAE. You can also use field output to generate an X–Y data plot. Only complete sets of basic variables (for example, all the stress or strain components) can be requested as field output. Field output is composed of a “cloud of data values” (e.g., stress tensors at each integration point for all elements). Each data value has a location, type, and value. You use the regions defined in the model data, such as an element set, to access subsets of the field output data. Figure 9–3 shows the field output data object model within an output database.

## history output
* requested for a small subset of the model
* typically requested at every increment during the anaylsis
* used to generate xy plots by step and time
* example -  the principal stress at a single node at the root of the beam

```python
import odbAccess
session.odbs[name].steps[name].historyRegions[name]
```

Abaqus Scripting User's Guide - 9.3.2  
History output is output defined for a single point or for values calculated for a portion of the model as a whole, such as energy. History output is intended for relatively frequent output requests for small portions of the model and can be displayed in the form of X–Y data plots in the Visualization module of Abaqus/CAE. Individual variables (such as a particular stress component) can be requested.
Depending on the type of output expected, a HistoryRegion object can be defined for one of the following:
* a node
* an integration point
* a region
* the whole model

The output from all history requests that relate to a particular point or region is then collected in one HistoryRegion object. Figure 9–4 shows the history output data object model within an output database. In contrast to field output, which is associated with a frame, history output is associated with a step. 

## Common Abaqus activities

### Accessing the help documentation
* Help -> Search and Browse Guides
* or type 'abaqus doc' in console

### execute a abaqus/python script  
* File -> Run Script  

### record a abaqus/python macro 
* File -> Macro Manager
* files stored in users home directory in abaqusMacros.py

### edit and execute a abaqus python script in using the provided python development environment (PDE)  
* File -> Abaqus PDE...

### set default working directory

### plugins
abaqus_plugins\reportGenerator\htmlReportGeneratorForm.py


### Change font of legend
Click on "Viewport" in the top Menu Bar
Select "Viewport Annotation Options"
Click on the "Triad" tab
Click on the "Set Font..." button to adjust the font
You can even change all 3 at once by checking on the "Legend",  "Title Block", or "State Block" check boxes under the "Apply To" heading

- - - -
# Abaqus/Python Scripting

## Abaqus specific Python modules imports
* Module - Abaqus/CAE functionality  
* assembly - The Assembly module  
* datum - The Datum toolset  
* interaction - The Interaction module  
* job - The Job module  
* load - The Load module  
* material - Materials in the Property module  
* mesh - The Mesh module  
* part - The Part module  
* partition - The Partition toolset  
* section - Sections in the Property module  
* sketch - The Sketch module  
* step - The Step module  
* visualization - The Visualization module  
* xyPlot - The X–Y toolset  

# built in abaqus gui example
```python
from abaqus import getInput
from math import sqrt
number = float(getInput('Enter a number:'))
print sqrt(number)
```

```python
def printCurrentVp(picname = 'file1'):

    from abaqus import session, getInputs
    from abaqusConstants import PNG
    name = getInputs( (('File name:', ''),), 
        'Print current viewport to PNG file')[0]
    vp = session.viewports[session.currentViewportName]
    session.printToFile(
        fileName=name, format=PNG, canvasObjects=(vp, ))
```


# viewport views
```python
viewVector    =(0,1,0)  #  x,y,z
cameraUpVector=(0,0,1)

myViewport.view.setValues(session.views['Front'])
myViewport.view.setViewpoint(viewVector=(0,0,1), cameraUpVector=(0,1,0))

myViewport.view.setValues(session.views['Back'])
myViewport.view.setViewpoint(viewVector=(0,0,-1), cameraUpVector=(0,1,0))

myViewport.view.setValues(session.views['Top'])
myViewport.view.setViewpoint(viewVector=(0,1,0), cameraUpVector=(0,0,-1))

myViewport.view.setValues(session.views['Bottom'])
myViewport.view.setViewpoint(viewVector=(0,-1,0), cameraUpVector=(0,0,1))

myViewport.view.setValues(session.views['Left'])
myViewport.view.setViewpoint(viewVector=(-1,0,0), cameraUpVector=(0,1,0))

myViewport.view.setValues(session.views['Right'])
myViewport.view.setViewpoint(viewVector=(1,0,0), cameraUpVector=(0,1,0))

myViewport.view.setValues(session.views['Iso'])
myViewport.view.setViewpoint(viewVector=(1,1,1), cameraUpVector=(0,1,0))

```

## Other random tests from the documentation
```bash
abaqus fetch job=odbElementConnectivity
abaqus fetch job=viewer_tutorial
abaqus cae odbElementConnectivity.py -- viewer_tutorial.odb
```

This document was generated from content in the abaqus documentation as a quick reference guide when using abaqus. To access  this content  

* Help -> Search and Browse Guides...
```


