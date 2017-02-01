# -*- coding: utf-8 -*-
"""
abaqus automation tool
Requires an abaqus license to run and must be executed in the abaqus environment
"""
__author__ = 'Neal Gordon <nealagordon@gmail.com>'
__date__ =   '2016-09-06'

# python general modules
import sys
import cStringIO
import numpy
import glob
import os
from textwrap import dedent
import time

# modules required by abaqus to open obb files
import visualization
import odbAccess
from abaqus import session
from abaqus import butterworthFilter
from abaqus import mdb
import textRepr
import displayGroupOdbToolset as dgo
import displayGroupMdbToolset as dgm
import xyPlot
from abaqusConstants import OFF, ON, FEATURE, CONTINUOUS, CONTOURS_ON_DEF,\
                            UNDEFORMED, SIZE_ON_SCREEN, FINE, LANDSCAPE, TIME_HISTORY,\
                            COLOR, FALSE, PNG, AVI, WHOLE_ELEMENT, \
                            INTEGRATION_POINT,COMPONENT, NODAL, INVARIANT,\
                            CYLINDRICAL, USER_SPECIFIED, DEFAULT, PNG, OFF,NONE, \
                            TIME_BASED, FRAME_BASED, PLAY_ONCE, SCALE_FACTOR, CURRENT_FRAME,\
                            DISCRETE, ALL_FRAMES, RECOMPUTE_EACH_FRAME, TIME,STRESS, STRAIN, DASHED,\
                            ANALYSIS, PERCENTAGE, SINGLE, SVG, TIFF, LARGE, UNLIMITED

class AbaqusReportTool(object):
    """REQUIRES AN ABAQUS LICENSE
        program to axtract data from an abaqus odb file"""

    def __init__(self, odbname1):
        """this is executed with the object is instantiated"""
        # odbname1 = 'C:/Users/ngordon/Desktop/abaqus-scripting/customBeamExample/customBeamExample.odb'

        if ' ' in odbname1:
            print('Remove all spaces from file path %s' % odbname1)
            sys.exit(0)
        else:
            if odbAccess.isUpgradeRequiredForOdb(upgradeRequiredOdbPath=odbname1):
                newodb = odbname1.name.replace('.odb','_new.odb')
                odbAccess.upgradeOdb(existingOdbPath=odbname, upgradedOdbPath=newodb)
                odbname1 = newodb

            self.myOdbDat =  session.openOdb(name=odbname1, readOnly=True)
            self.myAssembly = self.myOdbDat.rootAssembly

        self.xyp = session.XYPlot(name='XYPlot-1')
        ## can be run multiple times but the line >>>session.xyPlots['XYPlot-1'] , can only be run once >>>ession.XYPlot('XYPlot-1')
        chartName = self.xyp.charts.keys()[0]
        self.chart = self.xyp.charts[chartName]
        # self.chart.legend.textStyle.setValues(font='-*-verdana-medium-r-normal-*-*-340-*-*-p-*-*-*')
        self.chart.legend.setValues(show=False)
        self.chart.legend.titleStyle.setValues(font='-*-verdana-medium-r-normal-*-*-240-*-*-p-*-*-*')
        self.chart.gridArea.style.setValues(fill=False)
        self.xyp.title.style.setValues(font='-*-arial-medium-r-normal-*-*-240-*-*-p-*-*-*')

                        # 30 <= width <= 423           30 <= height <= 564
        self.myVp = session.Viewport(name='myVp1', origin=(0, 0), width=400, height=550)
        self.myVp.setValues(displayedObject=self.myOdbDat)

        self.csysCyn = self.myOdbDat.rootAssembly.DatumCsysByThreePoints(name='CSYS-CYN',
                                                            coordSysType=CYLINDRICAL,
                                                            origin=(0.0, 0.0, 0.0), point1=(1.0, 0.0,
                                                            0.0), point2=(0.0, 1.0, 0.0))

        # default view
        self.myView = 'Iso'
        # session settings
        session.printOptions.setValues(vpBackground=OFF,
                                       rendition=COLOR,
                                       vpDecorations=FALSE,
                                       compass=OFF)
        session.psOptions.setValues(orientation=LANDSCAPE)
        session.pageSetupOptions.setValues(imageSize=SIZE_ON_SCREEN,
                                           quality=FINE,
                                           orientation=LANDSCAPE,
                                           logo=OFF)

    def clearFiles(self, filetypes = ['png','csv','txt','avi','gif']):
        """clears the files """
        for fileext in filetypes:
            for f in glob.glob(os.path.dirname(self.myOdbDat.name)+'/*.'+fileext):
                try:
                    os.remove(f)
                except:
                    pass

    def clearFilesExcept(self, filetypes = ['.odb','.py','.cae','.stp']):
        """clears the files except filetypes"""
        for f in glob.glob(os.path.dirname(self.myOdbDat.name)+'/*'):
            if os.path.splitext(f)[1] not in filetypes:
                try:
                    os.remove(f)
                except:
                    pass

    def replaceElementSets(self, setsKeyword=''):
        """updates current viewport with element sets including the keywork provided. emepty string includes all sets"""
        ElementSets = self.myOdbDat.rootAssembly.elementSets.keys()
        ElementSet = [s for s in ElementSets if setsKeyword in s]
        myElementSet = tuple(ElementSet)
        myLeaf = dgo.LeafFromElementSets(elementSets=myElementSet)
        self.myVp.odbDisplay.displayGroup.replace(leaf=myLeaf)
        return myElementSet

    def replaceNodeSets(self, NodeSetsKeyword=''):
        """updates current viewport with element sets including the keywork provided. emepty string includes all sets"""
        nodeSets = self.myAssembly.nodeSets.keys()
        nodeSet = [s for s in nodeSets if NodeSetsKeyword in s]
        myNodeSet = tuple(nodeSet)
        myLeaf = dgo.LeafFromNodeSets(nodeSets=myNodeSet)
        self.myVp.odbDisplay.displayGroup.replace(leaf=myLeaf)
        return myNodeSet

    def xplot_history(self, s, x):
        """plots the historys based on a list
        x = 'Reaction force: RF1 at Node 58 in NSET SET-NODE-BOTTOM'
        
        Building the input from info from the odb
        
        
        """
        sName = self.myOdbDat.steps.keys()[s]
        xy1 = xyPlot.XYDataFromHistory(odb=self.myOdbDat,
                                        outputVariableName=x,
                                        steps=(sName, ), )
        xy1 = butterworthFilter(xyData=xy1, cutoffFrequency=0.1)
        c1 = session.Curve(xyData=xy1)
        self.chart.setValues(curvesToPlot=(c1, ), )
        self.myVp.setValues(displayedObject=self.xyp)
        chartName = self.xyp.charts.keys()[0]
        self.chart = self.xyp.charts[chartName]
        path_filename = '%s_Xplot_step-%s_x-%s.png' % \
            (self.myOdbDat.name.replace('.odb',''), sName, x.split(':')[1].split(' ')[1])
        print(path_filename)
        # fname = self.myOdbDat.name.replace('.odb','')+'_Xplot_step-'+sName+'_'+x.split(':')[1].split(' ')[1]+'.png'
        #session.printToFile(fileName=fname, format=PNG, canvasObjects=(self.myVp, ))
        self.myVp.setValues(height=250, width=350)
        try:
            session.printToFile(path_filename, PNG, (self.myVp,))
            print('%s saved' % path_filename)
        except:
            print('failed to save %s' % path_filename)

    def xyplot_history(self, s, x, y):
        """
        Display curves of theoretical and computed results in a new viewport
        x='Reaction force: RF1 at Node 58 in NSET SET-NODE-BOTTOM'
        y='Spatial displacement: U1 at Node 574 in NSET SET-NODE-MIDDLE'

        """
        sName = self.myOdbDat.steps.keys()[s]

        x1 = xyPlot.XYDataFromHistory(odb=self.myOdbDat,
                                       outputVariableName=x,
                                       steps=(sName, ), )

        y1 = xyPlot.XYDataFromHistory(odb=self.myOdbDat,
                                       outputVariableName=y,
                                       steps=(sName, ), )

        xy1 = xyPlot.combine(x1, y1)
        # xy3.setValues(sourceDescription='combine ( "XYData-strain","XYData-stress" )')
        # session.xyDataObjects.changeKey(xy3.name, 'XYData-stress-strain')

        c1 = session.Curve(xyData=xy1)
        self.chart.setValues(curvesToPlot=(c1, ), )
        self.myVp.setValues(displayedObject=self.xyp)
        chartName = self.xyp.charts.keys()[0]
        self.chart = self.xyp.charts[chartName]
        self.chart.axes1[0].labelStyle.setValues(font='-*-verdana-medium-r-normal-*-*-240-*-*-p-*-*-*')
        self.chart.axes1[0].titleStyle.setValues(font='-*-arial-medium-r-normal-*-*-240-*-*-p-*-*-*')
        self.chart.axes2[0].titleStyle.setValues(font='-*-arial-medium-r-normal-*-*-240-*-*-p-*-*-*')
        self.chart.axes2[0].labelStyle.setValues(font='-*-verdana-medium-r-normal-*-*-240-*-*-p-*-*-*')

        #session.printOptions.setValues(vpDecorations=OFF, reduceColors=False)
        path_filename = '%s_XYplot_step-%s_x-%s_y-%s.png' % \
            (self.myOdbDat.name.replace('.odb',''), sName, x.split(':')[1].split(' ')[1], y.split(':')[1].split(' ')[1] )
        print(path_filename)
        # fname = self.myOdbDat.name.replace('.odb','')+'_XYplot_step-'+sName+'_'+x.split(':')[1].split(' ')[1]+'_'+y.split(':')[1].split(' ')[1]+'.png'
        #session.printToFile(fileName=fname, format=PNG, canvasObjects=(self.myVp, ))
        self.myVp.setValues(height=250, width=350)
        self.simple_png_save(path_filename)

        path_filename = path_filename.replace('png','avi')
        session.animationController.setValues(animationType=TIME_HISTORY, viewports=(self.myVp.name, ))
        session.animationController.animationOptions.setValues(xyShowSymbol=True, xySymbolSize=LARGE)
        session.animationController.play(duration=UNLIMITED)
        self.simple_avi_save(path_filename)

    def simple_avi_save(self, path_filename):
        try:
            session.writeImageAnimation(path_filename, AVI, (self.myVp,))
            print('%s saved' % path_filename)
        except:
            print('failed to save %s' % path_filename)

    def simple_png_save(self, path_filename):
        try:
            session.printToFile(path_filename, PNG, (self.myVp,))
            print('%s saved' % path_filename)
        except:
            print('failed to save %s' % path_filename)

    def VP_edit(self, s=0, f=0, o='', c='', csys = '', setsKeyword = '',v='Iso1',
                        myLegend = ON, myState=OFF, myMaxValue = 0, myMinValue = 0):
        """
        VP_edit(s, f, o, c, csys, setsKeyword, v, myLegend, myState, myMaxValue, myMinValue)


        VP_edit(0, 0, '', '', '',   '','Iso1',   ON,    OFF,    0,  0):

        Undeformed
        self.VP_edit(0, 1, '', '', '',  '', 'Iso1',   ON,    0,  0)

        With Component
        self.VP_edit(0, 1, 'S', 'S11', '',   '', 'Iso1',  ON, ON,   0,  0) ; self.png_save()
        self.VP_edit(0, 1, 'S', 'S11', 'cyn',   '', 'Iso1',  ON, ON,   0,  0)

        self.VP_edit(0, 1, 'S', 'Mises', '',   '', 'Iso1',  ON,    0,  0)
        self.VP_edit(0, 1, 'U', 'Magnitude', '',   '',  'Iso1', ON,    0,  0)
        self.VP_edit(0, 1, 'U', 'U1', '',   '',   ON,  'Iso1',  0,  0)
        self.VP_edit(0, 1, 'U', 'U1', 'cyn',  '', 'Iso1',   ON,    0,  0)

        Without Component
        self.VP_edit(0, 1, 'EVOL', '', '',   '',  'Iso1', ON,    0,  0)

        general viewport settings - Ensure you add these constants to the 'from abaqusConstants import ...' list
        viewport settings specific to changing fields settings
            subset = UNDEFINED_POSITION, NODAL, ELEMENT_NODAL, INTEGRATION_POINT, ELEMENT_FACE_INTEGRATION_POINT,
                    WHOLE_ELEMENT, WHOLE_REGION, WHOLE_MODEL, CENTROID, SURFACE_INTEGRATION_POINT, SURFACE_NODAL,
                    ELEMENT_FACE or WHOLE_PART_INSTANCE
            scalar field = UNDEFINED_INVARIANT, TRESCA, PRESS, MAGNITUDE, MAX_PRINCIPAL, MID_PRINCIPAL,
                        MIN_PRINCIPAL, MAX_INPLANE_PRINCIPAL, MIN_INPLANE_PRINCIPAL, OUTOFPLANE_PRINCIPAL, MISES or INV3
            datatype = QUATERNION_3D, VECTOR, TENSOR_3D_FULL, TENSOR_3D_PLANAR, TENSOR_2D_PLANAR,
                        TENSOR_3D_SURFACE, TENSOR_2D_SURFACE, QUATERNION_2D, SCALAR or MATRIX
            subset = UNDEFINED_POSITION, NODAL, ELEMENT_NODAL, INTEGRATION_POINT,
                    ELEMENT_FACE_INTEGRATION_POINT, WHOLE_ELEMENT, WHOLE_REGION, WHOLE_MODEL, CENTROID,
                    SURFACE_INTEGRATION_POINT, SURFACE_NODAL, ELEMENT_FACE or WHOLE_PART_INSTANCE

            # DEFORMED, CONTOURS_ON_UNDEF, CONTOURS_ON_DEF, SYMBOLS_ON_UNDEF, SYMBOLS_ON_DEF, ORIENT_ON_UNDEF, ORIENT_ON_DE

            dir(cbe.myVp.odbDisplay) # list all methods
            self.myVp.odbDisplay.fieldSteps
            self.myVp.odbDisplay.fieldFrame
            self.myVp.odbDisplay.statusMaximum

        """
        self.myVp.setValues(displayedObject=self.myOdbDat, width=100, height=100)
        self.myVp.makeCurrent()

        myComponent = INVARIANT if c in INVComponentField else COMPONENT

        # myPlotstate = UNDEFORMED, CONTOURS_ON_DEF

        #print 'o=%s    c=%s  ' % (o,c)
        if o and c and c != ' ':
            self.myVp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))
            self.myVp.odbDisplay.setPrimaryVariable(variableLabel=o,
                                                     outputPosition=myOutputPosition[o],
                                                     refinement=(myComponent, c), )
        elif o and (c == ' ' or c == ''): # no refinement
            self.myVp.odbDisplay.display.setValues(plotState=(CONTOURS_ON_DEF, ))
            self.myVp.odbDisplay.setPrimaryVariable(variableLabel=o,outputPosition=myOutputPosition[o],)
        else:
            self.myVp.odbDisplay.display.setValues(plotState=(UNDEFORMED, ))

        # note any frame value given larger than the number of frames will floor to the last frame
        self.myVp.odbDisplay.setFrame(step=s, frame=f)

        self.myView = v
        myViews =  {'Front':  "self.myVp.view.setValues(session.views['Front'])",
                    'Back':   "self.myVp.view.setValues(session.views['Back'])",
                    'Top':    "self.myVp.view.setValues(session.views['Top'])",
                    'Bottom': "self.myVp.view.setValues(session.views['Bottom'])",
                    'Left':   "self.myVp.view.setValues(session.views['Left'])",
                    'Right':  "self.myVp.view.setValues(session.views['Right'])",
                    'Iso':    "self.myVp.view.setValues(session.views['Iso'])",
                    'Iso1':   "self.myVp.view.setViewpoint(viewVector=(1,1,1), cameraUpVector=(0,0,1))",
                    'Iso2':   "self.myVp.view.setViewpoint(viewVector=(1,1,-1), cameraUpVector=(0,1,0))"}
        eval(myViews[v])

        #self.myVp.odbDisplay.superimposeOptions.setValues(visibleEdges=NONE, translucencyFactor=0.15)
        self.myVp.odbDisplay.basicOptions.setValues(coordSystemDisplay=OFF, translucencySort=ON)
        self.myVp.odbDisplay.commonOptions.setValues(visibleEdges=FEATURE)  # NONE
        self.myVp.odbDisplay.contourOptions.setValues(contourStyle=DISCRETE, # DISCRETE CONTINUOUS
                                                      animationAutoLimits=ALL_FRAMES) # ALL_FRAMES RECOMPUTE_EACH_FRAME CURRENT_FRAME

        myMaxAutoCompute = ON if myMaxValue == 0 else OFF
        myMinAutoCompute = ON if myMinValue == 0 else OFF
        self.myVp.odbDisplay.contourOptions.setValues(maxAutoCompute=myMaxAutoCompute,
                                                      maxValue=myMaxValue,
                                                      minAutoCompute=myMinAutoCompute,
                                                      minValue=myMinValue,
                                                      animationAutoLimits=ALL_FRAMES,
                                                      showMinLocation=ON,
                                                      showMaxLocation=ON)
                                                      #intervalType=LOG)
        self.myVp.odbDisplay.contourOptions.setValues(numIntervals=14)

        # legendPosition=(0, 0) , 0-100% position from bottom left , legendPosition=(0, 100) = top left
        self.myVp.viewportAnnotationOptions.setValues(triad=OFF, title=OFF, state=myState,  compass=OFF,
                                                      legend=myLegend, legendPosition=(1, 99), legendBox=OFF,
                                                      legendFont='-*-verdana-medium-r-normal-*-*-120-*-*-p-*-*-*',
                                                      statePosition=(1, 15),
                                                      titleFont='-*-verdana-medium-r-normal-*-*-120-*-*-p-*-*-*',
                                                      stateFont='-*-verdana-medium-r-normal-*-*-120-*-*-p-*-*-*')
        if csys == 'cyn':
            self.myVp.odbDisplay.basicOptions.setValues(transformationType=USER_SPECIFIED, datumCsys=self.csysCyn)
        else:
            self.myVp.odbDisplay.basicOptions.setValues(transformationType=DEFAULT, datumCsys=None)

        self.replaceElementSets(setsKeyword)
        self.myVp.maximize()
        self.myVp.view.fitView()

    def txt_odb_summary(self):
        """Abaqus script that requires an odbfile as an argument and reads the contents
            and saves them to a text file of the same name

        #check if upgraded
        odbname = 'bottle_test/bottle_test.odb'
        import odbAccess
        if odbAccess.isUpgradeRequiredForOdb(upgradeRequiredOdbPath=odbname):
            odbAccess.upgradeOdb(existingOdbPath=odbname, upgradedOdbPath=odbname)
        else:
            pass

        newodbname = 'new_'+self.myOdbDat.name
        cmdout = os.system('abaqus upgrade -job='+newodbname+' -odb='+odbname)

    """

        stdout_ = sys.stdout #Keep track of the previous value.
        stream = cStringIO.StringIO()
        sys.stdout = stream
        textRepr.prettyPrint(self.myOdbDat,5)
        sys.stdout = stdout_ # restore the previous stdout.
        txtxout = stream.getvalue()  # This will get the printed values
        f = open(self.myOdbDat.name.replace('.odb','.txt'), 'w')
        f.write(txtxout)
        f.close()

    def csv_field(self, list_s=[], list_f=[],  dict_o_c={'S':['S11'],'E':['E11'] }):
        """ s=step , f = frames , o = fieldOutputs, c=component label
            list_s=['beamLoad']
            list_f = [1,6]
            subset = UNDEFINED_POSITION, NODAL, ELEMENT_NODAL, INTEGRATION_POINT,
                    ELEMENT_FACE_INTEGRATION_POINT, WHOLE_ELEMENT, WHOLE_REGION, WHOLE_MODEL,
                    CENTROID, SURFACE_INTEGRATION_POINT, SURFACE_NODAL, ELEMENT_FACE or WHOLE_PART_INSTANCE
            scalar field = UNDEFINED_INVARIANT, TRESCA, PRESS, MAGNITUDE, MAX_PRINCIPAL, MID_PRINCIPAL,
                MIN_PRINCIPAL, MAX_INPLANE_PRINCIPAL, MIN_INPLANE_PRINCIPAL, OUTOFPLANE_PRINCIPAL, MISES or INV3
            datatype = QUATERNION_3D, VECTOR, TENSOR_3D_FULL, TENSOR_3D_PLANAR, TENSOR_2D_PLANAR,
                TENSOR_3D_SURFACE, TENSOR_2D_SURFACE, QUATERNION_2D, SCALAR or MATRIX
            componentLabels #('S11', 'S22', 'S33', 'S12', 'S13', 'S23')"""

        # loop through data
        for s in self.myOdbDat.steps.keys():
            if s in list_s or (not list_s):
                ftot = len(self.myOdbDat.steps[s].frames)
                for f_i, f in enumerate(self.myOdbDat.steps[s].frames):
                    if f_i in list_f or (not list_f) or ((f_i+1==ftot) and -1 in list_f):
                        for o in dict_o_c:
                            fobj = self.myOdbDat.steps[s].frames[f_i].fieldOutputs[o]
                            myComponents = fobj.componentLabels
                            xlen = len(fobj.values)

                            if not myComponents: # components are empty, scalar value
                                try:
                                    nparraydat = numpy.array([fobj.values[k].data for k in range(xlen)]).T
                                    path_filename = '%s_field_%s_frame-%i_%s.csv' % (self.myOdbDat.name.replace('.odb',''),s,f_i,o)
                                    numpy.savetxt(path_filename, nparraydat, delimiter=",")
                                except:
                                    pass
                            else:
                                for c_i, c in enumerate(dict_o_c[o]):
                                    if c in myComponents:
                                        try:
                                            nparraydat = numpy.array([fobj.values[k].data[c_i] for k in range(xlen)]).T
                                            path_filename = '%s_field_%s_frame-%i_%s_%s.csv' % (self.myOdbDat.name.replace('.odb',''),s,f_i,o,c)
                                            numpy.savetxt(path_filename, nparraydat, delimiter=",")
                                        except:
                                            pass

    def txt_field_summary(self):
        """ create a text file as a sumamry of all the fields"""

        # create file stream to capture the print statements
        stdout_ = sys.stdout #Keep track of the previous value.
        stream = cStringIO.StringIO()
        sys.stdout = stream

        # Loop through all values
        for s in self.myOdbDat.steps.keys():
            print('step = %s' % s)
            for f_i, f in enumerate(self.myOdbDat.steps[s].frames):
                foutput = self.myOdbDat.steps[s].frames[f_i].fieldOutputs.keys()
                print('    frame=%i' % f_i)
                print('    output=%s' % foutput)
                for o in foutput:
                    fobj = self.myOdbDat.steps[s].frames[f_i].fieldOutputs[o]
                    myComponents = fobj.componentLabels
                    print('        fieldOutput=%s' % o)
                    print('        fieldDescription=%s' % fobj.description)
                    print('        frameName=%s' % fobj.name)
                    print('        componentLabels=%s' % list(myComponents))

        # close stream and write file
        sys.stdout = stdout_ # restore the previous stdout.
        txtxout = stream.getvalue()  # This will get the printed values
        path_filename = self.myOdbDat.name.replace('.odb','')+'_txt_field_summary'+'.txt'
        file1 = open(path_filename, 'w')
        file1.write(txtxout)
        file1.close()

    def txt_history_summary(self):
        """creates a text file that shows all the contents of the history"""

        # create file stream to capture the print statements
        stdout_ = sys.stdout #Keep track of the previous value.
        stream = cStringIO.StringIO()
        sys.stdout = stream

        # loops through using strings
        for s in self.myOdbDat.steps.keys():
            print('step=%s' % s)
            for r in self.myOdbDat.steps[s].historyRegions.keys():
                robj = self.myOdbDat.steps[s].historyRegions[r]
                routput = self.myOdbDat.steps[s].historyRegions[r].historyOutputs.keys()
                print('    region=%s' % r)
                print('    outputs=%s'%routput)
                for o in routput:
                    print('        historyOutput=%s' % o)
                    print('        description=%s' % robj.description)
                    print('        name=%s' % robj.name)
                    print('        position=%s\n' % robj.position)

        # close stream and write file
        sys.stdout = stdout_ # restore the previous stdout.
        txtxout = stream.getvalue()  # This will get the printed values
        path_filename = self.myOdbDat.name.replace('.odb','')+'_txt_history_summary'+'.txt'
        file1 = open(path_filename, 'w')
        file1.write(txtxout)
        file1.close()

    def csv_history(self, list_s=[], list_r=[], dict_o_c={'U':['U1']}):
        """s=step , r = historyRegion , o = historyOutputs
            if inputs are empty strings all inputs will be generated
            note , keys() gives a list of strings, where values() is the actual item

            # loops through using objects rather than strings
            for s in myOdb.steps.values():
                for r in s.historyRegions.values():
                    for o in r.historyOutputs.values():
                        t,val = zip(*o.data)
                        nparraydat = numpy.array([ t, val ]).T
                        path_filename = myOdb.name.replace('.odb','')+'_'+s.name+'_'+r.name+'_'+o.name+'.csv'
                        numpy.savetxt(path_filename, nparraydat, delimiter=",")
        """
        # loops through using strings
        for s in self.myOdbDat.steps.keys():
            if s in list_s or (not list_s):
                for r in self.myOdbDat.steps[s].historyRegions.keys():
                    if r in list_r or (not list_r):
                        for o in dict_o_c:
                             for c in dict_o_c[o]:
                                if c in self.myOdbDat.steps[s].historyRegions[r].historyOutputs.keys():
                                    t,val = zip(*self.myOdbDat.steps[s].historyRegions[r].historyOutputs[c].data)
                                    nparraydat = numpy.array([ t, val ]).T
                                    path_filename = '%s_history_%s_%s_%s.csv' %  (self.myOdbDat.name.replace('.odb',''),s,r,c)
                                    numpy.savetxt(path_filename, nparraydat, delimiter=",")

    def rpt_field(self, s=0, f=0, o='S', c=' ', setsKeyword='', csys = ''):
        """script to generate text file
            Menu -> Report -> Field Ouput
            if o='S', and c=' ', then all stress info will be saved
        """
        v = 'Iso' # doesn't matter, just need some input
        self.VP_edit(s, f, o, c, csys, setsKeyword, v ,ON, OFF, 0, 0)
        self.myVp.setValues(displayedObject=self.myOdbDat)
        session.fieldReportOptions.setValues(printTotal=OFF, printMinMax=OFF)
        rpt_name = '%s_step-%i_frame-%i_field-%s_elsets-%s_csys-%s.rpt' % \
                   (self.myOdbDat.name.replace('.odb','') , s , f , o, setsKeyword, csys)

        mySortItems = ['Element Label', 'Node Label']
        mySortItem = mySortItems[1]

        myComponent = INVARIANT if c in INVComponentField else COMPONENT

        if o and c and c != ' ':
            myVariable = ((o, myOutputPosition[o], ((myComponent, c), )), )
        elif o and (c == ' ' or not c): # no refinement
            myVariable = ((o, myOutputPosition[o]), )
        else:
            return

        session.writeFieldReport(fileName= rpt_name, append=OFF,
            sortItem=mySortItem, odb=self.myOdbDat, step=s, frame=f,
            outputPosition=myOutputPosition[o], variable=myVariable)

    def avi_save(self):
        """writes a video to file
            SCALE_FACTOR, TIME_HISTORY

            mytimeHistoryMode = TIME_BASED FRAME_BASED SCALE_FACTOR
        """
        #session.animationController.play(duration=UNLIMITED)
        # session.animationController.animationOptions.setValues(numScaleFactorFrames=7,
        #                                                        timeHistoryMode=TIME_BASED,
        #                                                        mode=PLAY_ONCE)
        # session.animationController.setValues(animationType=SCALE_FACTOR, viewports=(self.myVp.name, ))
        # session.imageAnimationOptions.setValues(vpDecorations=OFF,
        #                                         vpBackground=OFF,
        #                                         compass=OFF,
        #                                         timeScale=1,
        #                                         frameRate=30)
        session.animationController.setValues(animationType=TIME_HISTORY, viewports=(self.myVp.name, ))
        session.animationController.play(duration=UNLIMITED)
        session.imageAnimationOptions.setValues(vpDecorations=OFF,vpBackground=OFF,compass=OFF)
        o = self.myVp.odbDisplay.primaryVariable[0]
        c = self.myVp.odbDisplay.primaryVariable[5]
        s = self.myVp.odbDisplay.fieldSteps[0][0]
        v = self.myView
        cs = '' if self.myVp.odbDisplay.basicOptions.transformationType == DEFAULT else self.myVp.odbDisplay.basicOptions.datumCsys.name[1:]
        path_filename = '%s_%s_%s_%s_%s_%s.avi' % (self.myOdbDat.name.replace('.odb',''),s,o,c,cs,v)
        session.writeImageAnimation(path_filename, AVI, (self.myVp,))

    def png_save(self):
        """
            Captures images based on field outputs
        """
        o = self.myVp.odbDisplay.primaryVariable[0]
        c = self.myVp.odbDisplay.primaryVariable[5]
        f = self.myVp.odbDisplay.fieldFrame[1]
        s = self.myVp.odbDisplay.fieldSteps[0][0]
        v = self.myView
        cs = '' if self.myVp.odbDisplay.basicOptions.transformationType == DEFAULT else self.myVp.odbDisplay.basicOptions.datumCsys.name[1:]
        path_filename = '%s_step-%s_frame-%i_%s_%s_%s_%s.png' % (self.myOdbDat.name.replace('.odb',''),s,f,o,c,cs,v)
        try:
            session.printToFile(path_filename, PNG, (self.myVp,))
            print('%s saved' % path_filename)
        except:
            print('failed to save %s' % path_filename)

    def png_undeformed(self,v='Iso1'):
        """create a png of the mesh"""
        self.VP_edit(0, 0, '', '', '',   '', v,   ON,    OFF,    0,  0)
        path_filename = '%s_Undeformed_%s.png' % (self.myOdbDat.name.replace('.odb',''),v)
        try:
            session.printToFile(path_filename, PNG, (self.myVp,))
            print('%s saved' % path_filename)
        except:
            print('failed to save %s' % path_filename)

    def getMaxField(self, o = 'S', c='S11', elsetName='', csys = ''):
        """ gets max of every step and frame
            Test to develop a generic function to get the max/min of a specific field
            Can only filter a single element set.
            dir(fval) == 'inv3', 'magnitude', 'maxInPlanePrincipal', 'maxPrincipal', 'midPrincipal',
                            'minInPlanePrincipal', 'minPrincipal', 'mises', 'outOfPlanePrincipal', 'tresca',
            #fval.data == array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'f')

            self.getMaxField(o = 'U', c='U1', elsetName=None)
            self.getMaxField(o = 'U', c=0, elsetName=None)
            self.getMaxField(o = 'S', c='mises', elsetName=None)
            self.getMaxField(o = 'U', c='magnitude', elsetName=None)

            # fobj = self.myOdbDat.steps[sName].frames[f].fieldOutputs[o]
        """
        ElementSet = [s for s in self.myOdbDat.rootAssembly.elementSets.keys() if elsetName in s]
        if ElementSet and elsetName:
            elemset = self.myAssembly.elementSets[ElementSet[0]]
            print('elsetName %s ok' % elsetName)
        else:
            elemset = None
            print('using all elsetName')

        # return values, all max is the last value
        maxV = []
        maxE = []
        maxF = []
        maxS = []

        maxVal = None
        maxElem = None
        maxFrame= None
        maxStep = None
        for step in self.myOdbDat.steps.values():
            for frame in step.frames:
                if o in frame.fieldOutputs.keys():
                    fobj = frame.fieldOutputs[o]
                    if elemset:
                        fobj = fobj.getSubset(region=elemset)
                    if csys == 'cyn':
                        fobj = fobj.getTransformedField(datumCsys=self.csysCyn)
                    #x_csys = numpy.array([fobj.values[k].data[0] for k in range(len(fobj.values))]).T
                    #x = numpy.array([zip(*k.data)[1][c] for k in fieldOutputs])
                    for fval in fobj.values:
                        if c == ' ' or c == '':
                            dat = fval.data
                        elif isinstance(c,int):
                            dat = fval.data[c]
                        elif c in fobj.componentLabels:
                            dat = fval.data[fobj.componentLabels.index(c)]
                        else:
                            exec('dat=fval.'+c)
                        if dat > maxVal:
                            maxVal = dat
                            maxElem = fval.elementLabel
                            maxStep = step.name
                            maxFrame = frame.incrementNumber
                    maxV.append(maxVal)
                    maxE.append(maxElem)
                    maxF.append(maxFrame)
                    maxS.append(maxStep)
                    #print('step-%s frame=%i element=%s elementSet %s maxValue %f' % (maxStep, maxFrame,  maxElem, elsetName, maxVal) )
                    maxVal = None
                    maxElem = None
                    maxFrame= None
                    maxStep = None
                else:
                    print('field %s %s not found' % o)
        return maxV, maxE, maxS, maxF

    def CBE_makeReportData(self):
        """ Function to generate the report data for the customBeamTutorial (CBE)
            with an existing odb file create an example of outpus, such as text files, images and animations
            odbname = 'customBeamExample/customBeamExample.odb'
            # frames, [0,-1] will return the first and last frame"""

        dict_o_c = {'S':['S11','Mises'],
                    'U':['U1','U2','Magnitude'],
                    'E':['E11'],
                    'RF':['RF1'],
                    'CF': ['CF1'],
                    }
        list_v = ['Iso2','Left','Top']

        self.txt_history_summary()
        self.txt_field_summary()


        self.rpt_field( 0, 6, o='S', c='S11', setsKeyword='', csys = '')

        #self.csv_history([],     ['Element BEAMINSTANCE.42 Int Point 1'] , dict_o_c)

        s = 0
        f = -1
        self.csv_field( [],     [-1],   dict_o_c)
        self.txt_odb_summary()
        self.png_undeformed()

        csys=setsKeyword=''
        for o in dict_o_c:
            for c in dict_o_c[o]:
                for v in list_v:
                    print(o,c,v)
                    self.VP_edit(s, f, o, c, csys, setsKeyword ,v ,ON, ON, 0, 0)
                    self.png_save()
                    if v == 'Iso2':
                        self.VP_edit(s, f, o, c, csys, setsKeyword ,v ,OFF, OFF, 0, 0)
                        self.avi_save()

    def BE_makeReportData(self):
        """with an existing odb file create an example of outpus, such as text files, images and animations
            odbname = 'customBeamExample/customBeamExample.odb'"""

        self.txt_history_summary()
        self.txt_field_summary()

        #S,E acceptable values      'Magnitude','Max. Principal','Min. Principal','Mises',
        #                            'Tresca','Max. Principal (Abs)','Mid. Principal','Pressure','Third Invariant
        # historyRegions
        dict_o_c = {'S':['S11','Mises','Max. Principal (Abs)'],
                    'U':['U1','U2','Magnitude'],
                    'E':['E11'],
                    'RF':['RF1'],
                    'CF': ['CF1']}

        self.csv_history( [],     [-1],   dict_o_c)
        
        # frames, [0,-1] will return the first and last frame
        #                       list_s  list_f
        self.csv_field( [],     [-1],   dict_o_c)

        # general overview of odb file
        self.txt_odb_summary()

        # imaging
        self.png_undeformed()
        s = 0
        f = -1
        csys=setsKeyword=''
        list_v = ['Iso2','Left','Top']
        for o in dict_o_c:
            for c in dict_o_c[o]:
                for v in list_v:
                    print(o,c,v)
                    self.VP_edit(s, f, o, c, csys, setsKeyword ,v ,ON, ON, 0, 0)
                    self.png_save()
                    if v == 'Iso2':
                        self.VP_edit(s, f, o, c, csys, setsKeyword ,v ,OFF, OFF, 0, 0)
                        self.avi_save()

############################ Static Methods ###########################
def bottle_test():
    os.chdir('bottle_test')
    abaqus_cmd('abaqus job=bottle_test')
    time.sleep(45)  # program keeps running and wont have time to finish the job, so I added a manual wait
    # mdb.JobFromInputFile(name='bottle_test',
    #     inputFileName='bottle_test.inp',
    #     type=ANALYSIS, atTime=None, waitMinutes=0, waitHours=0, queue=None,
    #     memory=90, memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True,
    #     explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE,
    #     userSubroutine='', scratch='')
    # mdb.jobs['bottle_test'].submit(consistencyChecking=OFF)
    os.chdir('..')

def beamExample():
    """ demo of  executing abaqus commands"""
    if os.path.exists('beamExample'):
        os.chdir('beamExample')
    else:
        os.mkdir('beamExample')
        os.chdir('beamExample')
    print('beamExample created')
    abaqus_cmd('abaqus fetch job=beamExample')
    abaqus_cmd('abaqus cae noGUI=beamExample.py')
    # moves back one directory
    os.chdir('..')

    # makeReportData('beamExample/beamExample.odb')

def customBeamExample():
    """
    Reproduce the cantilever beam example from the
    Appendix of the Getting Started with
    Abaqus: Interactive Edition Manual.
    """

    customBeamExampleStr='''
    import os
    from abaqus import *
    from abaqusConstants import *
    #backwardCompatibility.setValues(includeDeprecated=True,reportDeprecated=False)
    # Create a model.
    myModel = mdb.Model(name='Beam')
    # Create a new viewport in which to display the model
    # and the results of the analysis.
    myViewport = session.Viewport(name='Cantilever Beam Example',
        origin=(0, 0), width=225, height=150)
    myViewport.maximize()
    #-----------------------------------------------------
    import part
    # Create a sketch for the base feature.
    mySketch = myModel.ConstrainedSketch(name='beamProfile',
        sheetSize=250.)
    # Create the rectangle.
    mySketch.rectangle(point1=(-100,10), point2=(100,-10))
    mySketch.rectangle(point1=(-95,5), point2=(95,-5))
    # Create a three-dimensional, deformable part.
    myBeam = myModel.Part(name='Beam', dimensionality=THREE_D,type=DEFORMABLE_BODY)
    # Create the part's base feature by extruding the sketch
    # through a distance of 25.0.
    myBeam.BaseSolidExtrude(sketch=mySketch, depth=25.0)
    #-----------------------------------------------------
    import material
    # Create a material.
    mySteel = myModel.Material(name='Steel')
    # Create the elastic properties: youngsModulus is 209.E3
    # and poissonsRatio is 0.3
    elasticProperties = (209.E3, 0.3)
    mySteel.Elastic(table=(elasticProperties, ) )
    #-------------------------------------------------------
    import section
    # Create the solid section.
    mySection = myModel.HomogeneousSolidSection(name='beamSection',
        material='Steel', thickness=1.0)
    # Assign the section to the region. The region refers
    # to the single cell in this model.
    region = (myBeam.cells,)
    myBeam.SectionAssignment(region=region,
        sectionName='beamSection')
    #-------------------------------------------------------
    import assembly
    # Create a part instance.
    myAssembly = myModel.rootAssembly
    myInstance = myAssembly.Instance(name='beamInstance',
        part=myBeam, dependent=OFF)
    #-------------------------------------------------------
    import step
    # Create a step. The time period of the static step is 1.0,
    # and the initial incrementation is 0.1; the step is created
    # after the initial step.
    myModel.StaticStep(name='beamLoad', previous='Initial',
        timePeriod=1.0, initialInc=0.1,
        description='Load the top of the beam.')
    #-------------------------------------------------------
    import load
    # Find the end face using coordinates.
    endFaceCenter = (-100,0,12.5)
    endFace = myInstance.faces.findAt((endFaceCenter,) )
    # Create a boundary condition that encastres one end
    # of the beam.
    endRegion = (endFace,)
    myModel.EncastreBC(name='Fixed',createStepName='beamLoad',
        region=endRegion)
    # Find the top face using coordinates.
    topFaceCenter = (0,10,12.5)
    topFace = myInstance.faces.findAt((topFaceCenter,) )
    # Create a pressure load on the top face of the beam.
    topSurface = ((topFace, SIDE1), )
    myModel.Pressure(name='Pressure', createStepName='beamLoad',
        region=topSurface, magnitude=0.5)
    #-------------------------------------------------------
    import mesh
    # Assign an element type to the part instance.
    region = (myInstance.cells,)
    elemType = mesh.ElemType(elemCode=C3D8I, elemLibrary=STANDARD)
    myAssembly.setElementType(regions=region, elemTypes=(elemType,))
    # Seed the part instance.
    myAssembly.seedPartInstance(regions=(myInstance,), size=9.0)
    # Mesh the part instance.
    myAssembly.generateMesh(regions=(myInstance,))
    # Display the meshed beam.
    myViewport.assemblyDisplay.setValues(mesh=ON)
    myViewport.assemblyDisplay.meshOptions.setValues(meshTechnique=ON)
    myViewport.setValues(displayedObject=myAssembly)
    #-------------------------------------------------------
    # add field output request for stress, strain and displacement
    myModel.FieldOutputRequest(name='F-Output-beamLoad', createStepName='beamLoad', variables=ALL)
    #-------------------------------------------------------
    # add history output request for stress, strain, and displacement
    p1 = mdb.models['Beam'].parts['Beam']
    session.viewports['Cantilever Beam Example'].setValues(displayedObject=p1)
    p = mdb.models['Beam'].parts['Beam']
    e = p.edges
    edges = e.getSequenceFromMask(mask=('[#100 ]', ), )
    p.Set(edges=edges, name='set-history-output')
    a = mdb.models['Beam'].rootAssembly
    a.regenerate()
    session.viewports['Cantilever Beam Example'].setValues(displayedObject=a)

    #regionDef=mdb.models['Beam'].rootAssembly.allInstances['beamInstance'].sets['set-history-output']
    regionDef=mdb.models['Beam'].rootAssembly.instances['beamInstance'].sets['set-history-output']

    mdb.models['Beam'].HistoryOutputRequest(name='H-Output-edge-set',
        createStepName='beamLoad', variables=('S11', 'S22', 'S33', 'E11',
        'E22', 'E33', 'U1', 'U2', 'U3', 'RF1', 'RF2', 'RF3'), region=regionDef,
        sectionPoints=DEFAULT, rebar=EXCLUDE)
    #-------------------------------------------------------
    import job
    # Create an analysis job for the model and submit it.
    jobName = 'customBeamExample'
    myJob = mdb.Job(name=jobName, model='Beam',
        description='Cantilever beam tutorial')
    # Wait for the job to complete.
    myJob.submit()
    myJob.waitForCompletion()
    # -------------------------------------------------------
    import visualization
    # Open the output database and display a
    # default contour plot.
    myOdb = visualization.openOdb(path=jobName + '.odb')
    myViewport.setValues(displayedObject=myOdb)
    myViewport.odbDisplay.display.setValues(plotState=CONTOURS_ON_DEF)
    myViewport.odbDisplay.contourOptions.setValues(numIntervals=15,  maxAutoCompute=ON,  minAutoCompute=ON)
    myViewport.odbDisplay.commonOptions.setValues(renderStyle=FILLED)
    session.printOptions.setValues(vpBackground=OFF)
    session.psOptions.setValues(orientation=LANDSCAPE)
    myViewport.viewportAnnotationOptions.setValues(triad=OFF,title=OFF,state=OFF,legend=OFF)
    myViewport.odbDisplay.basicOptions.setValues(coordSystemDisplay=OFF, )
    '''

    if os.path.exists('customBeamExample'):
        os.chdir('customBeamExample')
    else:
        os.mkdir('customBeamExample')
        os.chdir('customBeamExample')

    f = open('customBeamExample.py', 'w')
    f.write(dedent(customBeamExampleStr))
    f.close()
    abaqus_cmd('abaqus cae noGUI=customBeamExample.py')
    os.chdir('..')

def abaqus_cmd(mycmd):
    """ used to execute abaqus commands in the windows OS console
    inputs : mycmd, an abaqus command"""
    import subprocess, sys
    try:
        retcode = subprocess.call(mycmd,shell=True)
        if retcode < 0:
            print('>>' +sys.stderr, mycmd+"...failed during execution", -retcode)
        else:
            print('>>' + sys.stderr, mycmd+"...success")
    except OSError as e:
        print('>>' + sys.stderr, mycmd+"...failed at execution", e)

############################ Static Variables ###########################
myOutputPosition = {'E':INTEGRATION_POINT,
                    'S':INTEGRATION_POINT,
                    'PE':INTEGRATION_POINT,
                    'CF':NODAL,
                    'RF':NODAL,
                    'U':NODAL,
                    'EVOL':WHOLE_ELEMENT,
                    'STH':INTEGRATION_POINT}

INVComponentField = ['Magnitude','Max. Principal','Min. Principal',
                     'Mises','Tresca','Max. Principal (Abs)',
                     'Mid. Principal','Pressure','Third Invariant']

if __name__ == '__main__':
    """ run these commands if the script is explicitly called

    from abaqusConstants import CYLINDRICAL, ON, OFF, USER_SPECIFIED, INTEGRATION_POINT, INVARIANT, COMPONENT, NODAL, TIME, STRESS, STRAIN
    import odbAccess
    from abaqusConstants import TIME, STRESS, STRAIN

    import os
    os.chdir('C:/Users/ngordon/Desktop/abaqus-scripting')
    import AbaqusReportTool
    reload(AbaqusReportTool)
    myOdbTest = AbaqusReportTool.AbaqusReportTool('bottle_test/bottle_test.odb')

    x='Reaction force: RF1 at Node 58 in NSET SET-NODE-BOTTOM'
    y='Spatial displacement: U1 at Node 574 in NSET SET-NODE-MIDDLE'
    #self.xplot_history(0, x)
    self.xyplot_history(0, x, y)

    self.xplot_history()
    self.xyplot_history(0, 'Reaction force: RF1 at Node 58 in NSET SET-NODE-BOTTOM', 'Spatial displacement: U1 at Node 574 in NSET SET-NODE-MIDDLE')

    AbaqusReportTool.AbaqusReportTool.bottle_test()

    """

    odbname = sys.argv[-1]
    reportType = sys.argv[-2]

    # process  files if no demos are specified
    if os.path.splitext(odbname)[1] == '.odb':
        # odbname = 'bottle_test/bottle_test.odb'
        import AbaqusReportTool
        odb1 = AbaqusReportTool.AbaqusReportTool(odbname)

        if reportType == 'CBE':
            # Custom Beam Example
            odb1.CBE_makeReportData()

        elif reportType == 'BE':
            # Beam Example
            odb1.BE_makeReportData()

        elif reportType == 'BT':
            # Beam Test
            odb1.BT_makeReportData()

        else:
            print('no odb type detected, no report material created')
        odb1.myOdbDat.close()

    else: # Demo programs
        # if demos are specified, do them
        if odbname == 'customBeamExample':
            customBeamExample()
        elif odbname == 'beamExample':
            beamExample()
        else:
            print('No Demos run')


