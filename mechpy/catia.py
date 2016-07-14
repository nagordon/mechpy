# -*- coding: utf-8 -*-
'''
CATIA Python Module that connects to a Windows CATIA session through COM
Assumes that CATIA is running.

'''
import win32com.client
import random, os, math, glob

class Catia(object):
    ''' object to manipulate CATIA through windows COM '''

    # Initialize CATIA
    def __init__(self):
        print('attemping to connect to CATIA...')
        try:
            # faster, but no tool tips
            self.CATIA = win32com.client.Dispatch("CATIA.Application")
            print('CATIA connection success')
            ## slower but has tool tips
            # self.CATIA = win32com.client.gencache.EnsureDispatch("CATIA.Application")
            self.connect_doc()
        except:
            print('failed to connect to CATIA, error')

    #==============================================================================
    # public methods 
    #==============================================================================
    
    def connect_doc(self):
        '''connects to current active part'''
        try:
            self.myWindow = self.CATIA.ActiveWindow
            self.myDoc = self.CATIA.ActiveDocument
            #self.myPart = self.CATIA.ActiveDocument.Part
            print('connection to active document -' + self.myDoc.Name + '- success...')
        except:
            print('No Open Documents, Initialized Blank CATIA session')        

    ############ file open and close
    def launch_catia(self):
        #TODO - NEEDS TESTING
        ''' launches a quiet catia process'''
        ## alternative method
        #import subprocess
        #p = subprocess.Popen('Catia_R21_Launch.bat', creationflags=subprocess.CREATE_NEW_CONSOLE)      
        #return p
        os.system(r'Catia_R21_Launch.bat')
        
    def quit_catia(self):
        ''' terminates the current catia object'''
        self.CATIA.quit()

    def open_dialog(self):
        #self.CATIA.FileSelectionBox("File Open","*.CATPart", CATFileSelectionModeOpen)
        self.custom_cmd("Open")
        
    def save_dialog(self):
        #self.CATIA.FileSelectionBox('File Save','*.CATPart',CATFileSelectionModeSave)
        self.custom_cmd("Save")  
        
    def open_part(self, filepath='cube.CATPart'):
        ''' opens an existing CATIA part given an absolute filepath'''
        self.CATIA.Documents.Open(filepath)

    def new_from(self, filepath = "cube.CATPart"):
        '''creating a new part from an existing document'''
        self.CATIA.Documents.NewFrom(filepath)

    def read_from(self, filepath = "cube.CATPart"):
        '''loading a CATIA document, faster than opening but does not show it'''
        self.CATIA.Documents.Read(filepath)
        
    def save_current(self):
        '''saves the current file'''
        self.CATIA.ActiveDocument.Save()
        
    def save_current_as(self, file2='new1'):
        '''saves the current file as'''
        self.CATIA.ActiveDocument.SaveAs(file2)

    def update(self):
        ''' updates catia object'''
        self.CATIA.StartCommand("Update")

    def get_username(self):
        username = self.CATIA.SystemService.Environ("USERNAME")
        return username

    def toggle_show(self):
        ''' toggles the visiblity current CATIA session'''
        self.CATIA.Visible = False if self.CATIA.Visible else True
         
    def custom_cmd(self, cmdin):
        '''runs the command when you mouse over the icon in CATIA
            Open, Save, Fit All In, * iso '''
        self.CATIA.StartCommand(cmdin)
        
    def fit_window(self):
        self.custom_cmd("Fit All In")

    def show_selection(self):
        '''prints current selection in CATIA'''
        # prints any selection in CATIA
        for k in range(self.CATIA.ActiveDocument.Selection.Count):
            print(self.CATIA.ActiveDocument.Selection.Item(k+1).Value.Name)

    def search(self,s="Name='PLY-01',all"):
        '''search strings
        s=".Point.Name=Point.1*;All"
        '''
        #TODO - NEEDS TESTING
        #selection1 = self.myDoc.Selection
        mysel = self.CATIA.ActiveDocument.Selection # recognizing geometric elements
        sresult = mysel.Search(s)        


    def user_selection(self):
        Sel1 = self.CATIA.ActiveDocument.Selection
        Sel1.Clear
        what = ["Pad","Sketch"]
        out = Sel1.SelectElement2(what,"make a selection of a pad or a sketch",False)
        if out == 'Normal':
            print(Sel1.Item(1).Value.Name)
        else:
            print('Selection failed')        

    def add_part(self):
        ''' adds part to current catia object'''
        self.CATIA.Documents.Add("Part")

    def add_product(self):
        ''' adds part to current catia object'''
        self.CATIA.Documents.Add("Product")
        
    def add_drawing(self):
        '''creates a new drawing'''
        self.CATIA.Documents.Add("Drawing")
        self.myDwg = self.CATIA.ActiveDocument

    def add_geoset(self):
        ''' adds a flat geoset set give as a list'''
        self.CATIA.ActiveDocument.HybridBodies.Add()

    def export_dxf(self, fix1 = True):
        '''function to export a CATDrawing to dxf
        TODO - need to figure out how to export CATDrawing to dxf'''
        try:
            
            #c.myDoc.ExportData(os.getcwd()+'\\'+c.myDwg.Name, 'dxf')
            self.myDoc.ExportData(os.getcwd()+'\\'+self.myDoc.Name.split('.')[0], 'dxf')
        except:
            print('success creating dxf files')
        if fix1: # if files need a line removed, set this as true
            files = glob.glob('*.dxf')
            delete_list = ["isocp.shx"]            
            for f in files:    
                # f= 'f1.txt'
                fin = open(f)
                assert(f[0]!='_')  # no underlines in filename                
                fout = open('_'+f, "w+")
                for line in fin:
                    for word in delete_list:
                        line = line.replace(word, "")
                    fout.write(line)
                fin.close()
                fout.close()
                os.rename(f,f.split('.')[0]+'_old.dxf')
                os.rename('_'+f, f)
            

    def show_windows(self):
        ''' only one catia application can be running at a time
            all catia windows can be found using the Windows object'''
        print('---All windows---')
        for k in range(self.CATIA.Windows.Count):
            print(self.CATIA.Windows.Item(k+1).Name)
        print('---Active Window---')
        print(self.CATIA.ActiveWindow.Name)
        
    def show_product_info(self):
        ad = self.CATIA.ActiveDocument
        prod1 = ad.Product
        # lists all parts in a currently open CATProduct
        for i in range(prod1.Products.Count):
            print('Part Number:' + prod1.Products.Item(i+1).PartNumber)

    def show_docs(self):
        ''' a list of all documents or files can be generated with this command'''
        print('---All docs---')
        for k in range(self.CATIA.Documents.Count):
            print(self.CATIA.Documents.Item(k+1).Name)

    def show_geoset_tree(self, indent = ' '):
        ''' recursuvely returns a list of all the geosets in the part
        c.show_geoset_tree(c.CATIA.ActiveDocument.Part)
        part1 = c.CATIA.ActiveDocument.Part
        '''
        geoset_list = []
        part1 = self.CATIA.ActiveDocument.Part
        def geoset_tree(part1, indent = ' '):
            for k in range(part1.HybridBodies.Count):
                print(indent+part1.HybridBodies.Item(k+1).Name)
                geoset_list.append(part1.HybridBodies.Item(k+1).Name)
                try:#if part1.HybridBodies.Count > 1:
                    geoset_tree(part1.HybridBodies.Item(k+1),indent+'   ')
                except:
                    pass
        geoset_tree(part1)
        #print(geoset_list)
        return geoset_list
        
    def show_body_tree(self, indent = ' '):
        ''' returns a list of all the bodies in the part
        Bodies cannot be nested like geosets
        part1 = self.CATIA.ActiveDocument
        c.CATIA.ActiveDocument.Part
        c.CATIA.ActiveDocument.Part
        c.show_body_tree(c.CATIA.ActiveDocument.Part)
        '''
        part1 = self.CATIA.ActiveDocument.Part
        def body_tree(part1, indent = ' '):
            for k in range(part1.Bodies.Count):
                print(indent+part1.Bodies.Item(k+1).Name) 
        body_tree(part1)   

    def make_geosets(self):
        g = 3  # number of main geosets
        sg = 4 # number of subgeosets
        #k Main geosets
        for j in range(g):
            part1 = self.CATIA.ActiveDocument.Part
            part1 = part1.HybridBodies.Add()
            part1.Name = 'Main_geoset_'+str(j)
            # subgeoset
            for k in range(sg):
                part1 = part1.HybridBodies.Add()
                part1.Name = 'subgeoset_'+str(j)+'_'+str(k)                 
                
    def create_bom(self):
        #TODO - NEEDS TESTING
        '''' creates BOM for the current catia object'''
        prod1 = self.CATIA.ActiveDocument.Product
        prod1.Product.ExtractBOM(2, os.getcwd()+'\\'+"BOM.html")

    def print_parameters(self):
        ''' shows all parameters in catia'''
        # param_tree
        part1 = self.CATIA.ActiveDocument.Part
        for k in range(part1.Parameters.Count):
            print(part1.Parameters.Item(k+1).Name)        

    def get_ply_dir(self):
        part1 = self.CATIA.ActiveDocument.Part
        for k in range(part1.Parameters.Count):
            n = part1.Parameters.Item(k+1).Name
            if 'Rosette Name' in n or len(n) < 4:
                print(n)          

    def change_parameter(self, strMatch = '', inVal = 0):
        '''enter a value for strMatch and this will search through all the 
            parameters and try to match them. If more than one match is found,
            nothing will happen, if exactly one match is found the value will 
            update with the
            Example - this is find the shaft and change the angle to 180
            change_parameter(strMatch = 'Shaft.1\FirstAngle', inVal = 180)'''
        param_process = self.CATIA.ActiveDocument.GetItem("Process").Parameters
        slist = []
        for k in range(param_process.Count):
            s = param_process.Item(k+1).Name
            if strMatch in s:
                print(param_process.Item(k+1).Name)
                slist.append(s)
        #print slist
        #s = r'Part6\PartBody\Shaft.1\FirstAngle'
        if len(slist) == 1:
            print(slist[0])
            param_process.Item(slist[0]).Value = inVal
            self.update()

    def add_point(self, x=0, y=0, z=0):
        '''adds a point to the current part'''
        self.myPart = self.CATIA.ActiveDocument.Part
        try:
            hybridBody1 = self.myPart.HybridBodies.Item(1)
            print('found %s' % hybridBody1.Name)
        except:
            print('no geometric set found, creating a new one')
            hybridBody1 = self.myPart.HybridBodies.Add()  

        hybridShapePointCoord1 = self.myPart.HybridShapeFactory.AddNewPointCoord(x, y, z)        
        hybridBody1.AppendHybridShape(hybridShapePointCoord1)
        self.myPart.InWorkObject = hybridShapePointCoord1        
        #self.update()       
        
    def add_points(self):
        '''adds a bunch of random points'''
        for r in range(500):
            r = 50
            x,y,z = (random.randint(-r,r),random.randint(-r,r), random.randint(-r,r))        
            self.add_point(x, y, z)
            #self.update()

    def get_plies(self, stackName='Stacking', indent = ' '):
        ''' returns all plies on the Stacking geoset '''
        self.myPart = self.CATIA.ActiveDocument.Part
        plies = []        
        stack = self.myPart.HybridBodies.Item(stackName)
        for pg in range(1, stack.HybridBodies.Count+1):
            pliesgroup = stack.HybridBodies.Item(pg)
            for sq in range(1, pliesgroup.HybridBodies.Count+1):
                seq = pliesgroup.HybridBodies.Item(sq) 
                for pl in range(1, seq.HybridBodies.Count+1):
                    plies.append(seq.HybridBodies.Item(pl).Name)
        #print(plies)
        return plies   
       
    def plybook(self, stackName = "Stacking", pliesgroup = "Plies Group"):
        '''creates flat patterns in a catdrawing
        just have the CATPart composite part open
        given a geoset stackName, will generate a sheet for each ply'''

        self.myPart = self.myDoc.Part 
        
        # 1) Check that plies do not have duplicate names
        plies = self.get_plies()
        if set([x for x in plies if plies.count(x) > 1]):
            print('FYI, Duplicate Ply Names found. This is not allowed in Drawings, exiting...')
            return
        
        # 2) Add Drawing
        self.add_drawing()      # c.add_drawing()    
        mydwg1 = self.myDwg
        
        stack = self.myPart.HybridBodies.Item(stackName)
        
        for pg in range(1, stack.HybridBodies.Count+1):
            pliesgroup = stack.HybridBodies.Item(pg)
            
            for sq in range(1, pliesgroup.HybridBodies.Count+1):
                seq = pliesgroup.HybridBodies.Item(sq) 
                
                for pl in range(1, seq.HybridBodies.Count+1):
                    
                    mydwg1.Sheets.Add("AutomaticNaming") 
                    
                    mysheet1 = mydwg1.Sheets.Item(pl)  # mydwg1.Sheets.Item("Sheet.1")
                    mysheet1.Name = seq.HybridBodies.Item(pl).Name
                    
                    mysheet1.Activate()
                    #mysht1 = drawingSheets1.Item("Sheet.2")
                    myview1 = mysheet1.Views.Add("AutomaticNaming")
                    drawingViewGenerativeLinks1 = myview1.GenerativeLinks
                    drawingViewGenerativeBehavior1 = myview1.GenerativeBehavior
                    drawingViewGenerativeBehavior1.SetGPSName("CompositesGVSFlattenOnly.xml")
                    
                    pg1 = self.myPart.HybridBodies.Item(stackName).HybridBodies.Item(pg)
        
                    hybridBody4 = pg1.HybridBodies.Item(sq).HybridBodies.Item(pl)
                    drawingViewGenerativeLinks1.AddLink(hybridBody4)
                    drawingViewGenerativeBehavior1.DefineFrontView(1, 0, 0, 0, 1, 0)
                    drawingViewGenerativeBehavior1 = myview1.GenerativeBehavior
                    drawingViewGenerativeBehavior1.Update()
                    myview1.Activate
        
#                    # temp
#                    DrwTexts    = myview1.Texts
#                    ProductDrawn = mysheet1.Views.Item("Front view").GenerativeBehavior.Document                         
#                    DrwTexts.GetItem("TitleBlock_Text_Number_1").Text = ProductDrawn.Name
#                    ProductAnalysis = ProductDrawn.Analyze
#
#                    DrwTexts.GetItem("TitleBlock_Text_Weight_1").Text = ProductAnalysis.Mass
#                    textFormat = DrwTexts.GetItem("TitleBlock_Text_Size_1")
                    
                    
        
                    #self.change_parameter(strMatch = 'Text.1/DrwDressUp.1/Front view', inVal = 'test')        
                    self.fit_window()  #   c.fit_window()
        mydwg1.Sheets.Remove(mydwg1.Sheets.Count)
        
        
        # add geometry and text 
        # DrwDoc = self.CATIA.ActiveDocument
        
        # Get Selection Object and clear it
        #DrwSelect = DrwDoc.Selection
        #DrwSelect.Clear()            
        #DrwDoc.Sheets.ActiveSheet
        for s in range(1,mydwg1.Sheets.Count+1):

            DrwSheet = mydwg1.Sheets.Item(s)
            # Active Currentsheet
            DrwSheet.Activate()
            DrwViews = DrwSheet.Views
            #DrwView.SaveEdition()
            
             #3/ Scan all the view of the current sheet
            for v in range(1, DrwViews.Count+1):
                CurrentView = DrwViews.Item(v)
                #Active the current view
                CurrentView.Activate()
                DrwView     = DrwSheet.Views.ActiveView                
        	    #4/ Scan all the texts of the current view
                Texts = CurrentView.Texts
                DrwTexts = DrwView.Texts 
                   
                for t in range(1, Texts.Count+1):                   
                    CurrentText = Texts.Item(t)
                    #Texts.Item(t).x
                    #Texts.Item(t).y
                    Texts.Item(t).text = DrwSheet.Name + '\nply direction x'
                    #Texts.Item(t).GetFontName(0,0) 
                    Texts.Item(t).SetFontName(0,0, 'SSS4')
                    Texts.Item(t).SetFontSize(0,0, 12)
        
                    my2DFactory = CurrentView.Factory2D
                    Line = my2DFactory.CreateLine(0,0,50.8,0)
                    Line.Name = "0-ply-direction"        
        
        #self.save_current()
        self.save_current_as(os.getcwd()+'\\'+self.myDoc.Name.split('.')[0] + '.CATDrawing')
        

#==============================================================================
# Module Functions        
#==============================================================================
        
def bolt_test():
    '''make a bolt'''
    
    c = Catia()
    MyDocument = c.CATIA.Documents.Add("Part")
    print(MyDocument.Name)
    PartFactory = MyDocument.Part.ShapeFactory  # Retrieve the Part Factory.
    MyBody1 = MyDocument.Part.Bodies.Item("PartBody")
    c.CATIA.ActiveDocument.Part.InWorkObject = MyBody1 # Activate "PartDesign#
    
    # Creating the Shaft
    ReferencePlane1 = MyDocument.Part.CreateReferenceFromGeometry(MyDocument.Part.OriginElements.PlaneYZ)
      
    #Create the sketch1 on ReferencePlane1
    Sketch1 = MyBody1.Sketches.Add(ReferencePlane1)
    MyFactory1 = Sketch1.OpenEdition() # Define the sketch
    
    h1 = 80 # height of the bolt
    h2 = 300 # total height
    r1 = 120 # external radius
    r2 = 60 # Internal radius
    s1 = 20 # Size of the chamfer
      
    l101 = MyFactory1.CreateLine(0, 0, r1 - 20, 0)
    l102 = MyFactory1.CreateLine(r1 - 20, 0, r1, -20)
    l103 = MyFactory1.CreateLine(r1, -20, r1, -h1 + 20)
    l104 = MyFactory1.CreateLine(r1, -h1 + 20, r1 - 20, -h1)
    l105 = MyFactory1.CreateLine(r1 - 20, -h1, r2, -h1)
    l106 = MyFactory1.CreateLine(r2, -h1, r2, -h2 + s1)
    l107 = MyFactory1.CreateLine(r2, -h2 + s1, r2 - s1, -h2)
    l108 = MyFactory1.CreateLine(r2 - s1, -h2, 0, -h2)
    l109 = MyFactory1.CreateLine(0, -h2, 0, 0)
    Sketch1.CenterLine = l109
      
    Sketch1.CloseEdition
    AxisPad1 = PartFactory.AddNewShaft(Sketch1)
      
    #' Creating the Pocket
    ReferencePlane2 = MyDocument.Part.CreateReferenceFromGeometry(MyDocument.Part.OriginElements.PlaneXY)
        
    # Create the sketch2 on ReferencePlane2
    Sketch2 = MyBody1.Sketches.Add(ReferencePlane2)
    MyFactory2 = Sketch2.OpenEdition()
    D = 1 / 0.866
      
    l201 = MyFactory2.CreateLine(D * 100, 0, D * 50, D * 86.6)
    l202 = MyFactory2.CreateLine(D * 50, D * 86.6, D * -50, D * 86.6)
    l203 = MyFactory2.CreateLine(D * -50, D * 86.6, D * -100, 0)
    l204 = MyFactory2.CreateLine(D * -100, 0, D * -50, D * -86.6)
    l205 = MyFactory2.CreateLine(D * -50, D * -86.6, D * 50, D * -86.6)
    l206 = MyFactory2.CreateLine(D * 50, D * -86.6, D * 100, 0)
    
    #  ' Create a big circle around the form to get a Hole
    c2 = MyFactory2.CreateClosedCircle(0, 0, 300)
      
    Sketch2.CloseEdition
    AxisHole2 = PartFactory.AddNewPocket(Sketch2, h1)
      
    viewer3D1 = c.CATIA.ActiveWindow.ActiveViewer
    viewer3D1.Reframe
    viewpoint3D1 = viewer3D1.Viewpoint3D
    
    c.update() 
    c.fit_window()        

def cube_test():
    '''creates a cube
    https://gist.github.com/jl2/2704426'''
    
    c = Catia()
    part1 = c.CATIA.Documents.Add("Part").Part
    ad = c.CATIA.ActiveDocument
    part1 = ad.Part
    bod = part1.MainBody
    bod.Name="cube"

    cubeWidth = 10

    skts = bod.Sketches
    xyPlane = part1.CreateReferenceFromGeometry(part1.OriginElements.PlaneXY)
    shapeFact = part1.Shapefactory

    ms = skts.Add(xyPlane)
    ms.Name="Cube Outline"

    fact = ms.OpenEdition()
    fact.CreateLine(-cubeWidth, -cubeWidth,  cubeWidth, -cubeWidth)
    fact.CreateLine(cubeWidth, -cubeWidth,  cubeWidth, cubeWidth)
    fact.CreateLine(cubeWidth, cubeWidth,  -cubeWidth, cubeWidth)
    fact.CreateLine(-cubeWidth, cubeWidth,  -cubeWidth, -cubeWidth)
    ms.CloseEdition()
    mpad = shapeFact.AddNewPad(ms, cubeWidth)
    mpad.Name = "Python Pad"
    mpad.SecondLimit.Dimension.Value = cubeWidth

    sel = ad.Selection
    sel.Add(mpad)

    vp = sel.VisProperties
    vp.SetRealColor(random.randint(0,255),random.randint(0,255),random.randint(0,255), 0)
    part1.Update()
    
    hbs = part1.HybridBodies
    hBod = hbs.Add()
    hsf = part1.HybridShapeFactory

    sel.Search("Topology.Face,sel")
    faceCnt = sel.Count

    print("Found",faceCnt,"faces")
    hsd = hsf.AddNewDirectionByCoord(0,0,1)
    faces=[]
    for i in range(1, faceCnt+1):
        faces.append(sel.Item2(i))

    for fac in faces:
        sel.Clear()
        sel.Add(fac.Value)

        vp = sel.VisProperties
        vp.SetRealColor(random.randint(63,191), random.randint(63, 191), random.randint(63, 191), 0)

    sel.Clear()

    # Hide Planes   
    visPropertySet1 = sel.VisProperties
    part1 = c.CATIA.ActiveDocument.Part
    sel.Add(part1.OriginElements.PlaneXY)
    sel.Add(part1.OriginElements.PlaneYZ)
    sel.Add(part1.OriginElements.PlaneZX)
    visPropertySet1 = visPropertySet1.Parent
    bSTR1 = visPropertySet1.Name
    bSTR2 = visPropertySet1.Name
    visPropertySet1.SetShow(1)
    sel.Clear        
    
    part1.Update()
    ad.SaveAs(os.getcwd()+'\\'+'cube.CATPart')
    #ad.Close()
    
    ################ helix

    # make an assembly with helix
    '''makes a helix with the cube'''
    
    x = lambda t: 40.0*math.cos(t)
    y = lambda t: 40.0*math.sin(t)
    z = lambda t: 20*t
        
    prod1 = c.CATIA.Documents.Add("Product").Product
    #prod1 = c.CATIA.Documents.Add("Product").Product
    ad = c.CATIA.ActiveDocument
    #ad = c.CATIA.ActiveDocument
    
    prod1 = ad.Product
    prod_list = prod1.Products
    cn_list = [os.getcwd()+'\\'+'cube.CATPart']
    num_cubes = 100
    
    min_t = -math.pi*10
    max_t = math.pi*10
    t = min_t
    dt = (max_t-min_t)/(num_cubes-1)
    
    # original cube insertion
    '''
    for i in range(num_cubes):
        curName = 'Part1.{}'.format(i+1)
        prod_list.AddComponentsFromFiles(cn_list, "All")
        itm = prod_list.Item(curName)
        mvr = itm.Move
        mvr = mvr.MovableObject
        trans = [1.0,0.0,0.0,
                 0.0,1.0,0.0,
                 0.0,0.0,1.0,
                 x(t), y(t), z(t)]
        mvr.Apply(trans)
        trans = [math.cos(t),-math.sin(t),0.0,
                 math.sin(t),math.cos(t),0.0,
                 0.0,0.0,1.0,
                 0.0,0.0,0.0]
        mvr.Apply(trans)
        t += dt   
    '''
    
    for i in range(num_cubes):
        curName = 'Part1.{}'.format(i+1)
        prod_list.AddComponentsFromFiles(cn_list, "All")
        itm = prod_list.Item(curName)
        mvr = itm.Move
        mvr = mvr.MovableObject
        r = 200
        xt,yt,zt = (random.randint(-r,r), 
                     random.randint(-r,r), 
                     random.randint(-r,r))
        trans = [1.0,0.0,0.0,
                 0.0,1.0,0.0,
                 0.0,0.0,1.0,
                 xt, yt, zt]
        mvr.Apply(trans)
        t += dt             
        
    ad.SaveAs(os.getcwd()+'\\'+'cube_helix')
    #ad.Close()



if __name__ == '__main__':
    '''this code is executed if this script if explicitly run. This code is not
       executed if the file is imported as a module'''
    
    # grabs the current catia window and instantiates with c
    c = Catia()  
    
    # create a drawing and addinfo for the plybook
    #c.plybook()

    # work tbd
    # c.dwg_add_txt_geo()

    #save the drawing
    #c.save_current()
    
    # export dxf and remove a specific line    
    #c.export_dxf()
    
    bolt_test()
    
    #c.active_file_name()
    
    #c.export_dxf()
    
    #c.show_geoset_tree()    
    #c.print_parameters()
    #c.toggle_show()
    #cube_test()