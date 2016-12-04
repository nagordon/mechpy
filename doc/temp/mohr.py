#!/usr/bin/python
# https://github.com/estevaofon/mohrs-circle/blob/master/mohr.py
import Tkinter
import tkMessageBox
import math
import sys
import os
import subprocess


class Calculations:
    def __init__(self):
        self.sx = 0
        self.sy = 0
        self.txy = 0
        self.r = 0
        self.save = 0
        self.smax = 0
        self.smin = 0
        self.tmax = 0
        self.tetap = 0
        self.tetas = 0

    def run(self, sx, sy, txy, teta=0):
        sin = math.sin
        cos = math.cos
        rad = math.radians
        self.sx = sx
        self.sy = sy
        self.txy = txy
        self.teta = teta
        self.r = 0
        self.save = (self.sx + self.sy)/2.
        self.savem = (self.sx - self.sy)/2.
        self.r = ((((self.sx - self.sy)/2.)**2)+self.txy**2)**0.5
        self.nsx = (self.save + self.savem*cos(rad(2.*teta)) +
                    self.txy*sin(rad(2*teta)))
        self.nsy = (self.save - self.savem*cos(2.*rad(teta)) -
                    self.txy*sin(2.*rad(teta)))
        self.ntxy = -self.savem*sin(2.*rad(teta))+self.txy*cos(2.*rad(teta))
        self.smax = self.save + (((self.savem)**2) + self.txy**2)**0.5
        self.smin = self.save - (((self.savem)**2) + self.txy**2)**0.5
        self.tmax = ((self.savem**2) + self.txy**2)**0.5
        self.tetap = ((math.degrees(math.atan((2.*self.txy) /
                                              (self.sx-self.sy))))/2.)
        self.tetas = ((math.degrees(math.atan(-(self.sx-self.sy) /
                                              (2.*self.txy))))/2.)

    def conversion(self, radius):
        self.save_plot = (self.save*radius)/self.r
        self.sx_plot = (self.sx*radius)/self.r
        self.sy_plot = (self.sy*radius)/self.r
        self.txy_plot = (self.txy*radius)/self.r
        self.nsx_plot = (self.nsx*radius)/self.r
        self.nsy_plot = (self.nsy*radius)/self.r
        self.ntxy_plot = (self.ntxy*radius)/self.r


class Gui:
    def __init__(self, app):
        self.r_plot = 0
        app.title("Mohr Circle")
        app.geometry('850x500')
        self.calc = Calculations()
        self.circle_diameter = 400
        self.xo_circle = 120
        self.yo_circle = 50
        self.x1_circle = self.xo_circle + self.circle_diameter
        self.y1_circle = self.yo_circle + self.circle_diameter

        self.frame = Tkinter.Frame(app)
        self.frame.pack(side='right', pady=10)
        self.frame1 = Tkinter.Frame(self.frame)
        self.frame1.pack()
        self.frame2 = Tkinter.Frame(self.frame)
        self.frame2.pack()
        self.frame3 = Tkinter.Frame(self.frame)
        self.frame3.pack()
        self.frame4 = Tkinter.Frame(self.frame)
        self.frame4.pack()
        self.frame5 = Tkinter.Frame(self.frame)
        self.frame5.pack()

        top_draw_frame = Tkinter.Frame(app)
        top_draw_frame.pack(expand=Tkinter.YES, fill=Tkinter.BOTH, side='left')

        label_text = Tkinter.StringVar()
        label_text.set(u"\u03C3x")
        label1 = Tkinter.Label(self.frame1, textvariable=label_text, height=2)
        label1.pack(side='left', padx=7)
        cust_name = Tkinter.StringVar(None)
        self.entry1 = Tkinter.Entry(self.frame1,
                                    textvariable=cust_name, width=15)
        self.entry1.pack(side='left')
        self.entry1.focus_force()

        label_text2 = Tkinter.StringVar()
        label_text2.set(u"\u03C3y")
        label2 = Tkinter.Label(self.frame2, textvariable=label_text2, height=2)
        label2.pack(side='left', padx=7)
        cust_name2 = Tkinter.StringVar(None)
        self.entry2 = Tkinter.Entry(self.frame2,
                                    textvariable=cust_name2, width=15)
        self.entry2.pack(side='left')

        label_text3 = Tkinter.StringVar()
        label_text3.set(u"\u03C4xy")
        label3 = Tkinter.Label(self.frame3, textvariable=label_text3, height=2)
        label3.pack(side='left', padx=5)
        cust_name3 = Tkinter.StringVar(None)
        self.entry3 = Tkinter.Entry(self.frame3,
                                    textvariable=cust_name3, width=15)
        self.entry3.pack(side='left')

        label_text4 = Tkinter.StringVar()
        label_text4.set(u"\u03B8")
        label4 = Tkinter.Label(self.frame4, textvariable=label_text4, height=2)
        label4.pack(side='left', padx=12)
        cust_name4 = Tkinter.StringVar(None)
        self.entry4 = Tkinter.Entry(self.frame4,
                                    textvariable=cust_name4, width=15)
        self.entry4.pack(side='left')
        self.var = Tkinter.IntVar()
        R1 = Tkinter.Radiobutton(self.frame5, text="counterclockwise",
                                 variable=self.var, value=1)
        R1.pack(anchor='w')
        R2 = Tkinter.Radiobutton(self.frame5, text="clockwise",
                                 variable=self.var, value=2)
        R2.pack(anchor='w')
        R1.select()

        button1 = Tkinter.Button(self.frame, text="OK", width=10,
                                 command=self.execute)
        button1.pack(padx=10, pady=10)

        button2 = Tkinter.Button(self.frame, text="Log results", width=10,
                                 command=self.show_log)
        button2.pack(padx=10)

        menu_bar = Tkinter.Menu(app)
        file_menu = Tkinter.Menu(menu_bar, tearoff=0)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=app.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)
        about_menu = Tkinter.Menu(menu_bar, tearoff=0)
        about_menu.add_command(label="About the app", command=self.new_window)
        menu_bar.add_cascade(label="About", menu=about_menu)
        app.config(menu=menu_bar)

        self.r_plot = self.circle_diameter/2
        rx = self.xo_circle + self.r_plot
        ry = self.yo_circle + self.r_plot
        # left mohr circle
        self.canvas0 = Tkinter.Canvas(top_draw_frame, width=1000,
                                      height=500, bg='white')
        self.canvas0.pack(side='left')
        self.canvas0.create_oval(self.xo_circle, self.yo_circle, self.x1_circle,
                                 self.y1_circle, width=2,
                                 fill='#d2d2ff', tag='circle')

        self.canvas0.create_line(0, int(ry), 1000, int(ry),
                                 width=1, fill='black', tag='origin_line')
        self.canvas0.create_line(int(rx), 0, int(rx),
                                 1000, width=1, fill='black', tag='origin_line')
        diameter = 3
        self.rx2 = rx + diameter
        self.ry2 = ry + diameter
        self.canvas0.create_oval(rx-2, ry-2, self.rx2, self.ry2, width=2,
                                 fill='black', tag='center-dot')
        # convension canvas
        self.canvas1 = Tkinter.Canvas(self.frame, width=200,
                                      height=400, bg='#f0f0f0')
        self.canvas1.pack(expand=Tkinter.YES, fill='both')
        self.canvas1.create_rectangle(60, 100, 140, 180,
                                      fill='#b4b4ff', width=1)
        # sy arrow
        self.canvas1.create_line(100, 60, 100,
                                 100, width=1, fill='black', tag='origin_line')
        self.canvas1.create_line(100, 60, 95,
                                 70, width=1, fill='black', tag='origin_line')
        self.canvas1.create_line(100, 60, 105,
                                 70, width=1, fill='black', tag='origin_line')
        self.canvas1.create_text(100, 50, text=u"\u03C3y")
        # sx arrow
        self.canvas1.create_line(60, 140, 20,
                                 140, width=1, fill='black', tag='origin_line')
        self.canvas1.create_line(20, 140, 30,
                                 145, width=1, fill='black', tag='origin_line')
        self.canvas1.create_line(20, 140, 30,
                                 135, width=1, fill='black', tag='origin_line')
        self.canvas1.create_text(30, 125, text=u"\u03C3x")
        # txy up
        self.canvas1.create_line(150, 100, 150,
                                 140, width=1, fill='black', tag='origin_line')
        self.canvas1.create_line(150, 100, 155,
                                 110, width=1, fill='black', tag='origin_line')
        self.canvas1.create_line(150, 100, 145,
                                 110, width=1, fill='black', tag='origin_line')
        self.canvas1.create_text(150, 80, text=u"\u03C4xy")
        # txy right
        self.canvas1.create_line(100, 90, 140,
                                 90, width=1, fill='black', tag='origin_line')
        self.canvas1.create_line(140, 90, 130,
                                 95, width=1, fill='black', tag='origin_line')
        self.canvas1.create_line(140, 90, 130,
                                 85, width=1, fill='black', tag='origin_line')

    def execute(self):
        try:
            string_input1 = self.entry1.get()
            string_input2 = self.entry2.get()
            string_input3 = self.entry3.get()
            string_input4 = self.entry4.get()
            sx = float(string_input1)
            sy = float(string_input2)
            txy = float(string_input3)
            teta = float(string_input4)
            if self.var.get() == 2:
                teta = -teta
            self.calc.run(sx, sy, txy, teta)
            self.build_canvas()
        except ValueError:
            self.wrong_value()

    def wrong_value(self):
        tkMessageBox.showinfo("Status",
                              "Please insert only numbers!")

    def new_window(self):
        top = Tkinter.Toplevel()
        top.title("About the app")
        top.geometry('380x80')
        label_text = Tkinter.StringVar()
        label_text.set("App developed by Estevao Fonseca")
        label1 = Tkinter.Label(top, textvariable=label_text, height=2)
        label1.pack(side='top', padx=10, pady=15)
        top.mainloop()

    def file_not_created(self):
        top = Tkinter.Toplevel()
        top.title("Info")
        top.geometry('380x100')
        label_text = Tkinter.StringVar()
        label_text.set("There is no log yet")
        label1 = Tkinter.Label(top, textvariable=label_text, height=2)
        label1.pack(side='top', padx=10, pady=15)
        top.mainloop()

    def build_canvas(self):
        self.canvas0.delete('line1')
        self.canvas0.delete('line2')
        self.canvas0.delete('origin_line')
        self.canvas0.delete('center-dot')
        self.calc.conversion(self.r_plot)
        rx = self.xo_circle + self.r_plot
        ry = self.yo_circle + self.r_plot
        yo = ry
        xo = rx - self.calc.save_plot
        sx = xo + self.calc.sx_plot
        sy = xo + self.calc.sy_plot
        nsx = xo + self.calc.nsx_plot
        nsy = xo + self.calc.nsy_plot
        txy = yo + self.calc.txy_plot
        txym = yo - self.calc.txy_plot
        ntxy = yo + self.calc.ntxy_plot
        ntxym = yo - self.calc.ntxy_plot
        self.canvas0.create_line(int(rx), int(ry), int(nsx),
                                 int(ntxy), width=3, fill='red', tag='line2')
        self.canvas0.create_line(int(rx), int(ry), int(nsy),
                                 int(ntxym), width=3, fill='red', tag='line2')
        self.canvas0.create_line(int(rx), int(ry), int(sx),
                                 int(txy), width=3, fill='blue', tag='line1')
        self.canvas0.create_line(int(rx), int(ry), int(sy),
                                 int(txym), width=3, fill='blue', tag='line1')
        self.canvas0.create_line(0, int(yo), 1000, int(yo),
                                 width=1, fill='black', tag='origin_line')
        self.canvas0.create_line(xo, 0, xo,
                                 1000, width=1, fill='black', tag='origin_line')
        self.canvas0.create_oval(rx-2, ry-2, self.rx2, self.ry2, width=2,
                                 fill='black', tag='center-dot')
        try:
            with open("log.txt", 'w') as data:
                text = []
                text.append('tetap = '+str(self.calc.tetap)+'\n')
                text.append('tetas = '+str(self.calc.tetas)+'\n')
                text.append('smax = '+str(self.calc.smax)+'\n')
                text.append('smin = '+str(self.calc.smin)+'\n')
                text.append('tmax = '+str(self.calc.tmax)+'\n')
                text.append('smed = '+str(self.calc.save)+'\n')
                text.append('r = '+str(self.calc.r)+'\n')
                text.append('nsx = '+str(self.calc.nsx)+'\n')
                text.append('nsy = '+str(self.calc.nsy)+'\n')
                text.append('ntxy = '+str(self.calc.ntxy)+'\n')
                data.writelines(text)
                data.close()
        except:
            print "File erro"
            raise

    def show_log(self):
        if sys.platform == 'linux2':
            subprocess.call(["xdg-open", "log.txt"])
        else:
            os.startfile("log.txt")

app = Tkinter.Tk()
Gui(app)
app.mainloop()
