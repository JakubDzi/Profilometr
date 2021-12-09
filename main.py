import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import os
import gc
import numpy as np
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib import cm
import seaborn as sns
import scipy
from matplotlib.figure import Figure
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

class windowgauss:
    def gauss(self, x, a, x0, sigma):
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def prostokat(self, x, min, max, val):
        xr=[]
        for el in x:
            if ((el > min) & (el < max)):
                xr.append(val)
            else:
                xr.append(0)
        return xr

    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(master)
        self.title = tk.Label(self.frame, text="Otwórz plik i dopasuj krzywe do histogramu")
        self.title.pack()
        self.czytajframe=tk.Frame(self.frame)
        self.wyborlabel = tk.Label(self.czytajframe, text="Wybierz co którą daną wczytać (1 - wszystkie)")
        self.wyborlabel.pack()
        self.czytajprzyciskiframe=tk.Frame(self.czytajframe)
        self.danamin=tk.Button(self.czytajprzyciskiframe, text="-", command=self.minuswcz)
        self.danamin.pack(side=tk.RIGHT)
        self.danalab = tk.Label(self.czytajprzyciskiframe, text="1")
        self.danalab.pack(side=tk.RIGHT)
        self.danaplus = tk.Button(self.czytajprzyciskiframe, text="+", command=self.pluswcz)
        self.danaplus.pack(side=tk.RIGHT)
        self.czytajprzyciskiframe.pack()
        self.czytajframe.pack()
        self.opn = tk.Button(self.frame, text="Otwórz plik", command=self.openfile)
        self.opn.pack()
        self.opnprg = ttk.Progressbar(self.frame, orient='horizontal', mode='determinate', length=300)
        self.opnprg.pack()
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.plt = self.ax.plot(np.array([0, 0]), np.array([0, 0]))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.suw1fr = tk.Frame(self.frame)
        self.suw1min = tk.Scale(self.suw1fr, from_=0, to=100, orient=tk.HORIZONTAL, state=tk.DISABLED, command=self.su1min, showvalue=0)
        self.suw1min.grid(row=1, column=1)
        self.suw1max = tk.Scale(self.suw1fr, from_=0, to=100, orient=tk.HORIZONTAL, state=tk.DISABLED, command=self.su1max, showvalue=0)
        self.suw1max.grid(row=1, column=2)
        self.su1minl = tk.Label(self.suw1fr, text="Początek")
        self.su1minl.grid(row=2, column=1)
        self.su1maxl = tk.Label(self.suw1fr, text="Koniec")
        self.su1maxl.grid(row=2, column=2)
        self.suw1fr.pack()
        self.gau1lab = tk.Label(self.frame, text="Parametry gaussa poziomu odniesienia")
        self.gau1lab.pack()
        self.gau1fr = tk.Frame(self.frame)
        self.gau1ml = tk.Label(self.gau1fr, text="Wartość max")
        self.gau1ml.grid(row=1, column=1)
        self.gau1pl = tk.Label(self.gau1fr, text="Położenie max")
        self.gau1pl.grid(row=1, column=2)
        self.gau1sl = tk.Label(self.gau1fr, text="Sigma")
        self.gau1sl.grid(row=1, column=3)
        vcmd=(master.register(self.validate),'%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        self.gau1max = tk.Entry(self.gau1fr, validate = 'key', validatecommand=vcmd)
        self.gau1max.grid(row=2, column=1)
        self.gau1pol = tk.Entry(self.gau1fr, validate='key', validatecommand=vcmd)
        self.gau1pol.grid(row=2, column=2)
        self.gau1sig = tk.Entry(self.gau1fr, validate='key', validatecommand=vcmd)
        self.gau1sig.grid(row=2, column=3)
        self.gau1fr.pack()
        self.gpc = tk.Label(self.frame, text="Typ funkcji dla wgłębienia")
        self.gpc.pack()
        self.dziuratyp=tk.IntVar()
        self.dziuratyp.set(0)
        self.radfr=tk.Frame(self.frame)
        self.gbt = tk.Radiobutton(self.radfr, text="Gauss", variable=self.dziuratyp, value=1, command=self.selgau, state=tk.DISABLED)
        self.gbt.pack(side=tk.LEFT)
        self.pbt = tk.Radiobutton(self.radfr, text="Prostokąt", variable=self.dziuratyp, value=2, command=self.selgau, state=tk.DISABLED)
        self.pbt.pack(side=tk.LEFT)
        self.radfr.pack()
        self.f2fr = tk.Frame(self.frame)
        self.tx1 = tk.StringVar()
        self.tx2 = tk.StringVar()
        self.tx3 = tk.StringVar()
        self.f2ml = tk.Label(self.f2fr, textvariable=self.tx1)
        self.f2ml.grid(row=1, column=1)
        self.f2pl = tk.Label(self.f2fr, textvariable=self.tx2)
        self.f2pl.grid(row=1, column=2)
        self.f2sl = tk.Label(self.f2fr, textvariable=self.tx3)
        self.f2sl.grid(row=1, column=3)
        vcmd2 = (master.register(self.validate2), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        self.f2max = tk.Entry(self.f2fr, validate='key', validatecommand=vcmd2)
        self.f2max.grid(row=2, column=1)
        self.f2pol = tk.Entry(self.f2fr, validate='key', validatecommand=vcmd2)
        self.f2pol.grid(row=2, column=2)
        self.f2sig = tk.Entry(self.f2fr, validate='key', validatecommand=vcmd2)
        self.f2sig.grid(row=2, column=3)
        self.f2fr.pack()
        self.btn = tk.Button(self.frame, text="Przejdź dalej", command=self.dalej, state=tk.DISABLED)
        self.btn.pack()
        self.frame.grid(row=1, column=1)

        self.frame2 = tk.Frame(master)
        self.title2 = tk.Label(self.frame2, text="Mapa krateru")
        self.title2.pack()
        self.fig2 = Figure(figsize=(5, 4), dpi=100)
        self.ax2 = self.fig2.add_subplot(111, projection="3d")
        self.plt2 = self.ax2.plot_surface(np.zeros((1,1)), np.zeros((1,1)), np.zeros((1,1)))
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.frame2)
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.rad2frame = tk.Frame(self.frame2)
        self.r1f = tk.Frame(self.rad2frame)
        self.r2f = tk.Frame(self.rad2frame)
        self.datyp = tk.IntVar()
        self.datyp.set(1)
        self.w1bt = tk.Radiobutton(self.r1f, text="Normalne dane", variable=self.datyp, value=1, command=self.seldan, state=tk.DISABLED)
        self.w1bt.pack()
        self.w2bt = tk.Radiobutton(self.r2f, text="Wygładzone dane", variable=self.datyp, value=2, command=self.seldan, state=tk.DISABLED)
        self.w2bt.pack(side=tk.LEFT)
        self.w2sl = tk.Scale(self.r2f, from_=1, to=5, orient=tk.HORIZONTAL, state=tk.DISABLED, command=self.wyksl)
        self.w2sl.pack(side=tk.RIGHT)
        self.r1f.grid(row=1, column=1)
        self.r2f.grid(row=1, column=2)
        self.rad2frame.pack()
        self.wartyt=tk.Label(self.frame2, text="Parametry kształtu")
        self.wartyt.pack()
        self.parfr=tk.Frame(self.frame2)
        self.vminl=tk.Label(self.parfr, text="Objętość usuniętej części:")
        self.vminl.grid(row=1, column=1)
        self.vplul=tk.Label(self.parfr, text="Objętość wypukłości:")
        self.vplul.grid(row=2, column=1)
        self.maxdl=tk.Label(self.parfr, text="Maksymalna głębokość (Sv):")
        self.maxdl.grid(row=3, column=1)
        self.maxh=tk.Label(self.parfr, text="Maksymalna wysokość (Sp):")
        self.maxh.grid(row=4, column=1)
        self.mrozl=tk.Label(self.parfr, text="Rozpiętość wysokości (Sz):")
        self.mrozl.grid(row=5, column=1)
        self.adgll=tk.Label(self. parfr, text="Średnia głębokość wgłebienia:")
        self.adgll.grid(row=6, column=1)
        self.skl=tk.Label(self.parfr, text="Współczynnik skośności (Ssk):")
        self.skl.grid(row=7, column=1)
        self.swcpl=tk.Label(self.parfr, text="Średnia wysokość całego pomiaru (Sa):")
        self.swcpl.grid(row=8, column=1)
        self.stdl=tk.Label(self.parfr, text="Odchylenie standardowe wysokości (Sq):")
        self.stdl.grid(row=9, column=1)
        self.kurl=tk.Label(self.parfr, text="Kurtoza (Sku):")
        self.kurl.grid(row=10, column=1)
        self.nx1 = tk.StringVar()
        self.nx2 = tk.StringVar()
        self.nx3 = tk.StringVar()
        self.nx4 = tk.StringVar()
        self.nx5 = tk.StringVar()
        self.nx6 = tk.StringVar()
        self.nx7 = tk.StringVar()
        self.nx8 = tk.StringVar()
        self.nx9 = tk.StringVar()
        self.nx10 = tk.StringVar()
        self.vminw = tk.Label(self.parfr, textvariable=self.nx1)
        self.vminw.grid(row=1, column=2)
        self.vpluw = tk.Label(self.parfr, textvariable=self.nx2)
        self.vpluw.grid(row=2, column=2)
        self.maxdw = tk.Label(self.parfr, textvariable=self.nx3)
        self.maxdw.grid(row=3, column=2)
        self.maxhw = tk.Label(self.parfr, textvariable=self.nx4)
        self.maxhw.grid(row=4, column=2)
        self.mrozw = tk.Label(self.parfr, textvariable=self.nx5)
        self.mrozw.grid(row=5, column=2)
        self.adglw = tk.Label(self.parfr, textvariable=self.nx6)
        self.adglw.grid(row=6, column=2)
        self.skw = tk.Label(self.parfr, textvariable=self.nx7)
        self.skw.grid(row=7, column=2)
        self.swcpw = tk.Label(self.parfr, textvariable=self.nx8)
        self.swcpw.grid(row=8, column=2)
        self.stdw = tk.Label(self.parfr, textvariable=self.nx9)
        self.stdw.grid(row=9, column=2)
        self.kurw = tk.Label(self.parfr, textvariable=self.nx10)
        self.kurw.grid(row=10, column=2)
        self.parfr.pack()
        self.savbtn = tk.Button(self. frame2, text="Zapisz parametry do pliku", state=tk.DISABLED, command=self.zappar)
        self.savbtn.pack()
        self.frame2.grid(row=1, column=2)
        self.filename=''

    def su1min(self, event):
        if(self.suw1min.get()<self.suw1max.get()):
            self.minplot=int(len(self.binmids)*self.suw1min.get()/100)
            bintemp = self.binmids[self.minplot:self.maxplot]
            if self.dziuratyp.get() == 1:
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111)
                self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot],
                                        bintemp, self.gauss(bintemp, *self.popt),
                                        bintemp, self.gauss(bintemp, *self.dopt),
                                        bintemp, self.gauss(bintemp, *self.popt) + self.gauss(bintemp, *self.dopt))
                self.canvas.draw()
            if self.dziuratyp.get()==2:
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111)
                self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot],
                                        bintemp, self.gauss(bintemp, *self.popt),
                                        bintemp, self.prostokat(bintemp, *self.dkw),
                                        bintemp, self.gauss(bintemp, *self.popt) + self.prostokat(bintemp, *self.dkw))
                self.canvas.draw()
            if self.dziuratyp.get() == 0:
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111)
                self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot],
                                        bintemp,
                                        self.gauss(bintemp, *self.popt))
                self.canvas.draw()

    def su1max(self, event):
        if(self.suw1max.get()>self.suw1min.get()):
            self.maxplot=int(len(self.binmids)*self.suw1max.get()/100)
            bintemp = self.binmids[self.minplot:self.maxplot]
            if self.dziuratyp.get() == 1:
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111)
                self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot],
                                        bintemp,
                                        self.gauss(bintemp, *self.popt),
                                        bintemp,
                                        self.gauss(bintemp, *self.dopt),
                                        bintemp,
                                        self.gauss(bintemp, *self.popt) + self.gauss(
                                            bintemp, *self.dopt))
                self.canvas.draw()
            if self.dziuratyp.get()==2:
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111)
                self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot],
                                        bintemp,
                                        self.gauss(bintemp, *self.popt),
                                        bintemp,
                                        self.prostokat(bintemp, *self.dkw),
                                        bintemp,
                                        self.gauss(bintemp,
                                                   *self.popt) + self.prostokat(bintemp,
                                                                                *self.dkw))
                self.canvas.draw()
            if self.dziuratyp.get() == 0:
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111)
                self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot],
                                        bintemp,
                                        self.gauss(bintemp, *self.popt))
                self.canvas.draw()

    def zappar(self):
        savfile = asksaveasfilename(filetypes=[("Plik tekstowy", "*.txt")])
        if not savfile:
            return
        if not savfile.endswith('.txt'):
            savfile+='.txt'
        with open(savfile, 'w', encoding='utf-8') as f:
            f.write("Objętość usuniętej części: ")
            f.write(self.nx1.get())
            f.write("\n")
            f.write("Objętość wypukłości: ")
            f.write(self.nx2.get())
            f.write("\n")
            f.write("Maksymalna głębokość (Sv): ")
            f.write(self.nx3.get())
            f.write("\n")
            f.write("Maksymalna wysokość (Sp): ")
            f.write(self.nx4.get())
            f.write("\n")
            f.write("Rozpiętość wysokości (Sz): ")
            f.write(self.nx5.get())
            f.write("\n")
            f.write("Średnia głębokość wgłebienia: ")
            f.write(self.nx6.get())
            f.write("\n")
            f.write("Współczynnik skośności (Ssk): ")
            f.write(self.nx7.get())
            f.write("\n")
            f.write("Średnia wysokość całego pomiaru (Sa): ")
            f.write(self.nx8.get())
            f.write("\n")
            f.write("Odchylenie standardowe wysokości (Sq): ")
            f.write(self.nx9.get())
            f.write("\n")
            f.write("Kurtoza (Sku): ")
            f.write(self.nx10.get())

    def wyksl(self, event):
        if self.datyp.get() == 2:
            if self.w2sl.get()==1:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.plt2 = self.ax2.plot_surface(self.xws, self.yws, self.smth, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get()==2:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.plt2 = self.ax2.plot_surface(self.xws2, self.yws2, self.smth2, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get()==3:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.plt2 = self.ax2.plot_surface(self.xws3, self.yws3, self.smth3, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get()==4:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.plt2 = self.ax2.plot_surface(self.xws4, self.yws4, self.smth4, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get()==5:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.plt2 = self.ax2.plot_surface(self.xws5, self.yws5, self.smth5, cmap=cm.coolwarm)
                self.canvas2.draw()

    def seldan(self):
        if self.datyp.get()==1:
            self.canvas2.figure.clf()
            self.plt2 = []
            self.canvas2.draw()
            self.ax2 = self.fig2.add_subplot(111, projection="3d")
            self.plt2 = self.ax2.plot_surface(self.xw, self.yw, self.zw, cmap=cm.coolwarm)
            self.canvas2.draw()
        if self.datyp.get() == 2:
            if self.w2sl.get()==1:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.plt2 = self.ax2.plot_surface(self.xws, self.yws, self.smth, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get()==2:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.plt2 = self.ax2.plot_surface(self.xws2, self.yws2, self.smth2, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get()==3:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.plt2 = self.ax2.plot_surface(self.xws3, self.yws3, self.smth3, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get()==4:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.plt2 = self.ax2.plot_surface(self.xws4, self.yws4, self.smth4, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get()==5:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.plt2 = self.ax2.plot_surface(self.xws5, self.yws5, self.smth5, cmap=cm.coolwarm)
                self.canvas2.draw()
    def selgau(self):
        bintemp = self.binmids[self.minplot:self.maxplot]
        if self.dziuratyp.get()==1:
            self.tx1.set("Wartość max")
            self.tx2.set("Położenie max")
            self.tx3.set("Sigma")
            self.f2max.config(state=tk.NORMAL)
            self.f2max.delete(0, tk.END)
            self.f2max.insert(0, str(self.dopt[0]))
            self.f2pol.config(state=tk.NORMAL)
            self.f2pol.delete(0, tk.END)
            self.f2pol.insert(0, str(self.dopt[1]))
            self.f2sig.config(state=tk.NORMAL)
            self.f2sig.delete(0, tk.END)
            self.f2sig.insert(0, str(self.dopt[2]))
            self.canvas.figure.clf()
            self.plt = []
            self.canvas.draw()
            self.ax = self.fig.add_subplot(111)
            self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot], bintemp, self.gauss(bintemp, *self.popt),
                                    bintemp, self.gauss(bintemp, *self.dopt), bintemp, self.gauss(bintemp, *self.popt)+self.gauss(bintemp, *self.dopt))
            self.canvas.draw()
        if self.dziuratyp.get() == 2:
            self.tx1.set("Minimum")
            self.tx2.set("Maksimum")
            self.tx3.set("Wartość")
            self.f2max.config(state=tk.NORMAL)
            self.f2max.delete(0, tk.END)
            self.f2max.insert(0, str(self.dkw[0]))
            self.f2pol.config(state=tk.NORMAL)
            self.f2pol.delete(0, tk.END)
            self.f2pol.insert(0, str(self.dkw[1]))
            self.f2sig.config(state=tk.NORMAL)
            self.f2sig.delete(0, tk.END)
            self.f2sig.insert(0, str(self.dkw[2]))
            self.canvas.figure.clf()
            self.plt = []
            self.canvas.draw()
            self.ax = self.fig.add_subplot(111)
            self.plt = self.ax.plot(bintemp, self.ns, bintemp, self.gauss(bintemp, *self.popt),
                                    bintemp, self.prostokat(bintemp, *self.dkw), bintemp, self.gauss(bintemp, *self.popt)+self.prostokat(bintemp, *self.dkw))
            self.canvas.draw()
        self.btn.config(state=tk.NORMAL)

    def akt1(self, wn, val):
        bintemp = self.binmids[self.minplot:self.maxplot]
        self.gau1max.state='normal'
        self.gau1pol.state='normal'
        self.gau1sig.state='normal'
        if (wn == '.!frame.!frame3.!entry'):
            self.popt[0]=val
        if (wn == '.!frame.!frame3.!entry2'):
            self.popt[1]=val
        if (wn == '.!frame.!frame3.!entry3'):
            self.popt[2]=val
        self.canvas.figure.clf()
        self.plt = []
        self.canvas.draw()
        self.ax = self.fig.add_subplot(111)
        if self.dziuratyp.get()==0:
            self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot], bintemp, self.gauss(bintemp, *self.popt))
        if self.dziuratyp.get()==1:
            self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot], bintemp, self.gauss(bintemp, *self.popt), bintemp, self.gauss(bintemp, *self.dopt), bintemp, self.gauss(bintemp, *self.popt)+self.gauss(bintemp, *self.dopt))
        if self.dziuratyp.get()==2:
            self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot], bintemp, self.gauss(bintemp, *self.popt), bintemp, self.prostokat(bintemp, *self.dkw), bintemp, self.gauss(bintemp, *self.popt)+self.prostokat(bintemp, *self.dkw))
        self.canvas.draw()

    def akt2(self, wn, val):
        bintemp = self.binmids[self.minplot:self.maxplot]
        self.f2max.state = 'normal'
        self.f2pol.state = 'normal'
        self.f2sig.state = 'normal'
        if self.dziuratyp.get()==1:
            if (wn == '.!frame.!frame5.!entry'):
                self.dopt[0]=val
            if (wn == '.!frame.!frame5.!entry2'):
                self.dopt[1]=val
            if (wn == '.!frame.!frame5.!entry3'):
                self.dopt[2]=val
        if self.dziuratyp.get()==2:
            if (wn == '.!frame.!frame5.!entry'):
                self.dkw[0]=val
            if (wn == '.!frame.!frame5.!entry2'):
                self.dkw[1]=val
            if (wn == '.!frame.!frame5.!entry3'):
                self.dkw[2]=val
        self.canvas.figure.clf()
        self.plt = []
        self.canvas.draw()
        self.ax = self.fig.add_subplot(111)
        if self.dziuratyp.get() == 0:
            self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot], bintemp, self.gauss(bintemp, *self.popt))
        if self.dziuratyp.get() == 1:
            self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot], bintemp, self.gauss(bintemp, *self.popt),
                                    bintemp, self.gauss(bintemp, *self.dopt), bintemp, self.gauss(bintemp, *self.popt)+self.gauss(bintemp, *self.dopt))
        if self.dziuratyp.get() == 2:
            self.plt = self.ax.plot(bintemp, self.ns, bintemp, self.gauss(bintemp, *self.popt),
                                    bintemp, self.prostokat(bintemp, *self.dkw), bintemp, self.gauss(bintemp, *self.popt)+self.prostokat(bintemp, *self.dkw))
        self.canvas.draw()

    def validate(self, action, index, value_if_allowed,
                       prior_value, text, validation_type, trigger_type, widget_name):
        if prior_value!=None :
            try:
                float(value_if_allowed)
                self.akt1(widget_name, value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False

    def validate2(self, action, index, value_if_allowed,
                       prior_value, text, validation_type, trigger_type, widget_name):
        if prior_value!=None :
            try:
                float(value_if_allowed)
                self.akt2(widget_name, value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False

    def pluswcz(self):
        val=int(self.danalab["text"])
        self.danalab["text"]=f"{val+1}"
    def minuswcz(self):
        val = int(self.danalab["text"])
        if val>1:
            self.danalab["text"] = f"{val + -1}"

    def dalej(self):
        hei, wid = self.zw.shape
        vminus = 0
        vplus = 0
        if self.dziuratyp.get() == 1:
            mingaus = argrelextrema(self.gauss(self.binmids, *self.dopt) + self.gauss(self.binmids, *self.popt), np.less)[0]
            if len(mingaus)>0:
                mingaus = int(mingaus.item(0))
            else:
                indomaxo=0
                valmaxo=0
                for i in range(len(self.binmids)):
                    if valmaxo< (self.gauss(self.binmids[i], *self.dopt) + self.gauss(self.binmids[i], *self.popt)):
                        valmaxo=(self.gauss(self.binmids[i], *self.dopt) + self.gauss(self.binmids[i], *self.popt))
                        indomaxo=i
                for i in range(indomaxo):
                    if (self.gauss(self.binmids[indomaxo-i], *self.dopt) + self.gauss(self.binmids[indomaxo-i], *self.popt))<max(self.gauss(self.binmids, *self.dopt) + self.gauss(self.binmids, *self.popt))/100:
                        mingaus=indomaxo-i
            for i in range(hei - 1):
                for j in range(wid - 1):
                    if ((self.zw[(i, j)] < self.binmids[mingaus]) & (self.zw[(i + 1, j)] < self.binmids[mingaus]) & (
                            self.zw[(i, j + 1)] < self.binmids[mingaus]) & (self.zw[(i + 1, j + 1)] < self.binmids[mingaus])):
                        dw = max([self.zw[(i, j)], self.zw[(i + 1, j)], self.zw[(i, j + 1)], self.zw[(i + 1, j + 1)]])
                        gw = min([self.zw[(i, j)], self.zw[(i + 1, j)], self.zw[(i, j + 1)], self.zw[(i + 1, j + 1)]])
                        vminus += (self.popt[1] - dw) * self.resx * self.resy * 0.001 + (1 / 3) * self.resx * self.resy * (gw - dw) * 0.001
        if self.dziuratyp.get() == 2:
            for i in range(hei - 1):
                for j in range(wid - 1):
                    if ((self.zw[(i, j)] < self.binmids[self.dkw[1]]) & (self.zw[(i + 1, j)] < self.binmids[self.dkw[1]]) & (
                            self.zw[(i, j + 1)] < self.binmids[self.dkw[1]]) & (self.zw[(i + 1, j + 1)] < self.binmids[self.dkw[1]])):
                        dw = max([self.zw[(i, j)], self.zw[(i + 1, j)], self.zw[(i, j + 1)], self.zw[(i + 1, j + 1)]])
                        gw = min([self.zw[(i, j)], self.zw[(i + 1, j)], self.zw[(i, j + 1)], self.zw[(i + 1, j + 1)]])
                        vminus += (self.popt[1] - dw) * self.resx * self.resy * 0.001 + (1 / 3) * self.resx * self.resy * (gw - dw) * 0.001
        #print("Objętość usuniętej części: ", vminus, "mm\u00B3")
        self.nx1.set(str(vminus)+" mm\u00B3")
        for i in range(self.maxmaxind, len(self.binmids)):
            if self.gauss(self.binmids[i], *self.popt) < (self.popt[0] / 100):
                wypuk = self.binmids[i]
                pass
        for i in range(hei - 1):
            for j in range(wid - 1):
                if (self.zw[(i, j)] > wypuk or self.zw[(i + 1, j)] > wypuk or self.zw[(i, j + 1)] > wypuk or self.zw[
                    (i + 1, j + 1)] > wypuk):
                    dw = min([self.zw[(i, j)], self.zw[(i + 1, j)], self.zw[(i, j + 1)], self.zw[(i + 1, j + 1)]])
                    gw = max([self.zw[(i, j)], self.zw[(i + 1, j)], self.zw[(i, j + 1)], self.zw[(i + 1, j + 1)]])
                    vplus += self.resx * self.resy * (abs(dw - self.popt[1]) + (1 / 3) * (gw - dw)) * 0.001
        #print("Objętość wypukłości: ", vplus, "mm\u00B3")
        self.nx2.set(str(vplus)+" mm\u00B3")
        maxdep = 0
        maxhei = 0
        for i in range(hei):
            for j in range(wid):
                self.zw[(i, j)] -= self.popt[1]
                if maxdep > self.zw[(i, j)]:
                    maxdep = self.zw[(i, j)]
                if maxhei < self.zw[(i, j)]:
                    maxhei = self.zw[(i, j)]
        #print("Maksymalna głębokość (Sv): ", abs(maxdep), "\u03BCm")
        self.nx3.set(str(abs(maxdep))+" \u03BCm")
        #print("Maksymalna wysokość (Sp): ", maxhei, "\u03BCm")
        self.nx4.set(str(maxhei)+" \u03BCm")
        #print("Różnica między największą wysokością a maksymalną głębokością (Sz): ", maxhei - maxdep, "\u03BCm")
        self.nx5.set(str(maxhei - maxdep) + " \u03BCm")
        if self.dziuratyp.get() == 1:
            #print("Średnia głębokość wgłebienia: ", self.popt[1] - self.dopt[1], "\u03BCm")
            self.nx6.set(str(self.popt[1] - self.dopt[1])+" \u03BCm")
        if self.dziuratyp.get() == 2:
            #print("Średnia głębokość wgłębienia: ", float((self.dkw[1] - self.dkw[0]) / 2), "\u03BCm")
            self.nx6.set(str(float((self.dkw[1] - self.dkw[0]) / 2)) + " \u03BCm")
        quarts = np.percentile(self.zw, [25, 50, 75])
        #print("Współczynnik skośności (Ssk): ", (quarts[0] + quarts[2] - 2 * quarts[1]) / (quarts[2] - quarts[0]))
        self.nx7.set(str((quarts[0] + quarts[2] - 2 * quarts[1]) / (quarts[2] - quarts[0])))
        n = wid * hei
        zwsr = 0
        for i in range(hei):
            for j in range(wid):
                zwsr += self.zw[(i, j)] / n
        #print("Średnia wysokość całego pomiaru (Sa): ", zwsr, "\u03BCm")
        self.nx8.set(str(zwsr) + " \u03BCm")
        stdevsum = 0
        for i in range(hei):
            for j in range(wid):
                stdevsum += ((self.zw[(i, j)] - zwsr) ** 2 / n)
        stdev = np.sqrt(stdevsum)
        #print("Odchylenie standardowe wysokości (Sq): ", stdev)
        self.nx9.set(str(stdev))
        kurtsum = 0
        for i in range(hei):
            for j in range(wid):
                kurtsum += (((self.zw[(i, j)] - zwsr) / stdev) ** 4 * (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)))
        kurtoza = kurtsum - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
        #print("Kurtoza (Sku): ", kurtoza)
        self.nx10.set(str(kurtoza))
        tmp=wid
        wid=hei
        hei=tmp
        self.smth = np.zeros((wid - 1, hei - 1))
        self.xws = np.zeros((wid - 1, hei - 1))
        self.yws = np.zeros((wid - 1, hei - 1))
        self.smth2 = np.zeros((wid - 2, hei - 2))
        self.xws2 = np.zeros((wid - 2, hei - 2))
        self.yws2 = np.zeros((wid - 2, hei - 2))
        self.smth3 = np.zeros((wid - 3, hei - 3))
        self.xws3 = np.zeros((wid - 3, hei - 3))
        self.yws3 = np.zeros((wid - 3, hei - 3))
        self.smth4 = np.zeros((wid - 4, hei - 4))
        self.xws4 = np.zeros((wid - 4, hei - 4))
        self.yws4 = np.zeros((wid - 4, hei - 4))
        self.smth5 = np.zeros((wid - 5, hei - 5))
        self.xws5 = np.zeros((wid - 5, hei - 5))
        self.yws5 = np.zeros((wid - 5, hei - 5))
        for i in range(wid - 1):
            for j in range(hei - 1):
                self.smth[(i, j)] = (self.zw[(i, j)] + self.zw[(i + 1, j)] + self.zw[(i, j + 1)] + self.zw[(i + 1, j + 1)]) / 4
                self.xws[(i, j)] = (self.xw[(i, j)] + self.xw[(i + 1, j)]) / 2
                self.yws[(i, j)] = (self.yw[(i, j)] + self.yw[(i, j + 1)]) / 2
        for i in range(wid - 2):
            for j in range(hei - 2):
                self.smth2[(i, j)] = (self.smth[(i, j)] + self.smth[(i + 1, j)] + self.smth[(i, j + 1)] + self.smth[
                    (i + 1, j + 1)]) / 4
                self.xws2[(i, j)] = (self.xws[(i, j)] + self.xws[(i + 1, j)]) / 2
                self.yws2[(i, j)] = (self.yws[(i, j)] + self.yws[(i, j + 1)]) / 2
        for i in range(wid - 3):
            for j in range(hei - 3):
                self.smth3[(i, j)] = (self.smth2[(i, j)] + self.smth2[(i + 1, j)] + self.smth2[(i, j + 1)] + self.smth2[
                    (i + 1, j + 1)]) / 4
                self.xws3[(i, j)] = (self.xws2[(i, j)] + self.xws2[(i + 1, j)]) / 2
                self.yws3[(i, j)] = (self.yws2[(i, j)] + self.yws2[(i, j + 1)]) / 2
        for i in range(wid - 4):
            for j in range(hei - 4):
                self.smth4[(i, j)] = (self.smth3[(i, j)] + self.smth3[(i + 1, j)] + self.smth3[(i, j + 1)] + self.smth3[
                    (i + 1, j + 1)]) / 4
                self.xws4[(i, j)] = (self.xws3[(i, j)] + self.xws3[(i + 1, j)]) / 2
                self.yws4[(i, j)] = (self.yws3[(i, j)] + self.yws3[(i, j + 1)]) / 2
        for i in range(wid - 5):
            for j in range(hei - 5):
                self.smth5[(i, j)] = (self.smth4[(i, j)] + self.smth4[(i + 1, j)] + self.smth4[(i, j + 1)] + self.smth4[
                    (i + 1, j + 1)]) / 4
                self.xws5[(i, j)] = (self.xws4[(i, j)] + self.xws4[(i + 1, j)]) / 2
                self.yws5[(i, j)] = (self.yws4[(i, j)] + self.yws4[(i, j + 1)]) / 2
        self.btn.config(state=tk.DISABLED)
        self.gbt.config(state=tk.DISABLED)
        self.pbt.config(state=tk.DISABLED)
        self.gau1max.config(state=tk.DISABLED)
        self.gau1pol.config(state=tk.DISABLED)
        self.gau1sig.config(state=tk.DISABLED)
        self.f2max.config(state=tk.DISABLED)
        self.f2pol.config(state=tk.DISABLED)
        self.f2sig.config(state=tk.DISABLED)
        self.suw1min.config(state=tk.DISABLED)
        self.suw1max.config(state=tk.DISABLED)
        self.w1bt.config(state=tk.NORMAL)
        self.w2bt.config(state=tk.NORMAL)
        self.w2sl.config(state=tk.NORMAL)
        self.savbtn.config(state=tk.NORMAL)
        self.canvas2.figure.clf()
        self.plt2 = []
        self.canvas2.draw()
        self.ax2 = self.fig2.add_subplot(111, projection="3d")
        self.plt2 = self.ax2.plot_surface(self.xw, self.yw, self.zw, cmap=cm.coolwarm)
        self.canvas2.draw()

    def openfile(self):
        self.filename = ''
        self.filename = askopenfilename()
        if self.filename!='':
            self.opnprg['value']=0
            plik = self.filename
            dziel = int(self.danalab["text"])
            f = open(plik, "r")
            linie = f.readlines()
            del f
            gc.collect()
            x = []
            y = []
            z = []
            doblin = 0
            cnt = 0
            cw = int(0)
            for i in range(len(linie)):
                if(int(i/len(linie)*100)>cw):
                    cw+=1
                    self.opnprg['value']=cw/3
                    self.opnprg.update()
                if ((linie[i] != '\n') & (dziel == 0)):
                    nlin = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in linie[i])
                    lon = [float(j) for j in nlin.split()]
                    x.append(lon[0])
                    y.append(lon[1])
                    z.append(lon[2])
                if (dziel != 0):
                    if ((linie[i] != '\n') & (doblin == 0)):
                        if ((cnt % dziel) == 0):
                            nlin = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in linie[i])
                            lon = [float(j) for j in nlin.split()]
                            x.append(lon[0])
                            y.append(lon[1])
                            z.append(lon[2])
                        cnt += 1
                    if (linie[i] == '\n'):
                        doblin += 1
                        if (2 * dziel == doblin):
                            doblin = 0
                            cnt = 0
            i = 0
            x0 = x[0]
            rev = 0
            while (x[i] == x0):
                i = i + 1
            if (i >= 1):
                hei = i
                wid = int(len(z) / hei)
                rev = 1
                y0 = y[0]
                i = 0
                while (y[i] == y0):
                    i = i + 1
            else:
                y0 = y[0]
                while (y[i] == y0):
                    i = i + 1
                wid = i
                hei = int(len(z) / wid)
            self.xw = []
            self.yw = []
            self.zw = []
            if rev == 1:
                for i in range(hei):
                    self.yw.append(y[i])
                    self.zw.append([])
                    for j in range(wid):
                        self.zw[i].append(z[i + hei * j])
                for i in range(wid):
                    self.xw.append(x[hei * i])
                self.zw = np.array(self.zw, order='A')
            if rev == 0:
                for i in range(wid):
                    self.yw.append(y[i])
                    self.zw.append([])
                    for j in range(hei):
                        self.zw[i].append(z[i + wid * j])
                for j in range(hei):
                    self.xw.append(x[j * wid])
                self.zw = np.array(self.zw, order='A')
            del x, y, z, nlin, lon
            gc.collect()
            self.xw, self.yw = np.meshgrid(self.xw, self.yw, indexing='xy')
            for i in range(hei):
                for j in range(wid):
                    if self.zw[(i, j)] == 0:
                        #print("błąd odczytu dla punktu:", self.xw[(i, j)], self.yw[(i, j)])
                        sm = 0
                        ad = 0
                        if i < wid:
                            ad += self.zw[(i + 1, j)]
                            sm += 1
                        if i > 0:
                            ad += self.zw[(i - 1, j)]
                            sm += 1
                        if j < hei:
                            ad += self.zw[(i, j + 1)]
                            sm += 1
                        if j > 0:
                            ad += self.zw[(i, j - 1)]
                            sm += 1
                        self.zw[(i, j)] = ad / sm
            #print("wczytano. obracanie po x")
            ztx = []
            zty = []
            ytx = []
            xtx = []
            xty = []
            yty = []
            ztx2 = []
            zty2 = []
            ytx2 = []
            xtx2 = []
            xty2 = []
            yty2 = []
            for i in range(wid):
                ztx.append(self.zw[(0, i)])
                ytx.append(self.yw[(0, 0)])
                xtx.append(self.xw[(0, i)])
                ztx2.append(self.zw[(hei - 1, i)])
                ytx2.append(self.yw[(hei - 1, 0)])
                xtx2.append(self.xw[(hei - 1, i)])
            fx = np.polyfit(xtx, ztx, 1)
            fx2 = np.polyfit(xtx2, ztx2, 1)
            cw=0
            sw=self.opnprg['value']
            for i in range(hei):
                for j in range(wid):
                    if (int((i*wid+j) / (hei*wid) * 100) > cw):
                        cw += 1
                        self.opnprg['value'] =  sw+ (cw / 3)
                        self.opnprg.update()
                    self.zw[(i, j)] = self.zw[(i, j)] - (fx[0] + fx2[0]) / 2 * self.xw[(i, j)]
                    self.xw[(i, j)] = self.xw[(i, j)] * (1 / (np.cos(np.arctan((fx[0] + fx2[0]) / 2))) - 1)
            #print("obrócono. obracanie po y")
            for i in range(hei):
                zty.append(self.zw[(i, 0)])
                yty.append(self.yw[(i, 0)])
                xty.append(self.xw[(0, 0)])
                zty2.append(self.zw[(i, wid - 1)])
                yty2.append(self.yw[(i, wid - 1)])
                xty2.append(self.xw[(0, wid - 1)])
            fy = np.polyfit(yty, zty, 1)
            fy2 = np.polyfit(yty2, zty2, 1)
            cw = 0
            sw = self.opnprg['value']
            for i in range(hei):
                for j in range(wid):
                    if (int((i*wid+j) / (hei*wid) * 100) > cw):
                        cw += 1
                        self.opnprg['value'] =  sw+ (cw / 3)
                        self.opnprg.update()
                    self.zw[(i, j)] = self.zw[(i, j)] - (fy[0] + fy2[0]) / 2 * self.yw[(i, j)]
                    self.yw[(i, j)] = self.yw[(i, j)] * (1 / (np.cos(np.arctan((fy[0] + fy2[0]) / 2))) - 1)
            self.resx = self.xw[(0, 1)] - self.xw[(0, 0)]
            self.resy = self.yw[(1, 0)] - self.yw[(0, 0)]
            maxdep = 0
            for i in range(hei):
                for j in range(wid):
                    if maxdep > self.zw[(i, j)]:
                        maxdep = self.zw[(i, j)]
            for i in range(hei):
                for j in range(wid):
                    self.zw[(i, j)] += abs(maxdep)
            #print("obrócono. szukanie poziomu odniesienia")
            zh = []
            for i in range(hei):
                for j in range(wid):
                    zh.append(self.zw[(i, j)])
            zhist = sns.histplot(data=zh)
            n, bins = [], []
            for rect in zhist.patches:
                ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
                n.append(y1 - y0)
                bins.append(x0)
            bins.append(x1)
            binmid = []
            indohival=0
            cuhival=0
            for i in range(len(n)):
                if n[i]>cuhival:
                    indohival=i
                    cuhival=n[i]
            indmin=indohival
            indmax=indohival
            for i in range(indohival):
                if n[indohival-i]>=(max(n)/100):
                    indmin=indohival-i
            for i in range(len(n)-indohival):
                if n[indohival+i]>=(max(n)/100):
                    indmax=indohival+i

            for i in range(len(bins) - 1):
                binmid.append((bins[i] + bins[i + 1]) / 2)
            #self.binmids = []
            #self.ns = []
            #for i in range(indmax - indmin):
            #    self.binmids.append(binmid[i + indmin])
            #    self.ns.append(n[i + indmin])
            self.binmids=binmid
            self.ns=n
            self.minplot = 0
            self.maxplot = len(self.binmids)
            try:
                maxima = argrelextrema(np.array(self.ns), np.greater)[0]
                minima = argrelextrema(np.array(self.ns), np.less)[0]
                nmax = []
                nmin = []
                maxx = []
                minx = []
                for m in maxima:
                    nmax.append(self.ns[m])
                for m in minima:
                    nmin.append(self.ns[m])
                for i in range(len(maxima)):
                    maxx.append(self.binmids[maxima[i]])
                for i in range(len(minima)):
                    minx.append(self.binmids[minima[i]])
                self.maxmaxind = maxima[0]
                for i in range(len(maxima)):
                    if self.ns[maxima[i]] > self.ns[self.maxmaxind]:
                        self.maxmaxind = maxima[i]
                jminind = 0
                dminind = minima[0]
                for m in range(len(minima) - 1):
                    if minima[m] < self.maxmaxind:
                        jminind = minima[m]
                        dminind = minima[m + 1]
                if minima[len(minima) - 1] < self.maxmaxind:
                    jminind = minima[len(minima) - 1]
                    dminind = len(self.binmids) - 1
                xgauss = []
                ygauss = []
                for i in range(dminind - jminind - 2):
                    xgauss.append(self.binmids[i + 1 + jminind])
                    ygauss.append(self.ns[i + 1 + jminind])
                xgauss = np.array(xgauss)
                ygauss = np.array(ygauss)
                mean = sum(xgauss * ygauss) / sum(ygauss)
                sigma = np.sqrt(sum(ygauss * (xgauss - mean) ** 2) / sum(ygauss))
                self.popt, pcov = curve_fit(self.gauss, xgauss, ygauss, p0=[max(ygauss), mean, sigma])
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111)
                self.plt = self.ax.plot(self.binmids, self.ns, self.binmids, self.gauss(self.binmids, *self.popt))
                self.canvas.draw()
                menu = 1
                self.gau1max.delete(0,tk.END)
                self.gau1max.insert(0,self.popt[0])
                self.gau1pol.delete(0, tk.END)
                self.gau1pol.insert(0, self.popt[1])
                self.gau1sig.delete(0, tk.END)
                self.gau1sig.insert(0, self.popt[2])
                del binmid, bins, dminind, fx, fx2, fy, fy2, hei, indmax, indmin, i, j, jminind, linie, maxima, maxx, mean, minima, minx, n, nmax, nmin, pcov, plik, rect, sigma, sm, wid, x0, x1, xgauss, xtx, xtx2, xty, xty2, y0, y1, ygauss, ytx, ytx2, yty, yty2, zh, zhist, ztx, ztx2, zty, zty2
            #except ValueError or RuntimeError or TypeError or RuntimeWarning:
            except:
                gausspod = 0
                self.popt=np.ones(3)
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111)
                self.plt = self.ax.plot(self.binmids, self.ns, self.binmids, self.gauss(self.binmids, *self.popt))
                self.canvas.draw()
                self.gau1max.delete(0, tk.END)
                self.gau1max.insert(0, self.popt[0])
                self.gau1pol.delete(0, tk.END)
                self.gau1pol.insert(0, self.popt[1])
                self.gau1sig.delete(0, tk.END)
                self.gau1sig.insert(0, self.popt[2])
            self.dopt=np.ones(3)
            self.dkw=np.ones(3)
            gc.collect()
            self.danamin.config(state=tk.DISABLED)
            self.danalab.config(state=tk.DISABLED)
            self.danaplus.config(state=tk.DISABLED)
            self.opn.config(state=tk.DISABLED)
            self.gbt.config(state=tk.NORMAL)
            self.pbt.config(state=tk.NORMAL)
            self.suw1min.config(state=tk.NORMAL)
            self.suw1max.config(state=tk.NORMAL)
            self.suw1max.set(100)

root = tk.Tk()
root.title("Profilometr")
cls = windowgauss(root)
root.mainloop()