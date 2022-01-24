import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import os
import gc
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import cm
import seaborn as sns
from matplotlib.figure import Figure
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import warnings


class Windowgauss:
    def gauss(self, x, a, x0, sigma):   #funkcja zwracająca wartość rozkładu Gaussa
        return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    def prostokat(self, x, min, max, val):  #funkcja zwracająca wartość funkcji prostokątnej
        xr = []
        for el in x:
            if ((el > min) & (el < max)):
                xr.append(val)
            else:
                xr.append(0)
        return xr

    def __init__(self, master):
        self.master = master
        self.frame = tk.Frame(master)   #ramka odpowiadająca za lewą część interfejsu
        self.title = tk.Label(self.frame, text="Otwórz plik i dopasuj krzywe do histogramu")
        self.title.pack()
        self.czytajframe = tk.Frame(self.frame) #ramka odpowiadająca za część interfejsu odpowiadającą za wczytywanie pliku
        self.wyborlabel = tk.Label(self.czytajframe, text="Wybierz co którą daną wczytać (1 - wszystkie)")
        self.wyborlabel.pack()
        self.czytajprzyciskiframe = tk.Frame(self.czytajframe)  #ramka z przyciskami + i -, oraz wyświetleniem liczby ograniczającej wczytywane dane
        self.danamin = tk.Button(self.czytajprzyciskiframe, text="-", command=self.minuswcz)
        self.danamin.pack(side=tk.RIGHT)    #elementy są ustawiane od prawej do lewej
        self.danalab = tk.Label(self.czytajprzyciskiframe, text="1")    #domyślnie wczytywana jest każda wartość
        self.danalab.pack(side=tk.RIGHT)
        self.danaplus = tk.Button(self.czytajprzyciskiframe, text="+", command=self.pluswcz)
        self.danaplus.pack(side=tk.RIGHT)
        self.czytajprzyciskiframe.pack()
        self.czytajframe.pack()
        self.opn = tk.Button(self.frame, text="Otwórz plik", command=self.openfile)
        self.opn.pack()
        self.opnprg = ttk.Progressbar(self.frame, orient='horizontal', mode='determinate', length=300)  #pasek postępu, ustawiony na tryb gdzie wartość 100 to w pełni wypełniony pasek
        self.opnprg.pack()
        self.fig = Figure(figsize=(5, 4), dpi=100)  #obrazek odpowiedzialny za wyświetlanie wykresu 2D
        self.ax = self.fig.add_subplot(111) #dzielenie wykresu na obiekty
        self.plt = self.ax.plot(np.array([0, 0]), np.array([0, 0])) #inicjalizacja wartości na wykresie, w celu dalszej łatwiejszej manipulacji nimi
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)    #wstawienie wykresu na obrazek
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.suw1fr = tk.Frame(self.frame)  #ramka z suwakami odpowiadającymi za wybór wartości wyświetlanych na histogramie
        self.suw1min = tk.Scale(self.suw1fr, from_=0, to=100, orient=tk.HORIZONTAL, state=tk.DISABLED,
                                command=self.su1min, showvalue=0) #suwak wartości minimalnej
        self.suw1min.grid(row=1, column=1)
        self.suw1max = tk.Scale(self.suw1fr, from_=0, to=100, orient=tk.HORIZONTAL, state=tk.DISABLED,
                                command=self.su1max, showvalue=0) #suwak wartości maksymalnej
        self.suw1max.grid(row=1, column=2)
        self.su1minl = tk.Label(self.suw1fr, text="Początek")   #podpisy suwaków
        self.su1minl.grid(row=2, column=1)
        self.su1maxl = tk.Label(self.suw1fr, text="Koniec")
        self.su1maxl.grid(row=2, column=2)
        self.suw1fr.pack()
        self.gau1lab = tk.Label(self.frame, text="Parametry gaussa poziomu odniesienia")
        self.gau1lab.pack()
        self.gau1fr = tk.Frame(self.frame)  #ramka odpowiedzialna za ustawianie parametrów rozkładu Gaussa poziomu odniesienia
        self.gau1ml = tk.Label(self.gau1fr, text="Wartość max")
        self.gau1ml.grid(row=1, column=1)
        self.gau1pl = tk.Label(self.gau1fr, text="Położenie max")
        self.gau1pl.grid(row=1, column=2)
        self.gau1sl = tk.Label(self.gau1fr, text="Sigma")
        self.gau1sl.grid(row=1, column=3)
        vcmd = (master.register(self.validate), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        #komenda przekazująca wartości wpisywane w polach tekstowych do funkcji odpowiadającej za sprawdzenie czy są liczbami
        self.gau1max = tk.Entry(self.gau1fr, validate='key', validatecommand=vcmd)
        self.gau1max.grid(row=2, column=1)
        self.gau1pol = tk.Entry(self.gau1fr, validate='key', validatecommand=vcmd)
        self.gau1pol.grid(row=2, column=2)
        self.gau1sig = tk.Entry(self.gau1fr, validate='key', validatecommand=vcmd)
        self.gau1sig.grid(row=2, column=3)
        self.gau1fr.pack()
        self.gpc = tk.Label(self.frame, text="Typ funkcji dla wgłębienia")
        self.gpc.pack()
        self.dziuratyp = tk.IntVar()    #zmienna przechowująca typ funkcji wgłębienia
        self.dziuratyp.set(0)   #na początku żaden typ nie jest wybrany
        self.radfr = tk.Frame(self.frame)   #ramka z przyciskami odpowiadającymi za wybór typu funkcji
        self.gbt = tk.Radiobutton(self.radfr, text="Gauss", variable=self.dziuratyp, value=1, command=self.selgau,
                                  state=tk.DISABLED)
        self.gbt.pack(side=tk.LEFT)
        self.pbt = tk.Radiobutton(self.radfr, text="Prostokąt", variable=self.dziuratyp, value=2, command=self.selgau,
                                  state=tk.DISABLED)
        self.pbt.pack(side=tk.LEFT)
        self.radfr.pack()
        self.f2fr = tk.Frame(self.frame)    #ramka odpowiedzialna za ustawienia parametrów funkcji wgłębienia
        self.tx1 = tk.StringVar()   #zmienne tekstowe, opisujące wpisywane parametry (wyświetlane nad polami tekstowymi)
        self.tx2 = tk.StringVar()
        self.tx3 = tk.StringVar()
        self.f2ml = tk.Label(self.f2fr, textvariable=self.tx1)  #etykiety, o tekście którym są wyżej sdefiniowane zmienne
        self.f2ml.grid(row=1, column=1)
        self.f2pl = tk.Label(self.f2fr, textvariable=self.tx2)
        self.f2pl.grid(row=1, column=2)
        self.f2sl = tk.Label(self.f2fr, textvariable=self.tx3)
        self.f2sl.grid(row=1, column=3)
        vcmd2 = (master.register(self.validate2), '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')
        # druga komenda przekazująca wartości wpisywane w polach tekstowych do funkcji odpowiadającej za sprawdzenie czy są liczbami
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

        self.frame2 = tk.Frame(master)  #ramka odpowiadająca za prawą część interfejsu
        self.title2 = tk.Label(self.frame2, text="Mapa krateru")
        self.title2.pack()
        self.fig2 = Figure(figsize=(5, 4), dpi=100) #obrazek odpowiedzialny za wyświetlanie wykresu 3D
        self.ax2 = self.fig2.add_subplot(111, projection="3d")  #dzielenie wykresu na obiekty
        self.plt2 = self.ax2.plot_surface(np.zeros((1, 1)), np.zeros((1, 1)), np.zeros((1, 1))) #inicjalizacja wartości na wykresie, w celu dalszej łatwiejszej manipulacji nimi
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.frame2) #wstawienie wykresu na obrazek
        self.canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.rad2frame = tk.Frame(self.frame2)  #ramka z opcjami odpowiedzialnymi za wybór wyświetlanych danych
        self.r1f = tk.Frame(self.rad2frame) #ramka z przyciskiem "normalne dane"
        self.r2f = tk.Frame(self.rad2frame) #ramka z przyciskiem "wygładzone dane" i suwakiem
        self.datyp = tk.IntVar()    #zmienna przechowująca typ wyświetlanych danych
        self.datyp.set(1)   #domyślnie to normalne dane
        self.w1bt = tk.Radiobutton(self.r1f, text="Normalne dane", variable=self.datyp, value=1, command=self.seldan,
                                   state=tk.DISABLED)
        self.w1bt.pack()
        self.w2bt = tk.Radiobutton(self.r2f, text="Wygładzone dane", variable=self.datyp, value=2, command=self.seldan,
                                   state=tk.DISABLED)
        self.w2bt.pack(side=tk.LEFT)
        self.w2sl = tk.Scale(self.r2f, from_=1, to=5, orient=tk.HORIZONTAL, state=tk.DISABLED, command=self.wyksl) #suwak sterujący stopniem wygładzenia danych
        self.w2sl.pack(side=tk.RIGHT)
        self.r1f.grid(row=1, column=1)
        self.r2f.grid(row=1, column=2)
        self.rad2frame.pack()
        self.wartyt = tk.Label(self.frame2, text="Parametry kształtu")
        self.wartyt.pack()
        self.parfr = tk.Frame(self.frame2)  #ramka ze wszystkimi parametrami kształtu
        self.vminl = tk.Label(self.parfr, text="Objętość usuniętej części:")
        self.vminl.grid(row=1, column=1)
        self.vplul = tk.Label(self.parfr, text="Objętość wypukłości:")
        self.vplul.grid(row=2, column=1)
        self.maxdl = tk.Label(self.parfr, text="Maksymalna głębokość (Sv):")
        self.maxdl.grid(row=3, column=1)
        self.maxh = tk.Label(self.parfr, text="Maksymalna wysokość (Sp):")
        self.maxh.grid(row=4, column=1)
        self.mrozl = tk.Label(self.parfr, text="Rozpiętość wysokości (Sz):")
        self.mrozl.grid(row=5, column=1)
        self.adgll = tk.Label(self.parfr, text="Średnia głębokość wgłebienia:")
        self.adgll.grid(row=6, column=1)
        self.skl = tk.Label(self.parfr, text="Współczynnik skośności (Ssk):")
        self.skl.grid(row=7, column=1)
        self.swcpl = tk.Label(self.parfr, text="Średnia wysokość całego pomiaru (Sa):")
        self.swcpl.grid(row=8, column=1)
        self.stdl = tk.Label(self.parfr, text="Odchylenie standardowe wysokości (Sq):")
        self.stdl.grid(row=9, column=1)
        self.kurl = tk.Label(self.parfr, text="Kurtoza (Sku):")
        self.kurl.grid(row=10, column=1)
        self.nx1 = tk.StringVar()   #zmienne w których przechowywany jest tekst z wartością parametru
        self.nx2 = tk.StringVar()
        self.nx3 = tk.StringVar()
        self.nx4 = tk.StringVar()
        self.nx5 = tk.StringVar()
        self.nx6 = tk.StringVar()
        self.nx7 = tk.StringVar()
        self.nx8 = tk.StringVar()
        self.nx9 = tk.StringVar()
        self.nx10 = tk.StringVar()
        self.vminw = tk.Label(self.parfr, textvariable=self.nx1)    #etykiety z wartościami parametrów
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
        self.savbtn = tk.Button(self.frame2, text="Zapisz parametry do pliku", state=tk.DISABLED, command=self.zappar)
        self.savbtn.pack()
        self.frame2.grid(row=1, column=2)
        self.filename = ''

    def su1min(self, event):    #funkcja obsługująca sterowaniem suwakiem minimum dla wykresu 2D
        if (self.suw1min.get() < self.suw1max.get()): #sprawdzenie, czy suwak minimum ma mniejszą wartość niż maksimum
            self.minplot = int(len(self.binmids) * self.suw1min.get() / 100)    #przeliczenie nowego minimalnego indeksu
            bintemp = self.binmids[self.minplot:self.maxplot]   #wycięcie odpowiedniej części środków binów (tymczasowo)
            if self.dziuratyp.get() == 1:   #jeśli funkcja wgłębienia to gauss
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111)
                self.ax.set_xlabel("$z [\mu m]$")
                self.ax.set_ylabel("Ilość wystąpień")
                self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot],    #rysowanie odpowiednich danych
                                        bintemp, self.gauss(bintemp, *self.popt),
                                        bintemp, self.gauss(bintemp, *self.dopt),
                                        bintemp, self.gauss(bintemp, *self.popt) + self.gauss(bintemp, *self.dopt))
                self.canvas.draw()
            if self.dziuratyp.get() == 2:   #jeśli funkcja wgłębienia to prostokąt
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111)
                self.ax.set_xlabel("$z [\mu m]$")
                self.ax.set_ylabel("Ilość wystąpień")
                self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot],    #rysowanie odpowiednich danych
                                        bintemp, self.gauss(bintemp, *self.popt),
                                        bintemp, self.prostokat(bintemp, *self.dkw),
                                        bintemp, self.gauss(bintemp, *self.popt) + self.prostokat(bintemp, *self.dkw))
                self.canvas.draw()
            if self.dziuratyp.get() == 0:   #jeśli brak funkcji wgłębienia
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111)
                self.ax.set_xlabel("$z [\mu m]$")
                self.ax.set_ylabel("Ilość wystąpień")
                self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot],    #rysowanie odpowiednich danych
                                        bintemp,
                                        self.gauss(bintemp, *self.popt))
                self.canvas.draw()

    def su1max(self, event):    #funkcja obsługująca sterowaniem suwakiem minimum dla wykresu 2D
        if (self.suw1max.get() > self.suw1min.get()):   #sprawdzenie, czy suwak maksimum ma większą wartość od minimum
            self.maxplot = int(len(self.binmids) * self.suw1max.get() / 100)    #przeliczenie nowego maksymalnego indeksu
            bintemp = self.binmids[self.minplot:self.maxplot]   #wycięcie odpowiedniej części środków binów (tymczasowo)
            if self.dziuratyp.get() == 1:   #jeśli funkcja wgłębienia to gauss
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111)
                self.ax.set_xlabel("$z [\mu m]$")
                self.ax.set_ylabel("Ilość wystąpień")
                self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot],    #rysowanie odpowiednich danych
                                        bintemp,
                                        self.gauss(bintemp, *self.popt),
                                        bintemp,
                                        self.gauss(bintemp, *self.dopt),
                                        bintemp,
                                        self.gauss(bintemp, *self.popt) + self.gauss(
                                            bintemp, *self.dopt))
                self.canvas.draw()
            if self.dziuratyp.get() == 2:   #jeśli funkcja wgłębienia to prostokąt
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111)
                self.ax.set_xlabel("$z [\mu m]$")
                self.ax.set_ylabel("Ilość wystąpień")
                self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot],    #rysowanie odpowiednich danych
                                        bintemp,
                                        self.gauss(bintemp, *self.popt),
                                        bintemp,
                                        self.prostokat(bintemp, *self.dkw),
                                        bintemp,
                                        self.gauss(bintemp,
                                                   *self.popt) + self.prostokat(bintemp,
                                                                                *self.dkw))
                self.canvas.draw()
            if self.dziuratyp.get() == 0:   #jeśli brak funkcji wgłębienia
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111)
                self.ax.set_xlabel("$z [\mu m]$")
                self.ax.set_ylabel("Ilość wystąpień")
                self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot],    #rysowanie odpowiednich danych
                                        bintemp,
                                        self.gauss(bintemp, *self.popt))
                self.canvas.draw()

    def zappar(self):   #funkcja zapisująca paramtery do pliku
        savfile = asksaveasfilename(filetypes=[("Plik tekstowy", "*.txt")]) #zapytanie o nazwę pliku
        if not savfile: #jeśli nie podano nazwy
            return  #nie zapisuje nic
        if not savfile.endswith('.txt'):    #jeśli nazwa nie kończy się na ".txt"
            savfile += '.txt'   #dodaj ".txt" do końca nazwy pliku
        with open(savfile, 'w', encoding='utf-8') as f: #zapisywanie danych w kodowaniu utf-8
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

    def wyksl(self, event): #funkcja obsługująca suwak stopnia wygładzenia
        if self.datyp.get() == 2:   #jeśli jest wybrane wyświetlanie danych wygładzonych
            if self.w2sl.get() == 1:    #dla stopnia wygładzenia 1
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")  #rysuj odpowiednie dane
                self.ax2.set_xlabel("$x [mm]$")
                self.ax2.set_ylabel("$y [mm]$")
                self.ax2.set_zlabel("$z [\mu m]$")
                self.plt2 = self.ax2.plot_surface(self.xws, self.yws, self.smth, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get() == 2:    #dla stopnia wygładzenia 2 itd.
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.ax2.set_xlabel("$x [mm]$")
                self.ax2.set_ylabel("$y [mm]$")
                self.ax2.set_zlabel("$z [\mu m]$")
                self.plt2 = self.ax2.plot_surface(self.xws2, self.yws2, self.smth2, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get() == 3:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.ax2.set_xlabel("$x [mm]$")
                self.ax2.set_ylabel("$y [mm]$")
                self.ax2.set_zlabel("$z [\mu m]$")
                self.plt2 = self.ax2.plot_surface(self.xws3, self.yws3, self.smth3, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get() == 4:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.ax2.set_xlabel("$x [mm]$")
                self.ax2.set_ylabel("$y [mm]$")
                self.ax2.set_zlabel("$z [\mu m]$")
                self.plt2 = self.ax2.plot_surface(self.xws4, self.yws4, self.smth4, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get() == 5:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.ax2.set_xlabel("$x [mm]$")
                self.ax2.set_ylabel("$y [mm]$")
                self.ax2.set_zlabel("$z [\mu m]$")
                self.plt2 = self.ax2.plot_surface(self.xws5, self.yws5, self.smth5, cmap=cm.coolwarm)
                self.canvas2.draw()

    def seldan(self):   #obsługa przycisku do wyboru normalnych/wygładzonych danych
        if self.datyp.get() == 1:   #jeśli wciśnieto przycisk normalne dane
            self.canvas2.figure.clf()
            self.plt2 = []
            self.canvas2.draw()
            self.ax2 = self.fig2.add_subplot(111, projection="3d")  #narysuj normalne dane
            self.ax2.set_xlabel("$x [mm]$")
            self.ax2.set_ylabel("$y [mm]$")
            self.ax2.set_zlabel("$z [\mu m]$")
            self.plt2 = self.ax2.plot_surface(self.xw, self.yw, self.zw, cmap=cm.coolwarm)
            self.canvas2.draw()
        if self.datyp.get() == 2:   #jeśli wciśnieto przycisk wygładzone dane
            if self.w2sl.get() == 1:    #sprawdź aktualną wartość suwaka ze stopniem wygłądzenia, jeśli 1:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")  #narysuj dane o stopniu wygładzenia 1
                self.ax2.set_xlabel("$x [mm]$")
                self.ax2.set_ylabel("$y [mm]$")
                self.ax2.set_zlabel("$z [\mu m]$")
                self.plt2 = self.ax2.plot_surface(self.xws, self.yws, self.smth, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get() == 2:    #i analogicznie dla każdego stopnia wygładzenia
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.ax2.set_xlabel("$x [mm]$")
                self.ax2.set_ylabel("$y [mm]$")
                self.ax2.set_zlabel("$z [\mu m]$")
                self.plt2 = self.ax2.plot_surface(self.xws2, self.yws2, self.smth2, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get() == 3:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.ax2.set_xlabel("$x [mm]$")
                self.ax2.set_ylabel("$y [mm]$")
                self.ax2.set_zlabel("$z [\mu m]$")
                self.plt2 = self.ax2.plot_surface(self.xws3, self.yws3, self.smth3, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get() == 4:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.ax2.set_xlabel("$x [mm]$")
                self.ax2.set_ylabel("$y [mm]$")
                self.ax2.set_zlabel("$z [\mu m]$")
                self.plt2 = self.ax2.plot_surface(self.xws4, self.yws4, self.smth4, cmap=cm.coolwarm)
                self.canvas2.draw()
            if self.w2sl.get() == 5:
                self.canvas2.figure.clf()
                self.plt2 = []
                self.canvas2.draw()
                self.ax2 = self.fig2.add_subplot(111, projection="3d")
                self.ax2.set_xlabel("$x [mm]$")
                self.ax2.set_ylabel("$y [mm]$")
                self.ax2.set_zlabel("$z [\mu m]$")
                self.plt2 = self.ax2.plot_surface(self.xws5, self.yws5, self.smth5, cmap=cm.coolwarm)
                self.canvas2.draw()

    def selgau(self):   #funkcja obsługująca wciśnięcie przycisku z wyborem typu funkcji
        bintemp = self.binmids[self.minplot:self.maxplot]   #tymczasowe biny, ograniczone przez suwaki rysowanego zakresu
        if self.dziuratyp.get() == 1:   #jeśli wybrano Gaussa
            self.tx1.set("Wartość max") #ustawianie odpowiednich etykiet do zmiennych do wpisania
            self.tx2.set("Położenie max")
            self.tx3.set("Sigma")
            self.f2max.config(state=tk.NORMAL)  #zamień tekst w polach tekstowych na wartości zmiennych funkcji Gaussa
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
            self.ax.set_xlabel("$z [\mu m]$")
            self.ax.set_ylabel("Ilość wystąpień")   #narysuj wykres z funkcją Gaussa
            self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot], bintemp,
                                    self.gauss(bintemp, *self.popt),
                                    bintemp, self.gauss(bintemp, *self.dopt), bintemp,
                                    self.gauss(bintemp, *self.popt) + self.gauss(bintemp, *self.dopt))
            self.canvas.draw()
        if self.dziuratyp.get() == 2:   #jeśli wybrano prostokąt
            self.tx1.set("Minimum") #ustawianie odpowiednich etykiet do zmiennych do wpisania
            self.tx2.set("Maksimum")
            self.tx3.set("Wartość")
            self.f2max.config(state=tk.NORMAL)  #zamień tekst w polach tekstowych na wartości zmiennych funkcji prostokątnej
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
            self.ax.set_xlabel("$z [\mu m]$")
            self.ax.set_ylabel("Ilość wystąpień")   #narysuj wykres z funkcją prostokątną
            self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot], bintemp, self.gauss(bintemp, *self.popt),
                                    bintemp, self.prostokat(bintemp, *self.dkw), bintemp,
                                    self.gauss(bintemp, *self.popt) + self.prostokat(bintemp, *self.dkw))
            self.canvas.draw()
        self.btn.config(state=tk.NORMAL)    #odblokuj przycisk do przejścia dalej

    def akt1(self, wn, val):    # funkcja obsługująca wpisywanie wartości parametrów funkcji odniesienia
        warnings.filterwarnings("ignore")   #nie pokazuj ostrzeżeń (przy wpisywaniu 0)
        bintemp = self.binmids[self.minplot:self.maxplot]   #tymczasowe biny, zakres ustalony przez suwaki
        self.gau1max.state = 'normal'   #nie blokuj możliwości wpisywania
        self.gau1pol.state = 'normal'
        self.gau1sig.state = 'normal'
        if wn == '.!frame.!frame3.!entry':  #sprawdź w którym polu została wpisana wartość
            self.popt[0] = val  #i zmień odpowiednią wartość w tablicy
        if wn == '.!frame.!frame3.!entry2':
            self.popt[1] = val
        if wn == '.!frame.!frame3.!entry3':
            self.popt[2] = val
        self.canvas.figure.clf()
        self.plt = []
        self.canvas.draw()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("$z [\mu m]$")
        self.ax.set_ylabel("Ilość wystąpień")   #sprawdź aktualny typ funkcji wgłębienia
        if self.dziuratyp.get() == 0:   #i narysuj wykres z odpowiednimi danymi
            self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot], bintemp,
                                    self.gauss(bintemp, *self.popt))
        if self.dziuratyp.get() == 1:
            self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot], bintemp,
                                    self.gauss(bintemp, *self.popt), bintemp, self.gauss(bintemp, *self.dopt), bintemp,
                                    self.gauss(bintemp, *self.popt) + self.gauss(bintemp, *self.dopt))
        if self.dziuratyp.get() == 2:
            self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot], bintemp,
                                    self.gauss(bintemp, *self.popt), bintemp, self.prostokat(bintemp, *self.dkw),
                                    bintemp, self.gauss(bintemp, *self.popt) + self.prostokat(bintemp, *self.dkw))
        self.canvas.draw()

    def akt2(self, wn, val):     # funkcja obsługująca wpisywanie wartości parametrów funkcji wgłębienia
        warnings.filterwarnings("ignore")   #nie pokazuj ostrzeżeń (przy wpisywaniu 0)
        bintemp = self.binmids[self.minplot:self.maxplot]   #tymczasowe biny, zakres ustalony przez suwaki
        self.f2max.state = 'normal' #nie blokuj możliwości wpisywania
        self.f2pol.state = 'normal'
        self.f2sig.state = 'normal'
        if self.dziuratyp.get() == 1:   #jeżeli typ wgłębienia to Gauss
            if wn == '.!frame.!frame5.!entry':  #sprawdź w którym polu została wpisana wartość
                self.dopt[0] = val  #i zmień odpowiednią wartość w odpowiedniej tablicy
            if wn == '.!frame.!frame5.!entry2':
                self.dopt[1] = val
            if wn == '.!frame.!frame5.!entry3':
                self.dopt[2] = val
        if self.dziuratyp.get() == 2:   #jeżeli typ wgłębienia to prostokąt
            if wn == '.!frame.!frame5.!entry':  #zrób analogicznie
                self.dkw[0] = val
            if wn == '.!frame.!frame5.!entry2':
                self.dkw[1] = val
            if wn == '.!frame.!frame5.!entry3':
                self.dkw[2] = val
        self.canvas.figure.clf()
        self.plt = []
        self.canvas.draw()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("$z [\mu m]$")
        self.ax.set_ylabel("Ilość wystąpień")
        if self.dziuratyp.get() == 0:   #narysuj wykres z nowymi wartościami, z odpowiednim typem funkcji
            self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot], bintemp,
                                    self.gauss(bintemp, *self.popt))
        if self.dziuratyp.get() == 1:
            self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot], bintemp,
                                    self.gauss(bintemp, *self.popt),
                                    bintemp, self.gauss(bintemp, *self.dopt), bintemp,
                                    self.gauss(bintemp, *self.popt) + self.gauss(bintemp, *self.dopt))
        if self.dziuratyp.get() == 2:
            self.plt = self.ax.plot(bintemp, self.ns[self.minplot:self.maxplot], bintemp, self.gauss(bintemp, *self.popt),
                                    bintemp, self.prostokat(bintemp, *self.dkw), bintemp,
                                    self.gauss(bintemp, *self.popt) + self.prostokat(bintemp, *self.dkw))
        self.canvas.draw()

    def validate(self, action, index, value_if_allowed, #funkcja sprawdzająca wpisywane dane do funkcji odniesienia
                 prior_value, text, validation_type, trigger_type, widget_name):
        if prior_value is not None: #jeśli to nie jest nowa wartość
            try:
                float(value_if_allowed) #sprawdź czy to jest liczba typu float
                self.akt1(widget_name, value_if_allowed)    #jeśli tak, wykonaj funkcję z przypisaniem wartości i rysowaniem wykresu
                return True
            except ValueError:  #jeśli nie, odrzuć zmianę
                return False
        else:
            return False

    def validate2(self, action, index, value_if_allowed,    #funkcja sprawdzająca wpisywane dane do funkcji wgłębienia
                  prior_value, text, validation_type, trigger_type, widget_name):
        if prior_value is not None: #jeśli to nie jest nowa wartość
            try:
                float(value_if_allowed) #sprawdź czy to jest liczba typu float
                self.akt2(widget_name, value_if_allowed)    #jeśli tak, wykonaj funkcję z przypisaniem wartości i rysowaniem wykresu
                return True
            except ValueError:  #jeśli nie, odrzuć zmianę
                return False
        else:
            return False

    def pluswcz(self):  #funkcja obsługująca przycisk "+"
        val = int(self.danalab["text"]) #pobierz aktualną wartość z etykiety
        self.danalab["text"] = f"{val + 1}" #i zapisz na niej wartość zwiększoną o 1

    def minuswcz(self): #funkcja obsługująca przycisk "-"
        val = int(self.danalab["text"]) #pobierz aktualną wartość z etykiety
        if val > 1: #jeśli jest większa od 1 (żeby końcowa wartość była zawsze dodatnia)
            self.danalab["text"] = f"{val + -1}"    #zapisz na niej wartość zmniejszoną o 1

    def dalej(self):   #funkcja obsługująca przycisk "przejdź dalej"
        hei, wid = self.zw.shape    # pobierz aktualne rozmiary tablic, by ograniczać nimi pętle
        vminus = 0  #inicjalizacja zmiennych przechowujących wartości objętości (wypukłości i wgłębienia)
        vplus = 0
        if self.dziuratyp.get() == 1:   #liczenie objętości wgłębienia jak funkcja to Gauss
            mingaus = argrelextrema(self.gauss(self.binmids, *self.dopt) + self.gauss(self.binmids, *self.popt), np.less)[0]    #znajdź minima lokalne sumy obu funkcji Gaussa
            if len(mingaus) > 0:    #jeśli takie istnieje
                mingaus = int(mingaus.item(0))  #to to jest nasz poziom odniesienia
            else:   #jeśli nie istnieje
                indomaxo = 0    #zmienne przechowujące tymczasowe wartości indeksu oraz wartości funkcji od tego indeksu
                valmaxo = 0
                for i in range(len(self.binmids)):  #szukaj indeksu, dla którego wartość jest największa (maksimum funkcji odniesienia)
                    if valmaxo < (self.gauss(self.binmids[i], *self.dopt) + self.gauss(self.binmids[i], *self.popt)):
                        valmaxo = (self.gauss(self.binmids[i], *self.dopt) + self.gauss(self.binmids[i], *self.popt))
                        indomaxo = i
                for i in range(indomaxo):   #poruszaj się od maksimum funkcji odniesienia w stronę mniejszej wysokości
                    if (self.gauss(self.binmids[indomaxo - i], *self.dopt) + self.gauss(self.binmids[indomaxo - i],
                                                                                        *self.popt)) < max(
                            self.gauss(self.binmids, *self.dopt) + self.gauss(self.binmids, *self.popt)) / 100: #szukaj, kiedy wartość będzie mniejsza niż 1% największej wartości
                        mingaus = indomaxo - i  #zapisz nowy indeks odpowiadający za poziom określający wgłębienie
            for i in range(hei - 1):
                for j in range(wid - 1):    #poruszaj się po każdym kwadracie złożonym z 4 najbliższych punktów
                    if ((self.zw[(i, j)] < self.binmids[mingaus]) & (self.zw[(i + 1, j)] < self.binmids[mingaus]) & (
                            self.zw[(i, j + 1)] < self.binmids[mingaus]) & (
                            self.zw[(i + 1, j + 1)] < self.binmids[mingaus])):  #sprawdź czy conajmniej 1 z punktów należy do wgłębienia
                        dw = max([self.zw[(i, j)], self.zw[(i + 1, j)], self.zw[(i, j + 1)], self.zw[(i + 1, j + 1)]])  #szukaj najniższego poziomu z tych punktów
                        gw = min([self.zw[(i, j)], self.zw[(i + 1, j)], self.zw[(i, j + 1)], self.zw[(i + 1, j + 1)]])  #szukaj najwyższego poziomu z tych punktów
                        vminus += (self.popt[1] - dw) * self.resx * self.resy * 0.001 + (
                                    1 / 3) * self.resx * self.resy * (gw - dw) * 0.001  #policz przybliżoną objetość między tymi 4 punktami
        if self.dziuratyp.get() == 2:   #liczenie objętości wgłębienia jak funkcja to prostokąt
            for i in range(hei - 1):
                for j in range(wid - 1):    #poruszaj się po każdym kwadracie złożonym z 4 najbliższych punktów
                    if ((self.zw[(i, j)] < self.dkw[1]) & (
                            self.zw[(i + 1, j)] < self.dkw[1]) & (
                            self.zw[(i, j + 1)] < self.dkw[1]) & (
                            self.zw[(i + 1, j + 1)] < self.dkw[1])): #sprawdź czy conajmniej 1 z punktów należy do wgłębienia, czyli poniżej maksa prostokąta
                        dw = max([self.zw[(i, j)], self.zw[(i + 1, j)], self.zw[(i, j + 1)], self.zw[(i + 1, j + 1)]])  #szukaj najniższego poziomu z tych punktów
                        gw = min([self.zw[(i, j)], self.zw[(i + 1, j)], self.zw[(i, j + 1)], self.zw[(i + 1, j + 1)]])  #szukaj najwyższego poziomu z tych punktów
                        vminus += (self.popt[1] - dw) * self.resx * self.resy * 0.001 + (
                                    1 / 3) * self.resx * self.resy * (gw - dw) * 0.001  #policz przybliżoną objetość między tymi 4 punktami
        self.nx1.set(str(vminus) + " mm\u00B3") #wpisz wartość objętości usuniętej części na etykietę
        for i in range(self.maxmaxind, len(self.binmids)):  #szukanie poziomu od którego można uznać wypiętrzenie
            if self.gauss(self.binmids[i], *self.popt) < (self.popt[0] / 100):  # dla wartości, dla któej gauss ma 1% wartości maksymalnej
                wypuk = self.binmids[i]
                pass
        for i in range(hei - 1):    #poruszając się po każdym prostokącie z sąsiednich punktów
            for j in range(wid - 1):
                if (self.zw[(i, j)] > wypuk or self.zw[(i + 1, j)] > wypuk or self.zw[(i, j + 1)] > wypuk or self.zw[
                    (i + 1, j + 1)] > wypuk):   # jeżeli conajmniej jeden z punktów jest wypukłością
                    dw = min([self.zw[(i, j)], self.zw[(i + 1, j)], self.zw[(i, j + 1)], self.zw[(i + 1, j + 1)]])  #znajdź poziom dolny
                    gw = max([self.zw[(i, j)], self.zw[(i + 1, j)], self.zw[(i, j + 1)], self.zw[(i + 1, j + 1)]])  #i górny
                    vplus += self.resx * self.resy * (abs(dw - self.popt[1]) + (1 / 3) * (gw - dw)) * 0.001 #i policz przybliżoną objetość
        self.nx2.set(str(vplus) + " mm\u00B3")  #wpisz wartość objętości wypukłości na etykietę
        maxdep = 0  #tymczasowe zmienne do maksymalnej głębokości i wysokości
        maxhei = 0
        for i in range(hei):    #poruszając się po wszystkich punktach
            for j in range(wid):
                self.zw[(i, j)] -= self.popt[1] #popraw wartość, by poziom odniesienia miał wartość 0
                if maxdep > self.zw[(i, j)]:
                    maxdep = self.zw[(i, j)]    #sprawdż, czy aktualny punkt jest najniżej
                if maxhei < self.zw[(i, j)]:
                    maxhei = self.zw[(i, j)]    #sprawdż, czy aktualny punkt jest najwyżej
        self.nx3.set(str(abs(maxdep)) + " \u03BCm") #wpisz wartość maksymalnej głębokości na etykietę
        self.nx4.set(str(maxhei) + " \u03BCm")  #wpisz wartość maksymalnej głębokości na etykietę
        self.nx5.set(str(maxhei - maxdep) + " \u03BCm") #wpisz wartość rozpiętości wysokości na etykietę
        if self.dziuratyp.get() == 1:
            self.nx6.set(str(self.popt[1] - self.dopt[1]) + " \u03BCm") #wypisz średnią głębokość wgłebienia, zależnie od typu funkcji
        if self.dziuratyp.get() == 2:
            self.nx6.set(str(float((self.dkw[1] - self.dkw[0]) / 2)) + " \u03BCm")
        quarts = np.percentile(self.zw, [25, 50, 75])   #policz kwartyle
        self.nx7.set(str((quarts[0] + quarts[2] - 2 * quarts[1]) / (quarts[2] - quarts[0])))    #wypisz współczynnik skośności
        n = wid * hei   #tymczasowa zmienna, przechowująca liczbę punktów
        zwsr = 0    #tymczasowa zmienna odpowiadająca za średnią wysokość całego pomiaru
        for i in range(hei):
            for j in range(wid):
                zwsr += self.zw[(i, j)] / n
        self.nx8.set(str(zwsr) + " \u03BCm")    #wpisz wartość średniej wysokości całego pomiaru na etykietę
        stdevsum = 0    #tymczasowa zmienna odpowiadająca za sumowanie elementów pod pierwiastkiem odchylenia standardowego
        for i in range(hei):
            for j in range(wid):
                stdevsum += ((self.zw[(i, j)] - zwsr) ** 2 / n)
        stdev = np.sqrt(stdevsum)
        self.nx9.set(str(stdev))    #wpisz wartość średniego odchylenia standardowego na etykietę
        kurtsum = 0 #tymczasowa zmienna do wykonania sumy stojącej we wzorze na kurtozę
        for i in range(hei):
            for j in range(wid):
                kurtsum += (((self.zw[(i, j)] - zwsr) / stdev) ** 4 * (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)))
        kurtoza = kurtsum - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))    #końcowa wartość kurtozy
        self.nx10.set(str(kurtoza)) #wpisz wartość kurtozy na etykietę
        tmp = wid   #tymczasowa zamiana oznaczeń wielkości na osiach
        wid = hei
        hei = tmp
        self.smth = np.zeros((wid - 1, hei - 1))    #tworzenie pustych macierzy pod wygładzone dane każdego stopnia
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
        for i in range(wid - 1):    #przeliczanie wygładzonych danych pierwszego stopnia przez uśrednianie
            for j in range(hei - 1):
                self.smth[(i, j)] = (self.zw[(i, j)] + self.zw[(i + 1, j)] + self.zw[(i, j + 1)] + self.zw[(i + 1, j + 1
                                                                                                            )]) / 4
                self.xws[(i, j)] = (self.xw[(i, j)] + self.xw[(i + 1, j)]) / 2
                self.yws[(i, j)] = (self.yw[(i, j)] + self.yw[(i, j + 1)]) / 2
        for i in range(wid - 2):    #to samo, drugiego stopnia itd.
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
        self.btn.config(state=tk.DISABLED)  #deaktywacja elementów z pierwszej części interfejsu
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
        self.w1bt.config(state=tk.NORMAL)   #aktywacja elementów z drugiej części interfejsu
        self.w2bt.config(state=tk.NORMAL)
        self.w2sl.config(state=tk.NORMAL)
        self.savbtn.config(state=tk.NORMAL)
        self.canvas2.figure.clf()
        self.plt2 = []
        self.canvas2.draw()
        self.ax2 = self.fig2.add_subplot(111, projection="3d")
        self.ax2.set_xlabel("$x [mm]$")
        self.ax2.set_ylabel("$y [mm]$")
        self.ax2.set_zlabel("$z [\mu m]$")  #rysowanie mapy 3D z normalnymi danymi
        self.plt2 = self.ax2.plot_surface(self.xw, self.yw, self.zw, cmap=cm.coolwarm)
        self.canvas2.draw()

    def openfile(self): #funkcja obsługująca przycisk "otwórz plik"
        self.filename = ''
        self.filename = askopenfilename()   #pobierz nazwę pliku do otwarcia
        if self.filename != '': #jeśli wybrano plik
            self.opnprg['value'] = 0    #ustaw wartość paska postępu na 0
            plik = self.filename
            dziel = int(self.danalab["text"])   #sprawdź co którą daną wczytywać
            f = open(plik, "r") #otwórz plik
            linie = f.readlines()   #wczytaj tekst
            del f
            gc.collect()    #usuń zmienną z otwartym plikiem, by oszczędzić miejsce w pamięci
            x = []  #twórz tablice z czytanymi wartościami
            y = []
            z = []
            doblin = 0  #licznik pustych linii
            cnt = 0 #licznik wczytanych danych w linii, do ograniczenia wczytywanych danych
            cw = int(0) #aktualny postęp wczytywania
            for i in range(len(linie)): #dla każdej linii w pliku
                if (int(i / len(linie) * 100) > cw):    #sprawdź czy zwiększyć pasek postępu
                    cw += 1
                    self.opnprg['value'] = cw / 3   #jak tak, to zwiększ wartość (o 1/3 %)
                    self.opnprg.update()
                #if ((linie[i] != '\n') & (dziel == 0)):
                    #nlin = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in linie[i])
                    #lon = [float(j) for j in nlin.split()]
                    #x.append(lon[0])
                    #y.append(lon[1])
                    #z.append(lon[2])
                if (dziel != 0):
                    if ((linie[i] != '\n') & (doblin == 0)):    #jeśli linia nie jest pusta, oraz jeśli licznik wczytanych linii to 0
                        if ((cnt % dziel) == 0):    #jeśli wczytujemy dane z odpowiednią częstotliwością
                            nlin = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in linie[i])   #znajdź liczby w linii tekstu
                            lon = [float(j) for j in nlin.split()]  #podziel linię z liczbami na 3 zmienne
                            x.append(lon[0])    #zapisz je do tablic
                            y.append(lon[1])
                            z.append(lon[2])
                        cnt += 1
                    if (linie[i] == '\n'):  #jeśli linia jest pusta
                        doblin += 1
                        if (2 * dziel == doblin):   #jeśli następna linia powinna być wczytana (*2, ponieważ są 2 puste linie między danymi)
                            doblin = 0  #to zresetuj liczniki
                            cnt = 0
            i = 0
            x0 = x[0]   #najniższa wartość x
            rev = 0 #zmienna przechowująca kierunek (najpierw x cz y)
            while (x[i] == x0): #dopóki x pozostaje ten sam
                i = i + 1
            if i >= 1:  #jeśli jest dużo tych samych x (są pierwsze)
                hei = i #wyznacz kształt
                wid = int(len(z) / hei)
                rev = 1 #zapisz kierunek pomiarów
            else:   #jeśli x się zmienia
                y0 = y[0]   #szukaj jak długo jest ten sam y
                while y[i] == y0:
                    i = i + 1
                wid = i #wyznacz kształt
                hei = int(len(z) / wid)
            self.xw = []    #zmienne docelowe na macierze 2D
            self.yw = []
            self.zw = []
            if rev == 1:    #przepisz wartości, w zależności od kierunku
                for i in range(hei):
                    self.yw.append(y[i])    #x oraz y to tablice z unikalnymi wartościami
                    self.zw.append([])  #z to tablica dwuwymiarowa, o wymiarach policzonych wcześniej
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
            del x, y, z #usuń stare zmienne
            gc.collect()
            self.xw, self.yw = np.meshgrid(self.xw, self.yw, indexing='xy') #zamień x i y na macierze 2D
            for i in range(hei):
                for j in range(wid):    #sprawdź dla każdego punktu
                    if self.zw[(i, j)] == 0:    #czy ma wartość 0 (błąd pomiaru)
                        sm = 0
                        ad = 0
                        if i < wid: #zsumuj wartości wszystkich punktów dookoła, zapisując ile ich jest
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
                        self.zw[(i, j)] = ad / sm   #uśrednij na podstawie wcześniej otrzymanych wartości
            ztx = []
            zty = []    #tymczasowe zmienne na tablice z wartościami przy krawędziach
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
                ztx.append(self.zw[(0, i)]) #zapisuj wartości x, y i z na początku i końcu osi x
                ytx.append(self.yw[(0, 0)])
                xtx.append(self.xw[(0, i)])
                ztx2.append(self.zw[(hei - 1, i)])
                ytx2.append(self.yw[(hei - 1, 0)])
                xtx2.append(self.xw[(hei - 1, i)])
            fx = np.polyfit(xtx, ztx, 1)    #dopasuj do nich proste
            fx2 = np.polyfit(xtx2, ztx2, 1)
            cw = 0
            sw = self.opnprg['value']   #wartości do śledzenia paska postępu
            for i in range(hei):
                for j in range(wid):    #dla każdego punktu
                    if (int((i * wid + j) / (hei * wid) * 100) > cw):   #jeżeli trzeba zwiększyć wartość paska postępu
                        cw += 1
                        self.opnprg['value'] = sw + (cw / 3)
                        self.opnprg.update()
                    ztmp = self.zw[(i,j)]   #zapis starej wartości z, do obliczeń x
                    self.zw[(i, j)] = self.zw[(i, j)] - (fx[0] + fx2[0]) / 2 * self.xw[(i, j)] #nowe wartości z
                    self.xw[(i,j)] = self.xw[(i,j)] + 1/np.tan((np.pi-np.arcsin((self.zw[(i,j)]-ztmp)/ztmp))/2) #nowe wartości x
            for i in range(hei):
                zty.append(self.zw[(i, 0)]) #zapisuj wartości x, y i z na początku i końcu osi y
                yty.append(self.yw[(i, 0)])
                xty.append(self.xw[(0, 0)])
                zty2.append(self.zw[(i, wid - 1)])
                yty2.append(self.yw[(i, wid - 1)])
                xty2.append(self.xw[(0, wid - 1)])
            fy = np.polyfit(yty, zty, 1)    #dopasuj do nich proste
            fy2 = np.polyfit(yty2, zty2, 1)
            cw = 0
            sw = self.opnprg['value']   #wartości do śledzenia paska postępu
            for i in range(hei):
                for j in range(wid):    #dla każdego punktu
                    if (int((i * wid + j) / (hei * wid) * 100) > cw):   #jeżeli trzeba zwiększyć wartość paska postępu
                        cw += 1
                        self.opnprg['value'] = sw + (cw / 3)
                        self.opnprg.update()
                    self.zw[(i, j)] = self.zw[(i, j)] - (fy[0] + fy2[0]) / 2 * self.yw[(i, j)]  #nowe wartości z
                    ztmp = self.zw[(i, j)]  #zapis starej wartości z, do obliczeń y
                    self.yw[(i, j)] = self.yw[(i, j)] + 1 / np.tan(
                        (np.pi - np.arcsin((self.zw[(i, j)] - ztmp) / ztmp)) / 2)   #nowe wartości y
            self.resx = self.xw[(0, 1)] - self.xw[(0, 0)]   #liczenie rozdzielczości x i y (potrzebna do objętości)
            self.resy = self.yw[(1, 0)] - self.yw[(0, 0)]
            maxdep = 0  #szukanie najmniejszej wartości, domyślnie to 0
            for i in range(hei):
                for j in range(wid):
                    if maxdep > self.zw[(i, j)]:
                        maxdep = self.zw[(i, j)]
            for i in range(hei):
                for j in range(wid):
                    self.zw[(i, j)] += abs(maxdep)  #dodaj wartość bezwzględną z najmniejszej wartości (czyli jak wszystkie są dodatnie - nic nie dodawaj)
            zh = []
            for i in range(hei):
                for j in range(wid):
                    zh.append(self.zw[(i, j)])  #przepisz wszystkie wartości z na jedną tablicę
            zhist = sns.histplot(data=zh)   #stwórz z niej histogram
            n, bins = [], []    #zmienne przechowujące końce binów i wartości
            for rect in zhist.patches:
                ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
                n.append(y1 - y0)
                bins.append(x0) #pobieranie ich wartości z histogramu
            bins.append(x1)
            binmid = []
            #indohival = 0
            #cuhival = 0
            #for i in range(len(n)):
                #if n[i] > cuhival:
                    #indohival = i
                    #cuhival = n[i]
            # indmin = indohival
            # indmax = indohival
            # for i in range(indohival):
                # if n[indohival - i] >= (max(n) / 100):
                    # indmin = indohival - i
            # for i in range(len(n) - indohival):
                # if n[indohival + i] >= (max(n) / 100):
                    # indmax = indohival + i

            for i in range(len(bins) - 1):  #znajdowanie środków binów
                binmid.append((bins[i] + bins[i + 1]) / 2)
            # self.binmids = []
            # self.ns = []
            # for i in range(indmax - indmin):
            #    self.binmids.append(binmid[i + indmin])
            #    self.ns.append(n[i + indmin])
            self.binmids = binmid   #zapisanie wartości w klasie
            self.ns = n
            self.minplot = 0    #wstępne ograniczenia rysowania wykresu (cały zakres)
            self.maxplot = len(self.binmids)
            try:    #wyliczanie automatyczne funkcji Gaussa poziomu odniesienia
                maxima = argrelextrema(np.array(self.ns), np.greater)[0]    #szukanie ekstremów
                minima = argrelextrema(np.array(self.ns), np.less)[0]
                nmax = []
                nmin = []
                maxx = []
                minx = []
                for m in maxima:
                    nmax.append(self.ns[m]) #wartości ilości wystąpień dla maksimów i minimów
                for m in minima:
                    nmin.append(self.ns[m])
                for i in range(len(maxima)):    #wartości środków binów dla maksimów i minimów
                    maxx.append(self.binmids[maxima[i]])
                for i in range(len(minima)):
                    minx.append(self.binmids[minima[i]])
                self.maxmaxind = maxima[0]
                for i in range(len(maxima)):    #szukanie największego maksimum
                    if self.ns[maxima[i]] > self.ns[self.maxmaxind]:
                        self.maxmaxind = maxima[i]
                jminind = 0
                dminind = minima[0] #szuaknie minimów sąsiadujących do wyznaczego największego maksimum
                for m in range(len(minima) - 1):
                    if minima[m] < self.maxmaxind:
                        jminind = minima[m]
                        dminind = minima[m + 1]
                if minima[len(minima) - 1] < self.maxmaxind:
                    jminind = minima[len(minima) - 1]
                    dminind = len(self.binmids) - 1
                xgauss = []
                ygauss = [] #przepisywanie do nowych tablic wartości binów i wystąpień, w celu dopasowania do nich krzywej
                for i in range(dminind - jminind - 2):
                    xgauss.append(self.binmids[i + 1 + jminind])
                    ygauss.append(self.ns[i + 1 + jminind])
                xgauss = np.array(xgauss)
                ygauss = np.array(ygauss)   #konwersja na tablice biblioteki numpy
                mean = sum(xgauss * ygauss) / sum(ygauss)   #liczenie średniej
                sigma = np.sqrt(sum(ygauss * (xgauss - mean) ** 2) / sum(ygauss))   #liczenie sigmy
                self.popt, pcov = curve_fit(self.gauss, xgauss, ygauss, p0=[max(ygauss), mean, sigma])  #dopasowanie parametrów z dość dokładnymi paramterami początkowymi
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111) #rysowanie wykresu
                self.plt = self.ax.plot(self.binmids, self.ns, self.binmids, self.gauss(self.binmids, *self.popt))
                self.canvas.draw()
                self.gau1max.delete(0, tk.END)  #wstawianie zmiennych w pola tekstowe
                self.gau1max.insert(0, self.popt[0])
                self.gau1pol.delete(0, tk.END)
                self.gau1pol.insert(0, self.popt[1])
                self.gau1sig.delete(0, tk.END)
                self.gau1sig.insert(0, self.popt[2])
                del binmid, bins, dminind, fx, fx2, fy, fy2, hei, i, j, jminind, linie, maxima, maxx, mean, minima, minx, n, nmax, nmin, pcov, plik, rect, sigma, sm, wid, x0, x1, xgauss, xtx, xtx2, xty, xty2, y0, y1, ygauss, ytx, ytx2, yty, yty2, zh, zhist, ztx, ztx2, zty, zty2
            # except ValueError or RuntimeError or TypeError or RuntimeWarning:
            except: #jeśli w trakcie liczenia wystąpił błąd
                self.popt = np.ones(3)  #ustaw parametry Gaussa jako 1
                self.canvas.figure.clf()
                self.plt = []
                self.canvas.draw()
                self.ax = self.fig.add_subplot(111) #narysuj wykres
                self.plt = self.ax.plot(self.binmids, self.ns, self.binmids, self.gauss(self.binmids, *self.popt))
                self.canvas.draw()
                self.gau1max.delete(0, tk.END)  #wpisz parametry do pól tekstowych
                self.gau1max.insert(0, self.popt[0])
                self.gau1pol.delete(0, tk.END)
                self.gau1pol.insert(0, self.popt[1])
                self.gau1sig.delete(0, tk.END)
                self.gau1sig.insert(0, self.popt[2])
            self.dopt = np.ones(3)  #inicjalizacja tablic dla funkcji wgłębienia
            self.dkw = np.ones(3)
            gc.collect()
            self.danamin.config(state=tk.DISABLED)  #przełączanie stanu elemntów interfejsu
            self.danalab.config(state=tk.DISABLED)
            self.danaplus.config(state=tk.DISABLED)
            self.opn.config(state=tk.DISABLED)
            self.gbt.config(state=tk.NORMAL)
            self.pbt.config(state=tk.NORMAL)
            self.suw1min.config(state=tk.NORMAL)
            self.suw1max.config(state=tk.NORMAL)
            self.suw1max.set(100)   #ustawienie wartości paska postępu na 100
            self.suw1max.update()


root = tk.Tk()
root.title("Profilometr")   #uruchomienie aplikacji
cls = Windowgauss(root)
root.iconify()
root.update()   #tak żeby zawsze była na wierzchu
root.deiconify()
root.mainloop()
