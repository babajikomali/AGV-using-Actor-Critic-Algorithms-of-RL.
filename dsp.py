#essential libraries
import os
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
from turtle import color
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

import sys
import numpy as np
from pip import main
from scipy.fft import fftfreq
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

#suppresses warnings
import warnings
warnings.filterwarnings("ignore")

#reference dominant frequency in Hzs
ref_fre_jazz = 239
ref_fre_classical = 990
ref_fre_rock = 111

#custom FFT function
def fft_v(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError("must be a power of 2")    
    N_min = min(N, 2)
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))
    while X.shape[0] < N:
        X_even = X[:, :int(X.shape[1] / 2)]
        X_odd = X[:, int(X.shape[1] / 2):]
        terms = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + terms * X_odd,
                       X_even - terms * X_odd])
    return X.ravel()

#function for classifying the input audio signal
def classify(dominantfre, results_window):
    error_jazz = abs(np.array(ref_fre_jazz)-np.array(dominantfre))
    error_classical = abs(np.array(ref_fre_classical)-np.array(dominantfre))
    error_rock = abs(np.array(ref_fre_rock)-np.array(dominantfre))
    max_jazz = np.max(error_jazz)
    max_classical = np.max(error_classical)
    max_rock = np.max(error_rock)
    if max_jazz<=max_classical and max_jazz<=max_rock:
        print('The given audio signal genre is Jazz')
        result = Label(results_window, text='The given audio signal genre is Jazz',
                font=('Georgia 30'), background='black', foreground='white')
        result.pack()
    elif max_classical<=max_jazz and max_classical<=max_rock:
        print('The given audio signal genre is Classical')
        result = Label(results_window, text='The given audio signal genre is Classical',
                font=('Georgia 30'), background='black', foreground='white')
        result.pack()
    elif max_rock<=max_classical and max_rock <= max_jazz:
        print('The given audio signal genre is Rock')
        result = Label(results_window, text='The given audio signal genre is Rock',
                       font=('Georgia 30'), background='black', foreground='white')
        result.pack()

def analyse(filename):
    samplerate, data = read(filename)
    print('Sample Rate: ', samplerate)  # 16000 Hz

    duration = len(data)/samplerate
    time = np.arange(0, duration, 1/samplerate)  # time vector
    
    N = 262144
    customfft = fft_v(data[0:N])
    numpyfft = np.fft.fft(data[0:N])
    frequencies = fftfreq(N, 1 / samplerate)

    #Finding dominant peaks
    num_dominant_peaks = 10
    zipped_lists = zip(customfft, frequencies)
    sorted_zipped_lists = sorted(zipped_lists)
    sorted_list1 = [element for _, element in sorted_zipped_lists]
    sorted_list2 = [element for element, _ in sorted_zipped_lists]
    sorted_list1.reverse()
    sorted_list2.reverse()
    dominantfre = [
        fre for fre in sorted_list1[0:num_dominant_peaks] if fre >= 0]
    indices = [ind for ind in range(
        num_dominant_peaks) if sorted_list1[ind] >= 0]
    dominantampl = [sorted_list2[i]
                    for i in range(num_dominant_peaks) if i in indices]
    
    #window for graphs
    graphs_window = Tk()
    graphs_window.geometry("1000x800")
    graphs_window.title('FFT Based Music Genre Classification')
    graphs_window.configure(bg='black')
    header = Label(graphs_window, text='FFT BASED MUSIC GENRE CLASSIFICATION',
                   font=('Georgia 20'), background='black', foreground='white')
    header.pack()
    description = Label(graphs_window, text='Supported Music Genres : Jazz, Classical, Rock.',
                        font=('Georgia 10'), background='black', foreground='white')
    description.pack
    rate = Label(graphs_window, text='Samplerate: 16000 Hz.',
                 font=('Georgia 10'), background='black', foreground='white')
    rate.pack()
    problem = Label(graphs_window, text='The classification of music genre is being done by finding the dominant peaks in frequency spectrum.',
                    font=('Georgia 10'), background='black', foreground='white')
    problem.pack()
    #Verification of custom FF
    fig = Figure(figsize=(7, 6))
    a = fig.add_subplot(211)
    a.plot(frequencies, np.abs(customfft), color='green')
    a.set_title('Custom FFT', fontsize=12)
    a.set_ylabel("Amplitude", fontsize=10)
    b = fig.add_subplot(212)
    b.plot(frequencies, np.abs(numpyfft), color='red')
    b.set_title('Numpy FFT', fontsize=12)
    b.set_ylabel("Amplitude", fontsize=10)
    b.set_xlabel("Frequency(Hz)", fontsize=10)

    canvas = FigureCanvasTkAgg(fig, master=graphs_window)
    canvas.get_tk_widget().pack()
    canvas._tkcanvas.pack(side=tk.LEFT)
    canvas.draw()

    fig1 = Figure(figsize=(7, 6))
    c = fig1.add_subplot(211)
    c.plot(time, data, color='magenta')
    c.set_title('Audio Signal', fontsize=12)
    c.set_ylabel("Amplitude", fontsize=10)
    c.set_xlabel("Time(sec)", fontsize=10)
    d = fig1.add_subplot(212)
    d.stem(dominantfre, dominantampl)
    d.set_title('First 5 Dominant Peaks', fontsize=12)
    d.set_ylabel("Amplitude", fontsize=10)
    d.set_xlabel("Frequency(Hz)", fontsize=10)

    canvas1 = FigureCanvasTkAgg(fig1, master=graphs_window)
    canvas1.get_tk_widget().pack()
    canvas1._tkcanvas.pack(side=tk.RIGHT)
    canvas1.draw()


    print('Dominant Frequencies: ', dominantfre)
    print('Dominant Amplitudes: ', dominantampl)

    #window for results 
    results_window = Tk()
    results_window.geometry("700x350")
    results_window.title('FFT Based Music Genre Classification')
    results_window.configure(bg='black')
    header = Label(results_window, text='FFT BASED MUSIC GENRE CLASSIFICATION',
                font=('Georgia 20'), background='black', foreground='white')
    header.pack()
    description = Label(results_window, text='Supported Music Genres : Jazz, Classical, Rock.',
                        font=('Georgia 10'), background='black', foreground='white')
    description.pack
    rate = Label(results_window, text='Samplerate: 16000 Hz.',
                font=('Georgia 10'), background='black', foreground='white')
    rate.pack()
    problem = Label(results_window, text='The classification of music genre is being done by finding the dominant peaks in frequency spectrum.',
                    font=('Georgia 10'), background='black', foreground='white')
    problem.pack()

    dominant = Label(results_window, text='First 5 Dominant Peaks are located at: '+str(dominantfre[0]) +'    ' +str(dominantfre[1]) +'    '+str(dominantfre[2]) +'    '+str(dominantfre[3]) +'    '+str(dominantfre[4]),
                    font=('Georgia 15'), background='black', foreground='white')
    dominant.pack()

    classify(dominantfre[0], results_window)

    nameo = Label(results_window, text='Sundarapalli Harikrishna-121901045',
                    font=('Georgia 15'), background='black', foreground='white')
    nameo.pack()
    namet = Label(results_window, text='Komali Babaji Pattabhiram Gowd-121901022',
                    font=('Georgia 15'), background='black', foreground='white')
    namet.pack()

    results_window.mainloop()

def start():
    #setting up the window 
    main_window = Tk()
    main_window.geometry("1200x1000")
    main_window.title('FFT Based Music Genre Classification')
    main_window.configure(bg='black')
    header = Label(main_window, text='FFT BASED MUSIC GENRE CLASSIFICATION', 
                    font=('Georgia 20'),background='black', foreground='white')

    header.pack()
    description = Label(main_window, text='Supported Music Genres : Jazz, Classical, Rock.',
                    font=('Georgia 10'),background='black', foreground='white')
    description.pack
    rate = Label(main_window, text='Samplerate: 16000 Hz.',
                        font=('Georgia 10'), background='black', foreground='white')
    rate.pack()
    problem = Label(main_window, text='The classification of music genre is being done by finding the dominant peaks in frequency spectrum.',
                    font=('Georgia 10'),background='black', foreground='white')
    problem.pack()
    #getting filename through file explorer
    def browseFiles():
        filename = filedialog.askopenfilename(initialdir="/",title="Select a File",
                                            filetypes=[("All files", "*")])
        analyse(filename)

    #button for browsing files
    browse= Label(main_window, text='Here upload the audio file in .wav format from your system.',
                    font=('Georgia 10'),background='black', foreground='white')
    browse.pack()
    button_explore = Button(main_window,
                            text='Browse Files',
                            command=browseFiles,
                            font=('Georgia 10'),
                            background='black', foreground= 'white')
    button_explore.pack()
    main_window.mainloop()

if __name__ == '__main__':
    start()