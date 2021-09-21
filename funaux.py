print("importando modulos de ploteo, audio e interaccion")

#@title Importar Bilioteca de Ploteo
import numpy as np
import matplotlib.pyplot as plt


#@title Importar Biblioteca de Audio
from IPython.display import Audio
from scipy.io import wavfile
from io import BytesIO

#@title Importar Biblioteca Interact
from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

#@title Definir funciones auxiliares

print("definiendo funciones auxiliares")

def normalizar(x):
  '''
    reescala los valores de entrada de manera que los valores
    de salida esten entre 0 y 1
  '''
  xn=x-min(x)
  xn=xn/max(xn)
  return xn


def normalizarb(x):
  '''
     reescala los valores de entrada de manera que los
     valores de salida esten entre -1 y 1
  '''
  xn=2*normalizar(x)-1
  return xn


def graft(x,fs,xlim):
    '''
       grafica en el dominio del tiempo
       primer argumento, arreglo de valores de la funcion en el tiempo
       tercer argumento, periodo a muestrear [t_inicial, t_final]
       segundo argumento, frecuencia de muestreo
       tercer argumento, intervalo de tiempo a graficar [tmin, tmax]
    '''
    tmax = len(x)/fs
    t = np.arange(0, tmax, step=1. / fs)
    plt.plot(t, x)
    plt.xlim(xlim)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud")
    return

def graff1l(x,fs,xlim):
    '''
       grafica en el dominio de la frecuencia, espectro de un lado
       primer argumento, arreglo de valores de la funcion en el tiempo
       segundo argumento, frecuencia de muestreo
       tercer argumento, intervalo de frecuencias a graficar [finf, fsup]
       entrega componentes en frecuencia (nros. complejos)
    '''
    N = len(x)
    X=np.fft.fft(x)/N
    freqs = np.arange(0, fs / 2, step=fs / N)
    plt.plot(freqs[:(N // 2)], 2*np.abs(X[:(N // 2)]))
    plt.xlabel("Frequencia (Hz)")
    plt.ylabel("Amplitud")
    plt.xlim(xlim)
    return X

def graff2l(x,fs,xlim):
    '''
       grafica en el dominio de la frecuencia, espectro de dos lados
       primer argumento, arreglo de valores de la funcion en el tiempo
       segundo argumento, frecuencia de muestreo
       tercer argumento, intervalo de frecuencias a graficar [finf, fsup]
       entrega componentes en frecuencia (nros. complejos)
    '''
    N = len(x)
    X = np.fft.fft(x)/N
    # Plot the positive frequencies.
    freqsp = np.arange(0, fs / 2, step=fs / N)
    plt.plot(freqsp, np.abs(X[:(N // 2)]))
    # Plot the negative frequencies.
    freqsn = np.arange(-fs / 2, 0, step=fs / N)
    plt.plot(freqsn, np.abs(X[(N // 2):]))
    # Now we can label the x-axis.
    plt.xlabel("Frequencia (Hz)")
    plt.ylabel("Amplitud")
    plt.xlim(xlim)
    return X

def graff1lp(x,fs,xlim):
    '''
       grafica en el dominio de la frecuencia, espectro de un lado de potencias
       primer argumento, arreglo de valores de la funcion en el tiempo
       segundo argumento, frecuencia de muestreo
       tercer argumento, intervalo de frecuencias a graficar [finf, fsup]
       entrega componentes en frecuencia (nros. complejos)
    '''
    N = len(x)
    X=np.fft.fft(x)/N
    X_p = np.abs(X) ** 2
    freqs = np.arange(0, fs / 2, step=fs / N)
    plt.plot(freqs[:(N // 2)], (X_p[:(N // 2)]))
    plt.xlim(xlim)
    plt.xlabel("Frequencia (Hz)")
    plt.ylabel("Potencia")
    plt.xlim(xlim)
    return X
  
def graff2lp(x,fs,xlim):
    '''
       grafica en el dominio de la frecuencia, espectro de dos lados de potencias
       primer argumento, arreglo de valores de la funcion en el tiempo
       segundo argumento, frecuencia de muestreo}
       tercer argumento, intervalo de frecuencias a graficar [finf, fsup]
    '''
    N = len(x)
    X = np.fft.fft(x)/N
    X_p = np.abs(X)**2
    # Plot the positive frequencies.
    freqsp = np.arange(0, fs / 2, step=fs / N)
    plt.plot(freqsp, X_p[:(N // 2)])
    # Plot the negative frequencies.
    freqsn = np.arange(-fs / 2, 0, step=fs / N)
    plt.plot(freqsn, X_p[(N // 2):])
    #  etiquetado de ejes
    plt.xlabel("Frequencia (Hz)")
    plt.ylabel("Amplitud ")
    plt.xlim(xlim)
    return X

def espectro(x,fs,xlim):
    '''
       grafica en el dominio de la frecuencia, espectro de dos lados de potencias en escala logaritmica
       primer argumento, arreglo de valores de la funcion en el tiempo
       segundo argumento, frecuencia de muestreo
       tercer argumento, limites de frecuencias a graficar [finf, fsup]
       entrega componentes en frecuencia (nros. complejos)
    '''
    N = len(x)
    X = np.fft.fft(x)/N
    X_pl = np.log(np.abs(X)**2)
    # Plot the positive frequencies.
    freqsp = np.arange(0, fs / 2, step=fs / N)
    plt.plot(freqsp, X_pl[:(N // 2)])
    # Plot the negative frequencies.
    freqsn = np.arange(-fs / 2, 0, step=fs / N)
    plt.plot(freqsn, X_pl[(N // 2):])
    # Now we can label the x-axis.
    plt.xlabel("Frequencia (Hz)")
    plt.ylabel("Amplitud (log)")
    plt.xlim(xlim)
    return X

def pasabanda(X,fs,finf,fsup):
  '''
  Filtro pasa banda ideal de las componentes en frecuencia X
  con frecuencia de muestreo fs, y frecuencias inferior y superior
  de la banda de paso finf y fsup.
  '''

  #inversion de muestras de frecuencias
  X2=np.fft.fftshift(X)
  
  #calculo de indices componentes de frecuencias
  nc=len(X)
  ifni =int ( -fsup*nc/(fs) + nc//2)
  ifns= int (-finf*nc/(fs) + nc//2)
  ifpi=int ( finf*nc/(fs) + nc//2)
  ifps =int ( fsup*nc/(fs) + nc//2)
  #print(ifni,ifns,ifpi,ifps)

  #eliminacion de componentes fuera de la banda de paso
  X2[:ifni]=0
  X2[ifns:ifpi]=0
  X2 [ifps:]=0

  #inversion de muestras de frecuencias
  X3=np.fft.ifftshift(X2)
  return X3

print("listo!")
