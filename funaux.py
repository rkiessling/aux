# version 0.1 14/9/2022


from __future__ import print_function
print("importando modulos de ploteo, audio e interaccion")
   
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

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
    if len(xlim) >= 2:
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
    if len(xlim) >= 2:
      plt.xlim(xlim) 
    return X

def graff1lf(x,fs,xlim):
    '''
       grafica en el dominio de la frecuencia, espectro de fases de un lado
       primer argumento, arreglo de valores de la funcion en el tiempo
       segundo argumento, frecuencia de muestreo
       tercer argumento, intervalo de frecuencias a graficar [finf, fsup]
       entrega componentes en frecuencia (nros. complejos)
    '''
    N = len(x)
    X=np.fft.fft(x)/N
    freqs = np.arange(0, fs / 2, step=fs / N)
    plt.plot(freqs[:(N // 2)], np.angle(X[:(N // 2)],deg=True),'b*')
    plt.xlabel("Frequencia (Hz)")
    plt.ylabel("Amplitud")
    plt.ylim([-180,180]) 
    if len(xlim) >= 2:
        plt.xlim(xlim) 
    return X

def graff2lf(x,fs,xlim):
    '''
       grafica en el dominio de la frecuencia, espectro de fases de un lado
       primer argumento, arreglo de valores de la funcion en el tiempo
       segundo argumento, frecuencia de muestreo
       tercer argumento, intervalo de frecuencias a graficar [finf, fsup]
       entrega componentes en frecuencia (nros. complejos)
    '''
    N = len(x)
    X=np.fft.fft(x)/N
    # Plot the positive frequencies.
    freqsp = np.arange(0, fs / 2, step=fs / N)
    plt.plot(freqsp, np.angle(X[:(N // 2)],deg=True),'b*')
    # Plot the negative frequencies.
    freqsn = np.arange(-fs / 2, 0, step=fs / N)
    plt.plot(freqsn, np.angle(X[(N // 2):],deg=True),'b*')
    plt.xlabel("Frequencia (Hz)")
    plt.ylabel("Amplitud")
    plt.ylim([-180,180]) 
    if len(xlim) >= 2:
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
    if len(xlim) >= 2:
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
    if len(xlim) >= 2:
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
    if len(xlim) >= 2:
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
    # plotear frecuencias positivas
    freqsp = np.arange(0, fs / 2, step=fs / N)
    plt.plot(freqsp, X_pl[:(N // 2)])
    # plotear frecuencias negativas
    freqsn = np.arange(-fs / 2, 0, step=fs / N)
    plt.plot(freqsn, X_pl[(N // 2):])
    # etiquetar eje x
    plt.xlabel("Frequencia (Hz)")
    plt.ylabel("Amplitud (log)")
    if len(xlim) >= 2:
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
  #eliminacion de componentes fuera de la banda de paso
  X2[:ifni]=0
  X2[ifns:ifpi]=0
  X2 [ifps:]=0
  #inversion de muestras de frecuencias
  X3=np.fft.ifftshift(X2)
  return X3

def xtriang(fs,periodo, tmax=1, tipo='t', polaridad='b'):
  '''
    señal triangular o diente de  sierra:
    primer argumento, frecuencia de muestreo
    arreglo de valores de la funcion en el tiempo
    segundo argumento, periodo del diente de sierra
    tercer argumento, tiempo de simulacion
    cuarto argumento, tipo: 't'riangular /\/\/\, 'c'reciente  /|/|/|, 'd'ecreciente  |\|\|\
    quinto argumento, polaridad: 'u'nipolar (0,1), 'b'ipolar (-1,1)
    entrega x, t: valores de la funcion, valores de tiempo 
  '''
  t = np.arange(0, tmax, step=1. / fs)
  if tipo=='c':
    x = np.mod(t,periodo)
  elif tipo=='d':
    x = periodo-np.mod(t,periodo)
  elif tipo=='t':
    x = 1-np.abs(periodo/2-np.mod(t,periodo))
  if polaridad=='u':
    x=normalizar(x)
  else:
    x=normalizarb(x)
  return x,t

def xpulsos(fs, periodo, ciclo, tmax=1, polaridad='b'):
  '''
    señal pulsos
    primer argumento, frecuencia de muestreo
    segundo argumento, periodo de la señal
    tercer argumento, ciclo de trabajo (entre 0 y 1)
    cuarto argumento, tiempo de simulacion
    quinto argumento, polaridad: 'u'nipolar (0,1), 'b'ipolar (-1,1)
    entrega x, t: valores de la funcion, valores de tiempo 
  '''
  t = np.arange(0, tmax, step=1. / fs)
  xx=t-t
  x = normalizar(np.mod(t,periodo))
  xx[x<ciclo]=1
  if polaridad=='u':
    xx=normalizar(xx)
  else:
    xx=normalizarb(xx)
  return xx,t

def potencia(x):
  '''
    potencia de un vector de muestras,
    suma de los cuadrados de los elementos del vector x, dividido la longitud del vector
    valido para entradas reales y complejas, en dominio del tiempo y de la frecuencia
  '''
  p=np.sum(x*x.conjugate())/len(x)
  return p

def energia(x):
  '''
    energia de un vector de muestras,
    suma de los cuadrados de los elementos del vector x
    valido para entradas reales y complejas, en dominio del tiempo y de la frecuencia
  '''
  e=np.sum(x*x.conjugate())
  return e
  
  
print("listo!")
