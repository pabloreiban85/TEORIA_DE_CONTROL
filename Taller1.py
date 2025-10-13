#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import control as co
from scipy.interpolate import interp1d

# In[ ]:





# In[ ]:


datos = pd.read_csv("data_motor.csv", index_col=0, sep=",")


# In[ ]:


print(datos.head())


# In[ ]:


tiempo = datos["time(t)"]
print(tiempo)
# Copiar la señal de salida
respuesta = datos["system_response(y)"].copy()

# 🔧 Corregir la línea base (eliminar parte negativa)
respuesta[respuesta < 0] = 0

# 🔧 Normalizar: restar valor inicial si no parte en 0 exactamente
respuesta = respuesta - respuesta.iloc[0]


# In[ ]:


respuesta = datos["system_response(y)"]
print(respuesta[4:12])


# In[ ]:


print(datos["ex_signal(u)"].size)
line_100 = np.ones(100)
line_0 = np.zeros(100)
print(line_0)
print(line_100)


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(tiempo, respuesta, label='Respuesta del Proceso')
plt.plot(tiempo, line_0, label='Linea Base')
plt.plot(tiempo, line_100, label='Linea 100%')
plt.plot(tiempo, datos["ex_signal(u)"], label='Señal Escalón')
plt.legend()
plt.xlabel('Tiempo', fontsize='14')
plt.title('Identificación Gráfica')
plt.ylabel('Amplitud', fontsize='14')
plt.show()


# In[ ]:


# Suponiendo que df es tu DataFrame y 'columna' es la columna que quieres modificar
indices_a_reemplazar = [5, 6]  # Lista de índices de las filas a reemplazar

# Reemplazar los valores en los índices especificados por cero en la columna 'columna'
datos.iloc[indices_a_reemplazar, datos.columns.get_loc('system_response(y)')] = 0


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(tiempo, respuesta, label='Respuesta del Proceso')
plt.plot(tiempo, line_0, label='Linea Base')
plt.plot(tiempo, line_100, label='Linea 100%')
plt.plot(tiempo, datos["ex_signal(u)"], label='Señal Escalón')
plt.legend()
plt.xlabel('Tiempo', fontsize='14')
plt.title('Identificación Gráfica')
plt.ylabel('Amplitud', fontsize='14')
plt.show()


# In[ ]:


respuesta = datos["system_response(y)"]
print(respuesta[4:12])


# In[ ]:





# In[ ]:


# Supongamos que tienes tus arrays de numpy para y(t) y t
# y_t es tu señal y(t)
# t es tu array de tiempos t
# Asegúrate de que ambos arrays tengan la misma longitud
y_t = np.array(respuesta)
t = np.array(tiempo)
# Calcula la segunda derivada de y(t) utilizando np.gradient dos veces
dy = np.gradient(y_t, t)
d2y = np.gradient(dy, t)

# Encuentra los puntos de inflexión donde d2y cambia de signo
puntos_inflexion = np.where(np.diff(np.sign(d2y)))[0]

# Los valores de t en los puntos de inflexión
valores_t_inflexion = t[puntos_inflexion]

# Los valores de y en los puntos de inflexión
valores_y_inflexion = y_t[puntos_inflexion]

print(valores_y_inflexion)
print(valores_t_inflexion)


# In[ ]:


plt.figure(figsize=(10,5))
plt.scatter(valores_t_inflexion, valores_y_inflexion, label='Puntos de Inflexión')
plt.legend()
plt.xlabel('Tiempo', fontsize='14')
plt.title('Puntos de Inlexión')
plt.ylabel('Amplitud', fontsize='14')
plt.show()


# In[ ]:


#print(valores_y_inflexion)
print(valores_t_inflexion[2])

tiempo_especifico = valores_t_inflexion[2]

# Busca el dato de la señal en el tiempo específico utilizando el método .iloc[]
dato_en_tiempo_especifico = datos.loc[datos['time(t)'] == tiempo_especifico, 'system_response(y)'].iloc[0]

print(dato_en_tiempo_especifico)


# In[ ]:


print(dy)
print(t)


# In[ ]:


indice = datos['time(t)'].index[datos['time(t)'] == tiempo_especifico].tolist()
print(indice)


# In[ ]:


print(dy[13])


# In[ ]:


recta = []
for i in tiempo:
    if i <= tiempo[28]:
        recta.append(dato_en_tiempo_especifico + dy[13]*i - dy[13]*tiempo_especifico)
    else:
        recta.append(0)

print(recta)

posiciones = [i for i, x in enumerate(recta) if 0.95 < x < 1.1]
print(posiciones)
print(recta[27:30])


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(tiempo, respuesta, label='Respuesta del Proceso')
plt.plot(tiempo, line_0, label='Linea Base')
plt.plot(tiempo, line_100, label='Linea 100%')
plt.plot(tiempo, datos["ex_signal(u)"], label='Señal Escalón')
plt.plot(tiempo[0:29], recta[0:29], '--', label='Tangente')
plt.legend()
plt.xlabel('Tiempo', fontsize='14')
plt.title('Identificación Gráfica')
plt.ylabel('Amplitud', fontsize='14')
plt.show()


# In[ ]:


print(tiempo[28])


# In[ ]:


posiciones2 = [i for i, x in enumerate(recta) if 0 < x < 0.1]
print(posiciones2)


# In[ ]:


print(recta[5:8])


# In[ ]:


print(tiempo[6])


# In[ ]:


print(datos.head(10))


# In[ ]:


t_inicio = tiempo[5]
print(t_inicio)


# In[ ]:


theta = tiempo[6] #- tiempo[5]
#theta = 0.35
print(theta)


# In[ ]:


tau = tiempo[28] - theta
#tau = 0.963
print(tau)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


k=1
numerador = [k]
denominador = [tau,1]
G = co.tf(numerador, denominador)
print(G)


# In[ ]:


np = [-theta/2,1]
dp = [theta/2,1]
G_exp = co.tf(np, dp)

G_fodt = G*G_exp
print(G_fodt)


# In[ ]:


tiempo2,y1 = co.step_response(G_fodt)
y1[y1 < 0] = 0

tiempo_zn = [i for i in tiempo2 if i <=5]
Z_N = [y1[i] for i in range(0, len(tiempo_zn))]

plt.figure()
plt.plot(tiempo2,y1,label='y(t)')
#plt.plot([t1[0],t1[-1]], [numerador, denominador],'--','k')
plt.legend()
plt.xlabel('Tiempo')
plt.show()

print(tiempo_zn)
print(Z_N)


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(tiempo, respuesta, label='Respuesta del Proceso')





plt.plot(tiempo, line_0, label='Linea Base')
plt.plot(tiempo, line_100, label='Linea 100%')
plt.plot(tiempo, datos["ex_signal(u)"], label='Señal Escalón')
plt.plot(tiempo[0:29], recta[0:29], '--', label='Tangente')
plt.plot(tiempo_zn, Z_N, label='M. Ziegler & Nichols', color='pink')
plt.legend()
plt.xlabel('Tiempo', fontsize='14')
plt.title('Método de Ziegler & Nichols')
plt.ylabel('Amplitud', fontsize='14')
plt.show()


# In[ ]:


valor_máximo = max(respuesta)
porcentaje_63 = 0.6321*valor_máximo

print(valor_máximo)
print('63%=', porcentaje_63)


# In[ ]:


buscar = list(respuesta)

for i, x in enumerate(buscar):
    if x >= 0.6 and x <= 0.7:
        print(i, x)

print('63%=', porcentaje_63)


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(tiempo, respuesta, label='Respuesta del Proceso')
plt.plot(tiempo, line_0, label='Linea Base')
plt.plot(tiempo, line_100, label='Linea 100%')
plt.plot(tiempo, datos["ex_signal(u)"], label='Señal Escalón')
plt.plot(tiempo[0:29], recta[0:29], '--', label='Tangente')
plt.scatter(tiempo[20], respuesta[20], label='Punto 63%')
plt.legend()
plt.xlabel('Tiempo', fontsize='14')
plt.title('Identificación Gráfica')
plt.ylabel('Amplitud', fontsize='14')
plt.show()


# In[ ]:


print(tiempo[20])


# In[ ]:


tau_miller = tiempo[20]-theta
print(tau_miller)


# In[ ]:


#Nuevo Tau
tau_miller = tiempo[20]-theta

#Planteamiento de la funcion de transferencia
numerador = [k]
denominador = [tau_miller,1]
G_M = co.tf(numerador, denominador)
np = [-theta/2,1]
dp = [theta/2,1]
G_exp_M = co.tf(np, dp)

G_fodt_M = G_M*G_exp_M

print(G_fodt_M)


# In[ ]:


tiempo3,y2 = co.step_response(G_fodt_M)
y2[y2 < 0] = 0
tiempo_M = [i for i in tiempo3 if i <=5]
Miller = [y2[i] for i in range(0, len(tiempo_M))]

plt.figure()
plt.plot(tiempo3,y2,label='y(t)')
plt.legend()
plt.xlabel('Tiempo')
plt.show()

print(tiempo_M)
print(Miller)


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(tiempo, respuesta, label='Respuesta del Proceso')
plt.plot(tiempo, line_0, label='Linea Base')
plt.plot(tiempo, line_100, label='Linea 100%')
plt.plot(tiempo, datos["ex_signal(u)"], label='Señal Escalón')
plt.plot(tiempo[0:29], recta[0:29], '--', label='Tangente')
plt.plot(tiempo_M, Miller, label='M. Miller', color='black')
plt.legend()
plt.xlabel('Tiempo', fontsize='14')
plt.title('Método de Miller')
plt.ylabel('Amplitud', fontsize='14')
plt.show()


# In[ ]:


porcentaje_28 = 0.284*valor_máximo

print(valor_máximo)
print(porcentaje_28)


# In[ ]:


buscar = list(respuesta)

for i, x in enumerate(buscar):
    if x >= 0.2 and x <= 0.35:
        print(i, x)

print('28% = ', porcentaje_28)


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(tiempo, respuesta, label='Respuesta del Proceso')
plt.plot(tiempo, line_0, label='Linea Base')
plt.plot(tiempo, line_100, label='Linea 100%')
plt.plot(tiempo, datos["ex_signal(u)"], label='Señal Escalón')
plt.plot(tiempo[0:29], recta[0:29], '--', label='Tangente')
plt.scatter(tiempo[20], respuesta[20], label='Punto 63%')
plt.scatter(tiempo[12], respuesta[12], label='Punto 28%')
plt.legend()
plt.xlabel('Tiempo', fontsize='14')
plt.title('Identificación Gráfica')
plt.ylabel('Amplitud', fontsize='14')
plt.show()


# In[ ]:


print('Theta + Tau =', tiempo[20])
print('Theta + Tau/3 =', tiempo[12])


# In[ ]:


tau_analitico = (3*(tiempo[20]-tiempo[12]))/2
theta_analitico = tiempo[12]- (tiempo[20]-tiempo[12])/2

print('Tau =',tau_analitico)
print('Theta =',theta_analitico)


# In[ ]:


k=1

numerador = [k]
denominador = [tau_analitico,1]
G_A = co.tf(numerador, denominador)
np = [-theta_analitico/2,1]
dp = [theta_analitico/2,1]
G_exp_A = co.tf(np, dp)

G_fodt_A = G_A*G_exp_A

print(G_fodt_A)


# In[ ]:


tiempo4, y3 = co.step_response(G_fodt_A)
y3[y3 < 0] = 0
tiempo_A = [i for i in tiempo4 if i <=5]
Analitico = [y3[i] for i in range(0, len(tiempo_A))]

plt.figure()
plt.plot(tiempo4, y3, label='y(t)')
plt.legend()
plt.xlabel('Tiempo')
plt.show()

print(tiempo_A)
print(Analitico)


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(tiempo, respuesta, label='Respuesta del Proceso')
plt.plot(tiempo, line_0, label='Linea Base')
plt.plot(tiempo, line_100, label='Linea 100%')
plt.plot(tiempo, datos["ex_signal(u)"], label='Señal Escalón')
plt.plot(tiempo[0:29], recta[0:29], '--', label='Tangente')
#plt.plot(tiempo_zn, Z_N, label='y(t)')
#plt.plot(tiempo_M, Miller, label='y(t)')
plt.plot(tiempo_A, Analitico, label='M. Analítico', color='purple')
plt.legend()
plt.xlabel('Tiempo', fontsize='14')
plt.title('Método Analítico')
plt.ylabel('Amplitud', fontsize='14')
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(tiempo, respuesta, label='Respuesta del Proceso')
plt.plot(tiempo, line_0, label='Linea Base')
plt.plot(tiempo, line_100, label='Linea 100%')
plt.plot(tiempo, datos["ex_signal(u)"], label='Señal Escalón')
plt.plot(tiempo[0:29], recta[0:29], '--', label='Tangente')
plt.plot(tiempo_zn, Z_N, label='M. Ziegler & Nichols', color='pink')
plt.plot(tiempo_M, Miller, label='M. Miller', color='black')
plt.plot(tiempo_A, Analitico, label='M. Analítico', color='purple')
plt.legend()
plt.xlabel('Tiempo', fontsize='14')
plt.title('Funciones de Transferencia FODT')
plt.ylabel('Amplitud', fontsize='14')
plt.show()


# In[ ]:



# Interpolacion de las curvas al mismo eje temporal de la respuesta real
interp_ZN = interp1d(tiempo_zn, Z_N, bounds_error=False, fill_value="extrapolate")
interp_Miller = interp1d(tiempo_M, Miller, bounds_error=False, fill_value="extrapolate")
interp_Analitico = interp1d(tiempo_A, Analitico, bounds_error=False, fill_value="extrapolate")

# Obtener las señales interpoladas
Z_N_interp = interp_ZN(tiempo)
Miller_interp = interp_Miller(tiempo)
Analitico_interp = interp_Analitico(tiempo)

# Calculo manual del error cuadratico medio
mse_ZN = sum((respuesta - Z_N_interp)**2) / len(respuesta)
mse_Miller = sum((respuesta - Miller_interp)**2) / len(respuesta)
mse_Analitico = sum((respuesta - Analitico_interp)**2) / len(respuesta)

print("Error cuadratico medio:")
print("Ziegler-Nichols:", mse_ZN)
print("Miller:", mse_Miller)
print("Analitico:", mse_Analitico)

# Ver cual metodo tiene menor error
if mse_ZN < mse_Miller and mse_ZN < mse_Analitico:
    print("El metodo con menor error es Ziegler-Nichols")
elif mse_Miller < mse_Analitico:
    print("El metodo con menor error es Miller")
else:
    print("El metodo con menor error es Analitico")