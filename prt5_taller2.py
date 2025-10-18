#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import control as ct

# === Función de transferencia ===
num = [3]
den = [1, 2, 3]
Gs = ct.tf(num, den)

# === Parámetros de simulación ===
N = 1000  # cantidad de puntos
t = np.linspace(0, 40, N)  # vector de tiempo

# === Definición de tramos (idénticos al MATLAB) ===
p1 = np.zeros(int(0.125 * N))                   # 0–5 s -> constante 0
r1 = np.linspace(0, 10, int(0.125 * N))         # 5–10 s -> rampa ascendente
p2 = np.ones(int(0.125 * N)) * 20               # 10–15 s -> escalón 20
r2 = np.linspace(20, 5, int(0.125 * N))         # 15–20 s -> rampa descendente
p3 = np.zeros(int(0.125 * N))                   # 20–25 s -> bajada a 0
r3 = np.linspace(0, 15, int(0.125 * N))         # 25–30 s -> rampa ascendente
r4 = np.linspace(15, 5, int(0.25 * N))          # 30–40 s -> rampa descendente final

# === Señal total ===
mysignal = np.concatenate((p1, r1, p2, r2, p3, r3, r4))

# === Verificación ===
print(f"Longitud de señal: {len(mysignal)}, longitud de tiempo: {len(t)}")

# === Gráfico de la señal ===
plt.figure(figsize=(10, 4))
plt.plot(t, mysignal, 'b', label='Señal de entrada')
plt.title('Señal de entrada con múltiples rampas y escalones')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.show()

# === Simulación del sistema ===
t_out, y_out = ct.forced_response(Gs, T=t, U=mysignal)
t_out, y_out = ct.forced_response(Gs, T=t, U=mysignal)

# === Gráfico de la respuesta ===
plt.figure(figsize=(10, 5))
plt.plot(t, mysignal, 'b--', label='Entrada')
plt.plot(t_out, y_out, 'r', label='Salida del sistema')
plt.title('Respuesta del sistema G(s) ante señal con rampas y escalones')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.show()
