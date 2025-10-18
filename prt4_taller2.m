num = 3; % Numerador
den = [1 2 3]; % Denominador
Gs = tf(num,den); % Funcion de Transferencia

N = 1000; % Cantidad de puntos
t = linspace(0,40,N); % Vector de tiempo

% ==== Definición de tramos ====
p1 = zeros(1,0.125*N);             % 0–5 s -> constante 0
r1 = linspace(0,10,0.125*N);       % 5–10 s -> rampa ascendente
p2 = ones(1,0.125*N)*20;           % 10–15 s -> escalón de subida
r2 = linspace(20,5,0.125*N);       % 15–20 s -> rampa descendente
p3 = zeros(1,0.125*N);             % 20–25 s -> escalón de bajada
r3 = linspace(0,15,0.125*N);       % 25–30 s -> rampa ascendente
r4 = linspace(15,5,0.25*N);        % 30–40 s -> rampa descendente final

% ==== Señal total ====
mysignal = [p1 r1 p2 r2 p3 r3 r4];

% ==== Simulación ====
figure;
lsim(Gs, mysignal, t)
title('Respuesta del sistema G(s) ante señal con múltiples rampas y escalones')
xlabel('Tiempo (s)')
ylabel('Salida del sistema')
grid on
