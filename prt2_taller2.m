num = 3; % Numerador
den = [1 2 3]; % Denominador

Gs = tf(num,den); % Funcion de Transferencia

delay = 2; % Tiempo muerto
Gsdt = tf(num,den,'InputDelay',delay);

N = 999; % Cantidad de datos (divisible entre 3)
t = linspace(0,30,N); % Vector de tiempo total de 0 a 30 s

% ---- Definición de tramos ----
p1 = zeros(1, N/3);       % 0–10 s  -> constante 0
p2 = ones(1, N/3) * 5;    % 10–20 s -> constante 5
p3 = ones(1, N/3) * 10;   % 20–30 s -> constante 10

% ---- Señal completa ----
mysignal2 = [p1 p2 p3]; % Señal por tramos

% ---- Simulación ----
lsim(Gsdt, mysignal2, t)