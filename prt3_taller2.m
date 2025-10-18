num = 3; % Numerador
den = [1 2 3]; % Denominador

Gs = tf(num,den); % Funcion de Transferencia

delay = 2; % Tiempo muerto
Gsdt = tf(num,den,'InputDelay',delay);

N = 1000; % Cantidad de datos
t = linspace(0,40,N); % Vector de tiempo total de 0 a 40 s

% ---- Definición de tramos ----
p1 = zeros(1,0.25*N);        % 0–10 s  -> constante 0
p2 = ones(1,0.25*N)*5;       % 10–20 s -> constante 5
r  = linspace(15,25,0.25*N); % 20–30 s -> rampa de 15 a 25
p3 = ones(1,0.25*N)*25;      % 30–40 s -> constante 25

% ---- Señal completa ----
mysignal2 = [p1 p2 r p3]; % Señal por tramos
% ---- Simulación ----
lsim(Gsdt,mysignal2,t)