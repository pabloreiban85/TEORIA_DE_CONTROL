 num=[3];
 den=[1 2 3];
 ts=0.1;
 Gs=tf(num,den); %tiempo continuo
 Gz=tf(num,den,ts); %tiempo discreto
 delay=2  % Tiempo muerto
 Gs=tf(num,den)  % Funcion de Transferencia
 Gsdt = tf(num,den,'InputDelay',delay)

 % ---- Simulación step----
 [y,t] = step(Gs);
 figure();
 plot(t,y)
 % ---- Simulación impulse----
 [w,q] = impulse(Gs);
 figure();
 plot(q,w)
