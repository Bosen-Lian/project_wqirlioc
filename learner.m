function xdot=learner(t,x)
global A;
global B;
global u

x1=x(1);
x2=x(2);

xdot=[A*[x1;x2]+B*u];    

end