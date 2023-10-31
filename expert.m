function xdot=expert(t,x)
global A;
global B;
global udca
x1=x(1);
x2=x(2);

xdot=[A*[x1;x2]+B*udca];    

end