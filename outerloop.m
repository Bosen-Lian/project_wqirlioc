function xdot=outerloop(t,x)
global A;
global B;
global ud1;
global R;
global K2;


x1=x(1);
x2=x(2);

xdot=[A*[x1;x2]+B*ud1                                                 
      -2*ud1'*R*K2*[x1;x2]-[x1 x2]*K2'*R*K2*[x1;x2] ;                
      [x1*x1 2*x1*x2 x2*x2]'];                                       

              
end
