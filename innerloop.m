function xdot=innerloop(t,x)
global A;
global B;
global ud;
global Q;
global R;
global K;


x1=x(1);
x2=x(2);

xdot=[A*[x1;x2]+B*ud
      [x1;x2]'*Q*[x1;x2]+(K*[x1;x2])'*R*K*[x1;x2]                   
      kron([x1 x2],((ud+K*[x1;x2])'*R))'];                        

              
end



