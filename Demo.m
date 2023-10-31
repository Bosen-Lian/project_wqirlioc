close all;
clc;
clear all;
warning off;

% Inverse RL simulation for trajectory tracking control--Demo version
% Algorithm 2 with diagonal Q

global A;
global B;
global u;
global ud;
global ud1;
global K2;
global Q;
global R;
global K;
global udca;

step=1000;
learningstep=300;

%linear system
A=[-1 2;2.2 1.7];
B=[2;1.6];

%target expert system settings
R=1;
Qd=[8,0;0,8];
Wrs=[8 0 0 8];
Pstar=care(A,B,Qd,R);
Kstar=R\B'*Pstar;

%generate expert system data
xdc=[20;-20];
xd=[20;-20];
T=0.0008;
udcc=[];
for i=1:step 
    %probing noise
    edc(i)=0.0000001*(0.5*sin(2.0*i)^2*cos(10.1*i)+0.9*sin(1.102*i)^2*cos(4.001*i)...
              +0.3*sin(1.99*i)^2*cos(7*i)+0.3*sin(10.0*i)^3+0.7*sin(3.0*i)^2*cos(4.0*i)...
              +0.3*sin(3.00*i)*1*cos(1.2*i)^2+0.400*sin(1.12*i)^2+0.5*cos(2.4*i)*sin(8*i)^2+0.3*sin(1.000*i)^1*cos(0.799999*i)^2+0.3*sin(4*i)^3);
    udca=-Kstar*xdc(:,i)+edc(i);    
    udcc=[udcc,udca];
    tspanxd=[0 T];%integral time
    [td,dd]= ode45(@expert,tspanxd,xdc(:,i));
    xdc(:,i+1)=[dd(length(td),1); dd(length(td),2)]; 
end

%Initial learning settings
R=1;
Wr=[1 0 0 1]';%initial vectorized Q
P=[0.3854,0.3771;0.3771,1.08424];
K=[1.3740    2.4891];
x=[20;-20];%initial states

alpha=0.1;%step size of GD
bbb=0.00000001;%threthold of error E
ii=0;
j=0;
wstep=500;

uu=[];
udc=[];
Wrr=[];
eta=[];
dWr=[];
dK=1;
Kr=[];
dKs=[];
dPs=[];
dWrs=[];

%learning
for i=1:learningstep
        
    Wrr=[Wrr,Wr];
    Kr=[Kr;K];
    Pr(:,:,i)=P;   
    Klast=K;
    
    % off-policy RL (inner loop) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Q=[Wr(1,1) Wr(2,1)
       Wr(3,1) Wr(4,1)]; 
    Ps=care(A,B,Q,R);
    Ks=R\B'*Ps;
    
    %collect expert data
    ed(i)=0.011*(0.5*sin(2.0*i)^2*cos(10.1*i)+0.9*sin(1.102*i)^2*cos(4.001*i)...%probing noise
              +0.3*sin(1.99*i)^2*cos(7*i)+0.3*sin(10.0*i)^3+0.7*sin(3.0*i)^2*cos(4.0*i)...
              +0.3*sin(3.00*i)*1*cos(1.2*i)^2+0.400*sin(1.12*i)^2+0.5*cos(2.4*i)*sin(8*i)^2+0.3*sin(1.000*i)^1*cos(0.799999*i)^2+0.3*sin(4*i)^3+0.4*cos(2*i)*1*sin(5*i)^4+0.3*sin(10.00*i)^3);
    ud=-Kstar*xd(:,i)+ed(i); %can be any other stablizing control gains
    udc=[udc,ud];
    Xd(:,i)=[xd(:,i);0;0;0];
    tspanx9=[0 T];
    [t9,d9]= ode45(@innerloop,tspanx9,Xd(:,i));
    xd(:,i+1)=[d9(length(t9),1); d9(length(t9),2)]; 
    
    %Update P and K   
    ii=ii+1;
    dxx(ii,:)=[xd(1,i+1)^2 2*xd(1,i+1)*xd(2,i+1) xd(2,i+1)^2]-[xd(1,i)^2 2*xd(1,i)*xd(2,i) xd(2,i)^2];
    ixux(ii,:)=-2*[d9(length(t9),4) d9(length(t9),5)];
    ixu(ii)=-d9(length(t9),3);
    fai(ii,:)=[dxx(ii,:),ixux(ii,:)];
    rank(fai'*fai);
    
    if(rank(fai'*fai)==5&&dK(end)>0.01)
         
        wpk=(fai'*fai)\fai'*ixu';
        P=[wpk(1) wpk(2); wpk(2) wpk(3)]; %Calculate P 
        K = [wpk(4) wpk(5)]; %Calculate K
        ii=0;
        dxx=[];
        ixux=[];
        ixu=[];
        fai=[];
       
        dK=[dK,norm(K-Klast)];
    end


 
  % Update f(P) and Q (outer loop) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if(dK(end)<=0.01)
      
        dKs=[dKs,norm(K-Kstar)];
        j=j+1;        
        e(j,:)=-K-[udcc(:,i-1),udcc(:,i)]/[xdc(:,i-1),xdc(:,i)];
        E(j)=e(j,:)*e(j,:)'/2;
        if (E(j)>bbb)
            Pn(:,:,j)=P-alpha*0.5*((-K/P)'*e(j,:)+((-K/P)'*e(j,:))');%Calculate f(P)
            K2=K/P*Pn(:,:,j);
            
            %Update Q via model, only for verifying learning result
            hWr1=-(A'*Pn(:,:,j)+Pn(:,:,j)*A-Pn(:,:,j)*B*inv(R)*B'*Pn(:,:,j));
            Wr1=[hWr1(1,1) 0 0 hWr1(2,2)]';
            dWr=[dWr,norm(Wr-Wr1)];
            
            %Update Q
            for jj=1:wstep
                ud1=udcc(i+1-jj);
                Xd1(:,jj)=[xdc(:,i+1-jj);0;0;0;0];
                tspanx10=[0 T];
                [t10,d10]= ode45(@outerloop,tspanx10,Xd1(:,jj));
                xd1=[d10(length(t10),1); d10(length(t10),2)]; 
                dxx1(jj)=([xd1(1)^2 2*xd1(1)*xd1(2) xd1(2)^2]-[xdc(1,i+1-jj)^2 2*xdc(1,i+1-jj)*xdc(2,i+1-jj) xdc(2,i+1-jj)^2])*[Pn(1,1,j);Pn(1,2,j);Pn(2,2,j)];
                ixu1(jj)=d10(length(t10),3);
                ixx1(jj,:)=-[d10(length(t10),4) d10(length(t10),5) d10(length(t10),6)];
                eta(jj)=dxx1(jj)+ixu1(jj);
                                
                if(rank(ixx1'*ixx1)==3&&dKs(end)>0.0043)
                    hWr=(ixx1'*ixx1)\ixx1'*eta';
                    Wr=[hWr(1) 0 0 hWr(3)]';%calculate diagonal Q
                                 
                    dxx1=[];
                    ixu1=[];
                    ixx1=[];
                    eta=[];
                    
                    dK(end)=1;%can be any value larger than the stop threthold 0.01 for inner loops
                    
                    dPs=[dPs,norm(P-Pstar)];
                    dWrs=[dWrs,norm(Wr-Wrs')];
                    
                    break;

                end  
                
                if (dKs(end)<=0.0043)
                    dKs=[dKs,norm(K-Kstar)];
                    dPs=[dPs,norm(P-Pstar)];
                    dWrs=[dWrs,norm(Wr-Wrs')];
 
                    break;
                end
            end
        end
  end
end


%Test tracking performance under the learned result
for i=1:step
    u=-K*x(:,i);
    uu=[uu,u];
    tspanx2=[0 T];
    [t2,d2]= ode45(@learner,tspanx2,x(:,i));
    x(:,i+1)=[d2(length(t2),1); d2(length(t2),2)]; 
end



figure(1)
t=1:step;
plot(T*t,x(1,t),'LineWidth',1);
hold on;
plot(T*t,x(2,t),'LineWidth',1);
hold on;
plot(T*t,xdc(1,t),'--','LineWidth',1);
hold on;
plot(T*t,xdc(2,t),'--','LineWidth',1);
legend('$xd_1$','$xd_2$','$x_1$','$x_2$');
set (groot, 'defaultAxesTickLabelInterpreter','latex'); 
xlabel('Time');

figure(2)
plot(T*t,uu,'LineWidth',1);
hold on;
plot(T*t,udcc,'--','LineWidth',1);
legend('$u$','$u_d$');
set (groot, 'defaultAxesTickLabelInterpreter','latex'); 
xlabel('Time');

figure(3)
plot(dKs(1:40),'-*','LineWidth',1);
hold on;
plot(dWrs(1:40),'-*','LineWidth',1);
hold on;
plot(dPs(1:40),'-*','LineWidth',1);
legend('$\|K_{i+1}-K_d\|$','$\|Q_{i+1}-Q_d\|$','$\|P_{i+1}-P_d\|$')
set (groot, 'defaultAxesTickLabelInterpreter','latex'); 
xlabel('Iteration steps');


