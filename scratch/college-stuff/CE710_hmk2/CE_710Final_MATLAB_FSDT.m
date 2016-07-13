% ce 710 hmk2
clear all
clc
close all
%% Variables
% layer         1 (top)  ... nl (to bottom) 
theta = fliplr([0 45 -45 90 0].* pi/180);
thk = zeros(1,length(theta)) + 0.0025;
nl = length(thk);
a =   20;  % plate width;
b =   10;  % plate height
q0_ = -55.7; % plate load;
% Transversly isotropic material properties
Ell = 150e9;
Ett = 12.1e9;
vlt = 0.248;
Glt = 4.4e9;
vtt = 0.458;
Gtt = Ett / (2*(1+vtt));
K_ = 5/6;
% Failure Strengths
SLLt =  1500e6;
SLLc = -1250e6;
STTt =  50e6;
STTc = -200e6;
SLTs =  100e6;
Sxzs =  100e6;
Strength = [SLLt SLLc;
            STTt STTc;
            SLTs Sxzs];
%% Stiffness Matrix
syms th
% tranformation
Tij6 = [cos(th)^2 sin(th)^2 0 0 0 -sin(2*th);
        sin(th)^2 cos(th)^2 0 0 0  sin(2*th);
        0 0 1 0 0 0;
        0 0 0 cos(th) sin(th) 0;
        0 0 0 -sin(th) cos(th) 0;
        cos(th)*sin(th) -cos(th)*sin(th) 0 0 0 (cos(th)^2-sin(th)^2)];

Tij = [cos(th)^2 sin(th)^2 2*sin(th)*cos(th);
        sin(th)^2 cos(th)^2 -2*sin(th)*cos(th);
        -cos(th)*sin(th) sin(th)*cos(th) (cos(th)^2-sin(th)^2)];

% compliance matrix
Sij6 = [1/Ell -vlt/Ell -vlt/Ell 0 0 0;
       -vlt/Ell 1/Ett -vtt/Ett 0 0 0;
       -vlt/Ell -vtt/Ett 1/Ett 0 0 0;
       0 0 0 1/Gtt 0 0;
       0 0 0 0 1/Glt 0;
       0 0 0 0 0 1/Glt];
   
% Stiffnes matrix in material coordinates
Cijm6 = inv(Sij6);

% Stiffness matrix in Structural coordinates
Cij6 = Tij6*Cijm6*Tij6.';

% reduced stiffness in structural
Cij = [Cij6(1,1) Cij6(1,2) 0; Cij6(1,2) Cij6(2,2) 0; 0 0 Cij6(6,6)];
hlam = sum(thk);

% Create z dimensions of laminate
z_(1) = -hlam/2;
for i = 1:nl
   z_(i+1) =  z_(1) + sum(thk(1:i));
end
% extensional stiffness
Aij = zeros(6,6);
for i = 1:nl
    Aij = Aij + subs(Cij6,th,theta(i)) * (z_(i+1)-z_(i));
end
% coupling  stiffness
Bij = zeros(6,6);
for i = 1:nl
    Bij = Bij + 0.5* subs(Cij6,th,theta(i)) * (z_(i+1)^2-z_(i)^2);
end
% bending or flexural laminate stiffness relating moments to curvatures
Dij = zeros(6,6);
for i = 1:nl
    Dij = Dij + (1/3)* subs(Cij6,th,theta(i)) * (z_(i+1)^3-z_(i)^3);
end

%% First order shear deformation theory

% displacement in w (z direction)
syms x y z q0 C1 C2 C3 C4 C5 C6  K A11 B11 D11 A16 B16 A55 
pfun = (-A11 *q0* x^3 + 6* B11^2 *C2 - 6 *A11 *D11* C2 - 3* A11* A55* K *x^2 *C2 +6 *B11^2 *x* C5 - 6* A11* D11* x *C5 - 3* A11* A55* K *x^2 *C6) / (6 *(B11^2 - A11* D11))
ufun = (B11 *q0 *x^3 + 6* B11^2 *C1 - 6 *A11 *D11 *C1 + 3 *A55* B11* K *x^2 *C2 + 6* B11^2 *x *C4 - 6 *A11 *D11 *x *C4 + 3* A55* B11* K* x^2* C6) / (6 *(B11^2 - A11* D11))
wfun = -12*B11^2*q0*x^2 + 12*A11*D11*q0*x^2 - A11*A55*K*q0*x^4 - 4*A11*A55^2*K^2*x^3*C2 - 24*A55*B11^2*K*C3 + 24*A11*A55*D11*K*C3 + 12*A55*B11^2*K*x^2 *C5 - 12 *A11 *A55 *D11 *K *x^2 *C5 - 24*A55*B11^2*K*x*C6 + 24*A11*A55*D11*K*x*C6 - 4*A11*A55^2*K^2*x^3*C6 / (24*A55*(A11*D11-B11^2 )*K)

C1sol = solve(subs(ufun,x,0),C1)
C2sol = solve(subs(pfun,x,0),C2)
C3sol = solve(subs(wfun,x,0),C3) 

cond3 = subs(wfun,x,a)
cond5 = B11*(diff(ufun)+0.5*diff(wfun)^2) - D11*diff(pfun)
cond6 = A11*(diff(ufun)+0.5*diff(wfun)^2) - B11*diff(pfun)
Sol = solve(cond3,cond6,cond5,C4,C5,C6)
C4sol = subs(Sol.C4,[C2 C3],[C2sol C3sol])
C5sol = subs(Sol.C5,[C2 C3],[C2sol C3sol])
C6sol = subs(Sol.C6,[C2 C3],[C2sol C3sol])

% substitute integration constants with actual values( _ is actual number) 
C1_ = C1sol;
C2_ = C2sol;
C3_ = C3sol
C4_ = subs(C4sol,[q0 A11 B11 D11 K A16 B16 A55 ],[q0_ Aij(1,1) Bij(1,1) Dij(1,1) K_ Aij(1,6) Bij(1,6) Aij(5,5)  ]);
C5_ = subs(C5sol,[q0 A11 B11 D11 K A16 B16 A55 ],[q0_ Aij(1,1) Bij(1,1) Dij(1,1) K_ Aij(1,6) Bij(1,6) Aij(5,5)   ]);
C6_ = subs(C6sol,[q0 A11 B11 D11 K A16 B16 A55 ],[q0_ Aij(1,1) Bij(1,1) Dij(1,1) K_ Aij(1,6) Bij(1,6) Aij(5,5)  ]);

% function w(x) vertical displacement w along z with actual vaules
wsol = subs(wfun,[q0  C1  C2  C3  C4  C5  C6  K A11      B11      D11       A55],...
                 [q0_ C1_ C2_ C3_ C4_ C5_ C6_ K_ Aij(1,1) Bij(1,1) Dij(1,1) Aij(5,5)])
% function u(x) horizontal displacement u along x with actual vaules
usol = subs(ufun,[q0  C1  C2  C4  C5  C6   K  A11      B11      D11       A55   ],...
                 [q0_ C1_ C2_ C4_ C5_ C6_  K_ Aij(1,1) Bij(1,1) Dij(1,1)  Aij(5,5)])
ezsurf(x,y,wsol,[0,a,0,b])  
view(-45,30)
xlabel('x')
ylabel('y')
zlabel('z')
title('Cylindrical Bending -Displacement of a plate With CLPT')
wsol_opt = matlabFunction(wsol);
[xmax,wmax] = fminsearch(wsol_opt,0);
%% Strain calculation
% eq 3.3.8 (pg 116 reddy (pdf = 138))
epstotal = [diff(usol,x) + 0.5* diff(wsol,x)^2 - z*diff(wsol,x,2),0,0].';
epsx = epstotal(1);
%% Calculating and plotting Stress in each layer
res = 8; % accuracy of finding max and min stress
xplot = linspace(0,a,res);
yplot = linspace(0,b,res);
for kstress = 1:3 % stress state s_x, s_y, s_xz
    figure(kstress+1)
    hold on
    for klay = 1:nl % loop through all layers
        thplot = theta(klay);
        zplot = linspace(z_(klay),z_(klay+1),res);
        %% Calc Stresses
        if kstress == 3
            % Shear stresses
            syms G0
            G0_ = -int(diff(s_stress(1),x),z)+G0.';
            % solve for shear stresses from s_1
            s_xz = solve(G0_,G0);     
            % out of plane shear S_xz does not need to be transformed ??
            ezsurf(s_xz, [0, a, z_(klay), z_(klay+1)]) 
        else
            % normal stresses
            % Cij = reduced structural stiffness in strictural coordinates 3x3
            % stress in structural coordinates
            s_stress = subs(Cij,th,thplot)*epstotal;
            % stressin material coordinates
            m_stress = subs(Tij,th,thplot)*s_stress  ;          
            ezsurf(m_stress(kstress),[0,a,z_(klay),z_(klay+1)])
        end     
        %% find max stress in each layer
        ii=1;
        for i = xplot
            jj=1;
            for j = zplot
                if kstress == 3
                    stressplot(ii,jj) = subs(s_xz,[x z],[i j]);
                else
                    stressplot(ii,jj) = subs(m_stress(kstress),[x z],[i j]);
                end
                jj=jj+1;
            end
            ii=ii+1;
        end    
        Globalminstress(kstress,klay) = min(min(stressplot));
        Globalmaxstress(kstress,klay) = max(max(stressplot));  
        %
    end
    hold off
    axis auto
    title(strcat('\sigma_',num2str(kstress)))
    zlabel('stress(MPa)')
    view(-45,30)
end
%% Plot max stress and failure strength
figure
for i = 1:3
    subplot(1,3,i)
    bar(Globalmaxstress(i,:))
    hold on
    bar(Globalminstress(i,:))
    scatter(1:nl,ones(nl,1).*Strength(i,1),'filled')
    scatter(1:nl,ones(nl,1).*Strength(i,2),'filled')
    hold off
    xlabel('layer')
    title(strcat('\sigma',num2str(i)))
end
