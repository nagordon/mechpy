
%% Plate a*b*h simply supported under q = q0 CLPT
syms q0 a b m n x y
Qmn = 4/(a*b)*int(int(q0*sin(m*pi*x/a)*sin(n*pi*y/b),x,0,a),y,0,b);

dmn = pi^4 / b^4 * (DTij(1,1)*m^4*(b/a)^4 + 2* (DTij(1,2)+2*DTij(6,6)) *m^2*n^2*(b/a)^2 + DTij(2,2)*n^4);

Wmn = Qmn/dmn;

w0 = Wmn * sin(m*pi*x/a) * sin(n*pi*y/b);

w0_ = subs(w0,[q0 a b],[-q0_ a_ b_] );

figure
w0sum = 0;
for n_ = 1:10
    for m_ = 1:10
        w0sum = w0sum + subs(w0_,[n m],[n_ m_]);
    end
end
w0sum;

% xplot = linspace(0,a_,res);
% yplot = linspace(0,b_,res);

ii=1;
for i = xplot
    jj=1;
    for j = yplot
        w0plot(ii,jj) = subs(w0sum,[x y],[i j]);
        jj=jj+1;
    end
    ii=ii+1;
end

surf(xplot,yplot,w0plot)
colorbar
set(gca,'PlotBoxAspectRatio',[2 1 1]);
xlabel('length a, u(x)')
ylabel('length b, v(y)')
zlabel('w(z)')