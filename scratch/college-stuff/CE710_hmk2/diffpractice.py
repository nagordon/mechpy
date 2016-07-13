

import sympy as sp
import numpy as np
import matplotlib.pyplot as mp


syms y(x)
diff(y(x),2)
y(x) = dsolve(diff(y) == x*y)
diff(y(x))

% syms u(x)
% Du = diff(u);
% D2u = diff(u, 2);
% u(x) = dsolve(diff(u, 3) == u, u(0) == 1, Du(0) == -1, D2u(0) == pi)
% ezplot(u)

% % EQ 4.4.1a
% eq1   = A11*diff(ufun,x,2) - B11*diff(wfun,x,3) 
% % EQ 4.4.1b
% eq2   = A16*diff(ufun,x,2) - B16*diff(wfun,x,3);
% % EQ 4.4.1c
% eq3 = B11*diff(ufun,x,3) - D11*diff(wfun,x,4) + q0
% % solve eq1 eq2 and eq3 to get the w and u functions


% D2u = diff(u,2)
% D3w = diff(w,3)
% D4w = diff(w,4)
% D3u = diff(u,3)
% eqq1 = ode(B11^2*D4w / A11 - D11 * D4w -q0, w(x))
% sol = dsolve( B11^2*D4w / A11 - D11 * D4w -q0,w(x))
syms A11 B11 D11 A16 A66 B16 K A55 w(x) u(x) v(x) q0 phi(x) a
S = dsolve(A11*diff(u,2)+B11*diff(phi,2) == 0 ,...
           B11*diff(u,2)+D11*diff(phi,2)-K*A55*(diff(w)+phi==0),...
           K*A55*(diff(w,2)+diff(phi)) -q0 == 0)%,w(0)==0,phi(0)==0,w(a)==0,u(0)==0)

% without initial conditions
syms f(t) g(t)
S=dsolve(diff(f) == 3*f + 4*g, diff(g) == -4*f + 3*g)
f(t) = S.f
g(t) = S.g
% with initial conditions
[f(t), g(t)] = dsolve(diff(f) == 3*f + 4*g,...
diff(g) == -4*f + 3*g, f(0) == 0, g(0) == 1)