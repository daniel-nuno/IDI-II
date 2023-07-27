# %%
import sympy as sp
# %%
x, y = sp.symbols('x y')

f = x**2 + y - 1
df_dx = f.diff(x,1)
df_dy = f.diff(y,1)

g = x - 2*y**2
dg_dx = g.diff(x,1)
dg_dy = g.diff(y,1)

# sp.plotting.plot3d(f, g) #3d plot

plot_f = sp.plot_implicit(sp.Eq(f), x_var=(x, -2, 2), y_var=(y, -2,2), show=False)
plot_g = sp.plot_implicit(sp.Eq(g), x_var=(x, -2, 2), y_var=(y, -2,2), show=False)
plot_f.append(plot_g[0])
plot_g.show()

# %%
exactitud = 10e-4
iteraciones = 0
norma_dos = 1

x0 = 1 #x inicial
y0 = -1 #y inicial
X0 = sp.Matrix([x0, y0]) #x e y
F = sp.Matrix([f,g]) #matrix F(x,y) no evaluada
J_inverse = F.jacobian([x,y]).inv() #matrix jacobiana inversa no evaluada
F_xy_0 = sp.Matrix([f.evalf(subs={x: x0, y: y0}),
                g.evalf(subs={x: x0, y: y0})]) #matrix F(x,y) evaluada en X0]
list_X = [sp.N(X0, 5)]
list_F = [sp.N(F_xy_0, 5)]

while norma_dos >= exactitud:
    iteraciones += 1
    J_inverse_xy_0 = J_inverse.evalf(subs={x: x0, y: y0}) #matrix jacobiana inversa evaluada en X0

    X1 = list_X[-1] - J_inverse_xy_0*list_F[-1]
    x1 = X1[0]
    y1 = X1[1]
    F_xy_1 = F.evalf(subs={x: x1, y: y1}) #matrix F(x,y) evaluada en X1
    
    norma_dos = sp.N(( F_xy_1[0]**2 + F_xy_1[1]**2 )**(1/2), 5)
    list_X.append(sp.N(X1,5))
    list_F.append(sp.N(F_xy_1, 5))
    x0 = x1 #new x
    y0 = y1 #new y

print('El número de iteraciones fue: ' + str(iteraciones))
print('Usando x inicial: ' + str(list_X[0][0]))
print('Usando y inicial: ' + str(list_X[0][1]))
print('x final es: ' + str(list_X[-1][0]))
print('y final es: ' + str(list_X[-1][1]))
# %%
