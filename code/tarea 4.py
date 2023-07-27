#%%
import sympy as sp
from sympy.plotting.plot import plot_contour
x, y = sp.symbols('x y')
#%%

f = x*y + 1/x + 1/y
f_derivada = sp.Matrix([f.diff(x,1), f.diff(y,1)])
#sp.plotting.plot3d(f, (x, -1, 1), (y, -1, 1))
#plot_contour(f, (x, -1, 1), (y, -1, 1))
#sp.Piecewise( ( sp.sin(x)+sp.sin(y)+sp.sin(x+y), (0 <= x) & (x <= 2*sp.pi) & (0 <= y) & (y <= 2*sp.pi)  ), (0, True) )
f
 # %%

exactitud = 10**-3
iteraciones = 0
error = 1
xn=1
yn=-1
Xn = sp.Matrix([xn, yn])
lista_X = [xn]
lista_y = [yn]
alpha = 0.5
y_derivada = f_derivada.evalf(subs={x: Xn[0], y: Xn[1]})

while (error >= exactitud) and (iteraciones < 5):
    iteraciones += 1
    Xn_1 = Xn - alpha*y_derivada
    lista_X.append(Xn_1[0])
    lista_y.append(Xn_1[1])
    y_derivada = f_derivada.evalf(subs={x: Xn_1[0], y: Xn_1[1]})
    error = ( y_derivada[0]**2 + y_derivada[1]**2 )**(1/2)
    Xn = Xn_1

print("El minimo local {} esta en x {} e y {} fue calculado con {} iteraciones.".format(
                                                    sp.N(f.evalf(subs={x: Xn[0], y: Xn[1]}),4),
                                                                                sp.N(Xn[0],4),
                                                                                sp.N(Xn[1],4),
                                                                                iteraciones))
# %%
#'El minimo %s esta en x %s y fue calculado con %s iteraciones.' % (sp.N(f.evalf(subs={x: xn}),4), sp.N(xn,4), iteraciones)

#f'El minimo {sp.N(f.evalf(subs={x: xn}),4)} esta en x {sp.N(xn,4)} y fue calculado con {iteraciones} iteraciones.'
#print("El minimo {} esta en x {} y fue calculado con {} iteraciones.".format(sp.N(f.evalf(subs={x: xn}),4), sp.N(xn,4), iteraciones)
