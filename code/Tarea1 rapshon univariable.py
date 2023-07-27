# %%
import sympy as sp

# %%
x = sp.symbols('x')
f = sp.log(x-1) + sp.cos(x-1)
f_derivada = f.diff(x,1)
exactitud = 10e-4
iteraciones = 0
error = 1
sp.plotting.plot(f,  xlim=(-10, 10), ylim=(-10,10))
f
f_derivada

x0 = 1
list_resultados = [x0]

while error >= exactitud:
    iteraciones += 1
    y = f.evalf(subs={x: x0})
    y_derivada = f_derivada.evalf(subs={x: x0})
    x1 = sp.N(x0 - y/y_derivada, 10)
    list_resultados.append(x1)
    error = abs(x1 - x0)
    x0 = x1

print('El número de iteraciones fue: ' + str(iteraciones))
print('Usando x inicial: ' + str(list_resultados[0]))
print('La raíz es cercana a: ' + str(list_resultados[-1]))
# %%
