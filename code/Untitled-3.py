#%%
import sympy as sp

x = sp.symbols('x')
sp.init_printing(use_unicode=True)
f = 0.5*x + sp.sin(x) - 1
#%%
f_derivada = f.diff(x,1)
x0 = 0.2
exactitud = 10e-5
iteraciones = 0
list_resultados = [x0]
error = 1
sp.plotting.plot(f,  xlim=(-5, 5), ylim=(-10,10))

while error >= exactitud:
    iteraciones += 1
    y = f.evalf(subs={x: x0})
    y_derivada = f_derivada.evalf(subs={x: x0})
    x1 = sp.N(x0 - y/y_derivada, 5)
    list_resultados.append(x1)
    error = abs(x1 - x0)
    x0 = x1

print('El número de iteraciones fue: ' + str(iteraciones))
print('Usando x inicial: ' + str(list_resultados[0]))
print('La raíz es cercana a: ' + str(list_resultados[-1]))
# %%
