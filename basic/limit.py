import sympy

x = sympy.Symbol('x')
f = sympy.sin(x) / x
print(sympy.limit(f, x, sympy.oo))
print(sympy.limit(f, x, 0)) # lim x->0 sin(x) == x

