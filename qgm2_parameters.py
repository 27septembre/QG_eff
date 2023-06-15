# JM: 22 May 2021

"""
  qgm2 parameters, consistent with choice of non-dim etc.
  modify this (e.g. A and B values for changes in wind etc.) when inverting for 
  eddy force functions etc.
"""

from dolfin import Constant, Expression, DirichletBC

# define the usual parameters (may need to manually modify these when loading experiments)
L     = 3840.0e5
beta  = Constant(2.0e-13)
A     = 0.9
B     = 0.2
tau0  = Constant(0.80)
nu    = Constant(50.0e4)
r     = Constant(4.0e-8)
alpha = Constant(120.0e5)

# Stratification
H = [Constant(0.25e5), Constant(0.75e5), Constant(3.0e5)]
# Stratification parameters output by qgm2, yielding deformation radii of
# 32.2 km and 18.9 km
# can read this in from the "stratification" binary oputput directly too
s = [(None, Constant(1.7177956325348763e-13)), \
     (Constant(5.72598544178292e-14), Constant(1.1788387884961508e-13)), \
     (Constant(2.947096971240377e-14), None)]

layers = len(H)
ngrid = 512
dt = Constant(1200.0)
stime  = 3600.0 * 24.0 * 5000.0 # averaging length (don't need this if not dealing with t_int data)

# non-dim
scale = L / ngrid

def Scale(p, s):
  return p.assign(float(p) / s)

for i in range(layers):
  Scale(H[i], scale)

L /= scale
Scale(beta,   1.0 / (scale ** 2))
Scale(tau0,   (scale ** 2) / (scale ** 2))
Scale(nu,     (scale ** 2) / scale)
Scale(r,      1.0 / scale)
Scale(alpha,  scale)
if layers > 1:
  Scale(s[0][1],  1.0 / (scale ** 2))
  Scale(s[-1][0], 1.0 / (scale ** 2))
  for i in range(1, layers - 1):
    Scale(s[i][0], 1.0 / (scale ** 2))
    Scale(s[i][1], 1.0 / (scale ** 2))
Scale(dt, scale)
stime /= scale

# Wind forcing (TO CHECK)
def Wind(element):
  # wind forcing profile
  ex = "yVal < yMid ? " \
       "-pi * (1.0 / lL) * A * sin(pi * (lL + yVal) / (lL + yMid)) : " \
       "pi * (1.0 / lL) * (1.0 / A) * sin(pi * (yVal - yMid) / (lL - yMid))"
  for key, value in reversed([("lL", "L / 2.0"),
                              ("xVal", "x[0] - lL"),
                              ("yMid", "B * xVal"),
                              ("yVal", "x[1] - lL")]):
    ex = ex.replace(key, "(%s)" % value)
  return Expression(ex, A = A, B = B, L = L, element = element)

print("finished loading qgm2_parameters")

