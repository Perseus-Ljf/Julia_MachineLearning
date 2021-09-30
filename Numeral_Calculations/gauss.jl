using Revise
includet("calculation_methods.jl")
include("calculation_methods.jl")

A = [2 10 0 -3;
     -3 -4 -12 13;
     1 2 3 -4;
     4 14 9 -13]
b = [10.0, 5, -2, 7]

gauss(A, b)