using Revise
include("calculation_methods.jl")
includet("calculation_methods.jl")

A1 = [2 10 0 -3;
     -3 -4 -12 13;
     1 2 3 -4;
     4 14 9 -13]
b1 = [10.0, 5, -2, 7]
result1 = lu_solve(A1, b1)

A2 = [4 -1 1;
     -1 17/4 11/4;
     1 11/4 7/2]
b2 = [0,1,0]

result2 = lu_solve(A2, b2; method = cholesky)