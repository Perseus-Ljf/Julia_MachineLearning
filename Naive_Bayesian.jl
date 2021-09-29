using GLMakie
using Revise
using LinearAlgebra
# using RDatasets
using DataFrames
includet("MachineLearning.jl")
include("MachineLearning.jl")

x2_data = ["S", "M", "M", "S", "S", "S", "M", "M", "L", "L", "L", "M", "M", "L", "L"]
y_data = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]

df = DataFrame(x1 = repeat(1:3, inner = 5), x2 = x2_data, y = y_data)
test_data = [2, "S"]

naive_bayesian(df, test_data)