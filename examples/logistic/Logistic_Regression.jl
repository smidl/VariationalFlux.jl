using Flux
using LinearAlgebra
using Plots
using Base.Iterators

using DataFrames
using CSV
using VariationalAdam

# make sure to see the data! e.g. by (cd("path_to_data"))
df=CSV.read("data.csv",DataFrame);
X = Array(df)'
y = [ones(10); zeros(11)]
ndim = size(X,2)

niter=10000
θ=randn(ndim)
m(X) = σ.(X*θ)                  # pravdepodobnost ze y bude 0.
loss(X,y) = sum(Flux.Losses.binarycrossentropy.(m(X),y))/size(y,2) 
ps = Flux.params(θ)

opt = ADAM(0.1)
data = repeated((X, y ), niter)

PS=Vector()
AL=Vector()


qps, λ = vtrain_ardvb!(loss,ps,data,opt; σ0=1e1,
    cb=(ps,logα)->(push!(PS,deepcopy(hcat(ps...))); push!(AL,deepcopy(hcat(logα...)));))

pal = hcat(AL...)
pps = hcat(PS...)
plot(pps')

# test prediction
μ,σμ = qps.ps[1], qps.σps[1]
plot(μ,errorbar=sqrt.(σμ))

i_relevant = μ.>sqrt.(σμ)

model_relevant(X)=σ.(X[:,i_relevant]*θ[i_relevant])
scatter(y,marker=:star,fillalpha=0,label="data")
plot!(model_relevant(X),label="model")
