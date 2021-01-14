using LinearAlgebra
using Base.Iterators
using Flux
using Plots

# trivial least squares with one point at [0,0] and other at [1.5,1.5]
# + some noise around those 
x = [0*ones(1,5) ones(1,5)] + 0.01randn(1,10)
y = [zeros(5); 1.0*ones(5)] + 0.01randn(10)

X=[x; x.^2];
# y = X*p

w =ones(1,2)
w[2]=0.5
# model=Dense(2,1)
# ps = Flux.params(model)
# loss(X)=sum((y-model(X)[:]).^2 ./0.01^2) 
m(X) = w*X
ps = Flux.params(w)
loss(X)=sum((y-m(X)[:]).^2 ./0.01^2) 

# least squares with Spike&Slab  prior 
include("/home/smidl/GitHub/VariationalAdam/src/vadam.jl")
#

niter = 10000
opt = ADAM(0.1)
data = repeated((X,  ), niter)

PS=Vector()
AL=Vector()


qμ, λ = vtrain_ardvb!(loss,ps,data,opt; σ0=1e-2, λ0=1e-4, clip=1e2,
    cb=(ps,logα)->(push!(PS,deepcopy(hcat(ps...))); push!(AL,deepcopy(hcat(logα...)));))

pal = vcat(AL...)
pps = vcat(PS...)
plot(pps)
