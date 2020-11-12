using LinearAlgebra
using Distributions
using Base.Iterators
# trivial least squares with one point at [0,0] and other at [1.5,1.5]
# + some noise around those 
x = [0.2ones(1,5) ones(1,5)] + 0.01randn(1,10)
y = [zeros(5);1.5*ones(5)] + 0.01randn(10)

X=[x; x.^2];
# y = X*p

# least squares with Student-t prior (marginalized ARD)
σ1=100;
σ2=1.0/10;
ssl(x) = log(0.9*exp(-0.5*(x/σ1)^2)/σ1+0.1*exp(-0.5*(x/σ2)^2)/σ2)
logSS(p)=ssl(p[1])+ssl(p[2]);
logp(p)=-norm(y-(p'*X)[:]).^2 +sum(logSS(p))

using Flux
include("../src/vadam.jl")
#

μ = [1.0,1.0] 
ps = Flux.Params([μ])
opt = ADAM(0.1)
data = repeated((  ), 1000)

loss()=-logp(μ)
vtrain!(loss,ps,data,opt)