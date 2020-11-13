using LinearAlgebra
using Distributions
using Base.Iterators
using Flux

# trivial least squares with one point at [0,0] and other at [1.5,1.5]
# + some noise around those 
x = [0.2ones(1,5) ones(1,5)] + 0.01randn(1,10)
y = [zeros(5);1.5*ones(5)] + 0.01randn(10)

X=[x; x.^2];
# y = X*p

model=Dense(2,1)

# least squares with Spike&Slab  prior 
σ1=100;
σ2=1.0/10;
ssl(x) = log(0.9*exp(-0.5*(x/σ1)^2)/σ1+0.1*exp(-0.5*(x/σ2)^2)/σ2)
ssl(x::AbstractArray)=mapreduce(ssl,+,x)

loss(X)=sum((y-model(X)[:]).^2) - mapreduce(ssl,+,Flux.params(model))


include("src/vadam.jl")
#

ps = Flux.params(model)
opt = ADAM(0.1)
data = repeated((X,  ), 10000)


qμ = vtrain!(loss,ps,data,opt)
xp = (-1:0.1:3)'
Xp=[xp; xp.^2];
yp = predict(model, Xp, qμ)
ypp=vcat(yp...)
plot(xp',ypp')