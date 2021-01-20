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

#opt = ADAM(0.1)
data = repeated((X, y ), niter)

niter=100
min_ids=Vector()
LL=zeros(niter)
for i=1:niter
    opt = RMSProp(0.01)
    θ.=randn(ndim)

    qps, λ = vtrain_ardvb!(loss,ps,data,opt; σ0=1e1)
    μ,σμ = qps.ps[1], qps.σps[1]
    
    i_relevant = μ.>sqrt.(σμ)
    push!(min_ids,i_relevant)
    model_relevant(X)=σ.(X[:,i_relevant]*θ[i_relevant])

    LL[i]=sum(Flux.Losses.binarycrossentropy.(model_relevant(X),y))/size(y,2) 
end

Mid = hcat(min_ids...)
ProbSol = exp.(-LL)
GoodSol = findall(ProbSol.>maximum(ProbSol)*0.01)
plot(sum(Mid[GoodSol,:]),dims=2)
GoodMid = Mid[:,GoodSol]
ids=sum(GoodMid,dims=2).>0

GoodIds=map(i->findall(GoodMid[:,i]),1:size(GoodMid,2))
UGI=unique(GoodIds)
UGI2=UGI[length.(UGI).==2]
UGI3=UGI[length.(UGI).==3]
