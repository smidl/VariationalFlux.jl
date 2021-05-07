using Flux
using LinearAlgebra
using Plots
using Base.Iterators

using DataFrames
using CSV
using VariationalFlux

# make sure to see the data! e.g. by (cd("path_to_data"))
df=CSV.read("data.csv",DataFrame);
X = Array(df)'
y = [ones(10); zeros(11)]
ndim = size(X,2)

niter=10000
θ=randn(ndim)
m(X) = σ.(X*θ)                  # pravdepodobnost ze y bude 0.
# loss(X,y) = sum(Flux.Losses.binarycrossentropy.(m(X),y))/size(y,2) 
loss(X,y) = sum(Flux.Losses.binarycrossentropy.(m(X),y))
ps = Flux.params(θ)

#opt = ADAM(0.1)
data = repeated((X, y ), niter)


function fit_relevant(ids,θ)
    θ2=θ[ids]
    loss_2(θ2) = sum(Flux.Losses.binarycrossentropy.(σ.(X[:,ids]*θ2),y))
    opt = ADAM(0.1)
    niter = 1000
    for j=1:niter
        gs=gradient(loss_2,θ2)[1]
        Flux.Optimise.update!(opt,θ2,gs)
    end
    maxerr=maximum(abs.(σ.(X[:,ids]*θ2).-y))
    maxerr, loss_2(θ2)
end

niter=100
min_ids=Vector()
min_ids3=Vector()
LL=zeros(niter)
ME=zeros(niter)
LL3=zeros(niter)
ME3=zeros(niter)
for i=1:niter
    opt = RMSProp(0.01)
    θ.=randn(ndim)

    qps, λ = vtrain_ardvb!(loss,ps,data,opt,1; σ0=1e1)
    μ,σμ = qps.ps[1], qps.σps[1]
    
    rat=abs.(μ)./sqrt.(σμ)
    perm=sortperm(rat)

    i_relevant = perm[end-1:end]
    push!(min_ids,i_relevant)
    ME[i], LL[i]=fit_relevant(i_relevant,θ)
    
    i_relevant3 = perm[end-2:end]
    push!(min_ids3,i_relevant3)
    ME3[i], LL3[i]=fit_relevant(i_relevant3,θ)
end

using BSON: @save @load
@save "mc_results" ME LL min_ids


Mid = hcat(min_ids...)
ProbSol = exp.(-LL)
GoodSol = findall(ProbSol.>maximum(ProbSol)*0.0001)
plot(sum(Mid[GoodSol,:]),dims=2)
GoodMid = Mid[:,GoodSol]
ids=sum(GoodMid,dims=2).>0

GoodIds=map(i->findall(GoodMid[:,i]),1:size(GoodMid,2))
UGI=unique(GoodIds)
UGI2=UGI[length.(UGI).==2]
UGI3=UGI[length.(UGI).==3]

# logits
maxerr=zeros(length(UGI2))
