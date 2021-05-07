using Flux
using LinearAlgebra
using Plots
using Base.Iterators

using VariationalFlux

using BSON: @save @load
@load "mc_results" ME LL min_ids


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
