using LinearAlgebra
using Flux
using Flux: update!

using VariationalAdam

x = 1.0:0.1:6.0
# y = x.^3

y = -x.^3 .+ 4*x.^2 .- 16


f(x,p,w) = (p[1]*x.^w[1] .+ p[2]*x.^w[2] .+ p[3])
llik(p,w)=sum((y-f(x,p,w)).^2)

ntrial =100
# Pbad=Vector()
Pe=zeros(3,ntrial)
We=zeros(2,ntrial)
LL=zeros(ntrial)
for k=1:100
    p = 0.1*[randn() randn() randn()]
    w = 0.1*[randn() randn()]
    
    #
    
    niter = 10000
    opt = RMSProp(0.01)
    data = repeated((  ), niter)
    
    # PS=Vector()
    # AL=Vector()
    
    ps=params(p,w)
    qμ, λ = vtrain_ardvb!(()->llik(p,w),ps,data,opt; σ0=1e-2, λ0=1e-4)
        # cb=(ps,logα)->(push!(PS,deepcopy(hcat(ps...))); push!(AL,deepcopy(hcat(logα...)));))
    
    # pal = vcat(AL...)
    # pps = vcat(PS...)
    # plot(pps)
    Pe[:,k]=qμ.ps[1]
    We[:,k]=qμ.ps[2]
    LL[k]=llik(qμ.ps[1],qμ.ps[2])

end

function plotall(Pe,We,LL)
    pl=Vector()
    for i=1:size(Pe,1)
        push!(pl,histogram(Pe[i,:],nbins=1000,label="p$i"))
    end
    for i=1:size(We,1)
        push!(pl,histogram(We[i,:],nbins=1000,label="w$i"))
    end
    push!(pl,histogram(LL,nbins=1000,label="mse"))
    plot(pl...)
end

plotall(Pe,We,LL)
