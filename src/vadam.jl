using LinearAlgebra
using Flux

import Flux.Optimise:  batchmemaybe, update!

struct GaussPs
    ps
    σps
end

function clip!(A, tol)
    @inbounds for t in eachindex(A)
        if A[t] > tol
            A[t] = tol
        elseif A[t] < -tol
            A[t] = -tol
        end
    end
end
function clipnorm!( Δ, thresh)
    Δnrm = norm(Δ)
    if Δnrm > thresh
        rmul!(Δ, thresh / Δnrm)
    end
    return Δ
end


predict(model, x, qθ::GaussPs; N=100) =begin
    model_aux = deepcopy(model)
    psa = Flux.params(model_aux)
    
    function sample()
        for (θ,μ,ωsq) in zip(psa,qθ.ps,qθ.ωsq)
             θ .= μ .+ randn(size(μ)).*σps
        end
        model_aux(x) # evaluated at psa=[θ]
    end
    map((n)->sample(),1:N)
end

# variance of ADAM is 
variance_of(opt::ADAM,psi)=opt.state[psi][2]
variance_of(opt::RMSProp,psi)=opt.acc[psi]

# working code
function vtrain!(loss, ps, data, opt; cb = (ps) -> (), σ0=1e4)
    ps_mean = deepcopy(ps)
    σps = deepcopy(ps)
    map((x)->(x.=σ0) , σps)
    
    # cb = runall(cb)
    for d in data
        # reparametrization trick
        for i=1:length(ps)
            ps_mean[i].=ps[i]
            ps[i].=ps[i].+randn(size(ps[i])).* σps[i]
        end
        gs = gradient(ps) do
          loss(batchmemaybe(d)...)
        end
        for i=1:length(ps)
            ps[i].=ps_mean[i]
        end
        update!(opt, ps, gs)

        for i=1:length(ps)
            σps[i].=1.0./sqrt.(variance_of(opt,ps[i]))
        end
        cb(ps)
    end
    GaussPs(ps,σps)
end
  
# alternative 1, TODO

# working code
function vtrain_ard!(loss, ps, data, opt; cb = () -> (), ω0=1e4, logα0=1e-2)
    ps_mean = deepcopy(ps)
    ωsq = deepcopy(ps)
    logα = deepcopy(ps); # precision of every ps
    map((x)->(x.=ω0) , ωsq)
    map((x)->(x.=logα0) , logα)
    ard(x,logα) = 0.5*sum(x.^2 .* exp.(logα)) + 0.5*sum(logα)

    loss_ard()= mapreduce(pl->ard(pl[1],pl[2]),+, zip(ps,logα))
    optα = ADAM()


    # cb = runall(cb)
    for d in data
        # reparametrization trick
        for i=1:length(ps)
            ps_mean[i].=ps[i]
            ps[i].=ps[i].+randn(size(ps[i]))./ ωsq[i]
        end
        gs = gradient(ps) do
          loss(batchmemaybe(d)...) + loss_ard()
        end
        gα = gradient(logα) do 
            loss(batchmemaybe(d)...) + loss_ard()            
        end
        for i=1:length(ps)
            ps[i].=ps_mean[i]
        end
        update!(opt, ps, gs)
        update!(optα, logα, gα)

        for i=1:length(ps)
            ωsq[i].=sqrt.(variance_of(opt,ps[i]))
        end
        cb(ps,logα)
    end
    GaussPs(ps,ωsq), logα
end


# train model with ARD using VADAM on the model and VB update on precision
function vtrain_ardvb!(loss, ps, data, opt; cb = (a,b) -> (), σ0=1e2, λ0=1e-2, clip=0.0)
    ps_mean = deepcopy(ps)    # mean (i.e. ps as it is in adam)
    σps = deepcopy(ps)        # precision 
    λ = deepcopy(ps);         # precision of prior of every ps in ARD
    map((x)->(x.=σ0) , σps)
    map((x)->(x.=λ0) , λ)
    ard(x,λ) = 0.5*sum(x.^2 .* λ) 

    loss_ard()= mapreduce(pl->ard(pl[1],pl[2]),+, zip(ps,λ))

    # cb = runall(cb)
    for d in data
        # reparametrization trick
        for i=1:length(ps)
            ps_mean[i].=ps[i]
            ps[i].=ps[i].+randn(size(ps[i])).* σps[i]
        end # ps is random now
        gs = gradient(ps) do
          loss(batchmemaybe(d)...) + loss_ard()
        end

        if clip>0.0
            for i=1:length(ps)
                clipnorm!(gs[ps[i]],clip)
            end # ps is random now    
        end

        for i=1:length(ps)
            ps[i].=ps_mean[i]
        end # ps is the mean again
        update!(opt, ps, gs) # update the mean

        # store ADAM internals in σ
        for i=1:length(ps)
            σps[i].=1.0./sqrt.(variance_of(opt,ps[i]))
        end

        # VB update for λ
        for (λ,μ,σ) in zip(λ,ps,σps)
            λ .= 1.0 ./ (μ.^2 .+ σ.^2 )
        end  
        cb(ps,λ)
    end
    GaussPs(ps,σps), λ
end
