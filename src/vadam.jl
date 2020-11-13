using LinearAlgebra
using Distributions
using Flux

import Flux.Optimise:  batchmemaybe, update!

struct GaussPs
    ps
    ωsq
end

predict(model, x, qθ::GaussPs; N=100) =begin
    model_aux = deepcopy(model)
    psa = Flux.params(model_aux)
    
    function sample()
        for (θ,μ,ωsq) in zip(psa,qθ.ps,qθ.ωsq)
             θ .= μ .+ randn(size(μ))./ωsq
        end
        model_aux(x)
    end
    map((n)->sample(),1:N)
end


# working code
function vtrain!(loss, ps, data, opt; cb = () -> (), ω0=1e4)
    ps_mean = deepcopy(ps)
    ωsq = deepcopy(ps)
    map((x)->(x.=ω0) , ωsq)
    
    # cb = runall(cb)
    for d in data
        # reparametrization trick
        for i=1:length(ps)
            ps_mean[i].=ps[i]
            ps[i].=ps[i].+randn(size(ps[i]))./ ωsq[i]
        end
        gs = gradient(ps) do
          loss(batchmemaybe(d)...)
        end
        for i=1:length(ps)
            ps[i].=ps_mean[i]
        end
        update!(opt, ps, gs)

        for i=1:length(ps)
            ωsq[i].=sqrt.(opt.state[ps[i]][2])
        end
        cb()
    end
    GaussPs(ps,ωsq)
end
  
# alternative 1, TODO

