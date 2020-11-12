using LinearAlgebra
using Distributions
using Flux
import Flux: Params
import Flux.Optimise:  batchmemaybe, update!

struct GaussPs
    ps
    ω
end


# working code
function vtrain!(loss, ps, data, opt; cb = () -> ())
    ps = Flux.Params(ps)
    ps_mean = deepcopy(ps)
    ω = deepcopy(ps)
    map((x)->x.=1e8,ω)
    
    # cb = runall(cb)
    for d in data
        # reparametrization trick
        for i=1:length(ps)
            ps_mean[i].=ps[i]
            ps[i].=ps[i].+randn(length(ps[i]))./ .√ω[i]
        end
        gs = gradient(ps) do
          loss(batchmemaybe(d)...)
        end
        for i=1:length(ps)
            ps[i].=ps_mean[i]
        end
        update!(opt, ps, gs)

        for i=1:length(ps)
            ω[i].=opt.state[ps[i]][2]
        end
        cb()
    end
    GaussPs(ps,ω)
end
  
# alternative 1, TODO

