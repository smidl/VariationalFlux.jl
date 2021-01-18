# VariationalAdam
Implementation of Variational ADAM using Flux

The method is based on paper [1] that is proposing to use adaptive SDG algorithms such as ADAM / RMSProp to estimate variance of the variational approximation. Instead of estimating the parameters (ps) of conventional Flux models, the method provides estimates of the parameter and its variance:
g(param)=N(ps,σps).

```julia
ps = Flux.params(model)
qμ, λ = vtrain_ardvb!(loss,ps,data,opt)

```
where opt has to be optimizer with adaptive learning that can be considered as a n estimate of variance (ADAM or RMSProp)


[1] Khan, Mohammad Emtiyaz, Didrik Nielsen, Voot Tangkaratt, Wu Lin, Yarin Gal, and Akash Srivastava. "Fast and scalable bayesian deep learning by weight-perturbation in adam." arXiv preprint arXiv:1806.04854 (2018).
