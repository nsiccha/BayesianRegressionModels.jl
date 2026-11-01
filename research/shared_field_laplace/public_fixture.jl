# Synthetic model shared by public accuracy, scaling and streaming-cost checks.
# Callers load BRM, Distributions and Random before including this file.
function public_laplace_fixture(;groups=6,rows=12,temporal=4,spatial=2)
    rng=Xoshiro(20261101)
    data=(;probe=repeat(1:groups;inner=rows),
        time=repeat(collect(range(.1,2.;length=rows)),groups),
        sx=cos.(1.3 .* (1:groups)),sy=sin.(.7 .* (1:groups)),y=randn(rng,groups*rows))
    builder=BayesianRegressionModels.brm(quote
        mu ~ 1 + (1 | probe) + hsgp(time;k=$temporal,by=probe,
            cor=multigroup((sx,sy),2.;k=$spatial))
        @plate for p in groups(probe)
            y[p] ~ Normal(mu[p],.7)
        end
    end;mod=@__MODULE__)
    (;builder,data)
end
