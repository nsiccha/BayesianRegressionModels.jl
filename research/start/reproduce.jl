using BayesianRegressionModels

function intro_model()
    (@brm begin
        y ~ Normal(loc, err)
        loc ~ 1 + x + (1 | subj)
        sd(loc, subj) ~ LogNormal(-1, 0.35)
        err ~ LogNormal(-1, 0.35)
    end)((;
        x=[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
        subj=[1, 1, 2, 2, 3, 3],
        y=[-0.6, 0.9, -0.2, 1.4, -0.8, 0.5],
    ))
end
