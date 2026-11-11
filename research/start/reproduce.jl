using BayesianRegressionModels

function intro_model()
    (@brm begin
        y ~ Normal(loc, err)
        loc ~ 1 + age + sex + (1 + age | subj)
        err ~ Exponential(1)
    end)((;
        age=[21.0, 38.0, 55.0, 29.0, 47.0, 61.0],
        sex=[1, 2, 1, 2, 1, 2],
        subj=[1, 1, 2, 2, 3, 3],
        y=[0.2, 1.1, -0.4, 0.7, 1.4, 1.0],
    ))
end
