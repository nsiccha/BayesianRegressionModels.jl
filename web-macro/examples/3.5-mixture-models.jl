# label: 3.5 mixture models
# tier: 3
# status: open
# flag: sbbrm

loc ~ 1 + a
y1 ~ MixtureModel(Normal[Normal(loc, 0.5), Normal(0, 5)], [0.9, 0.1])
