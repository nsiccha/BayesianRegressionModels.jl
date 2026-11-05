include("gmrf_preparation.jl")
using StanBlocks

function gmrf_plate_model(; known=false)
    input = merge(data, (; unit=[10, 20, 10, 10]))
    if known
        return @brm input begin
            @plate for p in groups(unit)
                field ~ 0 + gmrf((east[p], north[p]), time[p];
                    mesh=(vertices=xy, cells=triangles))
                sd(field, gmrf((east[p], north[p]), time[p])) = 1.2
                length_scale(field, gmrf((east[p], north[p]), time[p]), space) = 0.4
                length_scale(field, gmrf((east[p], north[p]), time[p]), time) = 1.7
                y[p] ~ Normal(field, 0.2)
            end
        end
    end
    @brm input begin
        @plate for p in groups(unit)
            field ~ 0 + gmrf((east[p], north[p]), time[p];
                mesh=(vertices=xy, cells=triangles))
            sd(field, gmrf((east[p], north[p]), time[p])) ~ Exponential(0.7)
            length_scale(field, gmrf((east[p], north[p]), time[p]), space) ~ LogNormal(0.5, 0.8)
            length_scale(field, gmrf((east[p], north[p]), time[p]), time) ~ LogNormal(0.7, 0.6)
            y[p] ~ Normal(field, 0.2)
        end
    end
end

