# Shared helpers stay in the authored graph. Local curvature is conditional
# on their values; the marginal gradient must still differentiate those values.
include(joinpath(@__DIR__, "nlme_fixtures.jl"))
using LinearAlgebra, LogDensityProblems, Random

spl_shared_field(x, beta) = [exp(beta * t) for t in x]
spl_shared_gather(z, indices) = [z[i] for i in indices]
spl_shared_wrapper(z, indices) = spl_shared_gather(z, indices)
spl_shared_vector(w) = [exp(t) for t in w]

function shared_fd(f, x; h=1e-5)
    map(eachindex(x)) do i
        e = zeros(length(x)); e[i] = h
        (f(x+e)-f(x-e))/(2h)
    end
end

function shared_wide_fixture(n)
    data=(;n,y=ones(Int,n))
    RKBRMI(@brm data begin
        weights[1:n] .~ Normal.(0,1)
        z[1:n] .~ Normal.(0,1)
        field=spl_shared_vector(weights)
        @plate for i in eachindex(y)
            y[i] ~ Poisson(exp(z[i])*field[i])
        end
    end)
end

function shared_sweep(p,u)
    group=only(p.cells)
    _,packed=BRM._rk_spl_cell_pass(group.ad,u) do batch
        for c in group.cells
            BRM._rk_spl_cell_accumulate(batch,group.ad,2,ones(group.m),1.0,
                u,c.sel,c.i)
        end
    end
    packed
end


function shared_fixture(n; helper=true)
    data = (; n, grid=collect(range(0.1,0.9;length=n)), y=ones(Int,n))
    model = if helper
        @brm data begin
            beta ~ Normal(0,1)
            z[1:n] .~ Normal.(0,1)
            field = spl_shared_field(grid,beta)
            @plate for i in eachindex(grid)
                y[i] ~ Poisson(exp(z[i])*field[i])
            end
        end
    else
        @brm data begin
            beta ~ Normal(0,1)
            z[1:n] .~ Normal.(0,1)
            field = exp(beta*grid)
            @plate for i in eachindex(grid)
                y[i] ~ Poisson(exp(z[i])*field[i])
            end
        end
    end
    RKBRMI(model), data
end

@stestset "shared prefix exact mixed marginal derivatives" begin
    rk, data = shared_fixture(4)
    localrk, _ = shared_fixture(4;helper=false)
    full = rk_logdensity_problem(rk;ad_backend=NLME_AD)
    localfull = rk_logdensity_problem(localrk;ad_backend=NLME_AD)
    p = brm_laplace_problem(rk;integrate=:z,ad_backend=NLME_AD)
    localp = brm_laplace_problem(localrk;integrate=:z,ad_backend=NLME_AD)
    @test length(p.integrated)==4
    @test length(p.population)==1
    @test all(length(c.lz)==1 for group in p.cells for c in group.cells)
    @test all(all(i in p.integrated for i in c.sel) for group in p.cells for c in group.cells)
    for beta in (0.0,0.3,-0.2,0.0)
        u=0.08randn(Xoshiro(81),length(p.u)); u[p.population].=beta
        @test all(isapprox.(LogDensityProblems.logdensity_and_gradient(full,u),
            LogDensityProblems.logdensity_and_gradient(localfull,u)))
        value, gradient=LogDensityProblems.logdensity_and_gradient(p,[beta])
        expected, expected_gradient=LogDensityProblems.logdensity_and_gradient(localp,[beta])
        @test value ≈ expected atol=1e-9
        @test gradient ≈ expected_gradient atol=1e-9
        @test gradient ≈ shared_fd(x->LogDensityProblems.logdensity(p,x),[beta]) atol=2e-7 rtol=2e-7
    end
    u=zeros(length(p.u))
    @test only(LogDensityProblems.logdensity_and_gradient(full,u)[2][p.population])==0
    @test only(LogDensityProblems.logdensity_and_gradient(p,[0.0])[2]) ≈ -sum(data.grid)/8 atol=1e-10
    plan=brm_laplace_plan(rk;ad_backend=NLME_AD,integrate=:z,draws=12)
    @test plan.integrated_dimension==4
    automatic=brm_laplace_plan(rk;ad_backend=NLME_AD,integrate=:auto,draws=12,ess_floor=0.01)
    @test automatic.integrated_dimension==0
    @test all(t -> !t.admitted && t.importance_ess === nothing, automatic.terms)
    reused=brm_laplace_problem(rk;ad_backend=NLME_AD,integrate=plan)
    copied=deepcopy(reused)
    @test only(reused.cells).ad !== only(copied.cells).ad
    @test all(isapprox.(LogDensityProblems.logdensity_and_gradient(copied,[0.2]),
        LogDensityProblems.logdensity_and_gradient(p,[0.2])))
    @test all(isapprox.(LogDensityProblems.logdensity_and_gradient(reused,[-0.1]),
        LogDensityProblems.logdensity_and_gradient(p,[-0.1])))
end

function shared_sum_fixture(; nonlinear=false)
    data=(;n=5,grid=[0.1,0.7,-0.3,0.5,0.9],rows=[[1,2,2],[2,3],Int[],[4,5]],
        anchor=[1,3,2,4],y=[0.8,-0.1,0.4,1.1])
    model=if nonlinear
        @brm data begin
            beta ~ Normal(0,1)
            z[1:n] .~ Normal.(0,1)
            field=spl_shared_field(grid,beta)
            @plate for k in eachindex(rows)
                y[k] ~ Normal(sum(exp.(z[rows[k]]).*field[rows[k]])+field[anchor[k]],exp(beta))
            end
        end
    else
        @brm data begin
            beta ~ Normal(0,1)
            z[1:n] .~ Normal.(0,1)
            field=spl_shared_field(grid,beta)
            @plate for k in eachindex(rows)
                y[k] ~ Normal(sum(z[rows[k]].*field[rows[k]])+field[anchor[k]],exp(beta))
            end
        end
    end
    RKBRMI(model),data
end

@stestset "shared prefix sum-factor chain rule" begin
    for nonlinear in (false,true)
        rk,data=shared_sum_fixture(;nonlinear)
        p=brm_laplace_problem(rk;integrate=:z,ad_backend=NLME_AD)
        @test only(p.cells) isa BRM._BRMSumCellGroup
        for beta in (0.0,0.2,-0.15,0.0)
            value,gradient=LogDensityProblems.logdensity_and_gradient(p,[beta])
            @test isfinite(value) && all(isfinite,gradient)
            @test gradient ≈ shared_fd(x->LogDensityProblems.logdensity(p,x),[beta]) atol=3e-6 rtol=3e-6
            if !nonlinear
                field=spl_shared_field(data.grid,beta)
                A=[count(==(j),row)*field[j] for row in data.rows,j in 1:data.n]
                expected=logpdf(Normal(),beta)+logpdf(MvNormal(field[data.anchor],
                    Symmetric(A*A'+exp(2beta)*I)),data.y)
                @test value ≈ expected atol=1e-9
            end
        end
    end
end

@stestset "integrated ordinary helper keeps cell locality" begin
    data=(;n=4,idx=[4,2,1,3],y=ones(Int,4))
    rk=RKBRMI(@brm data begin
        beta ~ Normal(0,1)
        z[1:n] .~ Normal.(0,1)
        gathered=spl_shared_wrapper(z,idx)
        @plate for i in eachindex(idx)
            y[i] ~ Poisson(exp(beta+gathered[i]))
        end
    end)
    p=brm_laplace_problem(rk;integrate=:z,ad_backend=NLME_AD)
    @test all(length(c.lz)==1 for group in p.cells for c in group.cells)
    @test length(p.components)==4
    value,gradient=LogDensityProblems.logdensity_and_gradient(p,[0.2])
    @test isfinite(value)
    @test gradient ≈ shared_fd(x->LogDensityProblems.logdensity(p,x),[0.2]) atol=1e-7
end

function shared_subset_fixture(; summed=false)
    data=(; n=4,grid=[0.1,0.7,-0.3,0.5],rows=[[1,2],[2,3],Int[],[4,4]],
        y=[0.2,-0.1,0.4,0.7])
    model=if summed
        @brm data begin
            beta ~ Normal(0,1)
            z[1:n] .~ Normal.(0,1)
            w[1:n] .~ Normal.(0,1)
            field=spl_shared_field(grid,beta)
            @plate for k in eachindex(rows)
                y[k] ~ Normal(sum((z[rows[k]].+w[rows[k]]).*field[rows[k]]),exp(beta))
            end
        end
    else
        @brm data begin
            beta ~ Normal(0,1)
            z[1:n] .~ Normal.(0,1)
            w[1:n] .~ Normal.(0,1)
            field=spl_shared_field(grid,beta)
            @plate for k in eachindex(y)
                y[k] ~ Normal((z[k]+w[k])*field[k],exp(beta))
            end
        end
    end
    RKBRMI(model)
end

@stestset "shared prefix retained subsets keep newly sampled derivatives" begin
    for summed in (false,true)
        rk=shared_subset_fixture(;summed)
        plan=brm_laplace_plan(rk;ad_backend=NLME_AD,integrate=:z,draws=12,ess_floor=1.1)
        @test plan.integrated_dimension==4
        reused=brm_laplace_problem(rk;ad_backend=NLME_AD,integrate=plan)
        fresh=brm_laplace_problem(rk;ad_backend=NLME_AD,integrate=:z)
        @test reused.population==fresh.population
        @test length(reused.population)==5
        for scale in (0.0,0.1,-0.2)
            phi=scale*randn(Xoshiro(91),length(reused.population))
            value,gradient=LogDensityProblems.logdensity_and_gradient(reused,phi)
            expected,expected_gradient=LogDensityProblems.logdensity_and_gradient(fresh,phi)
            @test value ≈ expected atol=1e-9
            @test gradient ≈ expected_gradient atol=1e-8
            @test gradient ≈ shared_fd(x->LogDensityProblems.logdensity(reused,x),phi) atol=3e-6 rtol=3e-6
        end
    end
end

@stestset "shared-coordinate validation is batched across cells" begin
    allocations=Int[]
    for n in (32,128)
        rk=shared_wide_fixture(n)
        prepared=@timed brm_laplace_problem(rk;integrate=:z,ad_backend=NLME_AD)
        p=prepared.value
        u=zeros(length(p.u))
        @test length(p.population)==n
        @test all(length(c.sel)==1 for c in only(p.cells).cells)
        result=shared_sweep(p,u)
        @test result[p.integrated] ≈ -ones(n)
        @test result[p.population] ≈ -ones(n)
        bytes=minimum(@allocated(shared_sweep(p,u)) for _ in 1:3)
        push!(allocations,bytes)
        println("shared prefix n=",n," preparation_seconds=",prepared.time,
            " preparation_bytes=",prepared.bytes," sweep_bytes=",bytes)
    end
    @test allocations[2] <= 5allocations[1]+10000
end

function shared_preparation_fixture(n)
    data=(; n,rows=[[i,mod1(i+1,n)] for i in 1:n],y=ones(n))
    RKBRMI(@brm data begin
        weights[1:n] .~ Normal.(0,1)
        z[1:n] .~ Normal.(0,1)
        field=spl_shared_vector(weights)
        @plate for i in eachindex(rows)
            y[i] ~ Normal(sum(z[rows[i]].*field[rows[i]]),1)
        end
    end)
end

@stestset "shared prefix preparation retains only local supports" begin
    allocations=Int[]
    for n in (64,256)
        rk=shared_preparation_fixture(n)
        layout=BRM._brm_spl_layout(rk,(:z,))
        u=zeros(length(layout.names))
        q=BRM._rk_spl_factor_queries(rk,:y;layout.partitioned,u0=u)
        prepare()=BRM._brm_spl_prepare_factors(rk,:y,u,n,
            layout.integrated,layout.population;query=q)
        prepared=prepare()
        support=sum(c -> length(c.direct.sel)+
            sum(t -> length(t.sel),Iterators.flatten(c.terms);init=0),prepared.cells)
        @test support==2n
        @test all(c -> length(c.lz)==2,prepared.cells)
        u[layout.integrated].=0.2
        u[layout.population].=0.1
        expected=logpdf(Normal(0.4exp(0.1),1),1)
        values=BRM._rk_spl_cell_batch(q.factors,u) do batch
            [BRM._brm_spl_factor_values(prepared,u,c;context=batch).value
                for c in prepared.cells]
        end
        @test values ≈ fill(expected,n)
        @test u[layout.integrated]==fill(0.2,n)
        @test u[layout.population]==fill(0.1,n)
        prepare()
        bytes=minimum(@allocated(prepare()) for _ in 1:3)
        push!(allocations,bytes)
        println("shared preparation n=",n," total_support=",support," bytes=",bytes)
    end
    @test allocations[2] <= 5allocations[1]+50000
end

@stestset "shared prefix gradients preserve matrix-free contractions" begin
    for summed in (false,true)
        rk,_=summed ? shared_sum_fixture(;nonlinear=true) : shared_fixture(4)
        p=brm_laplace_problem(rk;integrate=:z,ad_backend=NLME_AD,
            solver=:matrixfree,matrixfree=MatrixFreeLaplace(;probes=1,seed=79))
        for beta in (0.0,0.15)
            value,gradient=LogDensityProblems.logdensity_and_gradient(p,[beta])
            @test isfinite(value)
            @test gradient ≈ shared_fd(x -> LogDensityProblems.logdensity(p,x),[beta]) atol=3e-5 rtol=3e-5
            @test LogDensityProblems.logdensity_and_gradient(deepcopy(p),[beta]) == (value,gradient)
        end
    end
end
