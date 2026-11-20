# A singleton contributes its own value/global derivative and the *whole*
# derivative on its factor, including observations and conditionals owned by
# other units. Independently map a joint point through the RK conditional
# ports and differentiate the resulting full density.
function check_centered_field(m, rk)
    t0 = BRM._brm_svi_target(m; field_states=:units, separator=:none)
    q0 = BRM._brm_svi_params(t0, m.unit[t0.globals], MeanFieldGlobal();
        global_scale=0.1, unit_scale=0.2)
    t, q = BRM._brm_svi_with_centering(t0, q0, LearnedCentering(; delay=false, fields=true))
    plan = t.centering
    rng = Xoshiro(17)
    for x in q.centering.x
        x .= randn(rng, length(x))
    end
    point = first(BRM._brm_nlme_prior_median_point(m)) .+ 0.15 .* randn(rng, length(m.unit))
    sampler = prepare_sampler(rk.model, rk_bound_plan(rk), m.unit; backend=FIELD_AD)
    record = BRM._brm_gmrf_footprint_record(rk, :field)
    ports = (Symbol(record.declaration, ".", record.locals.conditional_mean),
        Symbol(record.declaration, ".", record.locals.conditional_scale))
    read = BRM._rk_value_reader(rk, ports)
    field = only(t.field_sweep.fields)
    function reference(u)
        g = u[t.globals]
        frame = BRM._brm_svi_centering_frame(plan, t, g)
        y = copy(u)
        jac = 0.0
        for (i, block) in enumerate(t.units)
            yp = similar(u[block])
            jac += first(BRM._brm_svi_unit_centering!(yp, u[block], plan.instances[i],
                plan.ncells[i], BRM._brm_svi_unit_controls(q, i), frame, g, plan.base))
            y[block] .= yp
        end
        sd = last(Tuple(read(y)))
        c = [field.owner[k] == 0 ? 1.0 :
            BRM._brm_svi_unit_controls(q, field.owner[k])[field.cell[k]] for k in eachindex(sd)]
        D = sd .^ (1 .- c)
        y[field.field.positions] .*= D
        loc = first(Tuple(read(y)))
        y[field.field.positions] .+= (1 .- c .* D) .* loc
        jac += sum(log, D)
        first(sampler_value_and_gradient!(sampler, similar(y), y)) + jac
    end
    g = point[t.globals]
    frame = BRM._brm_svi_centering_frame(plan, t, g)
    s = t.field_sweep
    s.others .= point
    s.draw[] = map(f -> BRM._brm_svi_sweep_draw(f, q, g, 0:length(t.units)), s.fields)
    value, gg = BRM._brm_svi_global_term(t.prior, t, g)
    value += first(BRM._brm_svi_sweep_term!(gg, Float64[], plan, frame, t, q, 0,
        g, Float64[], t.plan))
    gradient = zero(point)
    parent_rows = value_rows = 0
    for (i, block) in enumerate(t.units)
        gi, gu = zero(g), zero(point[block])
        v, gc, adj = BRM._brm_svi_centered_unit_term!(gi, gu, plan, frame, t, q, i, g, point[block])
        @test all(iszero, gi[setdiff(eachindex(g), t.touched[i])])
        # A batch prepares rows for this owner and the owners it visits only. It
        # takes every derivative of the owner's output rows, the scale's alone of
        # a row the owner reads as a parent, and none of a row only others read.
        visited = [visit.o for visit in BRM._brm_svi_sweep_visits(s, i)]
        prepared = map(f -> BRM._brm_svi_sweep_draw(f, q, g, visited; focus=(i,)), s.fields)
        for (k, row) in only(prepared)
            size(row.J, 2) == 1 && (value_rows += 1)
            size(row.J, 2) > 1 && any(isnan, row.J) && (parent_rows += 1)
        end
        gp, up = zero(gi), zero(gu)
        vp, cp, _ = BRM._brm_svi_sweep_term!(gp, up, plan, frame, t, q, i, g, point[block],
            t.plan; draw=prepared)
        @test vp ≈ v rtol=1e-13
        @test gp ≈ gi atol=1e-12 rtol=1e-12
        @test up ≈ gu atol=1e-12 rtol=1e-12
        @test cp ≈ gc atol=1e-12 rtol=1e-12
        # A Newton proposal for this owner can also change its neighbors'
        # likelihoods and priors. Its acceptance uses that whole increment.
        function objective(z)
            extra = Ref(0.0)
            own = first(BRM._brm_svi_centered_unit_term!(zero(gi), zero(gu),
                plan, frame, t, q, i, g, z; neighbor_value=extra))
            own + extra[]
        end
        moved = copy(point)
        moved[block] .+= 0.01 .+ 0.002 .* eachindex(block)
        @test objective(moved[block]) - objective(point[block]) ≈
            reference(moved) - reference(point) atol=2e-10
        # The external owners contribute only local derivatives. Enumerate one
        # importance draw exactly; the average of two has the same expectation.
        proposal = s.neighbors[i]
        if !isempty(proposal.owners)
            probabilities = proposal.sampler.probabilities
            local_mean, control_mean = zero(gu), zero(gc)
            for (n, owner) in enumerate(proposal.owners)
                probability = probabilities[n]
                s.visits[i] = [(; o=i, weight=1.0), (; o=owner, weight=inv(probability))]
                gj, uj = zero(gi), zero(gu)
                vj, cj, _ = BRM._brm_svi_centered_unit_term!(gj, uj, plan, frame, t, q, i, g, point[block])
                @test vj ≈ v
                @test gj ≈ gi
                local_mean .+= probability .* uj
                control_mean .+= probability .* cj
            end
            empty!(s.visits[i])
            @test local_mean ≈ gu atol=2e-12
            @test control_mean ≈ gc atol=2e-12
        end
        value += v
        gg .+= gi .+ BRM._brm_svi_centering_batch_gradient(plan, t, frame,
            [(; gc, adjoints=(adj,))], [2.0])
        gradient[block] .= gu
        x = q.centering.x[i]
        for j in eachindex(x)
            old = x[j]
            x[j] = old + 1e-5
            plus = reference(point)
            x[j] = old - 1e-5
            minus = reference(point)
            x[j] = old
            c = BRM._brm_svi_control(old)
            @test gc[j] * c * (1-c) ≈ (plus-minus)/2e-5 atol=2e-6 rtol=2e-6
        end
    end
    gradient[t.globals] .= gg
    @test value ≈ reference(point) atol=1e-9
    # Collapsed-site probes displace a global with neighboring noises fixed.
    # The mapped global term and the unit probes must use the new law.
    for p in field.places
        shifted = copy(g)
        shifted[p] += 0.04
        full_point = copy(point)
        full_point[t.globals] .= shifted
        v, gs = BRM._brm_svi_collapsed_global_term(t, q, shifted)
        for (i, block) in enumerate(t.units)
            gi, gu = zero(g), zero(point[block])
            # Only field hyperparameters are displaced on this fixture.
            v += BRM._brm_svi_collapsed_term!(gi, gu, t, q, i, shifted, point[block],
                t.plan, first(t.work), plan, frame, p, 0.04)
            gs .+= gi
        end
        @test v ≈ reference(full_point) atol=2e-9
        h = zero(full_point)
        h[t.globals[p]] = 1e-5
        @test gs[p] ≈ (reference(full_point+h)-reference(full_point-h))/2e-5 atol=3e-6
    end
    for j in eachindex(point)
        h = zero(point)
        h[j] = 1e-5
        @test gradient[j] ≈ (reference(point+h)-reference(point-h))/2e-5 atol=2e-6 rtol=2e-6
    end
    (; parent_rows, value_rows)
end

function check_field_window(first_fit, new_fit)
    old = brm_svi_checkpoint(first_fit)
    m = new_fit.diagnostics.model
    t0 = BRM._brm_svi_target(m; previous=old, field_states=:units, separator=:none)
    t0, q0 = BRM._brm_svi_initialize(t0, old, old.q.family; start=nothing, inits=nothing,
        global_scale=0.2, unit_scale=0.2)
    t, q = BRM._brm_svi_with_centering(t0, q0, LearnedCentering(; delay=false, fields=true))
    levels = BRM._brm_svi_unit_levels(t)
    # Change the ordinary controls to exercise transport in a Gaussian factor
    # with cross-covariance between its ordinary and its field coordinates.
    for i in eachindex(t.units), (j,cell) in enumerate(t.centering.cells[i])
        cell.kind === :gmrf && continue
        q.centering.x[i][j] = 0.15
    end
    plan = t.centering
    f = only(t.field_sweep.fields)
    for k in eachindex(f.owner)
        i = f.owner[k]
        i == 0 && continue
        @test old.units[levels[i]].centering[f.slot[k]] ==
            new_fit.diagnostics.q.centering.x[i][f.cell[k]]
    end
    model = m.backend
    rec = BRM._brm_gmrf_footprint_record(model, :field)
    reader = BRM._rk_value_reader(model, Tuple(Symbol(rec.declaration,".",getproperty(rec.locals,p))
        for p in (:conditional_mean,:conditional_scale)))
    sampler = prepare_sampler(model.model, rk_bound_plan(model), m.unit; backend=FIELD_AD)
    point = first(BRM._brm_nlme_prior_median_point(m)) .+ 0.2randn(Xoshiro(503),length(m.unit))
    # Fix the noise, while allowing the saved conditional Gaussian of an
    # absent owner to move with g. The oracle reads its checkpoint directly.
    missing = Set(t.ownership.absent)
    coordinate = Dict(m.view.coordinates[j] => j for j in missing)
    absent = Dict{Int,Tuple{NamedTuple,Int}}()
    for label in setdiff(keys(old.units),levels)
        unit = old.units[label]
        for (slot,key) in enumerate(unit.keys)
            j = get(coordinate,get(key,:coordinate,nothing),0)
            j == 0 || (absent[j] = (unit,slot))
        end
    end
    function absent_values!(u,g,rng)
        for label in sort!(collect(setdiff(keys(old.units),levels));by=repr)
            saved = old.units[label]
            z = saved.M + LowerTriangular(saved.U)' \ randn(rng,length(saved.M))
            coupling = get(saved,:coupling,nothing)
            if coupling !== nothing
                z[coupling.rows] .+= coupling.W * (g[coupling.columns] .- coupling.center)
            end
            for (slot,key) in enumerate(saved.keys)
                j = get(coordinate,get(key,:coordinate,nothing),0)
                j == 0 || (u[j] = z[slot])
            end
        end
        u
    end
    old_control(k) = if f.owner[k] > 0
        BRM._brm_svi_control(old.units[levels[f.owner[k]]].centering[f.slot[k]])
    elseif haskey(absent,f.field.positions[k])
        saved,slot = absent[f.field.positions[k]]
        BRM._brm_svi_control(saved.centering[slot])
    else
        1.0
    end
    function reference(point)
        g = point[t.globals]
        frame = BRM._brm_svi_centering_frame(plan,t,g)
        y = copy(point)
        jac = 0.0
        for (i, block) in enumerate(t.units)
            yi = similar(point[block])
            jac += first(BRM._brm_svi_unit_centering!(yi, point[block], plan.instances[i],
                plan.ncells[i], BRM._brm_svi_unit_controls(q,i), frame, g, plan.base))
            y[block] .= yi
        end
        positions = f.field.positions
        sd = last(Tuple(reader(y)))
        # Columns of the physical map from the emitted location and scale. The
        # previous-density evaluation solves this independent joint map, even
        # across units. The optimizer cancels the unchanged field maps locally.
        function matrix(controls)
            D = sd .^ (1 .- controls)
            A = Matrix(Diagonal(D))
            for k in eachindex(positions)
                basis = copy(y)
                basis[positions] .= 0.0
                basis[positions[k]] = D[k]
                loc = first(Tuple(reader(basis)))
                A[:,k] .+= (1 .- controls .* D) .* loc
            end
            A
        end
        prev = old_control.(eachindex(f.owner))
        c = [o == 0 ? prev[k] : BRM._brm_svi_unit_controls(q,o)[f.cell[k]]
            for (k,o) in enumerate(f.owner)]
        A, previous = matrix(c), matrix(prev)
        absent_values!(y,g,Xoshiro(92))
        y[positions] .= A * y[positions]
        prior_coordinates = copy(y)
        prior_coordinates[positions] .= previous \ y[positions]
        whole = first(sampler_value_and_gradient!(sampler,similar(y),y))
        original = first(BRM._rk_nlme_population_prior_value_gradient(m.population_prior,y))
        variance = Diagonal(exp.(2 .* old.q.ρ)) + old.q.Β * old.q.Β'
        δ = g-old.q.μ
        value = whole-original+jac+logabsdet(A)[1]-logabsdet(previous)[1] -
            (dot(δ,variance\δ)+logdet(variance)+length(g)*log(2π))/2
        for (i,block) in enumerate(t.units)
            saved = old.units[levels[i]]
            z = copy(prior_coordinates[block])
            before = copy(z)
            controls = zeros(plan.ncells[i])
            for inst in plan.instances[i], (a,j) in enumerate(inst.positions)
                controls[inst.cells[a]] = BRM._brm_svi_control(saved.centering[j])
            end
            for inst in plan.instances[i]
                value += BRM._brm_svi_centering_inverse!(z, zero(controls),zero(controls),
                    before,inst,controls,frame,g,plan.base)
            end
            residual = z-saved.M
            lin = get(saved,:coupling,nothing)
            if lin !== nothing
                residual[lin.rows] .-= lin.W * (g[lin.columns] .- lin.center)
            end
            value += (logdet(saved.Λ)-length(z)*log(2π)-dot(residual,saved.Λ*residual))/2
        end
        value
    end
    g = point[t.globals]
    t.field_sweep.others .= absent_values!(copy(point),g,Xoshiro(92))
    t.field_sweep.draw[] = map(f -> BRM._brm_svi_sweep_draw(f,q,g,0:length(t.units)),
        t.field_sweep.fields)
    frame = BRM._brm_svi_centering_frame(plan,t,g)
    value, gg = BRM._brm_svi_global_term(t.prior,t,g)
    grad = zero(point)
    for (i,block) in enumerate(t.units)
        gi,gu = zero(g),zero(point[block])
        v,gc,a = BRM._brm_svi_centered_unit_term!(gi,gu,plan,frame,t,q,i,g,point[block])
        value += v
        gg .+= gi .+ BRM._brm_svi_centering_batch_gradient(plan,t,frame,
            [(;gc,adjoints=(a,))],[2.0])
        grad[block] .= gu
    end
    grad[t.globals] .= gg
    @test value ≈ reference(point) atol=1e-8
    for k in eachindex(point)
        h = zero(point)
        h[k] = 1e-5
        @test grad[k] ≈ (reference(point+h)-reference(point-h))/2e-5 atol=4e-6 rtol=3e-6
    end
    if !isempty(absent)
        seed = 164
        draw = only(eachcol(BRM._brm_svi_draws(Xoshiro(seed),t,q,1)))
        rng = Xoshiro(seed)
        ε,η = randn(rng,length(q.μ)), randn(rng,size(q.Β,2))
        g = BRM._brm_svi_draw_global(q,ε,η)
        raw = copy(m.unit)
        raw[t.globals] .= g
        for (i,block) in enumerate(t.units)
            noise = BRM._brm_svi_unit_noise(rng,q.U[i])
            raw[block] .= q.M[i] + BRM._brm_svi_unit_offset(q,i,noise)
        end
        absent_values!(raw,g,rng)
        positions = f.field.positions
        sd = last(Tuple(reader(raw)))
        c = old_control.(eachindex(f.owner))
        D = sd .^ (1 .- c)
        raw[positions] .*= D
        μ = first(Tuple(reader(raw)))
        @test draw[positions] ≈ raw[positions] + (1 .- c .* D) .* μ atol=1e-12
        @test !isempty(intersect(Set(keys(absent)),Set(f.field.positions[j]
            for k in eachindex(f.owner) if f.owner[k]>0 for j in f.parents[k])))
    end

end


# Recover the affine field map independently by evaluating the RK graph on
# coordinate basis vectors. This gives both the conditional map of one unit
# (all others fixed) and its marginal over the other unit Gaussians.
function check_field_start_and_projection(m, fit)
    t0 = BRM._brm_svi_target(m; field_states=:units, separator=:none)
    state = brm_svi_checkpoint(fit)
    q = fit.diagnostics.q
    t = BRM._brm_svi_centered_target(t0, q)
    f = only(t.field_sweep.fields)
    record = BRM._brm_gmrf_footprint_record(m.backend, :field)
    reader = BRM._rk_value_reader(m.backend, Tuple(Symbol(record.declaration, ".", name)
        for name in (record.locals.conditional_mean, record.locals.conditional_scale)))
    field_indices = f.field.positions
    c = [o == 0 ? 1.0 : BRM._brm_svi_unit_controls(q, o)[f.cell[k]] for (k,o) in enumerate(f.owner)]
    g = q.μ
    basis = copy(m.unit)
    basis[t.globals] .= g
    sd = last(Tuple(reader(basis)))
    D = sd .^ (1 .- c)
    A = Matrix(Diagonal(D))
    for k in eachindex(field_indices)
        basis[field_indices] .= 0.0
        basis[field_indices[k]] = D[k]
        A[:, k] .+= (1 .- c .* D) .* first(Tuple(reader(basis)))
    end

    levels = BRM._brm_svi_unit_levels(t)
    mu, covariance = zeros(length(D)), zeros(length(D), length(D))
    source = BRM._brm_svi_source("field regression", state, "checkpoint")
    for (i, label) in enumerate(levels)
        rows = f.owned[i + 1]
        slots = f.slot[rows]
        saved = state.units[label]
        mu[rows] .= saved.M[slots]
        covariance[rows, rows] .= inv(Symmetric(saved.Λ))[slots, slots]
    end
    shared = findall(iszero, f.owner)
    mu[shared] .= g[f.slot[shared]]
    projected = A * covariance * A'
    for (i, label) in enumerate(levels)
        rows = f.owned[i + 1]
        slots = f.slot[rows]
        @test source.units[label].M[slots] ≈ (A * mu)[rows] atol=1e-11
        @test inv(Symmetric(source.units[label].Λ))[slots, slots] ≈ projected[rows, rows] atol=1e-11
    end

    # A packed point fixes all units' means jointly. Give the initial Gaussians
    # correlated unit coordinates as a factor start does. Then propagate their
    # within-owner covariance with the surrounding units held at their means.
    point = first(BRM._brm_nlme_prior_median_point(m)) + 0.15 .* randn(Xoshiro(28), length(m.unit))
    q0 = BRM._brm_svi_params(t0, q.μ, MeanFieldGlobal(); global_scale=0.2, unit_scale=0.3)
    point[t0.globals] .= q0.μ
    for (i, block) in enumerate(t0.units)
        q0.M[i] .= point[block]
        q0.Λ[i] .+= 0.3 .* ones(length(block), length(block))
        q0.U[i] .= cholesky(Symmetric(q0.Λ[i])).L
    end
    initial = [inv(Symmetric(L)) for L in q0.Λ]
    # Set identical controls to the fit so A above also describes this start.
    started = merge(q0, (; centering=deepcopy(q.centering)))
    BRM._brm_svi_field_start!(t, started)
    BRM._brm_svi_field_start_factors!(t, started, eachindex.(started.M))
    z = copy(point)
    for (i, block) in enumerate(t0.units)
        z[block] .= started.M[i]
        rows = f.owned[i + 1]
        slots = f.slot[rows]
        within = Matrix{Float64}(I, length(block), length(block))
        within[slots, slots] .= A[rows, rows]
        @test within * inv(Symmetric(started.Λ[i])) * within' ≈ initial[i] atol=1e-11
    end
    @test A * z[field_indices] ≈ point[field_indices] atol=1e-11
end
