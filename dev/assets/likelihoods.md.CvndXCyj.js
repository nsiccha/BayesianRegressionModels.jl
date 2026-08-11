import{_ as a,o as n,c as i,a5 as p}from"./chunks/framework.BBrGrNbV.js";const o=JSON.parse('{"title":"Likelihoods","description":"","frontmatter":{},"headers":[],"relativePath":"likelihoods.md","filePath":"likelihoods.md","lastUpdated":null}'),l={name:"likelihoods.md"};function e(t,s,h,r,k,c){return n(),i("div",null,[...s[0]||(s[0]=[p(`<h1 id="Likelihoods" tabindex="-1">Likelihoods <a class="header-anchor" href="#Likelihoods" aria-label="Permalink to &quot;Likelihoods {#Likelihoods}&quot;">​</a></h1><h2 id="Complete-built-in-catalogue" tabindex="-1">Complete built-in catalogue <a class="header-anchor" href="#Complete-built-in-catalogue" aria-label="Permalink to &quot;Complete built-in catalogue {#Complete-built-in-catalogue}&quot;">​</a></h2><p>This is the exhaustive public likelihood catalogue for the default <a href="/BayesianRegressionModels.jl/dev/api#BayesianRegressionModels.SBBRMI"><code>SBBRMI</code></a> Stan backend. A <code>Distributions.jl</code> subtype not named here is not accepted merely because it is a distribution: BRM rejects it until its Julia constructor has an explicit Stan mapping. The pure-Julia <a href="/BayesianRegressionModels.jl/dev/api#BayesianRegressionModels.VBRMI"><code>VBRMI</code></a> backend is independent and does not implement every specialized family or response wrapper below.</p><h3 id="Direct-Distributions.jl-families" tabindex="-1">Direct <code>Distributions.jl</code> families <a class="header-anchor" href="#Direct-Distributions.jl-families" aria-label="Permalink to &quot;Direct \`Distributions.jl\` families {#Direct-Distributions.jl-families}&quot;">​</a></h3><table tabindex="0"><thead><tr><th style="text-align:right;">Outcome</th><th style="text-align:right;">Accepted constructors</th></tr></thead><tbody><tr><td style="text-align:right;">Continuous</td><td style="text-align:right;"><code>Normal</code>, <code>NormalCanon</code>, <code>Cauchy</code>, <code>TDist</code>, <code>Logistic</code>, <code>Gumbel</code>, <code>Chisq</code>, <code>Exponential</code>, <code>Gamma</code>, <code>Erlang</code>, <code>Beta</code>, <code>Uniform</code>, <code>LogNormal</code>, <code>Laplace</code>, <code>Frechet</code>, <code>Rayleigh</code>, <code>SkewNormal</code>, <code>Pareto</code>, <code>Weibull</code>, <code>InverseGamma</code>, <code>VonMises</code></td></tr><tr><td style="text-align:right;">Continuous, restricted parameterization</td><td style="text-align:right;"><code>Arcsine()</code> — the standard <code>[0, 1]</code> form only; <code>SkewedExponentialPower(mu, sigma, 1, alpha)</code> — only the literal shape <code>1</code></td></tr><tr><td style="text-align:right;">Discrete</td><td style="text-align:right;"><code>Bernoulli</code>, <code>BernoulliLogit</code>, <code>Binomial</code>, <code>BinomialLogit</code>, <code>BetaBinomial</code>, <code>Poisson</code>, <code>NegativeBinomial</code></td></tr></tbody></table><p>BRM normalizes constructor conventions where Julia and Stan differ. In particular, <code>Exponential</code>, <code>Gamma</code>, and <code>Erlang</code> use scale in <code>Distributions.jl</code> but rate in Stan; <code>Pareto(shape, scale)</code> is reordered to Stan&#39;s <code>(minimum, shape)</code> convention; <code>NormalCanon(eta, lambda)</code> becomes <code>normal(eta / lambda, inv(sqrt(lambda)))</code>; <code>NegativeBinomial(r, p)</code> is translated to Stan&#39;s shape/inverse-scale form; <code>TDist(nu)</code> becomes <code>student_t(nu, 0, 1)</code>; and <code>Laplace</code> becomes Stan&#39;s <code>double_exponential</code>. The standard <code>Arcsine()</code> is exactly <code>beta(1/2, 1/2)</code>. Shifted/scaled <code>Arcsine(a, b)</code> constructors need a Jacobian-aware custom implementation and are rejected rather than silently treated as a standard beta likelihood.</p><h3 id="BRM-families-and-structured-likelihoods" tabindex="-1">BRM families and structured likelihoods <a class="header-anchor" href="#BRM-families-and-structured-likelihoods" aria-label="Permalink to &quot;BRM families and structured likelihoods {#BRM-families-and-structured-likelihoods}&quot;">​</a></h3><table tabindex="0"><thead><tr><th style="text-align:right;">Constructor</th><th style="text-align:right;">Meaning / boundary</th></tr></thead><tbody><tr><td style="text-align:right;"><code>SkewDoubleExponential(mu, sigma, tau)</code></td><td style="text-align:right;">Stan-native asymmetric-Laplace parameterization</td></tr><tr><td style="text-align:right;"><code>LocationScale(mu, sigma, TDist(nu))</code></td><td style="text-align:right;">location-scale Student-t regression</td></tr><tr><td style="text-align:right;"><code>ZeroInflatedPoisson(lambda, zi)</code></td><td style="text-align:right;">zero-inflated Poisson</td></tr><tr><td style="text-align:right;"><code>NegativeBinomial2(mu, phi)</code></td><td style="text-align:right;">mean/precision negative binomial</td></tr><tr><td style="text-align:right;"><code>BetaBinomial2(n, mean, precision)</code></td><td style="text-align:right;">mean/precision beta-binomial</td></tr><tr><td style="text-align:right;"><code>CategoricalLogit(eta2, eta3, ...)</code> or <code>CategoricalLogit(@brm(...))</code></td><td style="text-align:right;">reference-class categorical logit</td></tr><tr><td style="text-align:right;"><code>OrderedLogistic(eta)</code></td><td style="text-align:right;">legacy cumulative-logit ordinal model</td></tr><tr><td style="text-align:right;"><code>Ordinal(structure, link, eta; ...)</code></td><td style="text-align:right;"><code>Cumulative()</code> or <code>StoppingRatio()</code> crossed with <code>LogitLink()</code>, <code>ProbitLink()</code>, or <code>CloglogLink()</code></td></tr><tr><td style="text-align:right;"><code>CircularVonMises(mu, kappa; interval=(-pi, pi))</code></td><td style="text-align:right;">von Mises on a fixed principal interval</td></tr><tr><td style="text-align:right;"><code>TruncatedNormal(mu, sigma, lower, upper)</code></td><td style="text-align:right;">legacy Bordet-only censored-Normal marker; new models should use <code>censored</code> below</td></tr></tbody></table><h3 id="Response-compositions-and-modifiers" tabindex="-1">Response compositions and modifiers <a class="header-anchor" href="#Response-compositions-and-modifiers" aria-label="Permalink to &quot;Response compositions and modifiers {#Response-compositions-and-modifiers}&quot;">​</a></h3><table tabindex="0"><thead><tr><th style="text-align:right;">Form</th><th style="text-align:right;">Exact supported surface</th></tr></thead><tbody><tr><td style="text-align:right;"><code>truncated(d; lower, upper)</code></td><td style="text-align:right;">base family <code>Normal</code>, <code>LogNormal</code>, <code>Exponential</code>, <code>Weibull</code>, or <code>Poisson</code></td></tr><tr><td style="text-align:right;"><code>censored(d; lower, upper)</code></td><td style="text-align:right;">the same five base families</td></tr><tr><td style="text-align:right;"><code>interval_censored(d; upper)</code></td><td style="text-align:right;">the same five base families; the response is the lower endpoint</td></tr><tr><td style="text-align:right;"><code>weighted(d, aweights(w))</code></td><td style="text-align:right;">analytic/precision weights for <code>Normal</code> only</td></tr><tr><td style="text-align:right;"><code>weighted(d, fweights(w))</code></td><td style="text-align:right;">frequency weights for direct mapped families in the first table</td></tr><tr><td style="text-align:right;"><code>weighted(d, weights(w))</code></td><td style="text-align:right;">power-likelihood weights for direct mapped families in the first table</td></tr><tr><td style="text-align:right;"><code>mi(y) ~ d</code></td><td style="text-align:right;">partly-missing continuous response imputation</td></tr></tbody></table><p>Every specialized family is expected to supply the fitted density, pointwise log likelihood, and posterior-predictive RNG used by BRM&#39;s generated quantities. Truncation and censoring additionally require matching CDF/CCDF paths; ragged observations additionally require a sized RNG.</p><h2 id="Adding-another-likelihood" tabindex="-1">Adding another likelihood <a class="header-anchor" href="#Adding-another-likelihood" aria-label="Permalink to &quot;Adding another likelihood {#Adding-another-likelihood}&quot;">​</a></h2><p>Yes—when Stan already has the distribution, adding it is usually small. BRM needs an explicit Julia-type-to-Stan-name entry, plus an argument translation when the two libraries use different parameterizations. It then inherits the ordinary model, pointwise-log-likelihood, and predictive-RNG paths from StanBlocks.</p><p>The work becomes larger when Stan has no native family: the implementation must provide and test a density/mass function, a pointwise companion, and an RNG. Supporting <code>truncated</code> or <code>censored</code> also needs the relevant CDF/CCDF functions, and ragged responses need a vector-sized RNG. These are finite implementation tasks, not an architectural prohibition; the explicit gates prevent a family from appearing to fit while prediction or likelihood diagnostics silently mean something else.</p><h2 id="Truncation-and-censoring" tabindex="-1">Truncation and censoring <a class="header-anchor" href="#Truncation-and-censoring" aria-label="Permalink to &quot;Truncation and censoring {#Truncation-and-censoring}&quot;">​</a></h2><p>BRM preserves the standard Distributions.jl RHS composition for mathematical truncation and threshold censoring:</p><div class="language-brm-comparison vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">brm-comparison</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Truncated, censored, and interval evidence</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">bounded_evidence </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@brm</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sigma) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_truncated </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> truncated</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Normal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mu, sigma); lower</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, upper</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_clamped </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> censored</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">LogNormal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mu, sigma); lower</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.25</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, upper</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # Genuine interval evidence: y_lower stores the open lower endpoint and</span></span>
<span class="line"><span style="--shiki-light:#6A737D;--shiki-dark:#6A737D;">    # y_upper stores the closed upper endpoint.</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_lower </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> interval_censored</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Normal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mu, sigma); upper</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y_upper)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)((;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_truncated</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_clamped</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.25</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_lower</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_upper</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">SBBRMI with data keys </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y_clamped</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y_lower</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y_truncated</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y_upper</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">emitted </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@slic</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> body</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)), x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_mu)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_mu</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_log_sigma </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_truncated)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_log_sigma </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_log_sigma)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    log_sigma </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_log_sigma</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    sigma </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> exp</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(log_sigma)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_truncated </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> truncated</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(normal, mu, sigma; lower </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, upper </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 2.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_clamped </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> censored</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(lognormal, mu, sigma; lower </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0.25</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, upper </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_lower </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> interval_censored</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(normal, y_lower, y_upper, mu, sigma)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-stan vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">stan</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>functions {</span></span>
<span class="line"><span>matrix hcat(</span></span>
<span class="line"><span>    vector x,</span></span>
<span class="line"><span>    vector y</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    if (dims(y)[1] != n) reject(&quot;hcat: dim mismatch — \`y\` dim 1 (= &quot;, dims(y)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    return append_col(x, y);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>matrix hcat(vector x) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    return to_matrix(x, n, 1);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real conditioning_normal_lpdf(</span></span>
<span class="line"><span>    vector y,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    vector args1,</span></span>
<span class="line"><span>    vector args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    return sum(conditioning_normal_lpdfs(y, lo, hi, args1, args2));</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector conditioning_normal_lpdfs(</span></span>
<span class="line"><span>    vector y,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    vector args1,</span></span>
<span class="line"><span>    vector args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    return jbroadcasted_conditioning_lpdf_normal(y, lo, hi, args1, args2);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector jbroadcasted_conditioning_lpdf_normal(</span></span>
<span class="line"><span>    vector x1,</span></span>
<span class="line"><span>    real x3,</span></span>
<span class="line"><span>    real x4,</span></span>
<span class="line"><span>    vector x5,</span></span>
<span class="line"><span>    vector x6</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(x1)[1];</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = conditioning_normal_lpdf(broadcasted_getindex(x1, i) | </span></span>
<span class="line"><span>            x3,</span></span>
<span class="line"><span>            x4,</span></span>
<span class="line"><span>            broadcasted_getindex(x5, i),</span></span>
<span class="line"><span>            broadcasted_getindex(x6, i)</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real conditioning_normal_lpdf(</span></span>
<span class="line"><span>    real y,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    real args1,</span></span>
<span class="line"><span>    real args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    array[1] real rv;</span></span>
<span class="line"><span>    rv[1] = negative_infinity();</span></span>
<span class="line"><span>    if((lo &gt;= hi)) {</span></span>
<span class="line"><span>        reject(&quot;truncated: lower bound must be less than upper bound&quot;);</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((y &gt;= lo)) {</span></span>
<span class="line"><span>            if((y &lt;= hi)) {</span></span>
<span class="line"><span>                rv[1] = (</span></span>
<span class="line"><span>                    normal_lpdf(y | args1, args2) -</span></span>
<span class="line"><span>                    log_diff_exp(normal_lcdf_stable(hi, args1, args2), normal_lcdf_stable(lo, args1, args2))</span></span>
<span class="line"><span>                );</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv[1];</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real normal_lcdf_stable(</span></span>
<span class="line"><span>    real x,</span></span>
<span class="line"><span>    real loc,</span></span>
<span class="line"><span>    real scale</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    return (log(erfc(((-(x - loc)) / (scale * sqrt(2.0))))) - log(2.0));</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real broadcasted_getindex(vector x, int i) {</span></span>
<span class="line"><span>    return x[i];</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector conditioning_vector_normal_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    vector args1,</span></span>
<span class="line"><span>    vector args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = anontok__1;</span></span>
<span class="line"><span>    return jbroadcasted_conditioning_cell_rng_normal_rng(rep_vector(0.0, n), lo, hi, args1, args2);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector jbroadcasted_conditioning_cell_rng_normal_rng(</span></span>
<span class="line"><span>    vector x1,</span></span>
<span class="line"><span>    real x3,</span></span>
<span class="line"><span>    real x4,</span></span>
<span class="line"><span>    vector x5,</span></span>
<span class="line"><span>    vector x6</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(x1)[1];</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = conditioning_cell_normal_rng(</span></span>
<span class="line"><span>            broadcasted_getindex(x1, i),</span></span>
<span class="line"><span>            x3,</span></span>
<span class="line"><span>            x4,</span></span>
<span class="line"><span>            broadcasted_getindex(x5, i),</span></span>
<span class="line"><span>            broadcasted_getindex(x6, i)</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real conditioning_cell_normal_rng(</span></span>
<span class="line"><span>    real dummy,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    real args1,</span></span>
<span class="line"><span>    real args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    return conditioning_normal_rng(lo, hi, args1, args2);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real conditioning_normal_rng(</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    real args1,</span></span>
<span class="line"><span>    real args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    vector[1] draw;</span></span>
<span class="line"><span>    array[1] int attempts;</span></span>
<span class="line"><span>    draw[1] = normal_rng(args1, args2);</span></span>
<span class="line"><span>    attempts[1] = 1;</span></span>
<span class="line"><span>    while((conditioning_outside(draw[1], lo, hi) == 1)) {</span></span>
<span class="line"><span>        if((attempts[1] &gt;= 100000)) {</span></span>
<span class="line"><span>            reject(&quot;truncated: rejection sampler exceeded 100000 draws&quot;);</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>        draw[1] = normal_rng(args1, args2);</span></span>
<span class="line"><span>        attempts[1] = (attempts[1] + 1);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return draw[1];</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>int conditioning_outside(</span></span>
<span class="line"><span>    real x,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    array[1] int rv;</span></span>
<span class="line"><span>    rv[1] = 0;</span></span>
<span class="line"><span>    if((x &lt; lo)) {</span></span>
<span class="line"><span>        rv[1] = 1;</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((x &gt; hi)) {</span></span>
<span class="line"><span>            rv[1] = 1;</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv[1];</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real clamping_lognormal_lpdf(</span></span>
<span class="line"><span>    vector y,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    vector args1,</span></span>
<span class="line"><span>    vector args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    return sum(clamping_lognormal_lpdfs(y, lo, hi, args1, args2));</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector clamping_lognormal_lpdfs(</span></span>
<span class="line"><span>    vector y,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    vector args1,</span></span>
<span class="line"><span>    vector args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    return jbroadcasted_clamping_lpdf_lognormal(y, lo, hi, args1, args2);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector jbroadcasted_clamping_lpdf_lognormal(</span></span>
<span class="line"><span>    vector x1,</span></span>
<span class="line"><span>    real x3,</span></span>
<span class="line"><span>    real x4,</span></span>
<span class="line"><span>    vector x5,</span></span>
<span class="line"><span>    vector x6</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(x1)[1];</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = clamping_lognormal_lpdf(broadcasted_getindex(x1, i) | </span></span>
<span class="line"><span>            x3,</span></span>
<span class="line"><span>            x4,</span></span>
<span class="line"><span>            broadcasted_getindex(x5, i),</span></span>
<span class="line"><span>            broadcasted_getindex(x6, i)</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real clamping_lognormal_lpdf(</span></span>
<span class="line"><span>    real y,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    real args1,</span></span>
<span class="line"><span>    real args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    array[1] real rv;</span></span>
<span class="line"><span>    if((lo &gt;= hi)) {</span></span>
<span class="line"><span>        reject(&quot;censored: lower bound must be less than upper bound&quot;);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    rv[1] = negative_infinity();</span></span>
<span class="line"><span>    if((y == lo)) {</span></span>
<span class="line"><span>        rv[1] = lognormal_lcdf(lo | args1, args2);</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((y == hi)) {</span></span>
<span class="line"><span>            rv[1] = lognormal_lccdf(hi | args1, args2);</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            if((y &gt; lo)) {</span></span>
<span class="line"><span>                if((y &lt; hi)) {</span></span>
<span class="line"><span>                    rv[1] = lognormal_lpdf(y | args1, args2);</span></span>
<span class="line"><span>                }</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv[1];</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector clamping_vector_lognormal_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    vector args1,</span></span>
<span class="line"><span>    vector args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = anontok__1;</span></span>
<span class="line"><span>    return jbroadcasted_clamping_cell_rng_lognormal_rng(rep_vector(0.0, n), lo, hi, args1, args2);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector jbroadcasted_clamping_cell_rng_lognormal_rng(</span></span>
<span class="line"><span>    vector x1,</span></span>
<span class="line"><span>    real x3,</span></span>
<span class="line"><span>    real x4,</span></span>
<span class="line"><span>    vector x5,</span></span>
<span class="line"><span>    vector x6</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(x1)[1];</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = clamping_cell_lognormal_rng(</span></span>
<span class="line"><span>            broadcasted_getindex(x1, i),</span></span>
<span class="line"><span>            x3,</span></span>
<span class="line"><span>            x4,</span></span>
<span class="line"><span>            broadcasted_getindex(x5, i),</span></span>
<span class="line"><span>            broadcasted_getindex(x6, i)</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real clamping_cell_lognormal_rng(</span></span>
<span class="line"><span>    real dummy,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    real args1,</span></span>
<span class="line"><span>    real args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    return clamping_lognormal_rng(lo, hi, args1, args2);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real clamping_lognormal_rng(</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    real args1,</span></span>
<span class="line"><span>    real args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    vector[1] draw;</span></span>
<span class="line"><span>    draw[1] = lognormal_rng(args1, args2);</span></span>
<span class="line"><span>    if((draw[1] &lt; lo)) {</span></span>
<span class="line"><span>        draw[1] = lo;</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((draw[1] &gt; hi)) {</span></span>
<span class="line"><span>            draw[1] = hi;</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return draw[1];</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real interval_evidence_impl_normal_lpdf(</span></span>
<span class="line"><span>    vector y,</span></span>
<span class="line"><span>    vector lo,</span></span>
<span class="line"><span>    vector hi,</span></span>
<span class="line"><span>    vector args1,</span></span>
<span class="line"><span>    vector args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    return sum(interval_evidence_impl_normal_lpdfs(y, lo, hi, args1, args2));</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector interval_evidence_impl_normal_lpdfs(</span></span>
<span class="line"><span>    vector y,</span></span>
<span class="line"><span>    vector lo,</span></span>
<span class="line"><span>    vector hi,</span></span>
<span class="line"><span>    vector args1,</span></span>
<span class="line"><span>    vector args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    return jbroadcasted_interval_evidence_impl_lpdf_normal(y, lo, hi, args1, args2);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector jbroadcasted_interval_evidence_impl_lpdf_normal(</span></span>
<span class="line"><span>    vector x1,</span></span>
<span class="line"><span>    vector x3,</span></span>
<span class="line"><span>    vector x4,</span></span>
<span class="line"><span>    vector x5,</span></span>
<span class="line"><span>    vector x6</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(x1)[1];</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = interval_evidence_impl_normal_lpdf(broadcasted_getindex(x1, i) | </span></span>
<span class="line"><span>            broadcasted_getindex(x3, i),</span></span>
<span class="line"><span>            broadcasted_getindex(x4, i),</span></span>
<span class="line"><span>            broadcasted_getindex(x5, i),</span></span>
<span class="line"><span>            broadcasted_getindex(x6, i)</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real interval_evidence_impl_normal_lpdf(</span></span>
<span class="line"><span>    real y,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    real args1,</span></span>
<span class="line"><span>    real args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    array[1] real rv;</span></span>
<span class="line"><span>    if((lo &gt;= hi)) {</span></span>
<span class="line"><span>        reject(&quot;interval_censored: lower bound must be less than upper bound&quot;);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    rv[1] = log_diff_exp(normal_lcdf_stable(hi, args1, args2), normal_lcdf_stable(lo, args1, args2));</span></span>
<span class="line"><span>    return rv[1];</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector interval_evidence_impl_vector_normal_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    vector lo,</span></span>
<span class="line"><span>    vector hi,</span></span>
<span class="line"><span>    vector args1,</span></span>
<span class="line"><span>    vector args2</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = anontok__1;</span></span>
<span class="line"><span>    return normal_vector_rng(n, args1, args2);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector normal_vector_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    vector a,</span></span>
<span class="line"><span>    vector b</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = anontok__1;</span></span>
<span class="line"><span>    return to_vector(normal_rng(a, b));</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>data {</span></span>
<span class="line"><span>    int x_n;</span></span>
<span class="line"><span>    vector[x_n] x;</span></span>
<span class="line"><span>    int y_truncated_n;</span></span>
<span class="line"><span>    vector[y_truncated_n] y_truncated;</span></span>
<span class="line"><span>    int y_clamped_n;</span></span>
<span class="line"><span>    vector[y_clamped_n] y_clamped;</span></span>
<span class="line"><span>    int y_lower_n;</span></span>
<span class="line"><span>    vector[y_lower_n] y_lower;</span></span>
<span class="line"><span>    int y_upper_n;</span></span>
<span class="line"><span>    vector[y_upper_n] y_upper;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed data {</span></span>
<span class="line"><span>    matrix[x_n, 2] X_mu = hcat(rep_vector(1.0, num_elements(x)), x);</span></span>
<span class="line"><span>    int pop_mu_n_covariates = 2;</span></span>
<span class="line"><span>    matrix[num_elements(y_truncated), 1] X_log_sigma = hcat(rep_vector(1.0, num_elements(y_truncated)));</span></span>
<span class="line"><span>    int pop_log_sigma_n_covariates = 1;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>parameters {</span></span>
<span class="line"><span>    vector[pop_mu_n_covariates] pop_mu_beta_pop;</span></span>
<span class="line"><span>    vector[pop_log_sigma_n_covariates] pop_log_sigma_beta_pop;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed parameters {</span></span>
<span class="line"><span>    vector[x_n] pop_mu = (X_mu * pop_mu_beta_pop);</span></span>
<span class="line"><span>    vector[x_n] mu = pop_mu;</span></span>
<span class="line"><span>    vector[num_elements(y_truncated)] pop_log_sigma = (X_log_sigma * pop_log_sigma_beta_pop);</span></span>
<span class="line"><span>    vector[num_elements(y_truncated)] log_sigma = pop_log_sigma;</span></span>
<span class="line"><span>    vector[num_elements(y_truncated)] sigma = exp(log_sigma);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>model {</span></span>
<span class="line"><span>    pop_mu_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    pop_log_sigma_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    y_truncated ~ conditioning_normal(0.0, 2.0, mu, sigma);</span></span>
<span class="line"><span>    y_clamped ~ clamping_lognormal(0.25, 1.8, mu, sigma);</span></span>
<span class="line"><span>    y_lower ~ interval_evidence_impl_normal(y_lower, y_upper, mu, sigma);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>generated quantities {</span></span>
<span class="line"><span>    vector[y_truncated_n] y_truncated_likelihood = conditioning_normal_lpdfs(y_truncated, 0.0, 2.0, mu, sigma);</span></span>
<span class="line"><span>    vector[y_truncated_n] y_truncated_gen = conditioning_vector_normal_rng(y_truncated_n, 0.0, 2.0, mu, sigma);</span></span>
<span class="line"><span>    vector[y_clamped_n] y_clamped_likelihood = clamping_lognormal_lpdfs(y_clamped, 0.25, 1.8, mu, sigma);</span></span>
<span class="line"><span>    vector[y_clamped_n] y_clamped_gen = clamping_vector_lognormal_rng(y_clamped_n, 0.25, 1.8, mu, sigma);</span></span>
<span class="line"><span>    vector[y_lower_n] y_lower_likelihood = interval_evidence_impl_normal_lpdfs(y_lower, y_lower, y_upper, mu, sigma);</span></span>
<span class="line"><span>    vector[y_lower_n] y_lower_gen = interval_evidence_impl_vector_normal_rng(y_lower_n, y_lower, y_upper, mu, sigma);</span></span>
<span class="line"><span>}</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing unsupported </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> this BRM example</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Gaussian scale prior must bind bare </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`sigma\`</span></span></code></pre></div><p>These are three different likelihood contracts:</p><ul><li><p><code>truncated(d; lower, upper)</code> conditions <code>d</code> on the inclusive bounds and predicts from that conditional distribution;</p></li><li><p><code>censored(d; lower, upper)</code> is the distribution of <code>clamp(X, lower, upper)</code> and predicts clamped values;</p></li><li><p><code>interval_censored(d; upper)</code> contributes <code>log(CDF(upper) - CDF(response))</code> for the genuine interval observation <code>(response, upper]</code>, while prediction remains on the uncoarsened base scale.</p></li></ul><p>Bounds may be numeric literals or observed row-wise columns. The initial family-gated surface covers <code>Normal</code>, <code>LogNormal</code>, <code>Exponential</code>, <code>Weibull</code>, and <code>Poisson</code>; BRM rejects other base families until their aggregate density, pointwise likelihood, CDF/CCDF, generated prediction, and stanc paths are all tested. BRM&#39;s eager two-sided bound check accepts <code>lower &lt;= upper</code>, while the StanBlocks producer requires a non-degenerate interval with <code>lower &lt; upper</code>; equal bounds are therefore rejected during Stan lowering.</p><p>This composition is implemented only by the <code>SBBRMI</code> Stan backend. <strong>Do not use <code>VBRMI</code> for these formulas:</strong> it currently does not reject <code>truncated</code> or <code>censored</code> at construction and can return log densities for a misinterpreted model; <code>interval_censored</code> may fail only when the density is evaluated. A <code>VBRMI</code> result is therefore not a valid cross-check of an <code>SBBRMI</code> fit. The legacy <code>TruncatedNormal</code> Bordet marker remains a separate censored-Normal compatibility surface.</p><p>The backend compatibility floor for this surface is StanBlocks <code>0eaebfae904d3bffab150dfa2c59632ac783b992</code>, where the public distribution-HOF tokens became <code>truncated</code>, <code>censored</code>, and <code>interval_censored</code> with no aliases. BRM revisions containing this lowering must be co-pinned with that StanBlocks commit or later; the preceding StanBlocks <code>9b879d5e</code> expects the older internal token spellings and is intentionally incompatible.</p><h2 id="Concise-categorical-regression" tabindex="-1">Concise categorical regression <a class="header-anchor" href="#Concise-categorical-regression" aria-label="Permalink to &quot;Concise categorical regression {#Concise-categorical-regression}&quot;">​</a></h2><p><code>CategoricalLogit</code> accepts an explicit nested <code>@brm(...)</code> predictor formula:</p><div class="language-brm-comparison vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">brm-comparison</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Nested categorical-logit formula</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">categorical_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;b&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;a&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;c&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;b&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;c&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;a&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">categorical_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @brm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> categorical_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CategoricalLogit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@brm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">SBBRMI with data keys </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">emitted </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@slic</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> body</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_y_nested_arg1_class2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)), x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_y_nested_arg1_class2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_y_nested_arg1_class2)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_nested_arg1_class2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_y_nested_arg1_class2</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_y_nested_arg1_class3 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)), x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_y_nested_arg1_class3 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_y_nested_arg1_class3)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_nested_arg1_class3 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_y_nested_arg1_class3</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_categorical_logits </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> adjoint</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)), y_nested_arg1_class2, y_nested_arg1_class3))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> categorical_logit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_categorical_logits)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-stan vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">stan</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>functions {</span></span>
<span class="line"><span>matrix hcat(</span></span>
<span class="line"><span>    vector x,</span></span>
<span class="line"><span>    vector y</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    if (dims(y)[1] != n) reject(&quot;hcat: dim mismatch — \`y\` dim 1 (= &quot;, dims(y)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    return append_col(x, y);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>matrix hcat(vector x, vector y, vector z) {</span></span>
<span class="line"><span>    return hcat(hcat(x, y), z);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>matrix hcat(</span></span>
<span class="line"><span>    matrix x,</span></span>
<span class="line"><span>    vector y</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int m = dims(x)[1];</span></span>
<span class="line"><span>    int n = dims(x)[2];</span></span>
<span class="line"><span>    if (dims(y)[1] != m) reject(&quot;hcat: dim mismatch — \`y\` dim 1 (= &quot;, dims(y)[1], &quot;) does not match \`m\` (= &quot;, m, &quot;)&quot;);</span></span>
<span class="line"><span>    return append_col(x, y);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real categorical_logit_lpmf(</span></span>
<span class="line"><span>    array[] int y,</span></span>
<span class="line"><span>    matrix eta</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    if (dims(eta)[2] != n) reject(&quot;categorical_logit_lpmf: dim mismatch — \`eta\` dim 2 (= &quot;, dims(eta)[2], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    real rv = 0.0;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv += categorical_logit_lpmf(y[i] | eta[:, i]);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector categorical_logit_lpmfs(</span></span>
<span class="line"><span>    array[] int y,</span></span>
<span class="line"><span>    matrix eta</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    if (dims(eta)[2] != n) reject(&quot;categorical_logit_lpmfs: dim mismatch — \`eta\` dim 2 (= &quot;, dims(eta)[2], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = categorical_logit_lpmf(y[i] | eta[:, i]);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>array[] int categorical_logit_int_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    matrix eta</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = anontok__1;</span></span>
<span class="line"><span>    if (dims(eta)[2] != n) reject(&quot;categorical_logit_rng: dim mismatch — \`eta\` dim 2 (= &quot;, dims(eta)[2], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    array[n] int rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = categorical_logit_rng(eta[:, i]);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>data {</span></span>
<span class="line"><span>    int x_n;</span></span>
<span class="line"><span>    vector[x_n] x;</span></span>
<span class="line"><span>    int y_n;</span></span>
<span class="line"><span>    array[y_n] int y;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed data {</span></span>
<span class="line"><span>    matrix[x_n, 2] X_y_nested_arg1_class2 = hcat(rep_vector(1.0, num_elements(x)), x);</span></span>
<span class="line"><span>    int pop_y_nested_arg1_class2_n_covariates = 2;</span></span>
<span class="line"><span>    matrix[x_n, 2] X_y_nested_arg1_class3 = hcat(rep_vector(1.0, num_elements(x)), x);</span></span>
<span class="line"><span>    int pop_y_nested_arg1_class3_n_covariates = 2;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>parameters {</span></span>
<span class="line"><span>    vector[pop_y_nested_arg1_class2_n_covariates] pop_y_nested_arg1_class2_beta_pop;</span></span>
<span class="line"><span>    vector[pop_y_nested_arg1_class3_n_covariates] pop_y_nested_arg1_class3_beta_pop;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed parameters {</span></span>
<span class="line"><span>    vector[x_n] pop_y_nested_arg1_class2 = (X_y_nested_arg1_class2 * pop_y_nested_arg1_class2_beta_pop);</span></span>
<span class="line"><span>    vector[x_n] y_nested_arg1_class2 = pop_y_nested_arg1_class2;</span></span>
<span class="line"><span>    vector[x_n] pop_y_nested_arg1_class3 = (X_y_nested_arg1_class3 * pop_y_nested_arg1_class3_beta_pop);</span></span>
<span class="line"><span>    vector[x_n] y_nested_arg1_class3 = pop_y_nested_arg1_class3;</span></span>
<span class="line"><span>    matrix[(2 + 1), x_n] y_categorical_logits = (hcat(rep_vector(0.0, num_elements(y)), y_nested_arg1_class2, y_nested_arg1_class3)&#39;);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>model {</span></span>
<span class="line"><span>    pop_y_nested_arg1_class2_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    pop_y_nested_arg1_class3_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    y ~ categorical_logit(y_categorical_logits);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>generated quantities {</span></span>
<span class="line"><span>    vector[x_n] y_likelihood = categorical_logit_lpmfs(y, y_categorical_logits);</span></span>
<span class="line"><span>    array[x_n] int y_gen = categorical_logit_int_rng(y_n, y_categorical_logits);</span></span>
<span class="line"><span>}</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing unsupported </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> this BRM example</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> executable families are </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Bernoulli(p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BernoulliLogit(eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Binomial(trials, p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BinomialLogit(trials, eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Poisson(exp(log_rate))\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`NegativeBinomial2(mu, phi)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`BetaBinomial2(trials, mean, precision)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and bounded </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; got </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`CategoricalLogit\`</span></span></code></pre></div><p>For an outcome with $K$ levels, BRM expands the marked formula to $K-1$ ordinary scalar linear predictors with distinct coefficients, then calls the same reference-class categorical lowering as the fully explicit form. The first fitted level has logit zero. Plain vectors use <code>sort(unique(y))</code> for the fitted order; a <code>CategoricalVector</code> uses its declared level order. The latter is the way to select a reference level deliberately.</p><p>For example, a three-level outcome above is equivalent in model structure to:</p><div class="language-brm-comparison vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">brm-comparison</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Explicit categorical-logit predictors</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">explicit_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;b&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;a&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;c&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;b&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;c&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">&quot;a&quot;</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">explicit_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @brm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> explicit_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_nested_arg1_class2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_nested_arg1_class3 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CategoricalLogit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_nested_arg1_class2, y_nested_arg1_class3)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">SBBRMI with data keys </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">emitted </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@slic</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> body</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_y_nested_arg1_class2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)), x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_y_nested_arg1_class2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_y_nested_arg1_class2)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_nested_arg1_class2 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_y_nested_arg1_class2</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_y_nested_arg1_class3 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)), x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_y_nested_arg1_class3 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_y_nested_arg1_class3)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_nested_arg1_class3 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_y_nested_arg1_class3</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_categorical_logits </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> adjoint</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)), y_nested_arg1_class2, y_nested_arg1_class3))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> categorical_logit</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_categorical_logits)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-stan vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">stan</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>functions {</span></span>
<span class="line"><span>matrix hcat(</span></span>
<span class="line"><span>    vector x,</span></span>
<span class="line"><span>    vector y</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    if (dims(y)[1] != n) reject(&quot;hcat: dim mismatch — \`y\` dim 1 (= &quot;, dims(y)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    return append_col(x, y);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>matrix hcat(vector x, vector y, vector z) {</span></span>
<span class="line"><span>    return hcat(hcat(x, y), z);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>matrix hcat(</span></span>
<span class="line"><span>    matrix x,</span></span>
<span class="line"><span>    vector y</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int m = dims(x)[1];</span></span>
<span class="line"><span>    int n = dims(x)[2];</span></span>
<span class="line"><span>    if (dims(y)[1] != m) reject(&quot;hcat: dim mismatch — \`y\` dim 1 (= &quot;, dims(y)[1], &quot;) does not match \`m\` (= &quot;, m, &quot;)&quot;);</span></span>
<span class="line"><span>    return append_col(x, y);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real categorical_logit_lpmf(</span></span>
<span class="line"><span>    array[] int y,</span></span>
<span class="line"><span>    matrix eta</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    if (dims(eta)[2] != n) reject(&quot;categorical_logit_lpmf: dim mismatch — \`eta\` dim 2 (= &quot;, dims(eta)[2], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    real rv = 0.0;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv += categorical_logit_lpmf(y[i] | eta[:, i]);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector categorical_logit_lpmfs(</span></span>
<span class="line"><span>    array[] int y,</span></span>
<span class="line"><span>    matrix eta</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    if (dims(eta)[2] != n) reject(&quot;categorical_logit_lpmfs: dim mismatch — \`eta\` dim 2 (= &quot;, dims(eta)[2], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = categorical_logit_lpmf(y[i] | eta[:, i]);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>array[] int categorical_logit_int_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    matrix eta</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = anontok__1;</span></span>
<span class="line"><span>    if (dims(eta)[2] != n) reject(&quot;categorical_logit_rng: dim mismatch — \`eta\` dim 2 (= &quot;, dims(eta)[2], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    array[n] int rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = categorical_logit_rng(eta[:, i]);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>data {</span></span>
<span class="line"><span>    int x_n;</span></span>
<span class="line"><span>    vector[x_n] x;</span></span>
<span class="line"><span>    int y_n;</span></span>
<span class="line"><span>    array[y_n] int y;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed data {</span></span>
<span class="line"><span>    matrix[x_n, 2] X_y_nested_arg1_class2 = hcat(rep_vector(1.0, num_elements(x)), x);</span></span>
<span class="line"><span>    int pop_y_nested_arg1_class2_n_covariates = 2;</span></span>
<span class="line"><span>    matrix[x_n, 2] X_y_nested_arg1_class3 = hcat(rep_vector(1.0, num_elements(x)), x);</span></span>
<span class="line"><span>    int pop_y_nested_arg1_class3_n_covariates = 2;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>parameters {</span></span>
<span class="line"><span>    vector[pop_y_nested_arg1_class2_n_covariates] pop_y_nested_arg1_class2_beta_pop;</span></span>
<span class="line"><span>    vector[pop_y_nested_arg1_class3_n_covariates] pop_y_nested_arg1_class3_beta_pop;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed parameters {</span></span>
<span class="line"><span>    vector[x_n] pop_y_nested_arg1_class2 = (X_y_nested_arg1_class2 * pop_y_nested_arg1_class2_beta_pop);</span></span>
<span class="line"><span>    vector[x_n] y_nested_arg1_class2 = pop_y_nested_arg1_class2;</span></span>
<span class="line"><span>    vector[x_n] pop_y_nested_arg1_class3 = (X_y_nested_arg1_class3 * pop_y_nested_arg1_class3_beta_pop);</span></span>
<span class="line"><span>    vector[x_n] y_nested_arg1_class3 = pop_y_nested_arg1_class3;</span></span>
<span class="line"><span>    matrix[(2 + 1), x_n] y_categorical_logits = (hcat(rep_vector(0.0, num_elements(y)), y_nested_arg1_class2, y_nested_arg1_class3)&#39;);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>model {</span></span>
<span class="line"><span>    pop_y_nested_arg1_class2_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    pop_y_nested_arg1_class3_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    y ~ categorical_logit(y_categorical_logits);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>generated quantities {</span></span>
<span class="line"><span>    vector[x_n] y_likelihood = categorical_logit_lpmfs(y, y_categorical_logits);</span></span>
<span class="line"><span>    array[x_n] int y_gen = categorical_logit_int_rng(y_n, y_categorical_logits);</span></span>
<span class="line"><span>}</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing unsupported </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> this BRM example</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> executable families are </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Bernoulli(p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BernoulliLogit(eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Binomial(trials, p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BinomialLogit(trials, eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Poisson(exp(log_rate))\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`NegativeBinomial2(mu, phi)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`BetaBinomial2(trials, mean, precision)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and bounded </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; got </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`CategoricalLogit\`</span></span></code></pre></div><p>The generated names are deterministic implementation names; use the explicit form when those predictor names are part of another formula. Only nested <code>@brm(...)</code> opts into predictor-formula interpretation. Thus <code>CategoricalLogit(1 + x)</code> remains an ordinary expression and is rejected by the categorical backend, rather than silently acquiring coefficients.</p><p>The marker is not categorical-specific. It selects formula interpretation at one family-argument position while surrounding expressions retain their usual meaning. For example, a distributional Normal model can make both predictors concise while keeping the positive scale link explicit:</p><div class="language-brm-comparison vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">brm-comparison</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Nested distributional predictors</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">distributional_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (; x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], y</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">distributional_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @brm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> distributional_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Normal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@brm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">exp</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@brm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">SBBRMI with data keys </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">emitted </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@slic</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> body</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_y_nested_arg1 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)), x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_y_nested_arg1 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> _popefs_coefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_y_nested_arg1)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_y_nested_arg2_arg1 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_y_nested_arg2_arg1 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_y_nested_arg2_arg1)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_nested_arg2_arg1 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_y_nested_arg2_arg1</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> normal_id_glm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(X_y_nested_arg1, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, pop_y_nested_arg1, (exp)(y_nested_arg2_arg1))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_nested_arg1 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_y_nested_arg1 </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_y_nested_arg1</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-stan vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">stan</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>functions {</span></span>
<span class="line"><span>matrix hcat(</span></span>
<span class="line"><span>    vector x,</span></span>
<span class="line"><span>    vector y</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    if (dims(y)[1] != n) reject(&quot;hcat: dim mismatch — \`y\` dim 1 (= &quot;, dims(y)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    return append_col(x, y);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>matrix hcat(vector x) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    return to_matrix(x, n, 1);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector normal_id_glm_lpdfs(</span></span>
<span class="line"><span>    vector y,</span></span>
<span class="line"><span>    matrix X,</span></span>
<span class="line"><span>    real alpha,</span></span>
<span class="line"><span>    vector beta,</span></span>
<span class="line"><span>    vector sigma</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    if (dims(X)[1] != n) reject(&quot;normal_id_glm_lpdfs: dim mismatch — \`X\` dim 1 (= &quot;, dims(X)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = normal_lpdf(y[i] | (alpha + (X[i, :] * beta)), sigma);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector normal_id_glm_vector_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    matrix X,</span></span>
<span class="line"><span>    real alpha,</span></span>
<span class="line"><span>    vector beta,</span></span>
<span class="line"><span>    vector sigma</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int m = anontok__1;</span></span>
<span class="line"><span>    if (dims(X)[1] != m) reject(&quot;normal_id_glm_rng: dim mismatch — \`X\` dim 1 (= &quot;, dims(X)[1], &quot;) does not match \`m\` (= &quot;, m, &quot;)&quot;);</span></span>
<span class="line"><span>    return normal_id_glm_rng(X, alpha, beta, sigma);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector normal_id_glm_rng(</span></span>
<span class="line"><span>    matrix X,</span></span>
<span class="line"><span>    real alpha,</span></span>
<span class="line"><span>    vector beta,</span></span>
<span class="line"><span>    vector sigma</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int m = dims(X)[1];</span></span>
<span class="line"><span>    return to_vector(normal_rng((rep_vector(alpha, m) + (X * beta)), sigma));</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>data {</span></span>
<span class="line"><span>    int x_n;</span></span>
<span class="line"><span>    vector[x_n] x;</span></span>
<span class="line"><span>    int y_n;</span></span>
<span class="line"><span>    vector[y_n] y;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed data {</span></span>
<span class="line"><span>    matrix[x_n, 2] X_y_nested_arg1 = hcat(rep_vector(1.0, num_elements(x)), x);</span></span>
<span class="line"><span>    int pop_y_nested_arg1_n_covariates = 2;</span></span>
<span class="line"><span>    matrix[num_elements(y), 1] X_y_nested_arg2_arg1 = hcat(rep_vector(1.0, num_elements(y)));</span></span>
<span class="line"><span>    int pop_y_nested_arg2_arg1_n_covariates = 1;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>parameters {</span></span>
<span class="line"><span>    vector[pop_y_nested_arg1_n_covariates] pop_y_nested_arg1_beta_pop;</span></span>
<span class="line"><span>    vector[pop_y_nested_arg2_arg1_n_covariates] pop_y_nested_arg2_arg1_beta_pop;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed parameters {</span></span>
<span class="line"><span>    vector[pop_y_nested_arg1_n_covariates] pop_y_nested_arg1 = pop_y_nested_arg1_beta_pop;</span></span>
<span class="line"><span>    vector[num_elements(y)] pop_y_nested_arg2_arg1 = (X_y_nested_arg2_arg1 * pop_y_nested_arg2_arg1_beta_pop);</span></span>
<span class="line"><span>    vector[num_elements(y)] y_nested_arg2_arg1 = pop_y_nested_arg2_arg1;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>model {</span></span>
<span class="line"><span>    pop_y_nested_arg1_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    pop_y_nested_arg2_arg1_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    y ~ normal_id_glm(X_y_nested_arg1, 0.0, pop_y_nested_arg1, exp(y_nested_arg2_arg1));</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>generated quantities {</span></span>
<span class="line"><span>    vector[x_n] y_likelihood = normal_id_glm_lpdfs(y, X_y_nested_arg1, 0.0, pop_y_nested_arg1, exp(y_nested_arg2_arg1));</span></span>
<span class="line"><span>    vector[x_n] y_gen = normal_id_glm_vector_rng(y_n, X_y_nested_arg1, 0.0, pop_y_nested_arg1, exp(y_nested_arg2_arg1));</span></span>
<span class="line"><span>    vector[x_n] y_nested_arg1 = (X_y_nested_arg1 * pop_y_nested_arg1);</span></span>
<span class="line"><span>}</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing unsupported </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> this BRM example</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> scale must be a direct named predictor; got ExprColumn{</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">typeof</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(exp), Tuple{NamedColumn{Symbol, ExprColumn{</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">typeof</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">), Tuple{NamedColumn{Symbol, MissingColumn}, Int64}, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">{}}}}, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@NamedTuple</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">{}}</span></span></code></pre></div><p>This introduces distinct scalar predictors for location and log-scale, then passes <code>exp(log_scale)</code> to <code>Normal</code>; nested <code>@brm</code> never inserts a link. A standalone fragment such as <code>@brm(1 + x)</code> is not yet a first-class value and produces a targeted error outside an enclosing model.</p><p>This is the same broad model structure expressed by a top-level categorical formula in brms (<code>y ~ 1 + x</code>, <code>family = categorical(link = &quot;logit&quot;)</code>) or Bambi (<code>&quot;y ~ 1 + x&quot;</code>, <code>family=&quot;categorical&quot;</code>). Defaults for priors, contrasts, and reference-level selection are package-specific; BRM does not import those defaults implicitly.</p><h2 id="Ordinal-structure-and-link-composition" tabindex="-1">Ordinal structure and link composition <a class="header-anchor" href="#Ordinal-structure-and-link-composition" aria-label="Permalink to &quot;Ordinal structure and link composition {#Ordinal-structure-and-link-composition}&quot;">​</a></h2><p>BRM treats the ordinal probability construction and inverse link as separate typed choices. A cumulative probit model is:</p><div class="language-brm-comparison vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">brm-comparison</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Ordinal cumulative-probit model</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ordinal_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (; x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], y</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">])</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ordinal_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @brm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ordinal_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    eta </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Ordinal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Cumulative</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ProbitLink</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), eta)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">SBBRMI with data keys </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">emitted </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@slic</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> body</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_eta </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_eta </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_eta)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    eta </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_eta</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_thresholds</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ordered</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> std_normal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_threshold_effect </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> rep_matrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> brm_ordinal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(eta, y_thresholds, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, y_threshold_effect)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-stan vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">stan</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>functions {</span></span>
<span class="line"><span>matrix hcat(vector x) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    return to_matrix(x, n, 1);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_lpmf(</span></span>
<span class="line"><span>    array[] int y,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    real discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    return brm_ordinal_lpmf(y | </span></span>
<span class="line"><span>        eta,</span></span>
<span class="line"><span>        thresholds,</span></span>
<span class="line"><span>        rep_vector(discrimination, n),</span></span>
<span class="line"><span>        structure,</span></span>
<span class="line"><span>        link,</span></span>
<span class="line"><span>        threshold_effect</span></span>
<span class="line"><span>    );</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_lpmf(</span></span>
<span class="line"><span>    array[] int y,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    vector discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(discrimination)[1] != n) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`discrimination\` dim 1 (= &quot;, dims(discrimination)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    real rv = 0.0;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv += brm_ordinal_lpmf(y[i] | </span></span>
<span class="line"><span>            eta[i],</span></span>
<span class="line"><span>            thresholds,</span></span>
<span class="line"><span>            discrimination[i],</span></span>
<span class="line"><span>            structure,</span></span>
<span class="line"><span>            link,</span></span>
<span class="line"><span>            to_vector(threshold_effect[i, :])</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_lpmf(</span></span>
<span class="line"><span>    int y,</span></span>
<span class="line"><span>    real eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    real discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    vector threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != k) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    int K = (k + 1);</span></span>
<span class="line"><span>    if((discrimination &lt;= 0.0)) {</span></span>
<span class="line"><span>        return negative_infinity();</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((y &lt; 1)) {</span></span>
<span class="line"><span>            return negative_infinity();</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            if((y &gt; K)) {</span></span>
<span class="line"><span>                return negative_infinity();</span></span>
<span class="line"><span>            } else {</span></span>
<span class="line"><span>                if((structure == 1)) {</span></span>
<span class="line"><span>                    if((link == 1)) {</span></span>
<span class="line"><span>                        return ordered_logistic_lpmf(y | (discrimination * eta), (discrimination .* thresholds));</span></span>
<span class="line"><span>                    } else {</span></span>
<span class="line"><span>                        if((y == 1)) {</span></span>
<span class="line"><span>                            real z_first = (discrimination * (thresholds[1] - eta));</span></span>
<span class="line"><span>                            return brm_ordinal_logcdf(z_first, link);</span></span>
<span class="line"><span>                        } else {</span></span>
<span class="line"><span>                            if((y == K)) {</span></span>
<span class="line"><span>                                real z_last = (discrimination * (thresholds[k] - eta));</span></span>
<span class="line"><span>                                return brm_ordinal_logccdf(z_last, link);</span></span>
<span class="line"><span>                            } else {</span></span>
<span class="line"><span>                                real z_hi = (discrimination * (thresholds[y] - eta));</span></span>
<span class="line"><span>                                real z_lo = (discrimination * (thresholds[(y - 1)] - eta));</span></span>
<span class="line"><span>                                return log_diff_exp(brm_ordinal_logcdf(z_hi, link), brm_ordinal_logcdf(z_lo, link));</span></span>
<span class="line"><span>                            }</span></span>
<span class="line"><span>                        }</span></span>
<span class="line"><span>                    }</span></span>
<span class="line"><span>                } else {</span></span>
<span class="line"><span>                    real rv = 0.0;</span></span>
<span class="line"><span>                    for(j in 1:k) {</span></span>
<span class="line"><span>                        real z_stage = (discrimination * ((thresholds[j] - eta) - threshold_effect[j]));</span></span>
<span class="line"><span>                        if((j &lt; y)) {</span></span>
<span class="line"><span>                            rv += brm_ordinal_logccdf(z_stage, link);</span></span>
<span class="line"><span>                        } else {</span></span>
<span class="line"><span>                            if((j == y)) {</span></span>
<span class="line"><span>                                rv += brm_ordinal_logcdf(z_stage, link);</span></span>
<span class="line"><span>                            }</span></span>
<span class="line"><span>                        }</span></span>
<span class="line"><span>                    }</span></span>
<span class="line"><span>                    return rv;</span></span>
<span class="line"><span>                }</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_logcdf(</span></span>
<span class="line"><span>    real z,</span></span>
<span class="line"><span>    int link</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    if((link == 1)) {</span></span>
<span class="line"><span>        return log_inv_logit(z);</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((link == 2)) {</span></span>
<span class="line"><span>            return normal_lcdf(z | 0.0, 1.0);</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            return log1m_exp((-exp(z)));</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_logccdf(</span></span>
<span class="line"><span>    real z,</span></span>
<span class="line"><span>    int link</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    if((link == 1)) {</span></span>
<span class="line"><span>        return log_inv_logit((-z));</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((link == 2)) {</span></span>
<span class="line"><span>            return normal_lccdf(z | 0.0, 1.0);</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            return (-exp(z));</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector brm_ordinal_lpmfs(</span></span>
<span class="line"><span>    array[] int y,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    real discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    return brm_ordinal_lpmfs(</span></span>
<span class="line"><span>        y,</span></span>
<span class="line"><span>        eta,</span></span>
<span class="line"><span>        thresholds,</span></span>
<span class="line"><span>        rep_vector(discrimination, n),</span></span>
<span class="line"><span>        structure,</span></span>
<span class="line"><span>        link,</span></span>
<span class="line"><span>        threshold_effect</span></span>
<span class="line"><span>    );</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector brm_ordinal_lpmfs(</span></span>
<span class="line"><span>    array[] int y,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    vector discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(discrimination)[1] != n) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`discrimination\` dim 1 (= &quot;, dims(discrimination)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = brm_ordinal_lpmf(y[i] | </span></span>
<span class="line"><span>            eta[i],</span></span>
<span class="line"><span>            thresholds,</span></span>
<span class="line"><span>            discrimination[i],</span></span>
<span class="line"><span>            structure,</span></span>
<span class="line"><span>            link,</span></span>
<span class="line"><span>            to_vector(threshold_effect[i, :])</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>array[] int brm_ordinal_int_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    real discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = anontok__1;</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_rng: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_rng: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_rng: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    return brm_ordinal_int_rng(</span></span>
<span class="line"><span>        n,</span></span>
<span class="line"><span>        eta,</span></span>
<span class="line"><span>        thresholds,</span></span>
<span class="line"><span>        rep_vector(discrimination, n),</span></span>
<span class="line"><span>        structure,</span></span>
<span class="line"><span>        link,</span></span>
<span class="line"><span>        threshold_effect</span></span>
<span class="line"><span>    );</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>array[] int brm_ordinal_int_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    vector discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = anontok__1;</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_rng: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(discrimination)[1] != n) reject(&quot;brm_ordinal_rng: dim mismatch — \`discrimination\` dim 1 (= &quot;, dims(discrimination)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_rng: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_rng: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    array[n] int rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = brm_ordinal_rng(</span></span>
<span class="line"><span>            eta[i],</span></span>
<span class="line"><span>            thresholds,</span></span>
<span class="line"><span>            discrimination[i],</span></span>
<span class="line"><span>            structure,</span></span>
<span class="line"><span>            link,</span></span>
<span class="line"><span>            to_vector(threshold_effect[i, :])</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>int brm_ordinal_rng(</span></span>
<span class="line"><span>    real eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    real discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    vector threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != k) reject(&quot;brm_ordinal_rng: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    int K = (k + 1);</span></span>
<span class="line"><span>    int rv = K;</span></span>
<span class="line"><span>    if((structure == 1)) {</span></span>
<span class="line"><span>        real u = uniform_rng(0.0, 1.0);</span></span>
<span class="line"><span>        for(j in 1:k) {</span></span>
<span class="line"><span>            if((rv == K)) {</span></span>
<span class="line"><span>                real z_cumulative = (discrimination * (thresholds[j] - eta));</span></span>
<span class="line"><span>                if((u &lt;= brm_ordinal_cdf(z_cumulative | link))) {</span></span>
<span class="line"><span>                    rv += (j - rv);</span></span>
<span class="line"><span>                }</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        for(j in 1:k) {</span></span>
<span class="line"><span>            if((rv == K)) {</span></span>
<span class="line"><span>                real z_stopping = (discrimination * ((thresholds[j] - eta) - threshold_effect[j]));</span></span>
<span class="line"><span>                if((bernoulli_rng(brm_ordinal_cdf(z_stopping | link)) == 1)) {</span></span>
<span class="line"><span>                    rv += (j - rv);</span></span>
<span class="line"><span>                }</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_cdf(</span></span>
<span class="line"><span>    real z,</span></span>
<span class="line"><span>    int link</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    if((link == 1)) {</span></span>
<span class="line"><span>        return inv_logit(z);</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((link == 2)) {</span></span>
<span class="line"><span>            return Phi(z);</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            return (-expm1((-exp(z))));</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>data {</span></span>
<span class="line"><span>    int x_n;</span></span>
<span class="line"><span>    vector[x_n] x;</span></span>
<span class="line"><span>    int y_n;</span></span>
<span class="line"><span>    array[y_n] int y;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed data {</span></span>
<span class="line"><span>    matrix[x_n, 1] X_eta = hcat(x);</span></span>
<span class="line"><span>    int pop_eta_n_covariates = 1;</span></span>
<span class="line"><span>    matrix[num_elements(y), 2] y_threshold_effect = rep_matrix(0.0, num_elements(y), 2);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>parameters {</span></span>
<span class="line"><span>    vector[pop_eta_n_covariates] pop_eta_beta_pop;</span></span>
<span class="line"><span>    ordered[2] y_thresholds;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed parameters {</span></span>
<span class="line"><span>    vector[x_n] pop_eta = (X_eta * pop_eta_beta_pop);</span></span>
<span class="line"><span>    vector[x_n] eta = pop_eta;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>model {</span></span>
<span class="line"><span>    pop_eta_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    y_thresholds ~ std_normal();</span></span>
<span class="line"><span>    y ~ brm_ordinal(eta, y_thresholds, 1.0, 1, 2, y_threshold_effect);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>generated quantities {</span></span>
<span class="line"><span>    vector[num_elements(y)] y_likelihood = brm_ordinal_lpmfs(y, eta, y_thresholds, 1.0, 1, 2, y_threshold_effect);</span></span>
<span class="line"><span>    array[num_elements(y)] int y_gen = brm_ordinal_int_rng(y_n, eta, y_thresholds, 1.0, 1, 2, y_threshold_effect);</span></span>
<span class="line"><span>}</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing unsupported </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> this BRM example</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> executable families are </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Bernoulli(p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BernoulliLogit(eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Binomial(trials, p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BinomialLogit(trials, eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Poisson(exp(log_rate))\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`NegativeBinomial2(mu, phi)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`BetaBinomial2(trials, mean, precision)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and bounded </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; got </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Ordinal\`</span></span></code></pre></div><p>The accepted structures are <code>Cumulative()</code> and <code>StoppingRatio()</code>. Each composes with <code>LogitLink()</code>, <code>ProbitLink()</code>, or <code>CloglogLink()</code>. This is intentionally a Julia-native typed surface: BRM does not copy R formula helper names or encode every structure/link pair in a new family type.</p><p>For <code>Cumulative()</code>, BRM estimates strictly ordered thresholds $c_1 &lt; \\cdots &lt; c_{K-1}$ and uses</p><p>$$P(Y \\le k) = F!\\left(d(c_k - \\eta)\\right).$$</p><p>For <code>StoppingRatio()</code>, the estimated stage intercepts need not be ordered and</p><p>$$q_k = P(Y=k \\mid Y\\ge k) = F!\\left(d(c_k - \\eta_k)\\right),\\qquad P(Y=k)=q_k\\prod_{j&lt;k}(1-q_j),$$</p><p>with the final category equal to the probability of continuing through every stage. <code>F</code> is logistic, standard normal, or complementary-log-log according to the link tag. Both threshold vectors currently receive element-wise standard normal priors.</p><p>The thresholds already supply the model location, so the composed surface requires an intercept-free common predictor (<code>eta ~ 0 + ...</code>). A positive discrimination parameter is explicit:</p><div class="language-brm-comparison vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">brm-comparison</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Ordinal discrimination model</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ordinal_disc_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    group</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], y</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">ordinal_disc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @brm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> ordinal_disc_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    eta </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(disc) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> group</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Ordinal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Cumulative</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ProbitLink</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), eta;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                discrimination</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">disc)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">SBBRMI with data keys </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:group</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:group_idx</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:group_n_levels</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">emitted </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@slic</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> body</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_eta </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_eta </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_eta)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    eta </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_eta</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cat_group </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> _sb_cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> group_idx, n_levels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> group_n_levels)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    log_disc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cat_group</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    disc </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> exp</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(log_disc)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_thresholds</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ordered</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> std_normal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_threshold_effect </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> rep_matrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> brm_ordinal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(eta, y_thresholds, disc, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, y_threshold_effect)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-stan vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">stan</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>functions {</span></span>
<span class="line"><span>matrix hcat(vector x) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    return to_matrix(x, n, 1);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_lpmf(</span></span>
<span class="line"><span>    array[] int y,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    vector discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(discrimination)[1] != n) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`discrimination\` dim 1 (= &quot;, dims(discrimination)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    real rv = 0.0;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv += brm_ordinal_lpmf(y[i] | </span></span>
<span class="line"><span>            eta[i],</span></span>
<span class="line"><span>            thresholds,</span></span>
<span class="line"><span>            discrimination[i],</span></span>
<span class="line"><span>            structure,</span></span>
<span class="line"><span>            link,</span></span>
<span class="line"><span>            to_vector(threshold_effect[i, :])</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_lpmf(</span></span>
<span class="line"><span>    int y,</span></span>
<span class="line"><span>    real eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    real discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    vector threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != k) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    int K = (k + 1);</span></span>
<span class="line"><span>    if((discrimination &lt;= 0.0)) {</span></span>
<span class="line"><span>        return negative_infinity();</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((y &lt; 1)) {</span></span>
<span class="line"><span>            return negative_infinity();</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            if((y &gt; K)) {</span></span>
<span class="line"><span>                return negative_infinity();</span></span>
<span class="line"><span>            } else {</span></span>
<span class="line"><span>                if((structure == 1)) {</span></span>
<span class="line"><span>                    if((link == 1)) {</span></span>
<span class="line"><span>                        return ordered_logistic_lpmf(y | (discrimination * eta), (discrimination .* thresholds));</span></span>
<span class="line"><span>                    } else {</span></span>
<span class="line"><span>                        if((y == 1)) {</span></span>
<span class="line"><span>                            real z_first = (discrimination * (thresholds[1] - eta));</span></span>
<span class="line"><span>                            return brm_ordinal_logcdf(z_first, link);</span></span>
<span class="line"><span>                        } else {</span></span>
<span class="line"><span>                            if((y == K)) {</span></span>
<span class="line"><span>                                real z_last = (discrimination * (thresholds[k] - eta));</span></span>
<span class="line"><span>                                return brm_ordinal_logccdf(z_last, link);</span></span>
<span class="line"><span>                            } else {</span></span>
<span class="line"><span>                                real z_hi = (discrimination * (thresholds[y] - eta));</span></span>
<span class="line"><span>                                real z_lo = (discrimination * (thresholds[(y - 1)] - eta));</span></span>
<span class="line"><span>                                return log_diff_exp(brm_ordinal_logcdf(z_hi, link), brm_ordinal_logcdf(z_lo, link));</span></span>
<span class="line"><span>                            }</span></span>
<span class="line"><span>                        }</span></span>
<span class="line"><span>                    }</span></span>
<span class="line"><span>                } else {</span></span>
<span class="line"><span>                    real rv = 0.0;</span></span>
<span class="line"><span>                    for(j in 1:k) {</span></span>
<span class="line"><span>                        real z_stage = (discrimination * ((thresholds[j] - eta) - threshold_effect[j]));</span></span>
<span class="line"><span>                        if((j &lt; y)) {</span></span>
<span class="line"><span>                            rv += brm_ordinal_logccdf(z_stage, link);</span></span>
<span class="line"><span>                        } else {</span></span>
<span class="line"><span>                            if((j == y)) {</span></span>
<span class="line"><span>                                rv += brm_ordinal_logcdf(z_stage, link);</span></span>
<span class="line"><span>                            }</span></span>
<span class="line"><span>                        }</span></span>
<span class="line"><span>                    }</span></span>
<span class="line"><span>                    return rv;</span></span>
<span class="line"><span>                }</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_logcdf(</span></span>
<span class="line"><span>    real z,</span></span>
<span class="line"><span>    int link</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    if((link == 1)) {</span></span>
<span class="line"><span>        return log_inv_logit(z);</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((link == 2)) {</span></span>
<span class="line"><span>            return normal_lcdf(z | 0.0, 1.0);</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            return log1m_exp((-exp(z)));</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_logccdf(</span></span>
<span class="line"><span>    real z,</span></span>
<span class="line"><span>    int link</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    if((link == 1)) {</span></span>
<span class="line"><span>        return log_inv_logit((-z));</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((link == 2)) {</span></span>
<span class="line"><span>            return normal_lccdf(z | 0.0, 1.0);</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            return (-exp(z));</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector brm_ordinal_lpmfs(</span></span>
<span class="line"><span>    array[] int y,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    vector discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(discrimination)[1] != n) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`discrimination\` dim 1 (= &quot;, dims(discrimination)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = brm_ordinal_lpmf(y[i] | </span></span>
<span class="line"><span>            eta[i],</span></span>
<span class="line"><span>            thresholds,</span></span>
<span class="line"><span>            discrimination[i],</span></span>
<span class="line"><span>            structure,</span></span>
<span class="line"><span>            link,</span></span>
<span class="line"><span>            to_vector(threshold_effect[i, :])</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>array[] int brm_ordinal_int_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    vector discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = anontok__1;</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_rng: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(discrimination)[1] != n) reject(&quot;brm_ordinal_rng: dim mismatch — \`discrimination\` dim 1 (= &quot;, dims(discrimination)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_rng: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_rng: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    array[n] int rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = brm_ordinal_rng(</span></span>
<span class="line"><span>            eta[i],</span></span>
<span class="line"><span>            thresholds,</span></span>
<span class="line"><span>            discrimination[i],</span></span>
<span class="line"><span>            structure,</span></span>
<span class="line"><span>            link,</span></span>
<span class="line"><span>            to_vector(threshold_effect[i, :])</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>int brm_ordinal_rng(</span></span>
<span class="line"><span>    real eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    real discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    vector threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != k) reject(&quot;brm_ordinal_rng: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    int K = (k + 1);</span></span>
<span class="line"><span>    int rv = K;</span></span>
<span class="line"><span>    if((structure == 1)) {</span></span>
<span class="line"><span>        real u = uniform_rng(0.0, 1.0);</span></span>
<span class="line"><span>        for(j in 1:k) {</span></span>
<span class="line"><span>            if((rv == K)) {</span></span>
<span class="line"><span>                real z_cumulative = (discrimination * (thresholds[j] - eta));</span></span>
<span class="line"><span>                if((u &lt;= brm_ordinal_cdf(z_cumulative | link))) {</span></span>
<span class="line"><span>                    rv += (j - rv);</span></span>
<span class="line"><span>                }</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        for(j in 1:k) {</span></span>
<span class="line"><span>            if((rv == K)) {</span></span>
<span class="line"><span>                real z_stopping = (discrimination * ((thresholds[j] - eta) - threshold_effect[j]));</span></span>
<span class="line"><span>                if((bernoulli_rng(brm_ordinal_cdf(z_stopping | link)) == 1)) {</span></span>
<span class="line"><span>                    rv += (j - rv);</span></span>
<span class="line"><span>                }</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_cdf(</span></span>
<span class="line"><span>    real z,</span></span>
<span class="line"><span>    int link</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    if((link == 1)) {</span></span>
<span class="line"><span>        return inv_logit(z);</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((link == 2)) {</span></span>
<span class="line"><span>            return Phi(z);</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            return (-expm1((-exp(z))));</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>data {</span></span>
<span class="line"><span>    int x_n;</span></span>
<span class="line"><span>    vector[x_n] x;</span></span>
<span class="line"><span>    int group_n_levels;</span></span>
<span class="line"><span>    int group_idx_n;</span></span>
<span class="line"><span>    array[group_idx_n] int group_idx;</span></span>
<span class="line"><span>    int y_n;</span></span>
<span class="line"><span>    array[y_n] int y;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed data {</span></span>
<span class="line"><span>    matrix[x_n, 1] X_eta = hcat(x);</span></span>
<span class="line"><span>    int pop_eta_n_covariates = 1;</span></span>
<span class="line"><span>    matrix[num_elements(y), 2] y_threshold_effect = rep_matrix(0.0, num_elements(y), 2);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>parameters {</span></span>
<span class="line"><span>    vector[pop_eta_n_covariates] pop_eta_beta_pop;</span></span>
<span class="line"><span>    vector[(group_n_levels - 1)] cat_group_beta;</span></span>
<span class="line"><span>    ordered[2] y_thresholds;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed parameters {</span></span>
<span class="line"><span>    vector[x_n] pop_eta = (X_eta * pop_eta_beta_pop);</span></span>
<span class="line"><span>    vector[x_n] eta = pop_eta;</span></span>
<span class="line"><span>    vector[group_idx_n] cat_group = append_row(0.0, cat_group_beta)[group_idx];</span></span>
<span class="line"><span>    vector[group_idx_n] log_disc = cat_group;</span></span>
<span class="line"><span>    vector[group_idx_n] disc = exp(log_disc);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>model {</span></span>
<span class="line"><span>    pop_eta_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    cat_group_beta ~ std_normal();</span></span>
<span class="line"><span>    y_thresholds ~ std_normal();</span></span>
<span class="line"><span>    y ~ brm_ordinal(eta, y_thresholds, disc, 1, 2, y_threshold_effect);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>generated quantities {</span></span>
<span class="line"><span>    vector[num_elements(y)] y_likelihood = brm_ordinal_lpmfs(y, eta, y_thresholds, disc, 1, 2, y_threshold_effect);</span></span>
<span class="line"><span>    array[num_elements(y)] int y_gen = brm_ordinal_int_rng(y_n, eta, y_thresholds, disc, 1, 2, y_threshold_effect);</span></span>
<span class="line"><span>}</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing unsupported </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> this BRM example</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> executable families are </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Bernoulli(p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BernoulliLogit(eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Binomial(trials, p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BinomialLogit(trials, eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Poisson(exp(log_rate))\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`NegativeBinomial2(mu, phi)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`BetaBinomial2(trials, mean, precision)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and bounded </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; got </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Ordinal\`</span></span></code></pre></div><p><code>discrimination</code> defaults to one. Literal or data-supplied values are checked for finiteness and strict positivity; a modeled value should use a positive-support prior or a link such as <code>log(disc) ~ ...</code>.</p><p>Stopping-ratio models may add non-proportional effects with a tuple of raw numeric predictors:</p><div class="language-brm-comparison vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">brm-comparison</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Sequential ordinal model</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequential_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    period</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], carry</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    treat</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], y</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">sequential </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @brm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> sequential_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    eta </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 0</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> period </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> carry</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Ordinal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">StoppingRatio</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">CloglogLink</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(), eta;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">                per_threshold</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(treat,))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">SBBRMI with data keys </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:carry</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:carry_idx</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:carry_n_levels</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:period</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:period_idx</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:period_n_levels</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:treat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">emitted </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@slic</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> body</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cat_period </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> _sb_cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> period_idx, n_levels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> period_n_levels)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    cat_carry </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> _sb_cat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> carry_idx, n_levels </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> carry_n_levels)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    eta </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cat_period </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">+</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> cat_carry</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_thresholds</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> std_normal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_threshold_X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(treat)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_threshold_beta</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">] </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> multi_std_normal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">()</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y_threshold_effect </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_threshold_X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> adjoint</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">ranef_b_matrix</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y_threshold_beta))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> brm_ordinal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(eta, y_thresholds, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, y_threshold_effect)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-stan vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">stan</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>functions {</span></span>
<span class="line"><span>matrix hcat(vector x) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    return to_matrix(x, n, 1);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real multi_std_normal_lpdf(</span></span>
<span class="line"><span>    array[] vector x</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int m = dims(x)[1];</span></span>
<span class="line"><span>    real rv = 0.0;</span></span>
<span class="line"><span>    for(i in 1:m) {</span></span>
<span class="line"><span>        rv += std_normal_lpdf(x[i, :]);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>matrix ranef_b_matrix(</span></span>
<span class="line"><span>    array[] vector b</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int m = dims(b)[1];</span></span>
<span class="line"><span>    int n = dims(b)[2];</span></span>
<span class="line"><span>    matrix[m, n] rv;</span></span>
<span class="line"><span>    for(i in 1:m) {</span></span>
<span class="line"><span>        rv[i, :] = (b[i]&#39;);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_lpmf(</span></span>
<span class="line"><span>    array[] int y,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    real discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    return brm_ordinal_lpmf(y | </span></span>
<span class="line"><span>        eta,</span></span>
<span class="line"><span>        thresholds,</span></span>
<span class="line"><span>        rep_vector(discrimination, n),</span></span>
<span class="line"><span>        structure,</span></span>
<span class="line"><span>        link,</span></span>
<span class="line"><span>        threshold_effect</span></span>
<span class="line"><span>    );</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_lpmf(</span></span>
<span class="line"><span>    array[] int y,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    vector discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(discrimination)[1] != n) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`discrimination\` dim 1 (= &quot;, dims(discrimination)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    real rv = 0.0;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv += brm_ordinal_lpmf(y[i] | </span></span>
<span class="line"><span>            eta[i],</span></span>
<span class="line"><span>            thresholds,</span></span>
<span class="line"><span>            discrimination[i],</span></span>
<span class="line"><span>            structure,</span></span>
<span class="line"><span>            link,</span></span>
<span class="line"><span>            to_vector(threshold_effect[i, :])</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_lpmf(</span></span>
<span class="line"><span>    int y,</span></span>
<span class="line"><span>    real eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    real discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    vector threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != k) reject(&quot;brm_ordinal_lpmf: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    int K = (k + 1);</span></span>
<span class="line"><span>    if((discrimination &lt;= 0.0)) {</span></span>
<span class="line"><span>        return negative_infinity();</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((y &lt; 1)) {</span></span>
<span class="line"><span>            return negative_infinity();</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            if((y &gt; K)) {</span></span>
<span class="line"><span>                return negative_infinity();</span></span>
<span class="line"><span>            } else {</span></span>
<span class="line"><span>                if((structure == 1)) {</span></span>
<span class="line"><span>                    if((link == 1)) {</span></span>
<span class="line"><span>                        return ordered_logistic_lpmf(y | (discrimination * eta), (discrimination .* thresholds));</span></span>
<span class="line"><span>                    } else {</span></span>
<span class="line"><span>                        if((y == 1)) {</span></span>
<span class="line"><span>                            real z_first = (discrimination * (thresholds[1] - eta));</span></span>
<span class="line"><span>                            return brm_ordinal_logcdf(z_first, link);</span></span>
<span class="line"><span>                        } else {</span></span>
<span class="line"><span>                            if((y == K)) {</span></span>
<span class="line"><span>                                real z_last = (discrimination * (thresholds[k] - eta));</span></span>
<span class="line"><span>                                return brm_ordinal_logccdf(z_last, link);</span></span>
<span class="line"><span>                            } else {</span></span>
<span class="line"><span>                                real z_hi = (discrimination * (thresholds[y] - eta));</span></span>
<span class="line"><span>                                real z_lo = (discrimination * (thresholds[(y - 1)] - eta));</span></span>
<span class="line"><span>                                return log_diff_exp(brm_ordinal_logcdf(z_hi, link), brm_ordinal_logcdf(z_lo, link));</span></span>
<span class="line"><span>                            }</span></span>
<span class="line"><span>                        }</span></span>
<span class="line"><span>                    }</span></span>
<span class="line"><span>                } else {</span></span>
<span class="line"><span>                    real rv = 0.0;</span></span>
<span class="line"><span>                    for(j in 1:k) {</span></span>
<span class="line"><span>                        real z_stage = (discrimination * ((thresholds[j] - eta) - threshold_effect[j]));</span></span>
<span class="line"><span>                        if((j &lt; y)) {</span></span>
<span class="line"><span>                            rv += brm_ordinal_logccdf(z_stage, link);</span></span>
<span class="line"><span>                        } else {</span></span>
<span class="line"><span>                            if((j == y)) {</span></span>
<span class="line"><span>                                rv += brm_ordinal_logcdf(z_stage, link);</span></span>
<span class="line"><span>                            }</span></span>
<span class="line"><span>                        }</span></span>
<span class="line"><span>                    }</span></span>
<span class="line"><span>                    return rv;</span></span>
<span class="line"><span>                }</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_logcdf(</span></span>
<span class="line"><span>    real z,</span></span>
<span class="line"><span>    int link</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    if((link == 1)) {</span></span>
<span class="line"><span>        return log_inv_logit(z);</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((link == 2)) {</span></span>
<span class="line"><span>            return normal_lcdf(z | 0.0, 1.0);</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            return log1m_exp((-exp(z)));</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_logccdf(</span></span>
<span class="line"><span>    real z,</span></span>
<span class="line"><span>    int link</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    if((link == 1)) {</span></span>
<span class="line"><span>        return log_inv_logit((-z));</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((link == 2)) {</span></span>
<span class="line"><span>            return normal_lccdf(z | 0.0, 1.0);</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            return (-exp(z));</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector brm_ordinal_lpmfs(</span></span>
<span class="line"><span>    array[] int y,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    real discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    return brm_ordinal_lpmfs(</span></span>
<span class="line"><span>        y,</span></span>
<span class="line"><span>        eta,</span></span>
<span class="line"><span>        thresholds,</span></span>
<span class="line"><span>        rep_vector(discrimination, n),</span></span>
<span class="line"><span>        structure,</span></span>
<span class="line"><span>        link,</span></span>
<span class="line"><span>        threshold_effect</span></span>
<span class="line"><span>    );</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector brm_ordinal_lpmfs(</span></span>
<span class="line"><span>    array[] int y,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    vector discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(discrimination)[1] != n) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`discrimination\` dim 1 (= &quot;, dims(discrimination)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_lpmfs: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = brm_ordinal_lpmf(y[i] | </span></span>
<span class="line"><span>            eta[i],</span></span>
<span class="line"><span>            thresholds,</span></span>
<span class="line"><span>            discrimination[i],</span></span>
<span class="line"><span>            structure,</span></span>
<span class="line"><span>            link,</span></span>
<span class="line"><span>            to_vector(threshold_effect[i, :])</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>array[] int brm_ordinal_int_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    real discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = anontok__1;</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_rng: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_rng: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_rng: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    return brm_ordinal_int_rng(</span></span>
<span class="line"><span>        n,</span></span>
<span class="line"><span>        eta,</span></span>
<span class="line"><span>        thresholds,</span></span>
<span class="line"><span>        rep_vector(discrimination, n),</span></span>
<span class="line"><span>        structure,</span></span>
<span class="line"><span>        link,</span></span>
<span class="line"><span>        threshold_effect</span></span>
<span class="line"><span>    );</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>array[] int brm_ordinal_int_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    vector eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    vector discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    matrix threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = anontok__1;</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(eta)[1] != n) reject(&quot;brm_ordinal_rng: dim mismatch — \`eta\` dim 1 (= &quot;, dims(eta)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(discrimination)[1] != n) reject(&quot;brm_ordinal_rng: dim mismatch — \`discrimination\` dim 1 (= &quot;, dims(discrimination)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != n) reject(&quot;brm_ordinal_rng: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(threshold_effect)[2] != k) reject(&quot;brm_ordinal_rng: dim mismatch — \`threshold_effect\` dim 2 (= &quot;, dims(threshold_effect)[2], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    array[n] int rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = brm_ordinal_rng(</span></span>
<span class="line"><span>            eta[i],</span></span>
<span class="line"><span>            thresholds,</span></span>
<span class="line"><span>            discrimination[i],</span></span>
<span class="line"><span>            structure,</span></span>
<span class="line"><span>            link,</span></span>
<span class="line"><span>            to_vector(threshold_effect[i, :])</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>int brm_ordinal_rng(</span></span>
<span class="line"><span>    real eta,</span></span>
<span class="line"><span>    vector thresholds,</span></span>
<span class="line"><span>    real discrimination,</span></span>
<span class="line"><span>    int structure,</span></span>
<span class="line"><span>    int link,</span></span>
<span class="line"><span>    vector threshold_effect</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int k = dims(thresholds)[1];</span></span>
<span class="line"><span>    if (dims(threshold_effect)[1] != k) reject(&quot;brm_ordinal_rng: dim mismatch — \`threshold_effect\` dim 1 (= &quot;, dims(threshold_effect)[1], &quot;) does not match \`k\` (= &quot;, k, &quot;)&quot;);</span></span>
<span class="line"><span>    int K = (k + 1);</span></span>
<span class="line"><span>    int rv = K;</span></span>
<span class="line"><span>    if((structure == 1)) {</span></span>
<span class="line"><span>        real u = uniform_rng(0.0, 1.0);</span></span>
<span class="line"><span>        for(j in 1:k) {</span></span>
<span class="line"><span>            if((rv == K)) {</span></span>
<span class="line"><span>                real z_cumulative = (discrimination * (thresholds[j] - eta));</span></span>
<span class="line"><span>                if((u &lt;= brm_ordinal_cdf(z_cumulative | link))) {</span></span>
<span class="line"><span>                    rv += (j - rv);</span></span>
<span class="line"><span>                }</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        for(j in 1:k) {</span></span>
<span class="line"><span>            if((rv == K)) {</span></span>
<span class="line"><span>                real z_stopping = (discrimination * ((thresholds[j] - eta) - threshold_effect[j]));</span></span>
<span class="line"><span>                if((bernoulli_rng(brm_ordinal_cdf(z_stopping | link)) == 1)) {</span></span>
<span class="line"><span>                    rv += (j - rv);</span></span>
<span class="line"><span>                }</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_ordinal_cdf(</span></span>
<span class="line"><span>    real z,</span></span>
<span class="line"><span>    int link</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    if((link == 1)) {</span></span>
<span class="line"><span>        return inv_logit(z);</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((link == 2)) {</span></span>
<span class="line"><span>            return Phi(z);</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            return (-expm1((-exp(z))));</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>data {</span></span>
<span class="line"><span>    int period_n_levels;</span></span>
<span class="line"><span>    int period_idx_n;</span></span>
<span class="line"><span>    array[period_idx_n] int period_idx;</span></span>
<span class="line"><span>    int carry_n_levels;</span></span>
<span class="line"><span>    int carry_idx_n;</span></span>
<span class="line"><span>    array[carry_idx_n] int carry_idx;</span></span>
<span class="line"><span>    int treat_n;</span></span>
<span class="line"><span>    vector[treat_n] treat;</span></span>
<span class="line"><span>    int y_n;</span></span>
<span class="line"><span>    array[y_n] int y;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed data {</span></span>
<span class="line"><span>    matrix[treat_n, 1] y_threshold_X = hcat(treat);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>parameters {</span></span>
<span class="line"><span>    vector[(period_n_levels - 1)] cat_period_beta;</span></span>
<span class="line"><span>    vector[(carry_n_levels - 1)] cat_carry_beta;</span></span>
<span class="line"><span>    vector[2] y_thresholds;</span></span>
<span class="line"><span>    array[2] vector[1] y_threshold_beta;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed parameters {</span></span>
<span class="line"><span>    vector[period_idx_n] cat_period = append_row(0.0, cat_period_beta)[period_idx];</span></span>
<span class="line"><span>    vector[carry_idx_n] cat_carry = append_row(0.0, cat_carry_beta)[carry_idx];</span></span>
<span class="line"><span>    vector[period_idx_n] eta = (cat_period + cat_carry);</span></span>
<span class="line"><span>    matrix[treat_n, 2] y_threshold_effect = (y_threshold_X * (ranef_b_matrix(y_threshold_beta)&#39;));</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>model {</span></span>
<span class="line"><span>    cat_period_beta ~ std_normal();</span></span>
<span class="line"><span>    cat_carry_beta ~ std_normal();</span></span>
<span class="line"><span>    y_thresholds ~ std_normal();</span></span>
<span class="line"><span>    y_threshold_beta ~ multi_std_normal();</span></span>
<span class="line"><span>    y ~ brm_ordinal(eta, y_thresholds, 1.0, 2, 3, y_threshold_effect);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>generated quantities {</span></span>
<span class="line"><span>    vector[treat_n] y_likelihood = brm_ordinal_lpmfs(y, eta, y_thresholds, 1.0, 2, 3, y_threshold_effect);</span></span>
<span class="line"><span>    array[treat_n] int y_gen = brm_ordinal_int_rng(y_n, eta, y_thresholds, 1.0, 2, 3, y_threshold_effect);</span></span>
<span class="line"><span>}</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing unsupported </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> this BRM example</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> executable families are </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Bernoulli(p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BernoulliLogit(eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Binomial(trials, p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BinomialLogit(trials, eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Poisson(exp(log_rate))\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`NegativeBinomial2(mu, phi)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`BetaBinomial2(trials, mean, precision)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and bounded </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; got </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Ordinal\`</span></span></code></pre></div><p>BRM estimates one coefficient per predictor and non-terminal stage, so here $\\eta_k = \\eta + \\mathtt{treat},\\beta_k$. <code>per_threshold</code> is deliberately restricted to stopping-ratio models for now: unrestricted cumulative category-specific effects can make cumulative probabilities non-monotone. The predictors must currently be raw numeric data columns.</p><p>Outcome categories follow the declared order of a <code>CategoricalVector</code>; plain vectors use sorted unique values. That fitted order is frozen for replay and prediction. The legacy <code>OrderedLogistic(eta)</code> spelling remains supported and continues to lower directly to Stan&#39;s native ordered-logistic distribution. The composed cumulative-logit kernel also delegates its scalar density to that native primitive; the other links use Stan&#39;s native stable CDF/log-CDF functions. Stopping ratio has no native Stan distribution, so BRM supplies the matching stable lpmf, pointwise log-likelihood, and RNG.</p><p>Outside <code>@brm</code>, <code>Ordinal(structure, link, eta, thresholds; discrimination=1)</code> is an executable <code>DiscreteUnivariateDistribution</code> with <code>params</code>, <code>probs</code>, <code>logpdf</code>, and <code>rand</code>. A stopping-ratio <code>eta</code> may be scalar or a vector with one stage-specific value per threshold.</p><p>For neutral comparison, brms exposes the same statistical axes through families such as cumulative-probit and stopping-ratio complementary-log-log, and calls threshold-varying terms category-specific effects. BRM preserves that statistical contract while using the typed composition and <code>per_threshold=(...)</code> tuple above rather than importing brms&#39;s R formula helpers.</p><h2 id="Median-regression-with-Laplace" tabindex="-1">Median regression with <code>Laplace</code> <a class="header-anchor" href="#Median-regression-with-Laplace" aria-label="Permalink to &quot;Median regression with \`Laplace\` {#Median-regression-with-Laplace}&quot;">​</a></h2><p>The StanBlocks backend accepts <code>Distributions.Laplace</code> as an ordinary likelihood. For example, a robust regression for the conditional median can be written as:</p><div class="language-brm-comparison vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">brm-comparison</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Median regression with Laplace</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">laplace_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">median_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @brm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> laplace_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    median_y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(laplace_scale) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> Laplace</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(median_y, laplace_scale)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">SBBRMI with data keys </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">emitted </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@slic</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> body</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_median_y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)), x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_median_y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_median_y)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    median_y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_median_y</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_log_laplace_scale </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_log_laplace_scale </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_log_laplace_scale)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    log_laplace_scale </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_log_laplace_scale</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    laplace_scale </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> exp</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(log_laplace_scale)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> double_exponential</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(median_y, laplace_scale)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-stan vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">stan</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>functions {</span></span>
<span class="line"><span>matrix hcat(</span></span>
<span class="line"><span>    vector x,</span></span>
<span class="line"><span>    vector y</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    if (dims(y)[1] != n) reject(&quot;hcat: dim mismatch — \`y\` dim 1 (= &quot;, dims(y)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    return append_col(x, y);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>matrix hcat(vector x) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    return to_matrix(x, n, 1);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector double_exponential_lpdfs(</span></span>
<span class="line"><span>    vector obs,</span></span>
<span class="line"><span>    vector mu,</span></span>
<span class="line"><span>    vector sigma</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    return jbroadcasted_double_exponential_lpdfs(obs, mu, sigma);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector jbroadcasted_double_exponential_lpdfs(</span></span>
<span class="line"><span>    vector x1,</span></span>
<span class="line"><span>    vector x2,</span></span>
<span class="line"><span>    vector x3</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(x1)[1];</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = double_exponential_lpdfs(</span></span>
<span class="line"><span>            broadcasted_getindex(x1, i),</span></span>
<span class="line"><span>            broadcasted_getindex(x2, i),</span></span>
<span class="line"><span>            broadcasted_getindex(x3, i)</span></span>
<span class="line"><span>        );</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real double_exponential_lpdfs(</span></span>
<span class="line"><span>    real args1,</span></span>
<span class="line"><span>    real args2,</span></span>
<span class="line"><span>    real args3</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    return double_exponential_lpdf(args1 | args2, args3);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real broadcasted_getindex(vector x, int i) {</span></span>
<span class="line"><span>    return x[i];</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector double_exponential_vector_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    vector a,</span></span>
<span class="line"><span>    vector b</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = anontok__1;</span></span>
<span class="line"><span>    return to_vector(double_exponential_rng(a, b));</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>data {</span></span>
<span class="line"><span>    int x_n;</span></span>
<span class="line"><span>    vector[x_n] x;</span></span>
<span class="line"><span>    int y_n;</span></span>
<span class="line"><span>    vector[y_n] y;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed data {</span></span>
<span class="line"><span>    matrix[x_n, 2] X_median_y = hcat(rep_vector(1.0, num_elements(x)), x);</span></span>
<span class="line"><span>    int pop_median_y_n_covariates = 2;</span></span>
<span class="line"><span>    matrix[num_elements(y), 1] X_log_laplace_scale = hcat(rep_vector(1.0, num_elements(y)));</span></span>
<span class="line"><span>    int pop_log_laplace_scale_n_covariates = 1;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>parameters {</span></span>
<span class="line"><span>    vector[pop_median_y_n_covariates] pop_median_y_beta_pop;</span></span>
<span class="line"><span>    vector[pop_log_laplace_scale_n_covariates] pop_log_laplace_scale_beta_pop;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed parameters {</span></span>
<span class="line"><span>    vector[x_n] pop_median_y = (X_median_y * pop_median_y_beta_pop);</span></span>
<span class="line"><span>    vector[x_n] median_y = pop_median_y;</span></span>
<span class="line"><span>    vector[num_elements(y)] pop_log_laplace_scale = (X_log_laplace_scale * pop_log_laplace_scale_beta_pop);</span></span>
<span class="line"><span>    vector[num_elements(y)] log_laplace_scale = pop_log_laplace_scale;</span></span>
<span class="line"><span>    vector[num_elements(y)] laplace_scale = exp(log_laplace_scale);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>model {</span></span>
<span class="line"><span>    pop_median_y_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    pop_log_laplace_scale_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    y ~ double_exponential(median_y, laplace_scale);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>generated quantities {</span></span>
<span class="line"><span>    vector[y_n] y_likelihood = double_exponential_lpdfs(y, median_y, laplace_scale);</span></span>
<span class="line"><span>    vector[y_n] y_gen = double_exponential_vector_rng(y_n, median_y, laplace_scale);</span></span>
<span class="line"><span>}</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing unsupported </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> this BRM example</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> executable families are </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Bernoulli(p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BernoulliLogit(eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Binomial(trials, p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BinomialLogit(trials, eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Poisson(exp(log_rate))\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`NegativeBinomial2(mu, phi)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`BetaBinomial2(trials, mean, precision)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and bounded </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; got </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Distributions.Laplace\`</span></span></code></pre></div><p>This lowers to Stan&#39;s native <code>y ~ double_exponential(median_y, laplace_scale)</code>. The second argument is a <strong>Laplace scale</strong>, not a standard deviation or rate:</p><p>$$f(y \\mid \\mu, \\theta) = \\frac{1}{2\\theta}\\exp!\\left(-\\frac{|y-\\mu|}{\\theta}\\right), \\qquad \\theta = \\mathtt{laplace_scale}.$$</p><p>This symmetric likelihood is exactly the $q = 0.5$ special case of the asymmetric-Laplace likelihood used for quantile regression, after accounting for parameterization:</p><ul><li><p>In the check-loss convention used by <code>brms</code>, with scale $s$ and density $q(1-q)s^{-1}\\exp[-\\rho_q((y-\\mu)/s)]$, use $\\theta = 2s$ at $q = 0.5$.</p></li><li><p>In the Bambi/PyMC convention <code>AsymmetricLaplace(mu, b, kappa)</code>, where $\\kappa = \\sqrt{q/(1-q)}$, use $\\kappa = 1$ and $\\theta = 1/b$ at $q = 0.5$.</p></li></ul><p>The <code>Laplace</code> spelling covers median regression only. It does <strong>not</strong> express an asymmetric likelihood for $q \\ne 0.5$. It also does not translate Bambi&#39;s historical <code>bs(age, knots=...)</code> term: that basis mapping is a separate unresolved formula-semantic question. Thus the response-family component of the catalogue&#39;s <code>quantile_p50</code> model is available, while the complete historical model remains unsupported.</p><h2 id="Quantile-regression-with-SkewDoubleExponential" tabindex="-1">Quantile regression with <code>SkewDoubleExponential</code> <a class="header-anchor" href="#Quantile-regression-with-SkewDoubleExponential" aria-label="Permalink to &quot;Quantile regression with \`SkewDoubleExponential\` {#Quantile-regression-with-SkewDoubleExponential}&quot;">​</a></h2><p>For a non-median quantile, BRM exposes the executable distribution <code>SkewDoubleExponential(mu, sigma, tau)</code>. Its arguments and scale exactly match Stan&#39;s native <code>skew_double_exponential</code> family:</p><p>$$f(y \\mid \\mu, \\sigma, \\tau) = \\frac{2\\tau(1-\\tau)}{\\sigma} \\exp!\\left[-\\frac{2}{\\sigma} \\left((1-\\tau)\\mathbf{1}<em>{y&lt;\\mu}(\\mu-y) +\\tau\\mathbf{1}</em>{y&gt;\\mu}(y-\\mu)\\right)\\right].$$</p><p>Thus <code>cdf(SkewDoubleExponential(mu, sigma, tau), mu) == tau</code>, and <code>SkewDoubleExponential(mu, sigma, 0.5)</code> is exactly <code>Laplace(mu, sigma)</code>. There is no hidden brms-scale conversion on this primary Julia surface.</p><div class="language-brm-comparison vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">brm-comparison</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Non-median quantile regression</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">quantile_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.6</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.4</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">quantile_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @brm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> quantile_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    q25_y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(native_scale) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SkewDoubleExponential</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(q25_y, native_scale, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.25</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">StanBlocks unsupported </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> this BRM example</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">AssertionError</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> tracetype not defined </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> y_gen</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">anything</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> =</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> skew_double_exponential_rng</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">array</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[] tokenof, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">real</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">anything!</span></span></code></pre></div><div class="language-stan vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">stan</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Stan emission unavailable because StanBlocks lowering failed.</span></span>
<span class="line"><span></span></span>
<span class="line"><span>AssertionError: tracetype not defined for y_gen::anything = skew_double_exponential_rng(::array[] tokenof, ::vector, ::vector, ::real)::anything!</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing unsupported </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> this BRM example</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> executable families are </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Bernoulli(p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BernoulliLogit(eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Binomial(trials, p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BinomialLogit(trials, eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Poisson(exp(log_rate))\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`NegativeBinomial2(mu, phi)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`BetaBinomial2(trials, mean, precision)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and bounded </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; got </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`SkewDoubleExponential\`</span></span></code></pre></div><p>The brms/check-loss scale $s$ translates explicitly as $\\sigma = 2s$. Distributions.jl&#39;s existing exact special case also remains available in formulas:</p><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">using</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Distributions</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> SkewedExponentialPower</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> SkewedExponentialPower</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mu, sigma_sepd, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, tau)</span></span></code></pre></div><p>BRM lowers that spelling with $\\sigma = 4,\\mathtt{sigma_sepd},\\tau(1-\\tau)$. The shape must be the literal value <code>1</code>; other <code>SkewedExponentialPower</code> shapes are rejected because Stan&#39;s asymmetric double-exponential family is not a native implementation of the general SEPD. Density, pointwise log likelihood, and predictive RNG all use the same translation.</p><h2 id="Circular-regression-with-VonMises" tabindex="-1">Circular regression with <code>VonMises</code> <a class="header-anchor" href="#Circular-regression-with-VonMises" aria-label="Permalink to &quot;Circular regression with \`VonMises\` {#Circular-regression-with-VonMises}&quot;">​</a></h2><p>BRM exposes two deliberately different von-Mises likelihoods. Use Distributions.jl&#39;s <code>VonMises</code> when its exact Julia contract is intended:</p><div class="language-brm-comparison vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">brm-comparison</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Moving-support von Mises</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">von_mises_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    direction</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">exact_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @brm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> von_mises_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(kappa) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    direction </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> VonMises</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mu, kappa)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">SBBRMI with data keys </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:direction</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">emitted </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@slic</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> body</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)), x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_mu)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_mu</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_log_kappa </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(direction)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_log_kappa </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_log_kappa)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    log_kappa </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_log_kappa</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    kappa </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> exp</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(log_kappa)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    direction </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> brm_von_mises</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mu, kappa, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-stan vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">stan</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>functions {</span></span>
<span class="line"><span>matrix hcat(</span></span>
<span class="line"><span>    vector x,</span></span>
<span class="line"><span>    vector y</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    if (dims(y)[1] != n) reject(&quot;hcat: dim mismatch — \`y\` dim 1 (= &quot;, dims(y)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    return append_col(x, y);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>matrix hcat(vector x) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    return to_matrix(x, n, 1);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_von_mises_lpdf(</span></span>
<span class="line"><span>    vector y,</span></span>
<span class="line"><span>    vector mu,</span></span>
<span class="line"><span>    vector kappa,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    int principal</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    if (dims(mu)[1] != n) reject(&quot;brm_von_mises_lpdf: dim mismatch — \`mu\` dim 1 (= &quot;, dims(mu)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(kappa)[1] != n) reject(&quot;brm_von_mises_lpdf: dim mismatch — \`kappa\` dim 1 (= &quot;, dims(kappa)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    real rv = 0.0;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv += brm_von_mises_lpdf(y[i] | mu[i], kappa[i], lo, hi, principal);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_von_mises_lpdf(</span></span>
<span class="line"><span>    real y,</span></span>
<span class="line"><span>    real mu,</span></span>
<span class="line"><span>    real kappa,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    int principal</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    if((kappa &lt;= 0.0)) {</span></span>
<span class="line"><span>        return negative_infinity();</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((principal == 1)) {</span></span>
<span class="line"><span>            if((y &lt; lo)) {</span></span>
<span class="line"><span>                return negative_infinity();</span></span>
<span class="line"><span>            } else {</span></span>
<span class="line"><span>                if((y &gt;= hi)) {</span></span>
<span class="line"><span>                    return negative_infinity();</span></span>
<span class="line"><span>                } else {</span></span>
<span class="line"><span>                    real wrapped_mu = (lo + fmod(((fmod((mu - lo), (hi - lo)) + hi) - lo), (hi - lo)));</span></span>
<span class="line"><span>                    return von_mises_lpdf(y | wrapped_mu, kappa);</span></span>
<span class="line"><span>                }</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            if((y &lt; (mu - 3.141592653589793))) {</span></span>
<span class="line"><span>                return negative_infinity();</span></span>
<span class="line"><span>            } else {</span></span>
<span class="line"><span>                if((y &gt; (mu + 3.141592653589793))) {</span></span>
<span class="line"><span>                    return negative_infinity();</span></span>
<span class="line"><span>                } else {</span></span>
<span class="line"><span>                    return von_mises_lpdf(y | mu, kappa);</span></span>
<span class="line"><span>                }</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector brm_von_mises_lpdfs(</span></span>
<span class="line"><span>    vector y,</span></span>
<span class="line"><span>    vector mu,</span></span>
<span class="line"><span>    vector kappa,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    int principal</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    if (dims(mu)[1] != n) reject(&quot;brm_von_mises_lpdfs: dim mismatch — \`mu\` dim 1 (= &quot;, dims(mu)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(kappa)[1] != n) reject(&quot;brm_von_mises_lpdfs: dim mismatch — \`kappa\` dim 1 (= &quot;, dims(kappa)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = brm_von_mises_lpdf(y[i] | mu[i], kappa[i], lo, hi, principal);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector brm_von_mises_vector_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    vector mu,</span></span>
<span class="line"><span>    vector kappa,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    int principal</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = anontok__1;</span></span>
<span class="line"><span>    if (dims(mu)[1] != n) reject(&quot;brm_von_mises_rng: dim mismatch — \`mu\` dim 1 (= &quot;, dims(mu)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(kappa)[1] != n) reject(&quot;brm_von_mises_rng: dim mismatch — \`kappa\` dim 1 (= &quot;, dims(kappa)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = brm_von_mises_rng(mu[i], kappa[i], lo, hi, principal);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_von_mises_rng(</span></span>
<span class="line"><span>    real mu,</span></span>
<span class="line"><span>    real kappa,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    int principal</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    if((kappa &lt;= 0.0)) {</span></span>
<span class="line"><span>        reject(&quot;brm_von_mises_rng: kappa must be strictly positive&quot;);</span></span>
<span class="line"><span>        return 0.0;</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        real draw = von_mises_rng(mu, kappa);</span></span>
<span class="line"><span>        if((principal == 1)) {</span></span>
<span class="line"><span>            return (lo + fmod(((fmod((draw - lo), (hi - lo)) + hi) - lo), (hi - lo)));</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            real support_lo = (mu - 3.141592653589793);</span></span>
<span class="line"><span>            return (</span></span>
<span class="line"><span>                support_lo +</span></span>
<span class="line"><span>                fmod((fmod((draw - support_lo), 6.283185307179586) + 6.283185307179586), 6.283185307179586)</span></span>
<span class="line"><span>            );</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>data {</span></span>
<span class="line"><span>    int x_n;</span></span>
<span class="line"><span>    vector[x_n] x;</span></span>
<span class="line"><span>    int direction_n;</span></span>
<span class="line"><span>    vector[direction_n] direction;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed data {</span></span>
<span class="line"><span>    matrix[x_n, 2] X_mu = hcat(rep_vector(1.0, num_elements(x)), x);</span></span>
<span class="line"><span>    int pop_mu_n_covariates = 2;</span></span>
<span class="line"><span>    matrix[num_elements(direction), 1] X_log_kappa = hcat(rep_vector(1.0, num_elements(direction)));</span></span>
<span class="line"><span>    int pop_log_kappa_n_covariates = 1;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>parameters {</span></span>
<span class="line"><span>    vector[pop_mu_n_covariates] pop_mu_beta_pop;</span></span>
<span class="line"><span>    vector[pop_log_kappa_n_covariates] pop_log_kappa_beta_pop;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed parameters {</span></span>
<span class="line"><span>    vector[x_n] pop_mu = (X_mu * pop_mu_beta_pop);</span></span>
<span class="line"><span>    vector[x_n] mu = pop_mu;</span></span>
<span class="line"><span>    vector[num_elements(direction)] pop_log_kappa = (X_log_kappa * pop_log_kappa_beta_pop);</span></span>
<span class="line"><span>    vector[num_elements(direction)] log_kappa = pop_log_kappa;</span></span>
<span class="line"><span>    vector[num_elements(direction)] kappa = exp(log_kappa);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>model {</span></span>
<span class="line"><span>    pop_mu_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    pop_log_kappa_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    direction ~ brm_von_mises(mu, kappa, 0.0, 0.0, 0);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>generated quantities {</span></span>
<span class="line"><span>    vector[num_elements(direction)] direction_likelihood = brm_von_mises_lpdfs(direction, mu, kappa, 0.0, 0.0, 0);</span></span>
<span class="line"><span>    vector[num_elements(direction)] direction_gen = brm_von_mises_vector_rng(direction_n, mu, kappa, 0.0, 0.0, 0);</span></span>
<span class="line"><span>}</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing unsupported </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> this BRM example</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> executable families are </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Bernoulli(p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BernoulliLogit(eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Binomial(trials, p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BinomialLogit(trials, eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Poisson(exp(log_rate))\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`NegativeBinomial2(mu, phi)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`BetaBinomial2(trials, mean, precision)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and bounded </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; got </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Distributions.VonMises\`</span></span></code></pre></div><p>This preserves the constructor order <code>(mu, kappa)</code>, the shorthand <code>VonMises(kappa) == VonMises(0, kappa)</code>, strict <code>kappa &gt; 0</code>, and the moving closed support <code>[mu - pi, mu + pi]</code>. The backend adds those support/domain guards around Stan&#39;s native <code>von_mises_lpdf</code>, and recenters native predictive draws onto the same moving interval. Because the support moves with <code>mu</code>, this surface is usually not the right choice for observations encoded once on a fixed principal interval.</p><p>For conventional circular regression on a fixed interval, use the distinct BRM distribution <code>CircularVonMises</code>:</p><div class="language-brm-comparison vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">brm-comparison</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Fixed-interval circular regression</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">circular_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    direction</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.8</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.5</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.9</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">circular_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> @brm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> circular_data </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(kappa) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    direction </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> CircularVonMises</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mu, kappa; interval</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">pi</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">pi</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">SBBRMI with data keys </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:direction</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">emitted </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@slic</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> body</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)), x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_mu)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_mu</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_log_kappa </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(direction)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_log_kappa </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_log_kappa)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    log_kappa </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_log_kappa</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    kappa </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> exp</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(log_kappa)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    direction </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> brm_von_mises</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mu, kappa, </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3.141592653589793</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">3.141592653589793</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-stan vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">stan</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>functions {</span></span>
<span class="line"><span>matrix hcat(</span></span>
<span class="line"><span>    vector x,</span></span>
<span class="line"><span>    vector y</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    if (dims(y)[1] != n) reject(&quot;hcat: dim mismatch — \`y\` dim 1 (= &quot;, dims(y)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    return append_col(x, y);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>matrix hcat(vector x) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    return to_matrix(x, n, 1);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_von_mises_lpdf(</span></span>
<span class="line"><span>    vector y,</span></span>
<span class="line"><span>    vector mu,</span></span>
<span class="line"><span>    vector kappa,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    int principal</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    if (dims(mu)[1] != n) reject(&quot;brm_von_mises_lpdf: dim mismatch — \`mu\` dim 1 (= &quot;, dims(mu)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(kappa)[1] != n) reject(&quot;brm_von_mises_lpdf: dim mismatch — \`kappa\` dim 1 (= &quot;, dims(kappa)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    real rv = 0.0;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv += brm_von_mises_lpdf(y[i] | mu[i], kappa[i], lo, hi, principal);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_von_mises_lpdf(</span></span>
<span class="line"><span>    real y,</span></span>
<span class="line"><span>    real mu,</span></span>
<span class="line"><span>    real kappa,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    int principal</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    if((kappa &lt;= 0.0)) {</span></span>
<span class="line"><span>        return negative_infinity();</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        if((principal == 1)) {</span></span>
<span class="line"><span>            if((y &lt; lo)) {</span></span>
<span class="line"><span>                return negative_infinity();</span></span>
<span class="line"><span>            } else {</span></span>
<span class="line"><span>                if((y &gt;= hi)) {</span></span>
<span class="line"><span>                    return negative_infinity();</span></span>
<span class="line"><span>                } else {</span></span>
<span class="line"><span>                    real wrapped_mu = (lo + fmod(((fmod((mu - lo), (hi - lo)) + hi) - lo), (hi - lo)));</span></span>
<span class="line"><span>                    return von_mises_lpdf(y | wrapped_mu, kappa);</span></span>
<span class="line"><span>                }</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            if((y &lt; (mu - 3.141592653589793))) {</span></span>
<span class="line"><span>                return negative_infinity();</span></span>
<span class="line"><span>            } else {</span></span>
<span class="line"><span>                if((y &gt; (mu + 3.141592653589793))) {</span></span>
<span class="line"><span>                    return negative_infinity();</span></span>
<span class="line"><span>                } else {</span></span>
<span class="line"><span>                    return von_mises_lpdf(y | mu, kappa);</span></span>
<span class="line"><span>                }</span></span>
<span class="line"><span>            }</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector brm_von_mises_lpdfs(</span></span>
<span class="line"><span>    vector y,</span></span>
<span class="line"><span>    vector mu,</span></span>
<span class="line"><span>    vector kappa,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    int principal</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    if (dims(mu)[1] != n) reject(&quot;brm_von_mises_lpdfs: dim mismatch — \`mu\` dim 1 (= &quot;, dims(mu)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(kappa)[1] != n) reject(&quot;brm_von_mises_lpdfs: dim mismatch — \`kappa\` dim 1 (= &quot;, dims(kappa)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = brm_von_mises_lpdf(y[i] | mu[i], kappa[i], lo, hi, principal);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector brm_von_mises_vector_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    vector mu,</span></span>
<span class="line"><span>    vector kappa,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    int principal</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = anontok__1;</span></span>
<span class="line"><span>    if (dims(mu)[1] != n) reject(&quot;brm_von_mises_rng: dim mismatch — \`mu\` dim 1 (= &quot;, dims(mu)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    if (dims(kappa)[1] != n) reject(&quot;brm_von_mises_rng: dim mismatch — \`kappa\` dim 1 (= &quot;, dims(kappa)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = brm_von_mises_rng(mu[i], kappa[i], lo, hi, principal);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>real brm_von_mises_rng(</span></span>
<span class="line"><span>    real mu,</span></span>
<span class="line"><span>    real kappa,</span></span>
<span class="line"><span>    real lo,</span></span>
<span class="line"><span>    real hi,</span></span>
<span class="line"><span>    int principal</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    if((kappa &lt;= 0.0)) {</span></span>
<span class="line"><span>        reject(&quot;brm_von_mises_rng: kappa must be strictly positive&quot;);</span></span>
<span class="line"><span>        return 0.0;</span></span>
<span class="line"><span>    } else {</span></span>
<span class="line"><span>        real draw = von_mises_rng(mu, kappa);</span></span>
<span class="line"><span>        if((principal == 1)) {</span></span>
<span class="line"><span>            return (lo + fmod(((fmod((draw - lo), (hi - lo)) + hi) - lo), (hi - lo)));</span></span>
<span class="line"><span>        } else {</span></span>
<span class="line"><span>            real support_lo = (mu - 3.141592653589793);</span></span>
<span class="line"><span>            return (</span></span>
<span class="line"><span>                support_lo +</span></span>
<span class="line"><span>                fmod((fmod((draw - support_lo), 6.283185307179586) + 6.283185307179586), 6.283185307179586)</span></span>
<span class="line"><span>            );</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>data {</span></span>
<span class="line"><span>    int x_n;</span></span>
<span class="line"><span>    vector[x_n] x;</span></span>
<span class="line"><span>    int direction_n;</span></span>
<span class="line"><span>    vector[direction_n] direction;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed data {</span></span>
<span class="line"><span>    matrix[x_n, 2] X_mu = hcat(rep_vector(1.0, num_elements(x)), x);</span></span>
<span class="line"><span>    int pop_mu_n_covariates = 2;</span></span>
<span class="line"><span>    matrix[num_elements(direction), 1] X_log_kappa = hcat(rep_vector(1.0, num_elements(direction)));</span></span>
<span class="line"><span>    int pop_log_kappa_n_covariates = 1;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>parameters {</span></span>
<span class="line"><span>    vector[pop_mu_n_covariates] pop_mu_beta_pop;</span></span>
<span class="line"><span>    vector[pop_log_kappa_n_covariates] pop_log_kappa_beta_pop;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed parameters {</span></span>
<span class="line"><span>    vector[x_n] pop_mu = (X_mu * pop_mu_beta_pop);</span></span>
<span class="line"><span>    vector[x_n] mu = pop_mu;</span></span>
<span class="line"><span>    vector[num_elements(direction)] pop_log_kappa = (X_log_kappa * pop_log_kappa_beta_pop);</span></span>
<span class="line"><span>    vector[num_elements(direction)] log_kappa = pop_log_kappa;</span></span>
<span class="line"><span>    vector[num_elements(direction)] kappa = exp(log_kappa);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>model {</span></span>
<span class="line"><span>    pop_mu_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    pop_log_kappa_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    direction ~ brm_von_mises(mu, kappa, -3.141592653589793, 3.141592653589793, 1);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>generated quantities {</span></span>
<span class="line"><span>    vector[num_elements(direction)] direction_likelihood = brm_von_mises_lpdfs(direction, mu, kappa, -3.141592653589793, 3.141592653589793, 1);</span></span>
<span class="line"><span>    vector[num_elements(direction)] direction_gen = brm_von_mises_vector_rng(direction_n, mu, kappa, -3.141592653589793, 3.141592653589793, 1);</span></span>
<span class="line"><span>}</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing unsupported </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> this BRM example</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> executable families are </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Bernoulli(p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BernoulliLogit(eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Binomial(trials, p)\`</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> /</span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;"> \`BinomialLogit(trials, eta)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Poisson(exp(log_rate))\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`NegativeBinomial2(mu, phi)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`BetaBinomial2(trials, mean, precision)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, and bounded </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`Normal(mu, sigma)\`</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">; got </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`CircularVonMises\`</span></span></code></pre></div><p><code>interval</code> is a compile-time pair of finite numbers with length <code>2pi</code>; it defaults to <code>(-pi, pi)</code>. Observations must lie in the half-open interval <code>[lo, hi)</code>. BRM wraps <code>mu</code> and generated draws into that interval, while the density itself remains Stan&#39;s native <code>von_mises_lpdf</code>. Both arguments are ordinary distributional parameters: BRM supplies no implicit link or prior. Outside a formula, <code>CircularVonMises(mu, kappa; interval=...)</code> is an executable <code>ContinuousUnivariateDistribution</code>: <code>params</code>, <code>logpdf</code>, and <code>rand</code> preserve the same fixed-interval contract used by the Stan lowering.</p><p>For comparison, <code>brms</code> uses a fixed <code>(-pi, pi)</code> response convention and makes both <code>mu</code> and <code>kappa</code> distributional, with default <code>tan_half</code> and <code>log</code> links. Those are a useful neutral baseline, but BRM requires links and priors to be written explicitly rather than silently changing Distributions.jl semantics. In particular, <code>Distributions.Gamma</code> takes a <strong>scale</strong>, whereas Stan/brms gamma syntax takes a <strong>rate</strong>: the brms prior <code>gamma(2, 0.01)</code> is spelled <code>Gamma(2, 100.0)</code> in a BRM formula.</p><h2 id="Typed-observation-weights" tabindex="-1">Typed observation weights <a class="header-anchor" href="#Typed-observation-weights" aria-label="Permalink to &quot;Typed observation weights {#Typed-observation-weights}&quot;">​</a></h2><p>Observation weights live in the <code>@brm</code> model beside the observation distribution:</p><div class="language-brm-comparison vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">brm-comparison</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>Analytic observation weights</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">weighted_model </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> (</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@brm</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> weighted</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Normal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mu, sigma), </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">aweights</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(replicate_k))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;"> +</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> x</span></span>
<span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">    log</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(sigma) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> 1</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)((;</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    x</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">], y</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">-</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.2</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.3</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.1</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    replicate_k</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">[</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">2.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">4.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">],</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">))</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">SBBRMI with data keys </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> [</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:brm_weight_y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:replicate_k</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:x</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">:y</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">]</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">emitted </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">@slic</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> body</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">begin</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(x)), x)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> _popefs_coefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_mu)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    X_log_sigma </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> hcat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">rep_vector</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">1.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">num_elements</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(y)))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    pop_log_sigma </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> popefs</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(; X </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_log_sigma)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    log_sigma </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_log_sigma</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    sigma </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> exp</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(log_sigma)</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    y </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">~</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> normal_id_glm</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(X_mu, </span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">0.0</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, pop_mu, sigma </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">./</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> sqrt</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(brm_weight_y))</span></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">    mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> X_mu </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">*</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> pop_mu</span></span>
<span class="line"><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">end</span></span></code></pre></div><div class="language-stan vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">stan</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>functions {</span></span>
<span class="line"><span>matrix hcat(</span></span>
<span class="line"><span>    vector x,</span></span>
<span class="line"><span>    vector y</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    if (dims(y)[1] != n) reject(&quot;hcat: dim mismatch — \`y\` dim 1 (= &quot;, dims(y)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    return append_col(x, y);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>matrix hcat(vector x) {</span></span>
<span class="line"><span>    int n = dims(x)[1];</span></span>
<span class="line"><span>    return to_matrix(x, n, 1);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector normal_id_glm_lpdfs(</span></span>
<span class="line"><span>    vector y,</span></span>
<span class="line"><span>    matrix X,</span></span>
<span class="line"><span>    real alpha,</span></span>
<span class="line"><span>    vector beta,</span></span>
<span class="line"><span>    vector sigma</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int n = dims(y)[1];</span></span>
<span class="line"><span>    if (dims(X)[1] != n) reject(&quot;normal_id_glm_lpdfs: dim mismatch — \`X\` dim 1 (= &quot;, dims(X)[1], &quot;) does not match \`n\` (= &quot;, n, &quot;)&quot;);</span></span>
<span class="line"><span>    vector[n] rv;</span></span>
<span class="line"><span>    for(i in 1:n) {</span></span>
<span class="line"><span>        rv[i] = normal_lpdf(y[i] | (alpha + (X[i, :] * beta)), sigma);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    return rv;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector normal_id_glm_vector_rng(</span></span>
<span class="line"><span>    int anontok__1,</span></span>
<span class="line"><span>    matrix X,</span></span>
<span class="line"><span>    real alpha,</span></span>
<span class="line"><span>    vector beta,</span></span>
<span class="line"><span>    vector sigma</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int m = anontok__1;</span></span>
<span class="line"><span>    if (dims(X)[1] != m) reject(&quot;normal_id_glm_rng: dim mismatch — \`X\` dim 1 (= &quot;, dims(X)[1], &quot;) does not match \`m\` (= &quot;, m, &quot;)&quot;);</span></span>
<span class="line"><span>    return normal_id_glm_rng(X, alpha, beta, sigma);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>vector normal_id_glm_rng(</span></span>
<span class="line"><span>    matrix X,</span></span>
<span class="line"><span>    real alpha,</span></span>
<span class="line"><span>    vector beta,</span></span>
<span class="line"><span>    vector sigma</span></span>
<span class="line"><span>) {</span></span>
<span class="line"><span>    int m = dims(X)[1];</span></span>
<span class="line"><span>    return to_vector(normal_rng((rep_vector(alpha, m) + (X * beta)), sigma));</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>data {</span></span>
<span class="line"><span>    int x_n;</span></span>
<span class="line"><span>    vector[x_n] x;</span></span>
<span class="line"><span>    int y_n;</span></span>
<span class="line"><span>    vector[y_n] y;</span></span>
<span class="line"><span>    int brm_weight_y_n;</span></span>
<span class="line"><span>    vector[brm_weight_y_n] brm_weight_y;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed data {</span></span>
<span class="line"><span>    matrix[x_n, 2] X_mu = hcat(rep_vector(1.0, num_elements(x)), x);</span></span>
<span class="line"><span>    int pop_mu_n_covariates = 2;</span></span>
<span class="line"><span>    matrix[num_elements(y), 1] X_log_sigma = hcat(rep_vector(1.0, num_elements(y)));</span></span>
<span class="line"><span>    int pop_log_sigma_n_covariates = 1;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>parameters {</span></span>
<span class="line"><span>    vector[pop_mu_n_covariates] pop_mu_beta_pop;</span></span>
<span class="line"><span>    vector[pop_log_sigma_n_covariates] pop_log_sigma_beta_pop;</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>transformed parameters {</span></span>
<span class="line"><span>    vector[pop_mu_n_covariates] pop_mu = pop_mu_beta_pop;</span></span>
<span class="line"><span>    vector[num_elements(y)] pop_log_sigma = (X_log_sigma * pop_log_sigma_beta_pop);</span></span>
<span class="line"><span>    vector[num_elements(y)] log_sigma = pop_log_sigma;</span></span>
<span class="line"><span>    vector[num_elements(y)] sigma = exp(log_sigma);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>model {</span></span>
<span class="line"><span>    pop_mu_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    pop_log_sigma_beta_pop ~ std_normal();</span></span>
<span class="line"><span>    y ~ normal_id_glm(X_mu, 0.0, pop_mu, (sigma ./ sqrt(brm_weight_y)));</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>generated quantities {</span></span>
<span class="line"><span>    vector[x_n] y_likelihood = normal_id_glm_lpdfs(y, X_mu, 0.0, pop_mu, (sigma ./ sqrt(brm_weight_y)));</span></span>
<span class="line"><span>    vector[x_n] y_gen = normal_id_glm_vector_rng(y_n, X_mu, 0.0, pop_mu, (sigma ./ sqrt(brm_weight_y)));</span></span>
<span class="line"><span>    vector[x_n] mu = (X_mu * pop_mu);</span></span>
<span class="line"><span>}</span></span></code></pre></div><div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing unsupported </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">for</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> this BRM example</span></span>
<span class="line"></span>
<span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">Turing backend</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">:</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> Gaussian scale prior must bind bare </span><span style="--shiki-light:#032F62;--shiki-dark:#9ECBFF;">\`sigma\`</span></span></code></pre></div><p>The StatsBase constructor determines the statistical meaning:</p><ul><li><p><code>aweights(k)</code> uses analytic/precision semantics. For a Normal response BRM emits <code>Normal(mu, sigma / sqrt(k))</code>; model density, pointwise likelihood, and predictive draws all use that adjusted scale.</p></li><li><p><code>fweights(n)</code> uses frequency/repeat semantics. BRM multiplies each model and pointwise log-likelihood contribution by <code>n</code>; predictive draws remain from the original distribution.</p></li><li><p><code>weights(w)</code> opts into a power likelihood with the same density/pointwise scaling and unchanged predictive distribution.</p></li></ul><p>The current analytic-weight implementation supports Normal observations. Frequency and power weights support likelihood families that lower through BRM&#39;s native Distributions.jl-to-Stan family map. Probability weights and unsupported family/type combinations error instead of silently changing meaning. Weight columns are rebuilt from each dataframe by the reusable <code>@brm</code> builder and by <code>reprocess</code>.</p>`,129)])])}const E=a(l,[["render",e]]);export{o as __pageData,E as default};
