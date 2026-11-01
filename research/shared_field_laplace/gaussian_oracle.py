"""Small exact controls for the shared-field/SVI integration design.

This is a statistical oracle, not a BRM execution test or an inference method.
All integrals are Gaussian, so conditional Laplace is exact. No approximation
error from mode finding or SVI optimization can explain the discrepancies.
"""
import json
import math


def shared_log_marginal(y, x, theta, noise_var=1.0, field_var=1.0):
    # y_i = x_i theta + z + epsilon_i, z ~ N(0, field_var).
    # Matrix determinant lemma and Sherman-Morrison, evaluated as scalars.
    n = len(y)
    r = [a - theta * b for a, b in zip(y, x)]
    logdet = n * math.log(noise_var) + math.log1p(n * field_var / noise_var)
    quadratic = sum(a*a for a in r) / noise_var
    quadratic -= field_var * sum(r)**2 / (noise_var * (noise_var+n*field_var))
    return -0.5 * (n * math.log(2*math.pi) + logdet + quadratic)


def powered_single_log_marginal(y, x, theta, power):
    # Integral N(z;0,1) N(y;x theta+z,1)^power dz.
    r = y - x*theta
    return -0.5*power*math.log(2*math.pi) - 0.5*math.log1p(power) - 0.5*power*r*r/(1+power)


def derivative(f, x):
    eps = 1e-5
    return (f(x+eps)-f(x-eps))/(2*eps)


def normal_logpdf(y, mean, var):
    return -0.5*(math.log(2*math.pi*var)+(y-mean)**2/var)


def main():
    y, x, theta = [1.0, -1.0], [1.0, -1.0], 0.25
    joint = lambda t: shared_log_marginal(y, x, t)
    independent = lambda t: sum(shared_log_marginal([yi], [xi], t) for yi, xi in zip(y, x))
    minibatch = lambda t: sum(powered_single_log_marginal(yi, xi, t, 2) for yi, xi in zip(y, x))/2
    # Closed-form derivatives prove that the discrepancy changes the target,
    # rather than only an irrelevant normalization constant.
    assert math.isclose(derivative(joint, theta), 1.5, abs_tol=1e-9)
    assert math.isclose(derivative(independent, theta), 0.75, abs_tol=1e-9)
    assert math.isclose(derivative(minibatch, theta), 0.5, abs_tol=1e-9)

    # Positive control: if the model really has independent z_i, collapsing
    # separately is correct, and sampling one collapsed term with N/B is unbiased.
    control = sum(2*shared_log_marginal([yi], [xi], theta) for yi, xi in zip(y, x))/2
    assert math.isclose(control, independent(theta), abs_tol=1e-14)

    # Even two observations require the field posterior between windows.
    # z~N(0,1), y1=2 gives z|y1~N(1,1/2); y2=2 has variance 3/2.
    correct_predictive = normal_logpdf(2, 1, 1.5)
    reset_predictive = normal_logpdf(2, 0, 2)
    joint_two = shared_log_marginal([2, 2], [0, 0], 0)
    assert math.isclose(joint_two, normal_logpdf(2, 0, 2)+correct_predictive, abs_tol=1e-14)
    assert not math.isclose(correct_predictive, reset_predictive, abs_tol=1e-6)

    # A tiny HSGP-style sine design: sparse prior does not imply sparse posterior.
    # H=I+A'A; this explicit tiny calculation is only an oracle. No production
    # algorithm should assemble growing coefficient-squared arrays.
    times = [-0.83, -0.49, -0.12, 0.26, 0.57, 0.91]
    support = []
    for d in (4, 8, 16):
        basis = [[math.sin(math.pi*j*(t+1.5)/3)/math.sqrt(1.5) for j in range(1,d+1)] for t in times]
        nnz = sum(abs((1.0 if j == k else 0.0)+sum(row[j]*row[k] for row in basis)) > 1e-12
                  for j in range(d) for k in range(d))
        assert nnz == d*d
        support.append(dict(coefficients=d, posterior_precision_nonzeros=nnz))

    print(json.dumps(dict(
        theta=theta,
        log_marginals=dict(joint=joint(theta), separate=independent(theta), expected_powered_minibatch=minibatch(theta)),
        gradients=dict(joint=derivative(joint, theta), separate=derivative(independent, theta), expected_powered_minibatch=derivative(minibatch, theta)),
        theta_posterior_with_standard_normal_prior=dict(joint=dict(mean=2/3, variance=1/3), separate=dict(mean=1/2, variance=1/2), expected_powered_minibatch=dict(mean=2/5, variance=3/5)),
        independent_block_positive_control=True,
        streaming=dict(correct_log_predictive=correct_predictive, reset_field_log_predictive=reset_predictive),
        dense_support=support,
        scope="Exact scalar/small-basis mathematical controls only; not BRM fit, performance, or AP3 acceptance."
    ), indent=2))


if __name__ == "__main__":
    main()
