import numpy as np
import pytest

import ot
import gw_networks as gwn


@pytest.mark.parametrize(
    "loss_fun",
    [
        (lambda x, y: (x - y) ** 2),  # L2 loss
        (lambda x, y: x * np.log(x + 1e-15) - x + y - x * np.log(y + 1e-15)),
    ],  # KL loss as implemented by POT
)
def test_entropic_gromov_crude(loss_fun, nx):
    n_samples = 10  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    G, log = gwn.entropic_gromov_wasserstein_crude(
        C1,
        C2,
        None,
        q,
        loss_fun,
        symmetric=None,
        G0=G0,
        epsilon=1e-2,
        max_iter=10,
        verbose=True,
        log=True,
    )
    Gb = nx.to_numpy(
        gwn.entropic_gromov_wasserstein_crude(
            C1b,
            C2b,
            pb,
            None,
            loss_fun,
            symmetric=True,
            G0=None,
            epsilon=1e-2,
            max_iter=10,
            verbose=True,
            log=False,
        )
    )

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-04)  # cf convergence gromov


@pytest.mark.parametrize(
    "loss_fun",
    [
        (lambda x, y: (x - y) ** 2),  # L2 loss
        (lambda x, y: x * np.log(x + 1e-15) - x + y - x * np.log(y + 1e-15)),
    ],  # KL loss as implemented by POT
)
def test_entropic_gromov2_crude(nx, loss_fun):
    n_samples = 10  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)

    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    gw, log = gwn.entropic_gromov_wasserstein2_crude(
        C1,
        C2,
        p,
        None,
        loss_fun,
        symmetric=True,
        G0=None,
        max_iter=10,
        epsilon=1e-2,
        log=True,
    )
    gwb, logb = gwn.entropic_gromov_wasserstein2_crude(
        C1b,
        C2b,
        None,
        qb,
        loss_fun,
        symmetric=None,
        G0=G0b,
        max_iter=10,
        epsilon=1e-2,
        log=True,
    )
    gwb = nx.to_numpy(gwb)

    G = log["T"]
    Gb = nx.to_numpy(logb["T"])

    np.testing.assert_allclose(gw, gwb, atol=1e-06)
    np.testing.assert_allclose(gw, 0, atol=1e-1, rtol=1e-1)

    # check constraints
    np.testing.assert_allclose(G, Gb, atol=1e-06)
    np.testing.assert_allclose(p, Gb.sum(1), atol=1e-04)  # cf convergence gromov
    np.testing.assert_allclose(q, Gb.sum(0), atol=1e-04)  # cf convergence gromov


@pytest.mark.parametrize(
    "loss_fun",
    [
        (lambda x, y: (x - y) ** 2, "square_loss"),  # L2 loss
        (lambda x, y: x * np.log(x + 1e-15) - x + y - x * np.log(y + 1e-15), "kl_loss"),
    ],  # KL loss as implemented by POT
)
def test_entropic_gromov_vs_pot(nx, loss_fun):
    """
    Comparing against the result from POT.
    """

    n_samples = 10  # nb samples

    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)
    xt = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=43)

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    G0 = p[:, None] * q[None, :]
    C1 = ot.dist(xs, xs)
    C2 = ot.dist(xt, xt)

    C1 /= C1.max()
    C2 /= C2.max()

    C1b, C2b, pb, qb, G0b = nx.from_numpy(C1, C2, p, q, G0)

    gw, log = gwn.entropic_gromov_wasserstein2_crude(
        C1,
        C2,
        p,
        None,
        loss_fun[0],
        symmetric=True,
        G0=None,
        max_iter=10,
        epsilon=1e-2,
        log=True,
    )

    gw_pot, log_pot = ot.gromov.entropic_gromov_wasserstein2(
        C1,
        C2,
        p,
        None,
        loss_fun[1],
        symmetric=True,
        G0=None,
        max_iter=10,
        epsilon=1e-2,
        log=True,
    )

    G = log["T"]
    G_pot = log_pot["T"]

    np.testing.assert_allclose(gw, gw_pot, atol=1e-03)
    np.testing.assert_allclose(G, G_pot, atol=1e-03)


@pytest.mark.parametrize(
    "loss_fun",
    [
        (lambda x, y: gwn.elementwise_norm(x - y, ord=2)),  # L2 loss
        (lambda x, y: gwn.elementwise_norm(x - y, ord=np.inf)),  # L-infinity loss
    ],
)
def test_entropic_gromov_hidim(nx, loss_fun):
    """
    Testing the high dimensional case i.e. the kernels are vector-valued.
    """

    n_samples = 10
    mu_s = np.array([0, 0])
    cov_s = np.array([[1, 0], [0, 1]])

    xs = ot.datasets.make_2D_samples_gauss(n_samples, mu_s, cov_s, random_state=42)
    xt = xs[::-1].copy()

    p = ot.unif(n_samples)
    q = ot.unif(n_samples)
    C1 = xs.reshape(-1, 1, 2) - xs.reshape(1, -1, 2)
    C2 = xt.reshape(-1, 1, 2) - xt.reshape(1, -1, 2)

    gw, log = gwn.entropic_gromov_wasserstein2_crude(
        C1,
        C2,
        p,
        q,
        loss_fun,
        symmetric=True,
        G0=None,
        max_iter=10,
        epsilon=1e-2,
        log=True,
    )

    G = log["T"]
    np.testing.assert_allclose(gw, 0, atol=1e-1, rtol=1e-1)
    np.testing.assert_allclose(p, G.sum(1), atol=1e-04)
    np.testing.assert_allclose(q, G.sum(0), atol=1e-04)
