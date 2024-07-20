"""
An entropic solver for network GW-distances
"""

# Author: Mao Nishino <mao1756@gmail.com>
# This code is adapted and modified from PythonOT/POT.
# Repository URL: https://github.com/PythonOT/POT

# License: MIT License

# TODO think about how to implment high-dim loss functions
# Idea: write a nice docstring that tells you how to make one
# have a set list of functions that you can use (e.g. lp losses)

# Fix the docstrings for loss_fun for all functions


import numpy as np
import warnings

from ot.bregman import sinkhorn
from ot.utils import list_to_array, unif
from ot.backend import get_backend


def init_matrix_crude(C1, C2, loss_fun):
    r"""Return the loss matrices and tensors for Gromov-Wasserstein computation

    Returns the 4-D tensor L_ijkl = L((C_1)_ik, (C_2)_jl).

    Parameters
    ----------
    C1 : array-like, shape (ns, ns, *dims)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt, *dims)
        Metric cost matrix in the target space
    loss_fun : function: :math:`\mathbb{R}^{len(dims)} \times \mathbb{R}^{len(dims)} \mapsto \mathbb{R}`
        Loss function used for the distance. It is expected that the loss function is vectorized i.e. the function should take an array input.
    """

    return loss_fun(C1[:, None, :, None, ...], C2[None, :, None, :, ...])


def tensor_product_crude(loss, T, nx=None):
    r"""Return the tensor product between the loss tensor and the coupling T.

    Parameters
    ----------
    loss: array-like, shape (ns, nt, ns, nt)
        The loss tensor L(C1, C2).
    T : array-like, shape (ns, nt)
        Current value of transport matrix :math:`\mathbf{T}`
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    tens : array-like, shape (`ns`, `nt`)
        :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` tensor-matrix multiplication result


    .. _references-tensor-product:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    if nx is None:
        loss, T = list_to_array(loss, T)
        nx = get_backend(loss, T)

    return nx.sum(loss * T[None, None, :, :], axis=(2, 3))


def gwloss_crude(loss, T, nx=None):
    r"""Return the Loss for Gromov-Wasserstein.

    Parameters
    ----------
    loss: array-like, shape (ns, nt, ns, nt)
        The loss tensor L(C1, C2).
    T : array-like, shape (ns, nt)
        Current value of transport matrix :math:`\mathbf{T}`
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    loss : float
        Gromov-Wasserstein loss


    .. _references-gwloss:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """
    tens = tensor_product_crude(loss, T, nx)
    if nx is None:
        tens, T = list_to_array(tens, T)
        nx = get_backend(tens, T)

    return nx.sum(tens * T)


def gwggrad_crude(loss, T, nx=None):
    r"""Return the gradient for Gromov-Wasserstein

    The gradient is computed as described in Proposition 2 in :ref:`[12] <references-gwggrad>`

    Parameters
    ----------
    loss: array-like, shape (ns, nt, ns, nt)
        The loss tensor L(C1, C2).
    T : array-like, shape (ns, nt)
        Current value of transport matrix :math:`\mathbf{T}`
    nx : backend, optional
        If let to its default value None, a backend test will be conducted.
    Returns
    -------
    grad : array-like, shape (`ns`, `nt`)
        Gromov-Wasserstein gradient


    .. _references-gwggrad:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    """

    return 2 * tensor_product_crude(loss, T, nx)


def elementwise_norm(x, ord=2):
    """Given an array of the shape (m, n, m, n, d), calculate the norm of the last dimension.

    Parameters
    ----------
    x : array-like, shape (m, n, m, n, d) or (m, n, m, n)
        The input array. If the input array is of shape (m, n, m, n), the last dimension is assumed to be 1.
    ord : float, optional
        The order of the norm. Default is 2. Should be positive.
    """
    nx = get_backend(x)

    if ord < 0:
        raise ValueError("Order of the norm should be positive.")
    if len(x.shape) != 5:
        if len(x.shape) == 4:
            x = x[..., None]
        else:
            raise ValueError("The input array should have 4 or 5 dimensions.")
    if x.shape[0] != x.shape[2] or x.shape[1] != x.shape[3]:
        raise ValueError(
            "The input array should be of shape (m, n, m, n, d) or (m, n, m, n)."
        )

    if ord == np.inf:
        return nx.max(nx.abs(x), axis=-1)
    else:
        return (nx.abs(x) ** ord).sum(axis=-1) ** (1 / ord)


def entropic_gromov_wasserstein_crude(
    C1,
    C2,
    p=None,
    q=None,
    loss_fun="square_loss",
    epsilon=0.1,
    symmetric=None,
    G0=None,
    max_iter=1000,
    tol=1e-9,
    solver="PGD",
    warmstart=False,
    verbose=False,
    log=False,
    **kwargs,
):
    r"""
    Returns the Gromov-Wasserstein transport between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`
    estimated using Sinkhorn projections *for any loss functions*. As the tensor product is calculated naively, the time complexity is O(n^4) instead of O(n^3) for the original algorithm.

    If `solver="PGD"`, the function solves the following entropic-regularized
    Gromov-Wasserstein optimization problem using Projected Gradient Descent [12]:

    .. math::
        \mathbf{T}^* \in \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon H(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Else if `solver="PPA"`, the function solves the following Gromov-Wasserstein
    optimization problem using Proximal Point Algorithm [51]:

    .. math::
        \mathbf{T}^* \in \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0
    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices
    - `H`: entropy

    .. note:: If the inner solver `ot.sinkhorn` did not convergence, the
        optimal coupling :math:`\mathbf{T}` returned by this function does not
        necessarily satisfy the marginal constraints
        :math:`\mathbf{T}\mathbf{1}=\mathbf{p}` and
        :math:`\mathbf{T}^T\mathbf{1}=\mathbf{q}`. So the returned
        Gromov-Wasserstein loss does not necessarily satisfy distance
        properties and may be negative.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns, *dims)
        Metric cost matrix in the source space with len(dims)-dimensional tensors as elements.
    C2 : array-like, shape (nt, nt, *dims)
        Metric cost matrix in the target space with len(dims)-dimensional tensors as elements.
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : function: :math:`\mathbb{R}^{len(dims)} \times \mathbb{R}^{len(dims)} \mapsto \mathbb{R}`
        Loss function used for the distance. It is expected that the loss function is vectorized i.e. the function should take an array input.
    epsilon : float, optional
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 will be used as initial transport of the solver. G0 is not
        required to satisfy marginal constraints but we strongly recommend it
        to correctly estimate the GW distance.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    solver: string, optional
        Solver to use either 'PGD' for Projected Gradient Descent or 'PPA'
        for Proximal Point Algorithm.
        Default value is 'PGD'.
    warmstart: bool, optional
        Either to perform warmstart of dual potentials in the successive
        Sinkhorn projections.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.
    **kwargs: dict
        parameters can be directly passed to the ot.sinkhorn solver.
        Such as `numItermax` and `stopThr` to control its estimation precision,
        e.g [51] suggests to use `numItermax=1`.
    Returns
    -------
    T : array-like, shape (`ns`, `nt`)
        Optimal coupling between the two spaces

    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.

    .. [47] Chowdhury, S., & Mémoli, F. (2019). The gromov–wasserstein
        distance between networks and stable network invariants.
        Information and Inference: A Journal of the IMA, 8(4), 757-787.

    .. [51] Xu, H., Luo, D., Zha, H., & Duke, L. C. (2019). Gromov-wasserstein
        learning for graph matching and node embedding. In International
        Conference on Machine Learning (ICML), 2019.
    """
    if solver not in ["PGD", "PPA"]:
        raise ValueError("Unknown solver '%s'. Pick one in ['PGD', 'PPA']." % solver)

    if solver == "PPA":
        raise NotImplementedError("Solver PPA is not implemented yet.")

    # if loss_fun not in ("square_loss", "kl_loss"):
    #    raise ValueError(
    #        f"Unknown `loss_fun='{loss_fun}'`. Use one of: {'square_loss', 'kl_loss'}."
    #    )

    C1, C2 = list_to_array(C1, C2)
    arr = [C1, C2]
    if p is not None:
        arr.append(list_to_array(p))
    else:
        p = unif(C1.shape[0], type_as=C1)
    if q is not None:
        arr.append(list_to_array(q))
    else:
        q = unif(C2.shape[0], type_as=C2)

    if G0 is not None:
        arr.append(G0)

    nx = get_backend(*arr)

    if G0 is None:
        G0 = nx.outer(p, q)

    T = G0
    loss = init_matrix_crude(C1, C2, loss_fun)
    # constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun, nx)
    assert loss.shape == (C1.shape[0], C2.shape[0], C1.shape[0], C2.shape[0])

    if symmetric is None:
        symmetric = nx.allclose(
            C1, nx.transpose(C2, axes=(0, 1)), atol=1e-10
        ) and nx.allclose(C2, nx.transpose(C2, axes=(0, 1)), atol=1e-10)
    if not symmetric:
        loss_T = init_matrix_crude(
            nx.transpose(C1, axes=(0, 1)),
            nx.transpose(C2, axes=(0, 1)),
            loss_fun,
        )

    cpt = 0
    err = 1

    if warmstart:
        # initialize potentials to cope with ot.sinkhorn initialization
        N1, N2 = C1.shape[0], C2.shape[0]
        mu = nx.zeros(N1, type_as=C1) - np.log(N1)
        nu = nx.zeros(N2, type_as=C2) - np.log(N2)

    if log:
        log = {"err": []}

    while err > tol and cpt < max_iter:

        Tprev = T

        # compute the gradient
        if symmetric:
            tens = gwggrad_crude(loss, T, nx)
        else:
            tens = 0.5 * (gwggrad_crude(loss, T, nx) + gwggrad_crude(loss_T, T, nx))

        # if solver == 'PPA':
        #    tens = tens - epsilon * nx.log(T)

        if warmstart:
            T, loginn = sinkhorn(
                p,
                q,
                tens,
                epsilon,
                method="sinkhorn",
                log=True,
                warmstart=(mu, nu),
                **kwargs,
            )
            mu = epsilon * nx.log(loginn["u"])
            nu = epsilon * nx.log(loginn["v"])

        else:
            T = sinkhorn(p, q, tens, epsilon, method="sinkhorn", **kwargs)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations
            err = nx.norm(T - Tprev)

            if log:
                log["err"].append(err)

            if verbose:
                if cpt % 200 == 0:
                    print("{:5s}|{:12s}".format("It.", "Err") + "\n" + "-" * 19)
                print("{:5d}|{:8e}|".format(cpt, err))

        cpt += 1

    if abs(nx.sum(T) - 1) > 1e-5:
        warnings.warn(
            "Solver failed to produce a transport plan. You might "
            "want to increase the regularization parameter `epsilon`."
        )
    if log:
        log["gw_dist"] = gwloss_crude(loss, T, nx)
        return T, log
    else:
        return T


def entropic_gromov_wasserstein2_crude(
    C1,
    C2,
    p=None,
    q=None,
    loss_fun="square_loss",
    epsilon=0.1,
    symmetric=None,
    G0=None,
    max_iter=1000,
    tol=1e-9,
    solver="PGD",
    warmstart=False,
    verbose=False,
    log=False,
    **kwargs,
):
    r"""
        Returns the Gromov-Wasserstein loss :math:`\mathbf{GW}` between :math:`(\mathbf{C_1}, \mathbf{p})` and :math:`(\mathbf{C_2}, \mathbf{q})`
        estimated using Sinkhorn projections *for any loss functions*.

    If `solver="PGD"`, the function solves the following entropic-regularized
    Gromov-Wasserstein optimization problem using Projected Gradient Descent [12]:

    .. math::
        \mathbf{T}^* \in \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l} - \epsilon H(\mathbf{T})

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0

    Else if `solver="PPA"`, the function solves the following Gromov-Wasserstein
    optimization problem using Proximal Point Algorithm [51]:

    .. math::
        \mathbf{T}^* \in \mathop{\arg\min}_\mathbf{T} \quad \sum_{i,j,k,l} L(\mathbf{C_1}_{i,k}, \mathbf{C_2}_{j,l}) \mathbf{T}_{i,j} \mathbf{T}_{k,l}

        s.t. \ \mathbf{T} \mathbf{1} &= \mathbf{p}

             \mathbf{T}^T \mathbf{1} &= \mathbf{q}

             \mathbf{T} &\geq 0
    Where :

    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{p}`: distribution in the source space
    - :math:`\mathbf{q}`: distribution in the target space
    - `L`: loss function to account for the misfit between the similarity matrices
    - `H`: entropy

    .. note:: If the inner solver `ot.sinkhorn` did not convergence, the
        optimal coupling :math:`\mathbf{T}` returned by this function does not
        necessarily satisfy the marginal constraints
        :math:`\mathbf{T}\mathbf{1}=\mathbf{p}` and
        :math:`\mathbf{T}^T\mathbf{1}=\mathbf{q}`. So the returned
        Gromov-Wasserstein loss does not necessarily satisfy distance
        properties and may be negative.

    Parameters
    ----------
    C1 : array-like, shape (ns, ns, d)
        Metric cost matrix in the source space with d-dimensional tensors as elements.
    C2 : array-like, shape (nt, nt, d)
        Metric cost matrix in the target space with d-dimensional tensors as elements.
    p : array-like, shape (ns,), optional
        Distribution in the source space.
        If let to its default value None, uniform distribution is taken.
    q : array-like, shape (nt,), optional
        Distribution in the target space.
        If let to its default value None, uniform distribution is taken.
    loss_fun : function: :math:`\mathbb{R}^{d} \times \mathbb{R}^{d} \mapsto \mathbb{R}`
        Loss function used for the distance. It is expected that the loss function is vectorized. That is,

    # TODO FIX THE DOCSTRING FOR ALL LOSS_FUN

    epsilon : float, optional
        Regularization term >0
    symmetric : bool, optional
        Either C1 and C2 are to be assumed symmetric or not.
        If let to its default None value, a symmetry test will be conducted.
        Else if set to True (resp. False), C1 and C2 will be assumed symmetric (resp. asymmetric).
    G0: array-like, shape (ns,nt), optional
        If None the initial transport plan of the solver is pq^T.
        Otherwise G0 will be used as initial transport of the solver. G0 is not
        required to satisfy marginal constraints but we strongly recommend it
        to correctly estimate the GW distance.
    max_iter : int, optional
        Max number of iterations
    tol : float, optional
        Stop threshold on error (>0)
    solver: string, optional
        Solver to use either 'PGD' for Projected Gradient Descent or 'PPA'
        for Proximal Point Algorithm.
        Default value is 'PGD'.
    warmstart: bool, optional
        Either to perform warmstart of dual potentials in the successive
        Sinkhorn projections.
    verbose : bool, optional
        Print information along iterations
    log : bool, optional
        Record log if True.
    **kwargs: dict
        parameters can be directly passed to the ot.sinkhorn solver.
        Such as `numItermax` and `stopThr` to control its estimation precision,
        e.g [51] suggests to use `numItermax=1`.
        Returns
        -------
        gw_dist : float
            Gromov-Wasserstein distance

        References
        ----------
        .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
            "Gromov-Wasserstein averaging of kernel and distance matrices."
            International Conference on Machine Learning (ICML). 2016.

        .. [51] Xu, H., Luo, D., Zha, H., & Duke, L. C. (2019). Gromov-wasserstein
            learning for graph matching and node embedding. In International
            Conference on Machine Learning (ICML), 2019.
    """

    T, logv = entropic_gromov_wasserstein_crude(
        C1,
        C2,
        p,
        q,
        loss_fun,
        epsilon,
        symmetric,
        G0,
        max_iter,
        tol,
        solver,
        warmstart,
        verbose,
        log=True,
        **kwargs,
    )

    logv["T"] = T

    if log:
        return logv["gw_dist"], logv
    else:
        return logv["gw_dist"]
