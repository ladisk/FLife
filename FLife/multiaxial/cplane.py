import numpy as np
import scipy.optimize as opt

def _plane_stress_to_voigt_3d(multiaxial_psd):
    """Promote plane-stress PSD [sx, sy, txy] to 3D Voigt [sx, sy, sz, txy, txz, tyz].

    The critical-plane implementation operates on 6x6 Voigt stress transforms.
    For documented 2D PSD input, treat the out-of-plane terms as zero and embed
    the in-plane terms at Voigt indices [0, 1, 3].
    """
    if multiaxial_psd.shape[-2:] == (6, 6):
        return multiaxial_psd
    if multiaxial_psd.shape[-2:] != (3, 3):
        raise ValueError('Input Error. PSD matrix should be the size of (f,6,6) for 3D stress state or (f,3,3) for 2D stress state')

    psd_3d = np.zeros(multiaxial_psd.shape[:-2] + (6, 6), dtype=multiaxial_psd.dtype)
    plane_stress_indices = [0, 1, 3]

    for row_2d, row_3d in enumerate(plane_stress_indices):
        for col_2d, col_3d in enumerate(plane_stress_indices):
            psd_3d[..., row_3d, col_3d] = multiaxial_psd[..., row_2d, col_2d]

    return psd_3d

def compute_lmn_angles(theta, phi, psi):
    """Direction cosines of a proper ZXZ-Euler rotation (theta, phi, psi).

    Returns the three rows (l_i, m_i, n_i) of an orthonormal direction-cosine
    matrix, so that (l1,m1,n1), (l2,m2,n2) and (l3,m3,n3) form a right-handed
    orthonormal triad for any input angles.
    """
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)

    l1 = cps * cph - cth * sph * sps
    m1 = cps * sph + cth * cph * sps
    n1 = sps * sth

    l2 = -sps * cph - cth * sph * cps
    m2 = -sps * sph + cth * cph * cps
    n2 = cps * sth

    l3 = sth * sph
    m3 = -sth * cph
    n3 = cth

    return l1, m1, n1, l2, m2, n2, l3, m3, n3


def _maximize(crit, x0, bounds, search_method, n_starts=16):
    """Maximize -crit over the plane orientation.

    'local' uses a deterministic multi-start SLSQP (the default start x0 plus
    seeded random starts) so the search is not trapped at a stationary start
    point; 'global' uses differential evolution. Results are reproducible.
    """
    if search_method == 'global':
        return opt.differential_evolution(crit, bounds=bounds, seed=0)
    elif search_method == 'local':
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        starts = [np.asarray(x0, dtype=float)]
        starts += [rng.uniform(lo, hi) for _ in range(n_starts - 1)]
        best = None
        for s0 in starts:
            res = opt.minimize(crit, s0, method='SLSQP', bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        return best
    else:
        raise ValueError('Search method')


def max_variance_old(multiaxial_psd, df, method, K=None):
    '''Determine critical plane
    based on max variance.

    Notes
    -----
    Took this from article (its older tough):
        - Lagoda, Macha and Nieslony; Fatigue Life Calculation
        by means of the cycle counting and spectral methods
        under multiaxial random loading.
    '''
    mu = np.sum(multiaxial_psd, axis=0) * df

    if 'maxnormal' == method:

        def crit(v):
            """Criterion for optimization."""
            v = v / np.sqrt(np.sum(v ** 2))
            l, m, n = v
            a = np.asarray([l**2, m**2, n**2,
                            2*l*m, 2*l*n, 2*m*n])

            return -np.abs(np.einsum('i,ij,j', a, mu, a).real)


        res = opt.minimize(fun=crit, x0=[0.33,0.33,0.33], method='SLSQP', bounds=[(0,1),(0,1),(0,1)], options={'disp':False})

    return res['x'] / np.sqrt(np.sum(res['x']**2))

# Plane search shared by the critical-plane criteria
#
# The variance of a stress component on a plane is a quadratic form of the
# stress covariance. With the traction covariance M(n) = Cov(t, t), t = sigma n,
# all three max-variance objectives reduce to a function of the plane normal n
# alone (s is the shear direction, s _|_ n):
#   maxnormal:       Var(sigma_nn)                    = n^T M n
#   maxshear:        max_s Var(2 tau_ns)              = 4 max_s s^T M s
#   maxnormalshear:  max_s Var(2 tau_ns + 2 K sigma_nn) / (1 + K)^2
# The inner maximum over s is solved in closed form. The outer search over n is
# a grid on the half sphere for all nodes at once, followed by a vectorised
# compass search on the sphere from the best grid point of each node.

_VOIGT_INDEX = np.array([[0, 3, 4],
                         [3, 1, 5],
                         [4, 5, 2]])
_VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]


def _covariance(multiaxial_psd, df):
    """Real covariance (..., 6, 6) of the Voigt stress, from a PSD (..., f, 6, 6)."""
    multiaxial_psd = _plane_stress_to_voigt_3d(multiaxial_psd)
    return np.real(np.sum(multiaxial_psd, axis=-3)) * df


def _normalise(*covariances):
    """Divide covariances by the trace of the first one (per node).

    The critical plane depends only on the relative stress distribution, so a
    common scale factor keeps the objective O(1) for any input magnitude.
    """
    scale = np.trace(covariances[0], axis1=-2, axis2=-1)
    scale = np.where(scale > 0, scale, 1.)[..., None, None]
    return [c / scale for c in covariances]


def _traction_operator(cov):
    """Matrix (N, 9, 9) that maps n_i n_k to the traction covariance M_jl (flattened)."""
    full = cov[..., _VOIGT_INDEX[:, :, None, None], _VOIGT_INDEX[None, None, :, :]]  # C_ijkl
    return full.transpose(0, 2, 4, 1, 3).reshape(len(cov), 9, 9)


def _traction_covariance(op, n):
    """Traction covariance M (N, K, 3, 3) for normals n (N, K, 3)."""
    nn = (n[..., :, None] * n[..., None, :]).reshape(n.shape[:-1] + (9,))
    M = np.matmul(nn, op.transpose(0, 2, 1))
    return M.reshape(M.shape[:-1] + (3, 3))


def _normal_vector(n):
    """Voigt coefficients a with a . sigma = n^T sigma n (tensor shear components)."""
    l, m, k = n[..., 0], n[..., 1], n[..., 2]
    return np.stack([l*l, m*m, k*k, 2*l*m, 2*l*k, 2*m*k], axis=-1)


def _shear_vector(s, n):
    """Voigt coefficients a with a . sigma = s^T sigma n."""
    return np.stack([s[..., 0]*n[..., 0], s[..., 1]*n[..., 1], s[..., 2]*n[..., 2],
                     s[..., 0]*n[..., 1] + s[..., 1]*n[..., 0],
                     s[..., 0]*n[..., 2] + s[..., 2]*n[..., 0],
                     s[..., 1]*n[..., 2] + s[..., 2]*n[..., 1]], axis=-1)


def _quadratic_form(a, cov, b=None):
    """a^T C b over the last axis; a, b (N, K, 6) and C (N, 6, 6)."""
    b = a if b is None else b
    return np.einsum('nki,nij,nkj->nk', a, cov, b)


def _basis(n):
    """Unit vectors u, w so that (n, u, w) is a right-handed orthonormal triad."""
    use_x = np.abs(n[..., 0]) < 0.9
    a = np.zeros_like(n)
    a[..., 0] = use_x
    a[..., 1] = ~use_x
    u = a - np.sum(a * n, axis=-1, keepdims=True) * n
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    return u, np.cross(n, u)


def _quad3(M, a, b):
    """a^T M b over the last axes (broadcasting); faster than einsum when a, b are shared."""
    ab = a[..., :, None] * b[..., None, :]
    return np.sum((M * ab).reshape(np.broadcast_shapes(M.shape, ab.shape)[:-2] + (9,)), axis=-1)


def _max_on_circle(m11, m12, m22, q1, q2, newton_steps=12):
    """Maximise x^T m x + 2 q^T x over unit vectors x in 2D.

    Solved with the secular equation of the trust-region subproblem.
    Returns the maximum and x (in the original 2D basis).
    """
    c = 0.5 * (m11 + m22)
    d = 0.5 * (m11 - m22)
    rad = np.hypot(d, m12)
    mu1, mu2, delta = c + rad, c - rad, 2 * rad
    # eigenvector (ct, st) of mu1, without trigonometric functions
    pos = d >= 0
    ct = np.where(pos, d + rad, m12)
    st = np.where(pos, m12, rad - d)
    norm = np.hypot(ct, st)
    zero = norm == 0
    ct = np.where(zero, 1., ct / np.where(zero, 1., norm))
    st = np.where(zero, 0., st / np.where(zero, 1., norm))
    p1 = ct * q1 + st * q2                      # q in the eigenvector basis
    p2 = -st * q1 + ct * q2

    if newton_steps:
        # Root lam - mu1 = t >= 0 of p1^2/t^2 + p2^2/(t + delta)^2 = 1. Newton on
        # 1/|x(t)| - 1, which is concave and increasing, from a lower bound of
        # the root converges monotonically.
        a1, a2 = p1**2, p2**2
        t = np.maximum(np.abs(p1), np.abs(p2) - delta)
        upper = np.sqrt(a1 + a2)
        with np.errstate(divide='ignore', invalid='ignore'):
            for _ in range(newton_steps):
                tb = t + delta
                f1 = a1 / t**2
                f2 = a2 / tb**2
                f1[t == 0] = 0.
                f2[tb == 0] = 0.
                F = f1 + f2
                step = F * (1 - np.sqrt(F)) / (f1 / t + f2 / tb)    # (F^-1/2 - 1) / (F^-3/2 (f1/t + f2/tb))
                t = np.clip(np.where(np.isfinite(step), t - step, t), 0., upper)
            x2 = p2 / (t + delta)
        x2 = np.clip(np.nan_to_num(x2, posinf=0., neginf=0.), -1., 1.)
    else:
        x2 = np.zeros_like(p1)
    sign = np.where(p1 >= 0, 1., -1.)
    x1 = sign * np.sqrt(1 - x2**2)
    value = mu1 * x1**2 + mu2 * x2**2 + 2 * (p1 * x1 + p2 * x2)

    # fall back to the top eigenvector (the exact answer in the hard case)
    fallback = mu1 + 2 * np.abs(p1)
    use_fallback = fallback > value
    value = np.where(use_fallback, fallback, value)
    x1 = np.where(use_fallback, sign, x1)
    x2 = np.where(use_fallback, 0., x2)
    return value, ct * x1 - st * x2, st * x1 + ct * x2


def _top_eigenvalue(m11, m12, m22):
    return 0.5 * (m11 + m22) + np.hypot(0.5 * (m11 - m22), m12)


def _objective_from_scalars(p, m11, m12, m22, r1, r2, method, K, newton_steps=12, value_only=False):
    """Max-variance objective from p = n^T M n, m = B^T M B and r = B^T M n, B = (u, w).

    Returns the objective and the shear direction (x1, x2) in the basis (u, w).
    """
    if method == 'maxnormal':
        return p, None, None
    k = 0. if method == 'maxshear' else K
    if k == 0 and value_only:
        return 4 * _top_eigenvalue(m11, m12, m22), None, None
    value, x1, x2 = _max_on_circle(m11, m12, m22, k * r1, k * r2,
                                   newton_steps=newton_steps if k != 0 else 0)
    return 4 * (value + k**2 * p) / (1 + k)**2, x1, x2


def _plane_objective(M, n, method, K, newton_steps=12, value_only=False):
    """Max-variance objective for normals n (..., 3) with traction covariance M (..., 3, 3).

    Returns the objective and the shear direction s (None for 'maxnormal').
    """
    p = _quad3(M, n, n)
    if method == 'maxnormal':
        return p, None
    u, w = _basis(n)
    value, x1, x2 = _objective_from_scalars(p, _quad3(M, u, u), _quad3(M, u, w), _quad3(M, w, w),
                                            _quad3(M, u, n), _quad3(M, w, n), method, K, newton_steps,
                                            value_only=value_only)
    if value_only:
        return value, None
    return value, x1[..., None] * u + x2[..., None] * w


def _grid_objective(op, grid, method, K, newton_steps=3):
    """Objective for all nodes (op (N, 9, 9)) on a shared grid of normals (G, 3), shape (N, G).

    Each quadratic form b^T M(n) a is linear in the entries of op, so all of
    them come from one matrix product.
    """
    G = len(grid)
    nn = (grid[:, :, None] * grid[:, None, :]).reshape(G, 9)
    if method == 'maxnormal':
        pairs = [(grid, grid)]
    else:
        u, w = _basis(grid)
        pairs = [(grid, grid), (u, u), (u, w), (w, w), (u, grid), (w, grid)]
    P = np.concatenate([((a[:, :, None] * b[:, None, :]).reshape(G, 9)[:, :, None] * nn[:, None, :]).reshape(G, 81)
                        for a, b in pairs]).T
    scalars = (op.reshape(len(op), 81) @ P).reshape(len(op), len(pairs), G)
    if method == 'maxnormal':
        return scalars[:, 0]
    return _objective_from_scalars(*np.moveaxis(scalars, 1, 0), method, K, newton_steps, value_only=True)[0]


def _half_sphere_grid(n_points):
    """Fibonacci grid of unit normals on the half sphere z > 0 (n and -n give the same plane)."""
    k = np.arange(n_points) + 0.5
    z = 1 - k / n_points
    r = np.sqrt(1 - z**2)
    phi = np.pi * (1 + 5**0.5) * k
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)


def _sphere_search(objective, n_nodes, n_grid=2000, tol=1e-8, max_iter=300, chunk=1000000,
                   grid_objective=None):
    """Maximise objective(normals (M, K, 3), nodes (M,)) -> (M, K) over the unit sphere, per node.

    grid_objective (same call) may be a cheaper lower bound used only to pick the start point.
    On the grid, the normals have shape (1, G, 3) and must broadcast against the nodes.

    Grid on the half sphere for all nodes at once, then a compass search in the
    tangent plane with step halving. The result is never worse than the best
    grid point.
    """
    grid = _half_sphere_grid(n_grid)
    nodes = np.arange(n_nodes)
    n = np.empty((n_nodes, 3))
    val = np.empty(n_nodes)
    step = max(1, chunk // n_grid)
    for i in range(0, n_nodes, step):
        idx = nodes[i:i+step]
        v = (grid_objective or objective)(grid[None], idx)   # (1, G, 3) broadcasts over nodes
        n[idx] = grid[np.argmax(v, axis=1)]
    val = objective(n[:, None], nodes)[:, 0]

    # Local refinement in the tangent plane (coordinates x, y along u, w): the 3x3
    # stencil gives the gradient and Hessian by central differences; the Newton
    # step is tried next to the 8 stencil points and the best point is kept. If
    # nothing improves, the stencil shrinks (compass search), so the value never
    # decreases and the search ends when the stencil is smaller than tol.
    dx, dy = np.meshgrid([-1., 0., 1.], [-1., 0., 1.])
    keep = (dx != 0) | (dy != 0)
    dx, dy = dx[keep], dy[keep]
    ix = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(dx, dy))}
    h = np.full(n_nodes, 2 * np.sqrt(2 * np.pi / n_grid))
    for it in range(max_iter):
        idx = np.flatnonzero(h > tol)
        if idx.size == 0:
            break
        u, w = _basis(n[idx])
        hh = h[idx]
        cand = n[idx, None, :] + hh[:, None, None] * (dx[None, :, None] * u[:, None, :] + dy[None, :, None] * w[:, None, :])

        v = objective(cand / np.linalg.norm(cand, axis=-1, keepdims=True), idx)
        v0 = val[idx]
        gx = (v[:, ix[1, 0]] - v[:, ix[-1, 0]]) / (2 * hh)
        gy = (v[:, ix[0, 1]] - v[:, ix[0, -1]]) / (2 * hh)
        hxx = (v[:, ix[1, 0]] - 2 * v0 + v[:, ix[-1, 0]]) / hh**2
        hyy = (v[:, ix[0, 1]] - 2 * v0 + v[:, ix[0, -1]]) / hh**2
        hxy = (v[:, ix[1, 1]] - v[:, ix[1, -1]] - v[:, ix[-1, 1]] + v[:, ix[-1, -1]]) / (4 * hh**2)
        det = hxx * hyy - hxy**2
        concave = (hxx < 0) & (det > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            sx = np.where(concave, -(hyy * gx - hxy * gy) / det, 0.)
            sy = np.where(concave, -(hxx * gy - hxy * gx) / det, 0.)
        length = np.hypot(sx, sy)
        clip = np.minimum(1., 2 * hh / np.where(length > 0, length, 1.))
        sx, sy = np.nan_to_num(sx * clip), np.nan_to_num(sy * clip)
        newton = n[idx] + sx[:, None] * u + sy[:, None] * w
        newton /= np.linalg.norm(newton, axis=-1, keepdims=True)
        vn = objective(newton[:, None], idx)[:, 0]

        cand = np.concatenate([cand, newton[:, None]], axis=1)
        v = np.concatenate([v, vn[:, None]], axis=1)
        best = np.argmax(v, axis=1)
        vb = v[np.arange(len(idx)), best]
        better = vb > v0 + 1e-15 * np.abs(v0)
        moved = idx[better]
        n[moved] = cand[better, best[better]] / np.linalg.norm(cand[better, best[better]], axis=-1, keepdims=True)
        val[moved] = vb[better]
        # after a Newton move, the next stencil is about as large as the step
        took_newton = better & (best == len(dx))
        step = np.hypot(sx, sy)
        h[idx[took_newton]] = np.minimum(hh[took_newton], np.maximum(2 * step[took_newton], 0.5 * tol))
        h[idx[~better]] *= 0.25
    return n, val


def _frames_from_plane(n, s):
    """Direction cosines (N, 3, 3), rows (l_i, m_i, n_i), i = 1, 2, 3.

    For the normal criterion eta_1 = n. For the shear criteria the plane normal
    is (eta_1 + eta_3)/sqrt(2) and the shear direction (eta_1 - eta_3)/sqrt(2).
    """
    if s is None:
        e1 = n
        e2, e3 = _basis(n)
    else:
        e1 = (n + s) / np.sqrt(2)
        e3 = (n - s) / np.sqrt(2)
        e2 = np.cross(e3, e1)
    return np.stack([e1, e2, e3], axis=-2)


def _euler_criterion(cov, method, K):
    """Objective of the old Euler-angle formulation (for search_method='global')."""
    def crit(v):
        l1, m1, n1, l2, m2, n2, l3, m3, n3 = compute_lmn_angles(*v)
        if method == 'maxnormal':
            a = np.asarray([l1**2, m1**2, n1**2, 2*l1*m1, 2*l1*n1, 2*m1*n1])
        else:
            k = 0. if method == 'maxshear' else K
            a = np.asarray([l1**2 - l3**2 + k * (l1 + l3)**2,
                            m1**2 - m3**2 + k * (m1 + m3)**2,
                            n1**2 - n3**2 + k * (n1 + n3)**2,
                            2 * (l1 * m1 - l3 * m3 + k * (l1 + l3) * (m1 + m3)),
                            2 * (l1 * n1 - l3 * n3 + k * (l1 + l3) * (n1 + n3)),
                            2 * (m1 * n1 - m3 * n3 + k * (m1 + m3) * (n1 + n3))]) / (1 + k)
        return -(a @ cov @ a)
    return crit


def max_variance_planes(multiaxial_psd, df, method, K=None, search_method='local'):
    '''Critical planes by max variance, for one or many nodes.

    :param multiaxial_psd: array (f, 6, 6), (f, 3, 3), (N, f, 6, 6) or (N, f, 3, 3)
    :param method: 'maxnormal', 'maxshear' or 'maxnormalshear' (needs K)
    :param search_method: 'local' (grid on the half sphere plus compass search,
        all nodes at once) or 'global' (additionally differential evolution per
        node; the better plane is kept)
    :returns: direction cosines, shape (3, 3) or (N, 3, 3); row i is (l_i, m_i, n_i)
    '''
    if method not in ('maxnormal', 'maxshear', 'maxnormalshear'):
        raise ValueError('Critical plane method')
    if method == 'maxnormalshear' and K is None:
        raise ValueError('K is needed for method maxnormalshear')
    if search_method not in ('local', 'global'):
        raise ValueError('Search method')

    multiaxial_psd = _plane_stress_to_voigt_3d(np.asarray(multiaxial_psd))
    single = multiaxial_psd.ndim == 3
    cov, = _normalise(_covariance(multiaxial_psd.reshape((-1,) + multiaxial_psd.shape[-3:]), df))
    op = _traction_operator(cov)

    def objective(n, idx, newton_steps=12):
        return _plane_objective(_traction_covariance(op[idx], n), n, method, K, newton_steps, value_only=True)[0]

    n, val = _sphere_search(objective, len(cov),
                            grid_objective=lambda n, idx: _grid_objective(op[idx], n[0], method, K))
    _, s = _plane_objective(_traction_covariance(op, n[:, None]), n[:, None], method, K)
    frames = _frames_from_plane(n, None if s is None else s[:, 0])

    if search_method == 'global':
        bounds = [(0, np.pi), (0, 2*np.pi), (0, 2*np.pi)]
        for i in range(len(cov)):
            crit = _euler_criterion(cov[i], method, K)
            res = opt.differential_evolution(crit, bounds=bounds, seed=0)
            if -res.fun > val[i]:
                frames[i] = np.array(compute_lmn_angles(*res.x)).reshape(3, 3)
                val[i] = -res.fun

    return frames[0] if single else frames


def max_variance(multiaxial_psd, df, method, K=None, search_method='local'):
    '''Determine critical plane
    based on max variance.

    returns: l1, m1, n1, l2, m2, n2, l3, m3, n3

        -Lagoda, Macha and Nieslony; Fatigue Life Calculation
        by means of the cycle counting and spectral methods
        under multiaxial random loading.

        -Bȩdkowski, W., and E. Macha; Fatigue fracture plane
        under multiaxial random loadings–prediction by variance
        of equivalent stress based on the maximum shear and normal stresses.
    '''
    frames = max_variance_planes(multiaxial_psd, df, method, K=K, search_method=search_method)
    return tuple(frames.ravel())


# Carpinteri-Spagnoli
#
# 6x6 Voigt stress transformation matrices derived from first principles.
# Stress vector ordering: {sigma_x, sigma_y, sigma_z, tau_xy, tau_xz, tau_yz}
#
# Direction cosine matrix convention (passive/coordinate transform):
#   Q_z(a) = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
#   Q_x(a) = [[1, 0, 0], [0, c, s], [0, -s, c]]
# where c = cos(a), s = sin(a).
#
# The C-S frame is the ZXZ rotation Tz(chi) Tx(delta) Tz(psi) Tx(theta) Tz(phi):
#   phi, theta -- direction of the 1-hat axis (w)
#   psi        -- rotation about w (shear direction v)
#   delta      -- material off-angle about the u axis (Eq. 19)
#   chi        -- rotation about the tilted w (shear direction on the critical plane)
# cs_planes finds the same frame by working on the axis vectors directly, which
# avoids the Euler-angle singularities. voigt_transform builds the 6x6 matrix.

def Tz(c, s):
    '''6x6 stress transformation for rotation about Z-axis.
    Leaves sigma_z (index 2) invariant.

    Parameters
    ----------
    c : float
        cos(angle)
    s : float
        sin(angle)
    '''
    return np.array([
        [ c**2,  s**2, 0,  2*c*s,      0,  0],
        [ s**2,  c**2, 0, -2*c*s,      0,  0],
        [ 0,     0,    1,  0,           0,  0],
        [-c*s,   c*s,  0,  c**2-s**2,  0,  0],
        [ 0,     0,    0,  0,           c,  s],
        [ 0,     0,    0,  0,          -s,  c]])

def Tx(c, s):
    '''6x6 stress transformation for rotation about X-axis.
    Leaves sigma_x (index 0) invariant.

    Parameters
    ----------
    c : float
        cos(angle)
    s : float
        sin(angle)
    '''
    return np.array([
        [1,  0,     0,    0,    0,     0        ],
        [0,  c**2,  s**2, 0,    0,     2*c*s    ],
        [0,  s**2,  c**2, 0,    0,    -2*c*s    ],
        [0,  0,     0,    c,    s,     0        ],
        [0,  0,     0,   -s,    c,     0        ],
        [0, -c*s,   c*s,  0,    0,     c**2-s**2]])


def spectral_moment(f, multiaxial_psd, n):
    """Calculate n-th spectral moment of a PSD.

    f: frequency array
    s: PSD array
    n: order of spectral moment
    
    Returns
    -------
    Scalar.
    """
    df = f[1] - f[0]
    
    #return np.tensordot(f**n, s, ([0], [0])) * df
    return np.einsum('i,i...->...', f**n, multiaxial_psd) * df


def voigt_transform(Q):
    '''6x6 Voigt stress transformation for direction cosines Q (..., 3, 3).

    sigma' = Q sigma Q^T in Voigt form {sigma_x, sigma_y, sigma_z, tau_xy, tau_xz, tau_yz}.
    Tz and Tx are special cases.
    '''
    Q = np.asarray(Q)
    T = np.empty(Q.shape[:-2] + (6, 6))
    for I, (i, j) in enumerate(_VOIGT_PAIRS):
        for J, (k, l) in enumerate(_VOIGT_PAIRS):
            T[..., I, J] = Q[..., i, k] * Q[..., j, l]
            if k != l:
                T[..., I, J] += Q[..., i, l] * Q[..., j, k]
    return T


def _davenport_extreme(k0, k2, T):
    """Davenport expected maximum sqrt(k0) * (sqrt(2 ln N1) + 0.5772 / sqrt(2 ln N1)).

    N1 = nu0 * T with nu0 = sqrt(k2 / k0) the mean up-crossing rate (moments in
    Hz). As before, the value is 0 where N1 <= 1 (the formula is undefined there).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        N1 = np.sqrt(np.where(k0 > 0, k2 / k0, 0.)) * T
        x = np.sqrt(2 * np.log(np.where(N1 > 1, N1, np.e)))
        value = np.sqrt(np.maximum(k0, 0.)) * (x + 0.5772 / x)
    return np.where((N1 > 1) & (k2 > 0), value, 0.)


def _cs_delta(s_af, tau_af):
    """Off-angle delta, Eq. 19; limited to [0, pi/4] as in the authors' code (C_S_Spettrale.m:382-393)."""
    r = tau_af / s_af
    if r < 1 / np.sqrt(3):
        return np.pi / 4
    elif r > 1:
        return 0.
    return 3 * np.pi * (1 - r**2) / 8


def _matlab_sign(x, y, z):
    """+1 where the vector comes first in the authors' grid order, else -1.

    In C_S_Spettrale.m the first strict maximum wins, so of two equivalent
    directions the one with the smaller angle (phi or theta) is kept.
    """
    first = np.where(y != 0, y > 0, np.where(x != 0, x > 0, z >= 0))
    return np.where(first, 1., -1.)


def cs_planes(multiaxial_psd, f, s_af, tau_af, T=1.):
    '''Critical-plane frames of the C-S random criterion, for one or many nodes.

    :param multiaxial_psd: array (f, 6, 6), (f, 3, 3), (N, f, 6, 6) or (N, f, 3, 3)
    :param f: frequency vector in Hz
    :param T: observation time in s, for the number of cycles N1 = nu0 * T
    :returns: direction cosines (3, 3) or (N, 3, 3), rows (u, v, w); w is the
        normal of the critical plane and v the shear direction (tau_vw).

    The steps follow the authors' MATLAB code C_S_Spettrale.m (Carpinteri,
    Spagnoli, Vantadori) and Marques et al. (2020), Sec. 2:
        1. 1-hat maximises the Davenport expected maximum of the normal stress,
           sqrt(lambda0) (sqrt(2 ln N1) + 0.5772 / sqrt(2 ln N1)), with
           N1 = nu0 T and nu0 = sqrt(lambda2 / lambda0) / (2 pi), lambda_m in
           rad/s (= sqrt(m2 / m0) with moments in Hz). The authors scan a 10 deg
           grid; here a finer grid plus a local refinement is used.
        2. 3-hat _|_ 1-hat maximises the variance of the shear stress on the
           plane 1-hat (closed form instead of the 10 deg scan).
        3. w = cos(delta) 1-hat + sin(delta) 3-hat (Eq. 19 for delta).
        4. v _|_ w maximises the variance of tau_vw (closed form).
    Where two directions are equivalent (n and -n), the one the authors' scan
    meets first is kept, so the frame convention matches C_S_Spettrale.m.
    '''
    multiaxial_psd = _plane_stress_to_voigt_3d(np.asarray(multiaxial_psd))
    single = multiaxial_psd.ndim == 3
    psd = multiaxial_psd.reshape((-1,) + multiaxial_psd.shape[-3:])
    f = np.asarray(f, dtype=float)
    m0 = np.real(spectral_moment(f, np.moveaxis(psd, 1, 0), 0))
    m2 = np.real(spectral_moment(f, np.moveaxis(psd, 1, 0), 2))
    m0, m2 = _normalise(m0, m2)
    op = _traction_operator(m0)

    def objective(n, idx):
        a = _normal_vector(n)
        return _davenport_extreme(_quadratic_form(a, m0[idx]), _quadratic_form(a, m2[idx]), T)

    # 1. 1-hat, as (sin g cos p, sin g sin p, cos g) with 0 <= p <= 2 pi, 0 <= g <= pi
    n1, _ = _sphere_search(objective, len(psd))
    n1 = n1 * _matlab_sign(n1[:, 0], n1[:, 1], n1[:, 2])[:, None]
    phi = np.arctan2(n1[:, 1], n1[:, 0])
    phi = np.where(np.hypot(n1[:, 0], n1[:, 1]) > 0, phi, 0.)
    a_v = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=-1)   # C_S_Spettrale.m:253-257
    a_u = np.cross(n1, a_v)                                                      # C_S_Spettrale.m:265-269

    # 2. 3-hat = cos(theta) a_u + sin(theta) a_v, 0 <= theta <= pi (C_S_Spettrale.m:320-360)
    M = _traction_covariance(op, n1[:, None])[:, 0]
    zero = np.zeros(len(n1))
    _, x1, x2 = _max_on_circle(_quad3(M, a_u, a_u), _quad3(M, a_u, a_v), _quad3(M, a_v, a_v),
                               zero, zero, newton_steps=0)
    sign = _matlab_sign(zero, x2, x1)
    n3 = sign[:, None] * (x1[:, None] * a_u + x2[:, None] * a_v)
    n2 = np.cross(n1, n3)                                                        # C_S_Spettrale.m:369-373

    # 3. critical plane normal (C_S_Spettrale.m:382-411)
    delta = _cs_delta(s_af, tau_af)
    w = np.cos(delta) * n1 + np.sin(delta) * n3

    # 4. shear direction on the critical plane (C_S_Spettrale.m:473-525)
    k = np.cross(w, n2)
    M = _traction_covariance(op, w[:, None])[:, 0]
    _, x1, x2 = _max_on_circle(_quad3(M, n2, n2), _quad3(M, n2, k), _quad3(M, k, k),
                               zero, zero, newton_steps=0)
    v = x1[:, None] * n2 + x2[:, None] * k

    frames = np.stack([np.cross(v, w), v, w], axis=-2)
    return frames[0] if single else frames


def csrandom(multiaxial_psd, df, s_af, tau_af, f=None, T=1.):
    '''Determine critical plane with the C-S random criterion.

    Returns the 6x6 Voigt transformation CC to the critical-plane frame:
    after transformation, sigma_w = S[2,2] and tau_vw = S[5,5]. See
    `cs_planes` for the steps.

    :param f: frequency vector in Hz; defaults to arange(len) * df
    :param T: time span in s for the number of cycles N1 = nu0 * T

    See also
    --------
    Carpinteri A, Spagnoli A and Vantadori S, Reformulation in the frequency
    domain of a critical plane-based multiaxial fatigue criterion,
    Int J Fat, 2014
    '''
    multiaxial_psd = np.asarray(multiaxial_psd)
    if f is None:
        f = np.arange(multiaxial_psd.shape[-3]) * df
    return voigt_transform(cs_planes(multiaxial_psd, f, s_af, tau_af, T=T))
