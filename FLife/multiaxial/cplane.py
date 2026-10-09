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
# The C-S algorithm uses ZXZ Euler angles (phi, theta, psi):
#   C_phi  = Tz(phi)   -- rotation about Z
#   C_theta = Tx(theta) -- rotation about X (line of nodes)
#   C_psi  = Tz(psi)   -- rotation about Z' (leaves sigma_z invariant)
# Plus two more rotations for the critical plane:
#   C_delta = Tx(delta) -- rotation about X' (2-hat axis)
#   C_chi   = Tz(chi)   -- rotation about Z (w-axis, leaves sigma_w invariant)

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


def csrandom(multiaxial_psd, df, s_af, tau_af):
    '''Determine critical plane with the C-S random criterion.

    Uses ZXZ Euler angles with 6x6 Voigt stress transformations derived
    from first principles. The 5-angle sequence is:
        phi (about Z), theta (about X), psi (about Z'),
        delta (about X', material angle), chi (about Z/w-axis).

    After transformation, sigma_w = S[2,2] and tau_vw = S[5,5].

    See also
    --------
    Carpinteri A, Spagnoli A and Vantadori S, Reformulation in the frequency
    domain of a critical plane-based multiaxial fatigue criterion,
    Int J Fat, 2014
    '''
    multiaxial_psd = _plane_stress_to_voigt_3d(multiaxial_psd)
    f = np.arange(0, multiaxial_psd.shape[0] * df, df)
    l0 = spectral_moment(f, multiaxial_psd, 0)
    m2 = spectral_moment(f, multiaxial_psd, 2)

    # Normalise the moments before the plane search. The critical plane depends
    # only on the relative stress distribution, but SLSQP uses absolute
    # tolerances, so an objective that scales with the PSD magnitude converges to
    # a slightly different plane at different load levels. Scaling l0 and m2 by a
    # common factor keeps the objective O(1) and leaves both the plane and the
    # moment ratio in N1 unchanged. The equivalent stress is computed later in
    # _cs from the unscaled stress.
    scale = np.real(np.trace(l0))
    if scale > 0:
        l0 = l0 / scale
        m2 = m2 / scale

    # Step 2: Find 1-hat direction (phi, theta).
    # Maximize Davenport expected extreme of sigma_z at [2,2].
    # C = Tx(theta) @ Tz(phi)  (psi=0 => Tz(psi) = I)
    def s33crit(inp):
        phi, th = inp
        C = Tx(np.cos(th), np.sin(th)) @ Tz(np.cos(phi), np.sin(phi))
        l0_ = C @ l0 @ C.T
        m2_ = C @ m2 @ C.T
        k0 = np.real(l0_[2, 2])
        k2 = np.real(m2_[2, 2])
        if k0 <= 0 or k2 <= 0:
            return 0.0
        N1 = np.sqrt(k2 / k0) / (2 * np.pi)
        if N1 <= 1:
            return 0.0
        return -(np.sqrt(k0) * (np.sqrt(2 * np.log(N1))
                 + 0.5772 / np.sqrt(2 * np.log(N1))))

    res = opt.minimize(s33crit, [0.33, 0.33], method='SLSQP',
                       bounds=[(0, 2*np.pi), (0, np.pi)])
    phi, th = res['x']

    # Step 3: Find 3-hat direction (psi).
    # Maximize variance of tau_yz at [5,5].
    # Tz(psi) leaves sigma_z invariant, so normal stress is preserved.
    def taucrit(psi):
        C = (Tz(np.cos(psi), np.sin(psi))
             @ Tx(np.cos(th), np.sin(th))
             @ Tz(np.cos(phi), np.sin(phi)))
        return -np.real((C @ l0 @ C.T)[5, 5])

    res = opt.minimize_scalar(taucrit, 0.33, method='bounded',
                              bounds=(0, 2*np.pi))
    psi = res['x'].real

    # Step 4: Off-angle delta (Eq. 19).
    delta = 3 * np.pi * (1 - (tau_af / s_af)**2) / 8

    # Build 3-angle rotation (phi, theta, psi).
    C = (Tz(np.cos(psi), np.sin(psi))
         @ Tx(np.cos(th), np.sin(th))
         @ Tz(np.cos(phi), np.sin(phi)))

    # Step 5: Find chi — maximize variance of tau_vw at [5,5].
    # CC = Tz(chi) @ Tx(delta) @ C
    # Tz(chi) leaves sigma_w invariant after delta tilt.
    def tau2crit(chi):
        CC = (Tz(np.cos(chi), np.sin(chi))
              @ Tx(np.cos(delta), np.sin(delta))
              @ C)
        return -np.real((CC @ l0 @ CC.T)[5, 5])

    res = opt.minimize_scalar(tau2crit, 0.33, method='bounded',
                              bounds=(0, 2*np.pi))
    chi = res['x'].real

    CC = (Tz(np.cos(chi), np.sin(chi))
          @ Tx(np.cos(delta), np.sin(delta))
          @ C)
    return CC
