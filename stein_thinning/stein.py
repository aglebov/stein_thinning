"""Core functions of Stein Points."""

import jax.numpy as np
from jax import jit, vmap, grad, jacfwd
from jax.ops import index_update

def fmin_grid(vf, x, vfs, grid):
    s = vfs(grid)
    val = vf(grid, s)
    i = np.argmin(val)
    return grid[i], s[i], grid.shape[0]

def fk_imq(a, b, linv):
    amb = a - b
    return (1 + np.dot(np.dot(amb, linv), amb)) ** (-0.5)

def make_kfuncs(fk):
    # Gradient functions
    fdka = jit(grad(fk, argnums=0))
    fdkb = jit(grad(fk, argnums=1))
    fjac = jit(jacfwd(fdka, argnums=1))
    fd2k = lambda a, b: np.diagonal(fjac(a, b))

    # Stein kernel
    @jit
    def fk0(a, b, sa, sb):
        k0i = fd2k(a, b) + \
            sa * fdkb(a, b) + \
            sb * fdka(a, b) + \
            sa * sb * fk(a, b)
        return np.sum(k0i)

    # Vectorised Stein kernel
    vfk0 = vmap(fk0, (0, 0, 0, 0), 0)

    # Partial sum
    def fps(x_new, s_new, x, s, i):
        a = np.tile(x_new, (i, 1))
        b = x[0:i]
        sa = np.tile(s_new, (i, 1))
        sb = s[0:i]
        k0ab = vfk0(a, b, sa, sb)
        k0aa = fk0(x_new, x_new, s_new, s_new)
        return np.sum(k0ab) * 2 + k0aa

    # Vectorised partial sum
    vfps = vmap(fps, (0, 0, None, None, None), 0)

    # Return the JAX transformations
    return fk0, vfk0, fps, vfps

def ksd(x, s, fk):
    # JAX kernel transformations
    fk0, _, fps, _ = make_kfuncs(fk)

    n = x.shape[0]
    ks = np.empty(n)

    # Initial point
    ps = fk0(x[0], x[0], s[0], s[0])
    ks = index_update(ks, 0, np.sqrt(ps))
    print('i = 0')

    # Subsequent points
    for i in range(1, n):
        ps += fps(x[i], s[i], x, s, i)
        ks = index_update(ks, i, np.sqrt(ps) / (i + 1))
        print(f'i = {i}')

    return ks

def greedy(d, vfs, fk, fmin, n):
    # JAX kernel transformations
    _, vfk0, _, vfps = make_kfuncs(fk)

    # Returning arrays
    x = np.empty((n, d))
    s = np.empty((n, d))
    e = np.empty(n)

    # Generate initial point
    vf = lambda x_new, s_new: vfk0(x_new, x_new, s_new, s_new)
    x_min, s_min, e_min = fmin(vf, x, vfs)
    x = index_update(x, 0, x_min)
    s = index_update(s, 0, s_min)
    e = index_update(e, 0, e_min)
    print('i = 0')

    # Generate subsequent points
    for i in range(1, n):
        vf = lambda x_new, s_new: vfps(x_new, s_new, x, s, i)
        x_min, s_min, e_min = fmin(vf, x, vfs)
        x = index_update(x, i, x_min)
        s = index_update(s, i, s_min)
        e = index_update(e, i, e_min)
        print(f'i = {i}')

    # Point-set, scores, evaluation-count
    return x, s, e
