# directory of module (delete if necessary):
import sys
sys.path.append("./defect_atom_diag")
# ------------------------------------------

import os
from typing import Tuple, Sequence, Any
import numpy as np
from apply_sym_wan import load_cube_files, load_xsf_files
from matplotlib import pyplot as plt
from scipy.special import sph_harm


def read_files(path: str) -> Tuple[Any, ...]:
    """
    Read all wavefunction files in a directory and dispatch
    to the appropriate loader (xsf or cube).

    Parameters:
        path (str): path to directory containing the files.

    Returns:
        Tuple[Any, ...]: whatever is returned by the loader
    """
    # strip away the last character if necessary:
    if path.endswith("/"):
        path = path[:-1]
    
    filenames = tuple(sorted(os.listdir(path)))

    # print out the files:
    print("Files:")
    for filename in filenames:
        print(filename)
    print()
    
    # handle hex files
    if filenames[0].endswith(".xsf"):
        from apply_sym_wan import load_xsf_files
        return load_xsf_files(filenames, path=path+"/")
        
    # handle cubic files
    from apply_sym_wan import load_cube_files
    return load_cube_files(filenames, path=path+"/")


def construct_axes(delr: Sequence[float], 
    N3: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct 3D coordinate grids centered at zero.

    Parameters:
        delr (Sequence[float]): (dx, dy, dz) grid spacings.
        N3 (Sequence[int]): (Nx, Ny, Nz) grid sizes.

    Returns:
        grids (xx, yy, zz) with shape (Nx, Ny, Nz).
    """
    dx, dy, dz = delr
    Nx, Ny, Nz = N3
    x = (np.arange(Nx) - (Nx-1)/2)*dx
    y = (np.arange(Ny) - (Ny-1)/2)*dy
    z = (np.arange(Nz) - (Nz-1)/2)*dz
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return xx, yy, zz


def integrate(A: np.ndarray, delr: Sequence[float]) -> complex:
    """
    Take the numerical integration (using trapezoid approximation)

    Parameters:
        A (3d array): matrix of values to be integrated
        delr (list[float]): (dx, dy, dz)

    Returns:
        Approximate integral of A over the entire grid.
    """
    return np.trapezoid(
        np.trapezoid(np.trapezoid(A, dx=delr[0], axis=0),
                     dx=delr[1], axis=0),
        dx=delr[2], axis=0
    )


def normalize(A: np.ndarray, delr: Sequence[float]) -> np.ndarray:
    """
    Normalize a wavefunction on the grid defined by delr.

    Parameters:
        A (np.ndarray): 3D wavefunction array.
        delr (Sequence[float]): (dx, dy, dz) grid spacings.

    Returns:
        np.ndarray: normalized wavefunction.
    """
    return A / np.sqrt(integrate(np.conjugate(A) * A, delr))


def apply_Lz(
        psi: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray,
        zz: np.ndarray,
        delr: Sequence[float],
        h_bar: float
    ) -> np.ndarray:
    """
    Apply the Lz operator in real space to a wavefunction.

    Parameters:
        psi (np.ndarray): 3D wavefunction array.
        xx (np.ndarray): x-coordinate grid (same shape as psi).
        yy (np.ndarray): y-coordinate grid (same shape as psi).
        zz (np.ndarray): z-coordinate grid (same shape as psi).
        delr (Sequence[float]): (dx, dy, dz) grid spacings (for np.gradient).
        h_bar (float): reduced Planck constant (ℏ).

    Returns:
        np.ndarray: Lz psi, with the same shape as psi.
    """
    dpsi_dx, dpsi_dy, dpsi_dz = np.gradient(psi, *delr)
    return -1j * h_bar * (xx * dpsi_dy - yy * dpsi_dx)


def get_Ac(path: str, h_bar: float = 1.0) -> np.ndarray:
    """
    Construct the Lz matrix in the basis of wavefunctions loaded from path.

    Parameters:
        path (str): directory containing wavefunction files.
        h_bar (float, optional): reduced Planck constant (ℏ). Defaults to 1.0.

    Returns:
        np.ndarray: complex matrix A_c of shape (N, N), where N is the number
        of wavefunctions.
    """
    # read files:
    files = read_files(path)
    wavefunctions, delr, N3, *_ = files
    xx, yy, zz = construct_axes(delr, N3)
    N = len(wavefunctions)
    
    # normalize the wavefunctions:
    for i, psi in enumerate(wavefunctions):
        wavefunctions[i] = normalize(psi, delr)
    
    A_c = np.zeros((N, N), dtype=np.complex128)
    for i, psi_i in enumerate(wavefunctions):
        psi_i_conj = psi_i.conj()
        for j, psi_j in enumerate(wavefunctions):
            Lz_psi_j = apply_Lz(psi_j, xx, yy, zz, delr, h_bar=1)
            A_c[i, j] = integrate(psi_i_conj * Lz_psi_j, delr)

    return A_c


def view_files(path: str, 
    axis: int = 0,
    nrows: int = 1, 
    cmap: str = "inferno") -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot 2D slices of all wavefunctions in a directory.

    Parameters:
        path (str): directory containing wavefunction files.
        axis (int, optional): axis along which to slice (0, 1, or 2).
            Defaults to 0.
        nrows (int, optional): number of subplot rows. Defaults to 1.
        cmap (str, optional): matplotlib colormap. Defaults to "inferno".

    Returns:
        Tuple[plt.Figure, np.ndarray]: the figure and the array of axes.
    """
    # check valid axis:
    if not (0 <= axis <= 2):
        raise ValueError(f"Invalid 'axis' argument: '{axis}'. Expected integer from 0 to 2.")

    # read files:
    files = read_files(path)
    wavefunctions, delr, N3, *_ = files
    n = len(wavefunctions)

    ncols = n // nrows
    fig, axes = plt.subplots(nrows, ncols)
    axes = axes.reshape(nrows, ncols)

    # start plotting
    idx = [slice(None)] * 3
    idx[axis] = wavefunctions[0].shape[axis] // 2 + 1
    
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            flat_idx = np.ravel_multi_index((i, j), (nrows, ncols))
            data = wavefunctions[flat_idx][*idx]  # 2d array
            ax.imshow(data, cmap=cmap)

    return fig, axes

# ------ functions for quadrupole moment ------

def get_Y_lm(l: int, m: int,
             theta: np.ndarray,
             phi: np.ndarray) -> np.ndarray:
    """
    Spherical harmonic 
    """
    return sph_harm(m, l, phi, theta)


def get_R_l_exp(l: int,
                r: np.ndarray,
                alpha: float) -> np.ndarray:
    """
    Toy model radial function R_l(r) ~ r^l * exp(-alpha * r),
    """
    r = np.asarray(r)
    R = (r**l) * np.exp(-alpha * r)

    # normalize: 
    integrand = np.abs(R)**2 * r**2
    norm = np.sqrt(np.trapz(integrand, x=r))
    return R / norm


def radial_part(l: int, l_p: int,
                r: np.ndarray,
                alpha: float) -> complex:
    """
    Compute ∫ R_l^*(r) R_{l'}(r) r^4 dr
    using the exponential toy model for the radial functions.
    """
    R_l   = get_R_l_exp(l,   r, alpha)
    R_lp  = get_R_l_exp(l_p, r, alpha)

    integrand = np.conj(R_l) * R_lp * r**4
    return np.trapz(integrand, x=r)


def angular_part(l: int, m: int,
                 l_p: int, m_p: int,
                 q: int,
                 theta: np.ndarray,
                 phi: np.ndarray) -> complex:
    """
    √(4π/5) ∫ Y_{l m}^* Y_{2 q} Y_{l' m'} dΩ
    with dΩ = sinθ dθ dφ
    """
    th, ph = np.meshgrid(theta, phi, indexing="ij")

    Y_l_m   = get_Y_lm(l,   m,  th, ph)
    Y_2_q   = get_Y_lm(2,   q,  th, ph)
    Y_lp_mp = get_Y_lm(l_p, m_p, th, ph)

    integrand = np.conj(Y_l_m) * Y_2_q * Y_lp_mp * np.sin(th)

    # integrate over phi (axis=1), then theta (axis=0)
    tmp = np.trapz(integrand, x=phi, axis=1)
    ang_int = np.trapz(tmp, x=theta, axis=0)

    return np.sqrt(4 * np.pi / 5) * ang_int


def quadrupole_moment(l: int, m: int,
                      l_p: int, m_p: int,
                      q: int,
                      r: np.ndarray,
                      theta: np.ndarray,
                      phi: np.ndarray,
                      alpha: float = 0.01,
                      e: float = 1.0) -> complex:
    """
    ⟨ l m | Q_{2 q} | l' m' ⟩ for a toy radial model.

    Radial: model orbitals with exp(-alpha r).
    Angular: exact spherical harmonics + quadrupole operator.
    """
    rad = radial_part(l, l_p, r, alpha)
    ang = angular_part(l, m, l_p, m_p, q, theta, phi)
    return e * rad * ang
