"""
Phase Hologram Global Shift Invariance Demonstration
=====================================================
Demonstrates that adding a constant phase offset (global shift) to a phase-only
hologram does not change the reconstructed image.

  theta, (theta + delta_1) % 2pi, (theta + delta_2) % 2pi

Figure layout (3 rows x 3 columns):
  Col 1: Phase hologram images
  Col 2: 1D phase profile along a horizontal line
  Col 3: Numerical reconstruction via BL-ASM
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft as fft
import skimage.io as sio
from time import time

cm, mm, um, nm = 1e-2, 1e-3, 1e-6, 1e-9

# ── Utility functions ────────────────────────────────────────────────────────

def pad_image(field, target_shape, padval=0, mode='constant'):
    size_diff = np.array(target_shape) - np.array(field.shape[-2:])
    odd_dim = np.array(field.shape[-2:]) % 2
    if (size_diff > 0).any():
        pad_total = np.maximum(size_diff, 0)
        pad_front = (pad_total + odd_dim) // 2
        pad_end = (pad_total + 1 - odd_dim) // 2
        pad_axes = [int(p) for tple in zip(pad_front[::-1], pad_end[::-1]) for p in tple]
        return nn.functional.pad(field, pad_axes, mode=mode, value=padval)
    else:
        return field


def crop_image(field, target_shape):
    if target_shape is None:
        return field
    size_diff = np.array(field.shape[-2:]) - np.array(target_shape)
    odd_dim = np.array(field.shape[-2:]) % 2
    if (size_diff > 0).any():
        crop_total = np.maximum(size_diff, 0)
        crop_front = (crop_total + 1 - odd_dim) // 2
        crop_end = (crop_total + odd_dim) // 2
        crop_slices = [slice(int(f), int(-e) if e else None) for f, e in zip(crop_front, crop_end)]
        return field[(..., *crop_slices)]
    else:
        return field


# ── BL-ASM propagation ──────────────────────────────────────────────────────

def propagation_blasm(u_in, feature_size, wavelength, z,
                      linear_conv=True, return_kernel=False, precomputed_kernel=None):
    if linear_conv:
        input_resolution = u_in.size()[-2:]
        conv_size = [i * 2 for i in input_resolution]
        u_in = pad_image(u_in, conv_size, padval=0)

    if precomputed_kernel is None:
        field_resolution = u_in.size()
        num_y, num_x = field_resolution[-2], field_resolution[-1]
        dy, dx = feature_size

        fy = np.linspace(-1 / (2 * dy), 1 / (2 * dy), num_y)
        fx = np.linspace(-1 / (2 * dx), 1 / (2 * dx), num_x)
        fxx, fyy = np.meshgrid(fx, fy)

        kernel = np.exp(1j * 2 * np.pi * z * np.sqrt(1 / wavelength**2 - (fxx**2 + fyy**2)))
        dv, du = 1 / (num_y * dy), 1 / (num_x * dx)
        bly = 1 / (wavelength * np.sqrt((2 * z * dv)**2 + 1))
        blx = 1 / (wavelength * np.sqrt((2 * z * du)**2 + 1))
        bl_filter = (np.abs(fxx) < blx) & (np.abs(fyy) < bly)
        bl_kernel = torch.tensor(bl_filter * kernel).to(u_in.device)
    else:
        bl_kernel = precomputed_kernel

    if return_kernel:
        return bl_kernel

    u_in_fft = fft.fftshift(fft.fftn(fft.fftshift(u_in)))
    u_out_fft = bl_kernel * u_in_fft
    u_out = fft.fftshift(fft.ifftn(fft.fftshift(u_out_fft)))
    return crop_image(u_out, input_resolution) if linear_conv else u_out


# ── SGD hologram generation ─────────────────────────────────────────────────

def stochastic_gradient_descent(u_in, feature_size, wave_length, z,
                                linear_conv=True, num_iters=500, seed=7777):
    np.random.seed(seed)
    u_in = torch.tensor(u_in)
    init_phase = torch.tensor(np.random.rand(u_in.shape[-2], u_in.shape[-1]))
    holo_phase = init_phase.requires_grad_(True)

    precomputed_kernel = propagation_blasm(
        torch.zeros_like(u_in), feature_size, wave_length, -z,
        linear_conv, return_kernel=True)
    precomputed_kernel.requires_grad = False

    criterion = nn.MSELoss()
    optimizer = optim.Adam([{'params': holo_phase}], lr=8e-3)

    start_time = time()
    for i in range(1, num_iters + 1):
        holo_field = torch.exp(1j * 2 * torch.pi * holo_phase)
        recon_field = propagation_blasm(holo_field, feature_size, wave_length, -z,
                                        linear_conv, precomputed_kernel=precomputed_kernel)
        optimizer.zero_grad()
        loss = criterion(torch.abs(recon_field), u_in)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print(f'  iter {i:4d}/{num_iters}  loss={loss.item():.6f}  ({time()-start_time:.1f}s)')

    # Return the final phase (in radians, range [0, 2pi))
    final_phase = (holo_phase.detach() * 2 * np.pi) % (2 * np.pi)
    return final_phase, precomputed_kernel


# ── Numerical reconstruction (BL-ASM) ───────────────────────────────────────

def reconstruct(phase, feature_size, wave_length, z, precomputed_kernel=None):
    """Reconstruct from a phase-only hologram and return normalised amplitude."""
    holo_field = torch.exp(1j * phase)
    recon_field = propagation_blasm(holo_field, feature_size, wave_length, -z,
                                    precomputed_kernel=precomputed_kernel)
    recon = torch.abs(recon_field)
    recon = (recon - torch.amin(recon)) / (torch.amax(recon) - torch.amin(recon))
    return recon.detach().numpy()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # --- Parameters ---
    feature_size = (6.4 * um, 6.4 * um)
    wave_length = 530 * nm
    prop_dist = 20 * cm
    num_iters = 200

    # --- Load target image ---
    img = sio.imread('../assets/sample1.png')
    if img.ndim == 3:
        img = img[:, :, 1]  # green channel
    img = img.astype(np.float64) / 255.0
    print(f'Target image shape: {img.shape}')

    # --- Generate phase hologram via SGD ---
    print('Running SGD optimisation ...')
    theta, precomputed_kernel = stochastic_gradient_descent(
        img, feature_size, wave_length, prop_dist,
        linear_conv=True, num_iters=num_iters)

    # --- Define global phase offsets ---
    delta_1 = 1.0  # radians
    delta_2 = 1.7  # radians

    theta_np = theta.numpy()
    theta_1 = (theta_np + delta_1) % (2 * np.pi)
    theta_2 = (theta_np + delta_2) % (2 * np.pi)

    phases = [theta_np, theta_1, theta_2]
    labels = [
        r'$\theta$',
        r'$\theta_1 = (\theta + \Delta_1)\;\mathrm{mod}\;2\pi$'
        + f'\n$\\Delta_1 = {delta_1:.1f}$ rad',
        r'$\theta_2 = (\theta + \Delta_2)\;\mathrm{mod}\;2\pi$'
        + f'\n$\\Delta_2 = {delta_2:.1f}$ rad',
    ]

    # --- Numerical reconstructions ---
    print('Computing numerical reconstructions (BL-ASM) ...')
    recons = []
    for ph in phases:
        recon = reconstruct(torch.tensor(ph), feature_size, wave_length,
                            prop_dist, precomputed_kernel=precomputed_kernel)
        recons.append(recon)

    # --- Pick a horizontal scan-line for 1D phase profile ---
    scan_row = theta_np.shape[0] // 2  # middle row

    # --- Plot figure (3 rows x 3 columns) ---
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    row_labels = ['(a)', '(b)', '(c)']

    for row_idx in range(3):
        ph = phases[row_idx]

        # Column 1: Phase hologram image
        ax = axes[row_idx, 0]
        im = ax.imshow(ph, cmap='gray', vmin=0, vmax=2*np.pi)
        ax.set_title(f'{row_labels[row_idx]}  {labels[row_idx]}', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_xlabel('')
            # Draw scan-line indicator
        ax.axhline(y=scan_row, color='red', linewidth=0.8, linestyle='--', alpha=0.7)

        # Column 2: 1D phase profile along the scan-line
        ax2 = axes[row_idx, 1]
        line_phase = ph[scan_row, :5]
        x_pixels = np.arange(5)
        ax2.plot(x_pixels, line_phase, linewidth=1.5, color='navy',
                 marker='o', markersize=6)
        ax2.set_ylim(-0.2, 2 * np.pi + 0.2)
        ax2.set_ylabel('Phase (rad)', fontsize=10)
        ax2.set_title(f'1D phase profile (row={scan_row})', fontsize=10)
        ax2.axhline(y=0, color='gray', linewidth=0.3)
        ax2.axhline(y=2*np.pi, color='gray', linewidth=0.3)
        if row_idx == 2:
            ax2.set_xlabel('Pixel position', fontsize=10)

        # Column 3: Numerical reconstruction
        ax3 = axes[row_idx, 2]
        ax3.imshow(recons[row_idx], cmap='gray')
        ax3.set_title('NR (BL-ASM)', fontsize=11)
        ax3.set_xticks([])
        ax3.set_yticks([])

    fig.suptitle('Global Phase Shift Invariance of Phase-Only Hologram',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('phase_hologram_invariance.png', dpi=200, bbox_inches='tight')
    print('Saved → phase_hologram_invariance.png')
    plt.show()


if __name__ == '__main__':
    main()
