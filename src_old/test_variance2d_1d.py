#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:19:05 2024

@author: fbrient
"""

import numpy as np

# Example 2D field
field = np.random.rand(100, 100)  # Replace with your 2D data
nx, ny = field.shape

# Compute the 2D FFT and the power spectrum
field_fft = np.fft.fftshift(np.fft.fft2(field))
power_spectrum_2d = np.abs(field_fft)**2

# Generate wave number arrays
kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny
kx, ky = np.meshgrid(kx, ky)

# Radial wave number
kr = np.sqrt(kx**2 + ky**2)
kr = kr.flatten()
power_spectrum_flat = power_spectrum_2d.flatten()

# Bin the 2D power spectrum into radial bins
bin_edges = np.arange(0, np.max(kr) + 1, 1)  # Define radial bins
radial_power, _ = np.histogram(kr, bins=bin_edges, weights=power_spectrum_flat)
counts, _ = np.histogram(kr, bins=bin_edges)

# Average power in each radial bin (1D radial power spectrum)
radial_power /= counts  # Avoid division by zero; counts > 0 for valid bins

# Recover total variance from the 1D radial spectrum
total_variance = np.sum(radial_power * counts) / (nx * ny)**2.  # Normalize by grid size

# Compute the total variance directly from the 2D field (for comparison)
direct_variance = np.sum(field**2) / (nx * ny)

print(f"Total variance from radial spectrum: {total_variance}")
print(f"Total variance from direct computation: {direct_variance}")
