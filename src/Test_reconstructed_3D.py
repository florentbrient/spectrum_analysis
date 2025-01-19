#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:44:26 2024

@author: fbrient
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
import cloudmetrics as cm
from random import randrange

# Generate a 2D synthetic signal (image)
N, M, P = 256, 256, 100
dx,dy,dz =1,1,1
# 10 cycles
data = np.zeros((P,N,M))
for ij in range(P):
    dpi = 0.1*P-ij #randrange(4)
    x = np.linspace(0, dpi * 2*np.pi, N)
    y = np.linspace(0, dpi * 2*np.pi, M)
    X, Y = np.meshgrid(x, y)
    data[ij,:,:] = np.sin(X) + np.sin(Y)
    data[ij,:,:] += 0*np.random.random((N,M))

#dpi=10
#x = np.linspace(0, dpi * 2*np.pi, N)
#y = np.linspace(0, dpi * 2*np.pi, M)
#X, Y = np.meshgrid(x, y)
#f2 = np.cos(X) + np.cos(Y)
##f=(f+f2)/2.

#f+=0*np.random.random((N,N))

# Compute the 2D FFT
F = np.fft.fftn(data)

# Shift the zero frequency component to the center
F_shifted = np.fft.fftshift(F)

# Compute the magnitude spectrum for visualization
magnitude_spectrum = np.abs(F_shifted)**2.

# Compute the inverse FFT to verify the result
f_reconstructed = np.fft.ifftn(F).real

# Use cloudmetrics
# dx=1
# k1d, psd_1d_rad, psd_1d_azi = cm.scalar.compute_spectra(
#     f,
#     dx=dx,
#     periodic_domain=False,
#     apply_detrending=False,
#     window=None,
# )

# plt.hist(f.flatten());plt.show()
# print('Real variance : ',np.var(f))
# #print(amp**2 * np.pi / 4) # Why? Ok Ratio between circle/square
# print(np.std(f)**2 * np.pi / 4) # Why? Ok Ratio between circle/square

# area = np.trapz(psd_1d_rad, x=k1d)
# print("Area under psd1d =", area)
# # integral under the Power Spectra
# variance_psd = np.sum(psd_1d_rad) * 2 * np.pi / (dx * N)
# print("variance_psd: ",variance_psd)
# variance_azi = np.sum(psd_1d_azi)
# print("variance_azi: ",variance_azi)



# anisotropy = cm.scalar.spectral_anisotropy(psd_1d_azi)
# l_spec_median = cm.scalar.spectral_length_median(k1d, psd_1d_rad)
# l_spec_moment = cm.scalar.spectral_length_moment(k1d, psd_1d_rad)

# print('Plot Graph Test')
# plt.loglog(k1d,psd_1d_rad,'k')
# print('variance_psd','anisotropy','beta','beta_binned','l_spec_median','l_spec_moment')
# print(variance_psd,anisotropy,l_spec_median,l_spec_moment)
# plt.show()

# Plot 2D Fourier transform
#F = np.fft.fft2(x)  # 2D FFT (no prefactor)
#F = np.fft.fftshift(F)  # Shift so k0 is centred
psd_3d = np.abs(F_shifted) ** 2 / np.prod(data.shape)  # Energy-preserving 2D PSD
print(psd_3d.shape)

# Calculate the frequency ranges for the x and y axes
freq_x = np.fft.fftfreq(N, d=dx)
freq_y = np.fft.fftfreq(M, d=dy)
freq_x_shifted = np.fft.fftshift(freq_x)
freq_y_shifted = np.fft.fftshift(freq_y)

freq_z = np.fft.fftfreq(P, d=dz)
freq_z_shifted = np.fft.fftshift(freq_z)

iz = 18
#plt.contourf(k_x,k_y,psd_2d,locator=ticker.LogLocator());plt.colorbar();plt.show()
plt.imshow(psd_3d[iz,:,:], extent=(freq_x_shifted[0], freq_x_shifted[-1], freq_y_shifted[0], freq_y_shifted[-1]), norm=LogNorm(), aspect='auto')
plt.colorbar()
plt.title('Squared Magnitude of 2D Fourier Transform')
plt.xlabel('Frequency X (k_x)')
plt.ylabel('Frequency Y (k_y)')
plt.show()


# Plot the original signal, magnitude spectrum, and reconstructed signal
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Original signal
ax1.imshow(data[iz,:,:], extent=(0, 4 * np.pi, 0, 4 * np.pi))
ax1.set_title('Original Signal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# Magnitude spectrum
ax2.imshow(magnitude_spectrum[iz,:,:], extent=(-N//2, N//2, -M//2, M//2), cmap='gray')
ax2.set_title('Magnitude Spectrum')
ax2.set_xlabel('u')
ax2.set_ylabel('v')

# Reconstructed signal
ax3.imshow(f_reconstructed[iz,:,:]) #, extent=(0, 4 * np.pi, 0, 4 * np.pi))
ax3.set_title('Reconstructed Signal')
ax3.set_xlabel('x')
ax3.set_ylabel('y')

plt.tight_layout()
plt.show()

# Print a comparison of the original and reconstructed signals
print("Original signal variance:", np.var(data))
print("Reconstructed signal variance:", np.var(f_reconstructed))
print("Difference:", np.max(np.abs(data - f_reconstructed)))
# Impossible because of the absolute value
#print("Reconstructed signal variance from psd2D:", np.var(np.fft.ifft2(psd_2d).real))



# Define wave numbers
# kx, ky = np.meshgrid(freq_x, freq_y)
# kr = np.sqrt(kx**2 + ky**2).flatten()  # Radial wave numbers
# power_spectrum_flat = magnitude_spectrum.flatten()

# # Bin the 2D power spectrum into radial bins
# bins = np.arange(0.01, np.max(kr), step=0.01)
# radial_power, _ = np.histogram(kr, bins=bins, weights=power_spectrum_flat)
# counts, _ = np.histogram(kr, bins=bins)

# # Average power in each radial bin
# radial_power /= counts  # This gives P(k)

# # To recover variance:
# total_variance = np.sum(radial_power * counts)/(N*M)**2.
# print(f"Total variance recovered: {total_variance}")



# Test 3D
# Generate frequency arrays
kx = freq_x#np.fft.fftfreq(nx) * nx / Lx
ky = freq_y#np.fft.fftfreq(ny) * ny / Ly
kz = freq_z#np.fft.fftfreq(nz) * nz / Lz
kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')

# Compute radial wavenumber
kr = np.sqrt(kx**2 + ky**2 + kz**2).flatten()
power_spectrum_3d   = psd_3d
power_spectrum_flat = power_spectrum_3d.flatten()

# Bin the 3D power spectrum into spherical shells
bin_edges = np.arange(0, np.max(kr), 0.01)  # Define radial bins
radial_power, _ = np.histogram(kr, bins=bin_edges, weights=power_spectrum_flat)
counts, _ = np.histogram(kr, bins=bin_edges)

# Avoid division by zero and compute spherical average
radial_power /= np.maximum(counts, 1)  # Counts > 0 ensures valid bins

# Optional: Recover total variance from the 1D radial spectrum
total_variance = np.sum(radial_power * counts)  # Total energy in Fourier space
direct_variance = np.sum(data**2)  # Total variance in spatial domain

print(f"Total variance from spherical average: {total_variance}")
print(f"Total variance from direct computation: {direct_variance}")

# Radial wavenumbers for plotting
radial_wavenumbers = (bin_edges[:-1] + bin_edges[1:]) / 2


plt.semilogx(1/radial_wavenumbers, radial_power, '-o')
plt.xlabel('length scale (-)')
plt.ylabel('Spherical Average of Power Spectrum')
plt.title('Spherical Average of 3D FFT Power Spectrum (Anisotropic)')
plt.grid()
plt.show()

plt.semilogx(radial_wavenumbers, radial_power, '-o')
plt.xlabel('Radial Wavenumber (k)')
plt.ylabel('Spherical Average of Power Spectrum')
plt.title('Spherical Average of 3D FFT Power Spectrum (Anisotropic)')
plt.grid()
plt.show()

