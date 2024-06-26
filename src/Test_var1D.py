#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:31:25 2024

@author: fbrient
"""

import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
N = 1024
dx = 0.01
x = np.arange(0, N * dx, dx)
f = np.sin(2 * np.pi * 5 * x) + 0.5 * np.random.normal(size=N)
print('real variance: ', np.var(f))

# Compute the FFT and PSD
f_hat = np.fft.fft(f)
frequencies = np.fft.fftfreq(N, dx)
psd = np.abs(f_hat)**2 / (N / dx)

# Compute the variance using PSD
variance_psd = np.sum(psd) * (frequencies[1] - frequencies[0])
print(f"Variance from PSD: {variance_psd:.4f}")

# Compute the variance using k * PSD
k_psd = frequencies * psd
variance_k_psd = np.sum(k_psd[frequencies >= 0]) * (frequencies[1] - frequencies[0])
print(f"Variance from k * PSD: {variance_k_psd:.4f}")

# Plot the PSD and k * PSD
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.plot(frequencies, psd, label='PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('Power Spectral Density (PSD)')
plt.legend()

plt.subplot(122)
plt.plot(frequencies, k_psd, label='k * PSD')
plt.xlabel('Frequency (Hz)')
plt.ylabel('k * Power Spectral Density')
plt.title('k * Power Spectral Density')
plt.legend()

plt.tight_layout()
plt.show()
