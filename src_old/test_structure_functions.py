import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

def compute_power_spectrum(u, sampling_rate,nperseg=256):
    """
    Compute the power spectrum using the Welch method.
    """
    freq, power_spectrum = welch(u, fs=sampling_rate,nperseg=nperseg) #, nperseg=len(u)//8)
    return freq, power_spectrum

def compute_structure_function(u, x, r_values):
    """
    Compute the third-order structure function S(r).
    """
    S = np.zeros_like(r_values, dtype=np.float64)
    for i, r in enumerate(r_values):
        diffs = []
        for j in range(len(u) - 1):
            idx = np.searchsorted(x, x[j] + r)  # Find index for x[j] + r
            #print(i,r,j,idx,idx < len(u))
            if idx < len(u):
                diff = u[idx] - u[j]
                diffs.append(diff**3)
        if diffs:
            S[i] = np.mean(diffs)
    return S

# Example usage
if __name__ == "__main__":
    # Simulated data
    np.random.seed(0)
    n_points = 1000
    x = np.linspace(0, 100, n_points)  # Position array
    u = np.sin(2 * np.pi * x / 10) + 0.1 * np.random.randn(n_points)  # Velocity with noise
    
    # Parameters
    sampling_rate = n_points / (x[-1] - x[0])  # Assuming uniform spacing in x
    r_values = np.linspace(1, 10, 50)  # Lag distances to evaluate S(r)
    
    # Compute power spectrum
    freq, power_spectrum = compute_power_spectrum(u, sampling_rate)
    
    # Compute structure function
    S_r = compute_structure_function(u, x, r_values)
    
    # Compute k and k*r
    k_values = 2 * np.pi / r_values
    kr_values = k_values * r_values
    S_kr = S_r * r_values  # Scale S(r) by r for S3 as function of kr
    
    # Plotting results
    plt.figure(figsize=(15, 5))
    
    # Power Spectrum
    plt.subplot(1, 3, 1)
    plt.loglog(freq, power_spectrum)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectrum')
    plt.title('Power Spectrum of Wind Velocity')
    
    # Structure Function as S(r)
    plt.subplot(1, 3, 2)
    plt.semilogx(k_values, S_r, marker='o')
    plt.xlabel('Wavenumber k [1/m]')
    plt.ylabel('S3(k)')
    plt.title('Third-Order Structure Function S3 vs k')
    
    # Structure Function as S3(k*r)
    plt.subplot(1, 3, 3)
    plt.semilogx(kr_values, S_r, marker='o')
    plt.xlabel('Scaled Wavenumber k*r')
    plt.ylabel('S3(k*r)')
    plt.title('Third-Order Structure Function S3 vs k*r')
    
    plt.tight_layout()
    plt.show()
