import numpy as np
import matplotlib.pyplot as plt

# Assuming velocity fields u, v (2D) or u, v, w (3D) are defined
# Here is a simplified 2D example

# Radially average the transfer function
def radial_average_transfer(k,T_k):
    
    k_flat = k.flatten()
    T_k_flat = T_k.flatten()
    
    k_bins = np.arange(0.5, np.max(k_flat.shape) // 2)
    
    print(T_k_flat.size,k_flat.shape)
    print(k_bins)

    T_k_radial = np.zeros(k_bins.size)
#    for i in range(k_bins.size):
#    for i,k in enumerate(k_bins):
#        mask = (k >= k_bins[i] - 0.5) & (k < k_bins[i] + 0.5)
#        T_k_radial[i] = np.mean(T_k[mask])
        
    for i in range(k_bins.size):
        mask = (k_flat >= k_bins[i] - 0.5) & (k_flat < k_bins[i] + 0.5)
        if np.any(mask):
            T_k_radial[i] = np.mean(T_k_flat[mask])
    
    
    return k_bins, T_k_radial
    

def test_injection_rate2D(u,v,dx=1,dy=1,k=None):

    # Fourier transform the velocity fields
    U_hat = np.fft.fft2(u)
    V_hat = np.fft.fft2(v)
    
    # Compute the wavenumbers
    ny, nx = u.shape
    kx = np.fft.fftfreq(nx,d=1./nx)
    ky = np.fft.fftfreq(ny,d=1./ny)
    # Change to wavenumber
    kx = (2.0*np.pi)*kx/(nx*dx)
    ky = (2.0*np.pi)*ky/(ny*dy)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    
    #kx = np.fft.fftfreq(u.shape[0])
    #ky = np.fft.fftfreq(u.shape[1])
    #kx, ky = np.meshgrid(kx, ky)
    k2 = kx**2 + ky**2
    k = np.sqrt(k2)
    
    print('k ',k)
    
    # Compute the nonlinear terms (simplified example)
    U_hat_star = np.conjugate(U_hat)
    V_hat_star = np.conjugate(V_hat)
    
    nonlinear_term_u = -1j * (kx * (U_hat * U_hat_star + V_hat * V_hat_star))
    nonlinear_term_v = -1j * (ky * (U_hat * U_hat_star + V_hat * V_hat_star))
    
    T_u = np.real(U_hat * nonlinear_term_u)
    T_v = np.real(V_hat * nonlinear_term_v)
    
    # Sum to get the total energy transfer function
    T_k = T_u + T_v
        
    k_bins, T_k_radial = radial_average_transfer(k,T_k)
    
    # Plot the transfer function
    plt.plot(k_bins, T_k_radial)
    plt.xlabel('Wavenumber k')
    plt.ylabel('Energy Transfer Function T(k)')
    plt.title('Energy Transfer Function')
    plt.axhline(0, color='gray', linestyle='--')
    
    plt.show()
    
    # Identify the injection scale
    positive_indices = np.where(T_k_radial > 0)[0]
    if positive_indices.size > 0:
        injection_scale_k = k_bins[positive_indices[0]]
        injection_scale_l = 2 * np.pi / injection_scale_k
        print(f'Injection scale (wavenumber): {injection_scale_k}')
        print(f'Injection scale (physical length): {injection_scale_l}')
    else:
        print("No positive energy transfer found, check your data.")
    return None
