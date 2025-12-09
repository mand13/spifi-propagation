# objects.py

import numpy as np
import matplotlib.pyplot as plt

class Wavefront:
    def __init__(self, size_m, resolution, wavelength):
        self.L = size_m # Phyisical size of the grid (meters)
        self.N = resolution # number of pixels N x N
        self.lam = wavelength # wavelength (meters)
        self.k = 2 * np.pi / self.lam # wave number (1/meters)

        # create coordinate grid
        dx = self.L / self.N
        x = np.linspace(-self.L/2, self.L/2 - dx, self.N)
        self.X, self.Y = np.meshgrid(x, x)

        # initial planar wave
        self.field = np.ones((self.N, self.N), dtype=complex)

    def propagate(self, distance):
        """ Propagate the wavefront by distance using Angular Spectrum method. """
        # Fourier transform of the field (fftshift to center zero frequency)
        U_f = np.fft.fftshift(np.fft.fft2(self.field))

        # Frequency coordinates
        fx = np.fft.fftshift(np.fft.fftfreq(self.N, d=self.L/self.N))
        FX, FY = np.meshgrid(fx, fx)

        # Transfer function for propagation
        argument = (1/self.lam**2) - FX**2 - FY**2
        argument = np.maximum(argument, 0) # avoid negative values under sqrt (no evanescent waves)
        # TODO modify to handle evanescent waves if needed
        phase = 2*np.pi * distance * np.sqrt(argument)
        H = np.exp(1j * phase)

        # Apply transfer function
        U_f_propagated = U_f * H

        # Inverse Fourier transform to get propagated field
        self.field = np.fft.ifft2(np.fft.ifftshift(U_f_propagated))
    
    def get_intensity(self):
        """ Return the intensity of the wavefront."""
        return np.abs(self.field)**2

# --- Component Classes ---

class OpticalElement:
    def apply(self, wavefront):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
class Lens(OpticalElement):
    def __init__(self, focal_length):
        self.focal_length = focal_length
    
    def apply(self, wavefront):
        """ Apply lens phase transformation to the wavefront. (Thin lens approximation) """
        k = wavefront.k
        X, Y = wavefront.X, wavefront.Y
        phase = - (k / (2 * self.focal_length)) * (X**2 + Y**2)
        lens_phase = np.exp(1j * phase)
        wavefront.field *= lens_phase

