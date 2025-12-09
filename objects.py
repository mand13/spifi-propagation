# objects.py

import numpy as np
import matplotlib.pyplot as plt
import copy

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
    """ Base class for optical elements: things that ineract with the wavefront. """
    def apply(self, wavefront):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
class ConvergingLens(OpticalElement):
    def __init__(self, focal_length):
        self.focal_length = focal_length
    
    def apply(self, wavefront):
        """ Apply lens phase transformation to the wavefront. (Thin lens approximation) """
        k = wavefront.k
        X, Y = wavefront.X, wavefront.Y
        phase = - (k / (2 * self.focal_length)) * (X**2 + Y**2)
        lens_phase = np.exp(1j * phase)
        wavefront.field *= lens_phase

class CylindricalLens(OpticalElement):
    def __init__(self, focal_length, orientation='horizontal'):
        self.focal_length = focal_length
        self.orientation = orientation
    
    def apply(self, wavefront):
        """ Apply cylindrical lens phase transformation to the wavefront. (Thin lens approximation) """
        k = wavefront.k
        X, Y = wavefront.X, wavefront.Y
        if self.orientation == 'horizontal':
            phase = - (k / (2 * self.focal_length)) * (Y**2)
        else: # vertical
            phase = - (k / (2 * self.focal_length)) * (X**2)
        lens_phase = np.exp(1j * phase)
        wavefront.field *= lens_phase

class Target(OpticalElement):
    """ Base class for targets: objects that we want to image using wavefronts. """
    def apply(self, wavefront):
        raise NotImplementedError("This method should be implemented by subclasses.")

class SiemensStar(Target):
    def __init__(self, radius):
        self.radius = radius

    def create_siemens_star(resolution, num_spokes, supersample=8):
        """
        NOTE: function generated entirely by Google Gemini

        Generates a Siemens star with anti-aliasing (partial pixel values).
        
        Args:
            resolution (int): The width and height of the output image in pixels.
            num_spokes (int): The number of spokes (cycles of black/white). 
                            Note: A "spoke" here usually implies a pair of black/white wedges.
            supersample (int): The factor by which to upscale for area calculation.
                            Higher values = more accurate partial pixel values.
                            Default is 8 (64 sub-pixels per pixel).
        
        Returns:
            np.ndarray: A 2D numpy array with values between 0.0 and 1.0.
        """
        # 1. Calculate the high-resolution grid size
        high_res = resolution * supersample
        
        # 2. Create the coordinate grid
        # We center the grid so (0,0) is in the middle of the image
        x = np.linspace(-1, 1, high_res)
        y = np.linspace(-1, 1, high_res)
        xv, yv = np.meshgrid(x, y)
        
        # 3. Calculate the polar angle (theta) for every sub-pixel
        # arctan2 handles the quadrants correctly
        theta = np.arctan2(yv, xv)
        
        # 4. Generate the binary mask
        # The sine of (num_spokes * theta) creates the alternating pattern.
        # We use a threshold > 0 to make it binary (0 or 1).
        # We add a small phase shift if desired, but here we stick to standard alignment.
        star_pattern = np.sin(num_spokes * theta)
        
        # Convert to binary (0.0 or 1.0)
        binary_high_res = (star_pattern > 0).astype(np.float64)
        
        # 5. Downsample to calculate proportional area (average pooling)
        # We reshape the array to separate the super-sampled blocks
        # Shape becomes (Resolution, Supersample, Resolution, Supersample)
        reshaped = binary_high_res.reshape(resolution, supersample, resolution, supersample)
        
        # We take the mean over the supersampling axes (axis 1 and 3)
        # This results in the average value (area coverage) for that pixel block
        anti_aliased_image = reshaped.mean(axis=(1, 3))
        
        return anti_aliased_image
    
    def apply(self, wavefront):
        """ Apply the Siemens star mask to the wavefront intensity. """
        N = wavefront.N
        star_mask = SiemensStar.create_siemens_star(N, num_spokes=32, supersample=8)
        wavefront.field *= star_mask
    
class PhotoDiode(OpticalElement):
    def __init__(self, radius):
        self.radius = radius
    
    def apply(self, wavefront):
        """ Simulate photodiode detection by integrating intensity over its area. """
        intensity = wavefront.get_intensity()
        dx = wavefront.L / wavefront.N
        radius_pixels = int(self.radius / dx)
        center = wavefront.N // 2

        y, x = np.ogrid[-center:wavefront.N-center, -center:wavefront.N-center]
        mask = x**2 + y**2 <= radius_pixels**2

        detected_signal = np.sum(intensity[mask]) * (dx**2) # integrate intensity over area
        return detected_signal

class IdealSPIFIMask(OpticalElement):
    """ SPIFI mask that is idealized (true sine wave pattern)."""
    def apply(self, wavefront, total_time, dt, min_grating_period):
        """
        Takes in a wavefront, which has X,Y
        Outputs total_time/dt frames of the wavefront after applying the SPIFI mask at each time step.
        """
        N = wavefront.N
        x = np.linspace(-wavefront.L/2, wavefront.L/2, N)
        X, Y = np.meshgrid(x, x)
        frames = []
        for t in np.arange(-total_time/2, total_time/2, dt):
            spatial_freq = (2 / (total_time * min_grating_period)) * t
            spifi_pattern = 0.5 * (1 + np.cos(2 * np.pi * spatial_freq * X))
            wavefront_copy = copy.deepcopy(wavefront)
            wavefront_copy.field *= spifi_pattern
            frames.append(wavefront_copy)
        return frames

class RealSPIFIMask(OpticalElement):
    """ SPIFI mask that is more realistic (binary pattern with partial pixel values)."""

class SPIFISignalToImageConverter:
    """ Converts the time-varying signal from the photodiode into an image. """
    def signal_to_image(self, signal, dt, image_size):
        """
        Docstring for signal_to_image
        
        :param signal: Description
        :param dt: time between each frame in signal (in seconds)
        :param image_size: Description
        """




#TODO spifi mask, spifi signal to image converter, (maybe more samples if I have time?)

