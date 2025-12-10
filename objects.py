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

        # initial gaussian beam
        self.field = np.ones((self.N, self.N), dtype=complex)
        self.field *= np.exp(- (self.X**2 + self.Y**2) / (0.1 * self.L)**2)

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
    
    def plot_wavefront(self, title="Wavefront Intensity"):
        """ Plot the intensity of the wavefront. """
        intensity = self.get_intensity()
        plt.imshow(intensity, extent=(-self.L/2, self.L/2, -self.L/2, self.L/2))
        plt.colorbar(label='Intensity')
        plt.title(title)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.show()

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
    # TODO make size of star correct and fix weird bugs

    def __init__(self, radius, vertical_shift=0.0, resolution=512, num_spokes=32, supersample=8):
        self.radius = radius
        self.vertical_shift = vertical_shift
        self.resolution = resolution
        self.num_spokes = num_spokes
        self.supersample = supersample

    def create_siemens_star(self):
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
        high_res = self.resolution * self.supersample
        
        # 2. Create the coordinate grid
        # We center the grid so (0,0) is in the middle of the image
        # shift coordinates vertically
        x = np.linspace(-1, 1, high_res)
        y = np.linspace(-1 + self.vertical_shift, 1 + self.vertical_shift, high_res)
        xv, yv = np.meshgrid(x, y)
        
        # 3. Calculate the polar angle (theta) for every sub-pixel
        # arctan2 handles the quadrants correctly
        theta = np.arctan2(yv, xv)
        
        # 4. Generate the binary mask
        # The sine of (num_spokes * theta) creates the alternating pattern.
        # We use a threshold > 0 to make it binary (0 or 1).
        # We add a small phase shift if desired, but here we stick to standard alignment.
        star_pattern = np.sin(self.num_spokes * theta)
        
        # Convert to binary (0.0 or 1.0)
        binary_high_res = (star_pattern > 0).astype(np.float64)

        # apply circular mask to limit to radius
        R = np.sqrt(xv**2 + yv**2)
        binary_high_res[R > 1] = 1.0
        
        # 5. Downsample to calculate proportional area (average pooling)
        # We reshape the array to separate the super-sampled blocks
        # Shape becomes (Resolution, Supersample, Resolution, Supersample)
        reshaped = binary_high_res.reshape(self.resolution, self.supersample, self.resolution, self.supersample)
        
        # We take the mean over the supersampling axes (axis 1 and 3)
        # This results in the average value (area coverage) for that pixel block
        anti_aliased_image = reshaped.mean(axis=(1, 3))
        
        return anti_aliased_image
    
    def apply(self, wavefront):
        """ Apply the Siemens star mask to the wavefront intensity. """
        star_mask = self.create_siemens_star()
        wavefront.field *= star_mask

    def plot(self, resolution=512):
        """ Plot the Siemens star target. """
        star_image = self.create_siemens_star()
        plt.imshow(star_image, extent=(-self.radius, self.radius, -self.radius, self.radius), cmap='gray')
        plt.colorbar(label='Transmission')
        plt.title("Siemens Star Target")
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.show()
    
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
    """ Converts the time-varying signal from the photodiode into a 1D image. """
    def signal_to_image(self, signal, dt, image_size):
        """
        Docstring for signal_to_image
        
        :param signal: signal as a function of time (1D numpy array)
        :param dt: time between each frame in the signal (in seconds)
        :param image_size: sidelength of the output square image in m (wavelength.L)
        """
        # Perform Fourier Transform on the signal
        freq_domain = np.fft.fftshift(np.fft.fft(signal))
        magnitude = np.abs(freq_domain)

        # TODO determine if this works/makes sense # Normalize and reshape to form an image
        # image = magnitude / np.max(magnitude)
        # image_2d = image.reshape((int(np.sqrt(len(image))), int(np.sqrt(len(image)))))

        # plot 1d image
        plt.figure()
        plt.plot(magnitude)
        plt.title("SPIFI Signal Frequency Domain")
        plt.xlabel("Frequency (arbitrary units)")
        plt.ylabel("Magnitude (arbitrary units)")
        plt.grid()
        plt.show()

        return magnitude
    







#TODO spifi mask, spifi signal to image converter, (maybe more samples if I have time?)

