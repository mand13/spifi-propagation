# objects.py

# EXPERIMENTAL ON DAYSTROM LINES
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for plotting
# ^^ END OF DAYSTROM LINES ^^

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Wavefronts:
    def __init__(self, size_m, resolution, total_time, frames, wavelength, flat=False):
        self.L = size_m # Phyisical size of the grid (meters)
        self.N = resolution # number of pixels N x N
        self.lam = wavelength # wavelength (meters)
        self.k = 2 * np.pi / self.lam # wave number (1/meters)
        self.T = total_time # total time for time-varying wavefronts (seconds)
        self.frames = frames # number of time frames

        # create coordinate grid
        dx = self.L / self.N
        x = np.linspace(-self.L/2, self.L/2 - dx, self.N)
        self.X, self.Y = np.meshgrid(x, x)

        # initial gaussian beam
        init_field = np.ones((self.N, self.N), dtype=complex)
        if not flat:
            init_field *= np.exp(- ((self.X)**2 + (self.Y)**2) / (0.3 * self.L)**2)

        # 3d numpy array to track field through time
        self.fields = np.empty((frames, resolution, resolution), dtype=complex)
        self.fields[:] = init_field

    def propagate(self, distance):
        """ Propagate the wavefront by distance using Angular Spectrum method. """
        # Calculate frequency grid once (avoid recalculation in loop)
        fx = np.fft.fftshift(np.fft.fftfreq(self.N, d=self.L/self.N))
        FX, FY = np.meshgrid(fx, fx)
        
        # Pre-compute transfer function (same for all frames)
        argument = (1/self.lam**2) - FX**2 - FY**2
        argument = np.maximum(argument, 0) # avoid negative values under sqrt (no evanescent waves)
        phase = 2*np.pi * distance * np.sqrt(argument)
        H = np.exp(1j * phase)
        
        # Apply propagation to all frames
        for i in range(len(self.fields)):
            # Fourier transform of the fields (fftshift to center zero frequency)
            U_f = np.fft.fftshift(np.fft.fft2(self.fields[i]))
            # Apply transfer function
            U_f_propagated = U_f * H
            # Inverse Fourier transform to get propagated field
            self.fields[i] = np.fft.ifft2(np.fft.ifftshift(U_f_propagated))
    
    def get_intensities(self):
        """ Return the intensity of the wavefront."""
        return np.abs(self.fields)**2
    
    def plot_wavefront(self, title="Wavefront Intensity", filename=None, show=False):
        """ Plot the intensity of the wavefront for frame 0. """
        intensity = np.abs(self.fields[0])**2
        plt.imshow(intensity, extent=(0, self.L, 0, self.L))
        plt.colorbar(label='Intensity')
        plt.title(title)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        if not filename is None:
            plt.savefig(filename)
        if show:
            plt.show()
        plt.close()
    
    def animate_wavefront(self, title="Wavefront Intensity Animation", filename=None, show=False):
        """ Create an animation of the wavefront intensity over time. """
        fig, ax = plt.subplots()
        intensities = self.get_intensities()
        im = ax.imshow(intensities[0], extent=(0, self.L, 0, self.L), animated=True)
        plt.colorbar(im, ax=ax, label='Intensity')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(title)

        def update(frame):
            im.set_array(intensities[frame])
            return [im]
        
        ani = animation.FuncAnimation(fig, update, frames=self.frames, blit=True, interval=25)
        if not filename is None:
            ani.save(filename, writer='ffmpeg')
        if show:
            plt.show()
        plt.close()


# --- Component Classes ---

class OpticalElement:
    """ Base class for optical elements: things that ineract with the wavefront. """
    def apply(self, wavefront):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    
class ConvergingLens(OpticalElement):
    def __init__(self, focal_length):
        self.focal_length = focal_length
    
    def apply(self, wavefronts):
        """ Apply lens phase transformation to the wavefront. (Thin lens approximation) """
        k = wavefronts.k
        X, Y, L = wavefronts.X, wavefronts.Y, wavefronts.L
        phase = - (k / (2 * self.focal_length)) * ((X)**2 + (Y)**2)
        lens_phase = np.exp(1j * phase)
        # Broadcast lens phase to all frames efficiently
        wavefronts.fields *= lens_phase[np.newaxis, :, :]


class CylindricalLens(OpticalElement):
    def __init__(self, focal_length, orientation='horizontal'):
        self.focal_length = focal_length
        self.orientation = orientation
    
    def apply(self, wavefronts):
        """ Apply cylindrical lens phase transformation to the wavefront. (Thin lens approximation) """
        k = wavefronts.k
        X, Y, L = wavefronts.X, wavefronts.Y, wavefronts.L
        if self.orientation == 'horizontal':
            phase = - (k / (2 * self.focal_length)) * ((Y)**2)
        else: # vertical
            phase = - (k / (2 * self.focal_length)) * ((X)**2)
        lens_phase = np.exp(1j * phase)
        # Broadcast lens phase to all frames efficiently
        wavefronts.fields *= lens_phase[np.newaxis, :, :]


class Target(OpticalElement):
    """ Base class for targets: objects that we want to image using wavefronts. """
    def apply(self, wavefront):
        raise NotImplementedError("This method should be implemented by subclasses.")
    

class IdealSPIFIMask(OpticalElement):
    """ SPIFI mask that is idealized (true sine wave pattern)."""
    @staticmethod
    def apply(wavefronts, K, f_c):
        """
        Takes in a wavefront, which has T,X,Y, and applies the SPIFI mask over time.
        K is the spifi chirp parameter
        f_c is the center frequency of the SPIFI mask
        """
        total_time = wavefronts.T
        dt = total_time / wavefronts.frames
        N = wavefronts.N
        L = wavefronts.L
        dx = L / N
        x = wavefronts.X[0]  # x coordinates along one axis
        X = np.meshgrid(x, x)[0]  # Only need X coordinate
        
        # Pre-compute time array for all frames
        frame_indices = np.arange(wavefronts.frames)
        t_array = -total_time/2 + frame_indices * dt
        spifi_pattern = 0.5 * (1 + np.cos(2 * np.pi * (K * X + f_c) * t_array[:, np.newaxis, np.newaxis]))
        wavefronts.fields *= spifi_pattern


class RealSPIFIMask(OpticalElement):
    """ SPIFI mask that is more realistic (binary pattern with partial pixel values)."""
    

class SiemensStar(Target):
    """ Siemens star target with anti-aliasing. """
    def __init__(self, wavefront, radius, vertical_shift=0.0, num_spokes=32, supersample=8):
        self.image = SiemensStar.create_siemens_star(radius=radius, vertical_shift=vertical_shift, num_spokes=num_spokes, size_m=wavefront.L, resolution=wavefront.N, supersample=supersample)
        self.L = wavefront.L
        self.N = wavefront.N

    @staticmethod
    def create_siemens_star(radius, vertical_shift=0.0, num_spokes=32, size_m=512, resolution=512, supersample=8):
        """
        NOTE: function adapted largely from Google Gemini output

        Generates a Siemens star with anti-aliasing (partial pixel values).
        """
        # 1. Calculate the high-resolution grid size
        high_res = resolution * supersample
        
        # 2. Create the coordinate grid
        # We center the grid so (0,0) is in the middle of the image
        # shift coordinates vertically
        x = np.linspace(-size_m/2, size_m/2, high_res)
        y = np.linspace(-size_m/2 + vertical_shift, size_m/2 + vertical_shift, high_res)
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

        # apply circular mask to limit to radius
        R = np.sqrt(xv**2 + yv**2)
        binary_high_res[R > radius] = 1.0
        
        # 5. Downsample to calculate proportional area (average pooling)
        # We reshape the array to separate the super-sampled blocks
        # Shape becomes (Resolution, Supersample, Resolution, Supersample)
        reshaped = binary_high_res.reshape(resolution, supersample, resolution, supersample)
        
        # We take the mean over the supersampling axes (axis 1 and 3)
        # This results in the average value (area coverage) for that pixel block
        anti_aliased_image = reshaped.mean(axis=(1, 3))
        
        return anti_aliased_image
    
    def apply(self, wavefronts):
        """ Apply the Siemens star mask to the wavefront intensity. """
        wavefronts.fields *= self.image

    def plot(self, wavefronts=None, filename=None, show=False):
        """ Plot the Siemens star target with optionally overlayed wavefront intensity. """
        plt.imshow(self.image, extent=(-self.L/2, self.L/2, -self.L/2, self.L/2), cmap='gray')
        # plot horizontal line at y=0
        plt.axhline(0, color='red', linestyle='--', label='y=0 line')
        plt.title("Siemens Star Target")
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        if not filename is None:
            plt.savefig(filename)
        if show:
            plt.show()
        plt.close()

    
class Photodiode():
    def __init__(self):
        self.signal = np.array([]) # used to store detected signal over time
        self.T = 0
    
    def detect(self, wavefronts):
        """ Simulate photodiode detection by integrating intensity over its area. """
        self.signal = np.empty(wavefronts.frames)
        self.T = wavefronts.T
        intensities = wavefronts.get_intensities()
        
        # Pre-compute circular mask once (avoid recalculation per frame)
        dx = wavefronts.L / wavefronts.N

        for i in range(wavefronts.frames):
            detected_signal = np.sum(intensities[i]) * (dx**2) # integrate intensity over area
            self.signal[i] = detected_signal
        
    def plot_signal(self, filename=None, show=False):
        """ Plot the detected photodiode signal over time. """
        plt.figure()
        time = np.linspace(0, self.T, len(self.signal))
        plt.plot(time, self.signal)
        plt.title("Photodiode Detected Signal Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Detected Signal (arbitrary units (energy?))")
        plt.grid()
        if not filename is None:
            plt.savefig(filename)
        if show:
            plt.show()
        plt.close()
    
    def image(self, filename=None, show=False):
        """ Convert the detected signal into a SPIFI image using Fourier Transform. """
        # Perform Fourier Transform on the signal
        freq_domain = np.fft.fftshift(np.fft.fft(self.signal))
        dt = self.T / len(self.signal)
        f = np.fft.fftshift(np.fft.fftfreq(len(self.signal), d=dt))
        magnitude = np.abs(freq_domain)

        # plot 1d image
        plt.figure()
        plt.plot(f, magnitude)
        plt.title("SPIFI Signal Frequency Domain")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude (arbitrary units)")
        plt.yscale('log')
        plt.grid()
        if not filename is None:
            plt.savefig(filename)
        if show:
            plt.show()
        plt.close()

        return magnitude
    

