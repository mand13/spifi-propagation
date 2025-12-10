# apparatus.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import objects

# --- Simulation Setup ---
total_time = 0.01  # total simulation time in seconds
dt = 0.0001        # time step in seconds
size_m = 0.01    # size of the wavefront in meters
resolution = 512   # resolution of the wavefront grid (num pixels per side)
wavelength = 670e-9 # wavelength of light in meters (red light)

# --- SPIFI Mask Parameters ---
min_grating_period = (size_m / resolution) * 20 # minimum grating period in meters

# --- Target Parameters ---
siemens_radius = 0.001 # radius of Siemens star in meters
num_spokes = 32 # number of spokes in Siemens star
vertical_shift = 0.0005 # vertical shift of Siemens star center in meters

# --- Photodiode Parameters ---
photodiode_radius = 0.01 # radius of photodiode in meters

# initalize wavefront
wavefront = objects.Wavefront(size_m=size_m, resolution=resolution, wavelength=wavelength)
wavefront.plot_wavefront(title="Initial Wavefront")

# Propagate the wavefront by 0.1 meters to the cylindrical lens
wavefront.propagate(distance=0.1)
wavefront.plot_wavefront(title="Wavefront after 0.1 m Propagation")

# Apply a cylindrical lens
cylindrical_lens = objects.CylindricalLens(focal_length=0.1, orientation='horizontal')
cylindrical_lens.apply(wavefront)

# propagate to focal plane
wavefront.propagate(distance=0.1)
wavefront.plot_wavefront(title="Wavefront after Cylindrical Lens")

# apply spifi mask
spifi_mask = objects.IdealSPIFIMask()
wavefronts = spifi_mask.apply(wavefront, total_time=total_time, dt=dt, min_grating_period=min_grating_period) # make sure min_grating period is at least 10x pixel size
wavefronts[0].plot_wavefront(title="Wavefront after SPIFI Mask (first frame)")

# propagate wavefront at each time step to the next lens
for wf in wavefronts:
    wf.propagate(distance=0.2)
wavefronts[0].plot_wavefront(title="Wavefront after SPIFI Mask and Propagation (first frame)")

wavefronts[0].plot_wavefront(title="Wavefront after SPIFI Mask and Propagation (first frame)")

# apply second lens and propagate to object plane (imaging lens)
imaging_lens = objects.ConvergingLens(focal_length=0.1)
for wf in wavefronts:
    imaging_lens.apply(wf)
    wf.propagate(distance=0.2)
wavefronts[0].plot_wavefront(title="Wavefront after Imaging Lens (first frame)")

# wavefronts hit the target
target = objects.SiemensStar(wavefront, radius=siemens_radius, vertical_shift=vertical_shift, num_spokes=num_spokes)
#target.plot()
target.plot(wavefronts[0])
for wf in wavefronts:
    target.apply(wf)
wavefronts[0].plot_wavefront(title="Wavefront after hitting Siemens Star Target (first frame)")

# Propagate to collection lens
for wf in wavefronts:
    wf.propagate(distance=0.2)
wavefronts[0].plot_wavefront(title="Wavefront at Collection Lens (first frame)")

# Apply collection lens and propagate to detector
collection_lens = objects.ConvergingLens(focal_length=0.1)
for wf in wavefronts:
    collection_lens.apply(wf)
    wf.propagate(distance=0.2)
wavefronts[0].plot_wavefront(title="Wavefront after Collection Lens (first frame)")

# photodiode detects signal
detector = objects.Photodiode(radius=photodiode_radius)
signal = np.empty(len(wavefronts))
for i, wf in enumerate(wavefronts):
    signal[i] = detector.detect(wf)

# plot detected signal over time
time = np.linspace(0, total_time, len(wavefronts))
plt.figure()
plt.plot(time, signal)
plt.xlabel("Time (s)")
plt.ylabel("Detected Signal (a.u.)")
plt.title("Photodiode Detected Signal Over Time")
plt.show()

image = objects.SPIFISignalToImageConverter.signal_to_image(signal, dt)










