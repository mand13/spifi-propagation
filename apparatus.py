# apparatus.py

import time
import logging
import os

import objects

# estimated run time ~ 1 hour

# --- Simulation Setup ---
total_time = 0.1  # total simulation time in seconds
dt = 0.0001       # time step in seconds
size_m = 0.01    # size of the wavefront in meters
resolution = 2048   # resolution of the wavefront grid (num pixels per side)
wavelength = 670e-9 # wavelength of light in meters (red light)

# --- SPIFI Mask Parameters ---
min_grating_period = (size_m / resolution) * 10 # minimum grating period in meters

# --- Target Parameters ---
siemens_radius = 0.001 # radius of Siemens star in meters
num_spokes = 32 # number of spokes in Siemens star
vertical_shift = 0.0005 # vertical shift of Siemens star center in meters

# --- Photodiode Parameters ---
photodiode_radius = 0.01 # radius of photodiode in meters

# --- Plotting and Output ---
plot_dir = "high_res_long_term_test_01" # use None to not save plots
show_plots = False
logging.basicConfig(level=logging.DEBUG) # Set the root logger level to INFO

fast_debug = False
# -- SUPER FAST SIM PARAMETERS FOR TESTING ---
if fast_debug:
    dt = 0.001
    resolution = 256
    min_grating_period = (size_m / resolution) * 10
    num_spokes = 8

PLOT_COUNT = 0

# enter data directory
if not os.path.exists("data"):
    os.makedirs("data")
os.chdir("data")

# create dir if it doesn't exist
if plot_dir is not None and not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# time the simulation
start_time = time.time()

# initalize wavefront
wavefronts = objects.Wavefronts(size_m=size_m, resolution=resolution, total_time=total_time, frames=int(total_time/dt), wavelength=wavelength)
if plot_dir is not None:
    wavefronts.plot_wavefront(title="Initial Wavefront", filename=f"{plot_dir}/{PLOT_COUNT:02d}_initial_wavefront.png", show=show_plots)
    PLOT_COUNT += 1
elif show_plots:
    wavefronts.plot_wavefront(title="Initial Wavefront", show=show_plots)
logging.debug("Initialized wavefront.")

# Propagate the wavefront by 0.1 meters to the cylindrical lens
wavefronts.propagate(distance=0.1)
if plot_dir is not None:
    wavefronts.plot_wavefront(title="Wavefront after 0.1 m Propagation", filename=f"{plot_dir}/{PLOT_COUNT:02d}_after_0.1m_propagation.png", show=show_plots)
    PLOT_COUNT += 1
elif show_plots:
    wavefronts.plot_wavefront(title="Wavefront after 0.1 m Propagation", show=show_plots)
logging.debug("Propagated wavefront to cylindrical lens.")

# Apply a cylindrical lens
cylindrical_lens = objects.CylindricalLens(focal_length=0.1, orientation='horizontal')
cylindrical_lens.apply(wavefronts)
logging.debug("Applied cylindrical lens.")

# propagate to focal plane
wavefronts.propagate(distance=0.1)
if plot_dir is not None:
    wavefronts.plot_wavefront(title="Wavefront after Cylindrical Lens", filename=f"{plot_dir}/{PLOT_COUNT:02d}_after_cylindrical_lens.png", show=show_plots)
    PLOT_COUNT += 1
elif show_plots:
    wavefronts.plot_wavefront(title="Wavefront after Cylindrical Lens", show=show_plots)
logging.debug("Propagated wavefront to focal plane of cylindrical lens, where the SPIFI mask is.")

# apply spifi mask
objects.IdealSPIFIMask.apply(wavefronts, min_grating_period=min_grating_period)
if plot_dir is not None:
    wavefronts.animate_wavefront(title="Wavefront after SPIFI Mask Animation", filename=f"{plot_dir}/{PLOT_COUNT:02d}_after_spifi_mask_animation.mp4", show=show_plots)
    PLOT_COUNT += 1
elif show_plots:
    wavefronts.animate_wavefront(title="Wavefront after SPIFI Mask Animation", show=show_plots)
logging.debug("Applied SPIFI mask.")

# propagate to the next lens
wavefronts.propagate(distance=0.2)
if plot_dir is not None:
    wavefronts.animate_wavefront(title="Wavefront before Imaging Lens Animation", filename=f"{plot_dir}/{PLOT_COUNT:02d}_before_imaging_lens_animation.mp4", show=show_plots)
    PLOT_COUNT += 1
elif show_plots:
    wavefronts.animate_wavefront(title="Wavefront before Imaging Lens Animation", show=show_plots)
logging.debug("Propagated wavefront to imaging lens.")

# apply second lens and propagate to object plane (imaging lens)
imaging_lens = objects.ConvergingLens(focal_length=0.1)
imaging_lens.apply(wavefronts)
wavefronts.propagate(distance=0.2)
if plot_dir is not None:
    wavefronts.animate_wavefront(title="Wavefront before Hitting Target (first frame)", filename=f"{plot_dir}/{PLOT_COUNT:02d}_before_hitting_target_animation.mp4", show=show_plots)
    PLOT_COUNT += 1
elif show_plots:
    wavefronts.animate_wavefront(title="Wavefront before Hitting Target (first frame)", show=show_plots)
logging.debug("Applied imaging lens and propagated to object plane.")

# wavefronts hit the target
target = objects.SiemensStar(wavefronts, radius=siemens_radius, vertical_shift=vertical_shift, num_spokes=num_spokes)
target.apply(wavefronts)
if plot_dir is not None:
    target.plot(wavefronts)
    wavefronts.animate_wavefront(title="Wavefront after Hitting Target Animation", filename=f"{plot_dir}/{PLOT_COUNT:02d}_after_hitting_target_animation.mp4", show=show_plots)
    PLOT_COUNT += 1
elif show_plots:
    wavefronts.animate_wavefront(title="Wavefront after Hitting Target Animation", show=show_plots)
logging.debug("Applied Siemens star target.")

# Propagate to collection lens
wavefronts.propagate(distance=0.2)
if plot_dir is not None:
    wavefronts.animate_wavefront(title="Wavefront before Collection Lens Animation", filename=f"{plot_dir}/{PLOT_COUNT:02d}_before_collection_lens_animation.mp4", show=show_plots)
    PLOT_COUNT += 1
elif show_plots:
    wavefronts.animate_wavefront(title="Wavefront before Collection Lens Animation", show=show_plots)
logging.debug("Propagated wavefront to collection lens.")

# Apply collection lens and propagate to detector
collection_lens = objects.ConvergingLens(focal_length=0.1)
collection_lens.apply(wavefronts)
wavefronts.propagate(distance=0.2)
if plot_dir is not None:
    wavefronts.animate_wavefront(title="Wavefront at Photodiode Animation", filename=f"{plot_dir}/{PLOT_COUNT:02d}_at_photodiode_animation.mp4", show=show_plots)
    PLOT_COUNT += 1
elif show_plots:
    wavefronts.animate_wavefront(title="Wavefront at Photodiode Animation", show=show_plots)
logging.debug("Applied collection lens and propagated to photodiode.")

# photodiode detects signal
detector = objects.Photodiode(radius=photodiode_radius)
detector.detect(wavefronts)
logging.debug("Photodiode detected signal.")

time_elapsed = time.time() - start_time
print(f"Simulation completed in {time_elapsed:.2f} seconds.")

detector.plot_signal(filename=f"{plot_dir}/{PLOT_COUNT:02d}_photodiode_signal.png", show=show_plots)
PLOT_COUNT += 1
detector.image(filename=f"{plot_dir}/{PLOT_COUNT:02d}_photodiode_image.png", show=show_plots)










