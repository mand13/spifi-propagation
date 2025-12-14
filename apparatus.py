# apparatus.py

# EXPERIMENTAL ON DAYSTROM LINES
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for plotting
# ^^ END OF DAYSTROM LINES ^^

import time
import logging
import os
import matplotlib.pyplot as plt 
import numpy as np

import objects




# --- MODIFY SIMULATION PARAMETERS BELOW ---

# --- Simulation Setup ---
total_time = 12  # total simulation time in seconds
num_frames = 2048 # total number of time frames
dt = total_time / num_frames       # time step in seconds
size_m = 0.025    # size of the wavefront in meters
resolution = 1024   # resolution of the wavefront grid (num pixels per side)
dx = size_m / resolution  # spatial step in meters
wavelength = 1e-6 # wavelength of light in meters

# --- Target Parameters ---
siemens_radius = 0.004 # radius of Siemens star in meters
num_spokes = 32 # number of spokes in Siemens star
vertical_shift = 0.002 # vertical shift of Siemens star center in meters

# --- Photodiode Parameters ---
photodiode_radius = 0.01 # radius of photodiode in meters

# --- Plotting and Output ---
plot_dir = "newer7"
show_plots = False
normalize = True # run a second simulation without the target for normalization
logging.basicConfig(level=logging.INFO) # Set the root logger level to INFO

fast_debug = False
# -- SUPER FAST SIM PARAMETERS FOR TESTING ---
if fast_debug:
    num_frames = 256
    dt = total_time / num_frames
    resolution = 128
    dx = size_m / resolution
    num_spokes = 8

# --- Calculate ideal SPIFI parameters based on resolution ---
S = 7 # resolution spifi safety factor (determined to make any value of K work well)
K = min(2 / (S * total_time * dx), 1 / (2 * size_m * S * dt), 1 / (14 * size_m * dt)) # spifi spatial chirp parameter
f_c = 1.5 * K * size_m # center frequency of SPIFI mask

# --- END OF MODIFIABLE PARAMETERS ---













global_start_time = time.time()

PLOT_COUNT = 0

# enter data directory
if not os.path.exists("data"):
    os.makedirs("data")
os.chdir("data")

# create dir if it doesn't exist
if plot_dir is not None and not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


def run_simulation(use_target=True):
    global PLOT_COUNT

    # time the simulation
    start_time = time.time()

    # initalize wavefront
    wavefronts = objects.Wavefronts(size_m=size_m, resolution=resolution, total_time=total_time, frames=int(total_time/dt), wavelength=wavelength)
    if plot_dir is not None:
        wavefronts.plot_wavefront(title="Initial Wavefront", filename=f"{plot_dir}/{PLOT_COUNT:02d}_initial_wavefront.png", show=show_plots)
        PLOT_COUNT += 1
    elif show_plots:
        wavefronts.plot_wavefront(title="Initial Wavefront", show=show_plots)
    logging.info("Initialized wavefront.")

    # # Propagate the wavefront by 0.1 meters to the cylindrical lens
    # wavefronts.propagate(distance=0.1)
    # if plot_dir is not None:
    #     wavefronts.plot_wavefront(title="Wavefront after 0.1 m Propagation", filename=f"{plot_dir}/{PLOT_COUNT:02d}_after_0.1m_propagation.png", show=show_plots)
    #     PLOT_COUNT += 1
    # elif show_plots:
    #     wavefronts.plot_wavefront(title="Wavefront after 0.1 m Propagation", show=show_plots)
    # logging.info("Propagated wavefront to cylindrical lens.")

    # Apply a cylindrical lens
    cylindrical_lens = objects.CylindricalLens(focal_length=0.1, orientation='horizontal')
    cylindrical_lens.apply(wavefronts)
    logging.info("Applied cylindrical lens.")

    # propagate to focal plane
    wavefronts.propagate(distance=0.1)
    if plot_dir is not None:
        wavefronts.plot_wavefront(title="Wavefront after Cylindrical Lens", filename=f"{plot_dir}/{PLOT_COUNT:02d}_after_cylindrical_lens.png", show=show_plots)
        PLOT_COUNT += 1
    elif show_plots:
        wavefronts.plot_wavefront(title="Wavefront after Cylindrical Lens", show=show_plots)
    logging.info("Propagated wavefront to focal plane of cylindrical lens, where the SPIFI mask is.")

    # apply spifi mask
    objects.IdealSPIFIMask.apply(wavefronts, K=K, f_c=f_c)
    if plot_dir is not None:
        wavefronts.animate_wavefront(title="Wavefront after SPIFI Mask Animation", filename=f"{plot_dir}/{PLOT_COUNT:02d}_after_spifi_mask_animation.mp4", show=show_plots)
        PLOT_COUNT += 1
    elif show_plots:
        wavefronts.animate_wavefront(title="Wavefront after SPIFI Mask Animation", show=show_plots)
    logging.info("Applied SPIFI mask.")

    # propagate to the next lens
    wavefronts.propagate(distance=0.2)
    if plot_dir is not None:
        wavefronts.animate_wavefront(title="Wavefront before Imaging Lens Animation", filename=f"{plot_dir}/{PLOT_COUNT:02d}_before_imaging_lens_animation.mp4", show=show_plots)
        PLOT_COUNT += 1
    elif show_plots:
        wavefronts.animate_wavefront(title="Wavefront before Imaging Lens Animation", show=show_plots)
    logging.info("Propagated wavefront to imaging lens.")

    # apply second lens and propagate to object plane (imaging lens)
    imaging_lens = objects.ConvergingLens(focal_length=0.1)
    imaging_lens.apply(wavefronts)
    wavefronts.propagate(distance=0.2)
    if plot_dir is not None:
        wavefronts.animate_wavefront(title="Wavefront before Hitting Target (first frame)", filename=f"{plot_dir}/{PLOT_COUNT:02d}_before_hitting_target_animation.mp4", show=show_plots)
        PLOT_COUNT += 1
    elif show_plots:
        wavefronts.animate_wavefront(title="Wavefront before Hitting Target (first frame)", show=show_plots)
    logging.info("Applied imaging lens and propagated to object plane.")

    # wavefronts hit the target
    if use_target:
        target = objects.SiemensStar(wavefronts, radius=siemens_radius, vertical_shift=vertical_shift, num_spokes=num_spokes)
        target.apply(wavefronts)
        if plot_dir is not None:
            target.plot(wavefronts, filename=f"{plot_dir}/{PLOT_COUNT:02d}_siemens_star.png", show=show_plots)
            PLOT_COUNT += 1
            wavefronts.animate_wavefront(title="Wavefront after Hitting Target Animation", filename=f"{plot_dir}/{PLOT_COUNT:02d}_after_hitting_target_animation.mp4", show=show_plots)
            PLOT_COUNT += 1
        elif show_plots:
            wavefronts.animate_wavefront(title="Wavefront after Hitting Target Animation", show=show_plots)
        logging.info("Applied Siemens star target.")

    # Propagate to collection lens
    wavefronts.propagate(distance=0.2)
    if plot_dir is not None:
        wavefronts.animate_wavefront(title="Wavefront before Collection Lens Animation", filename=f"{plot_dir}/{PLOT_COUNT:02d}_before_collection_lens_animation.mp4", show=show_plots)
        PLOT_COUNT += 1
    elif show_plots:
        wavefronts.animate_wavefront(title="Wavefront before Collection Lens Animation", show=show_plots)
    logging.info("Propagated wavefront to collection lens.")

    # # Apply collection lens and propagate to detector
    # collection_lens = objects.ConvergingLens(focal_length=0.1)
    # collection_lens.apply(wavefronts)
    # wavefronts.propagate(distance=0.2)
    # if plot_dir is not None:
    #     wavefronts.animate_wavefront(title="Wavefront at Photodiode Animation", filename=f"{plot_dir}/{PLOT_COUNT:02d}_at_photodiode_animation.mp4", show=show_plots)
    #     PLOT_COUNT += 1
    # elif show_plots:
    #     wavefronts.animate_wavefront(title="Wavefront at Photodiode Animation", show=show_plots)
    # logging.info("Applied collection lens and propagated to photodiode.")

    # photodiode detects signal
    detector = objects.Photodiode()
    detector.detect(wavefronts)
    logging.info("Photodiode detected signal.")

    time_elapsed = time.time() - start_time
    print(f"Simulation completed in {time_elapsed:.2f} seconds.")

    if not plot_dir is None:
        detector.plot_signal(filename=f"{plot_dir}/{PLOT_COUNT:02d}_photodiode_signal.png", show=show_plots)
        PLOT_COUNT += 1
        magnitude = detector.image(filename=f"{plot_dir}/{PLOT_COUNT:02d}_spifi_image.png", show=show_plots)
        np.save(f"{plot_dir}/{PLOT_COUNT:02d}_spifi_image.npy", magnitude)
        PLOT_COUNT += 1

    print(f"\n\n\nSimulation completed in {time_elapsed:.2f} seconds.\n\n\n")

    return magnitude

target_image = run_simulation(use_target=True)

if (normalize):
    no_target_image = run_simulation(use_target=False)
    normalized_image = target_image / no_target_image
    # Perform Fourier Transform on the signal
    freq_domain = np.fft.fftshift(np.fft.fft(normalized_image))
    dt = total_time / len(normalized_image)
    f = np.fft.fftshift(np.fft.fftfreq(len(normalized_image), d=dt))
    magnitude = np.abs(freq_domain)
    if plot_dir is not None:
        np.save(f"{plot_dir}/{PLOT_COUNT:02d}_spifi_image_normalized.npy", normalized_image)

    # plot 1d image
    plt.figure()
    plt.plot(f, magnitude)
    plt.title("SPIFI Signal Frequency Domain")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (arbitrary units)")
    plt.yscale('log')
    plt.grid()
    if not plot_dir is None:
        plt.savefig(f"{plot_dir}/{PLOT_COUNT:02d}_spifi_image_normalized.png")
        PLOT_COUNT += 1
    if show_plots:
        plt.show()
    plt.close()

global_elapsed_time = time.time() - global_start_time
print(f"Total script time: {global_elapsed_time:.2f} seconds.")

with open(f"{plot_dir}/apparatus_parameters.txt", "w") as f:
    f.write(f"total_time = {total_time}\n")
    f.write(f"dt = {dt}\n")
    f.write(f"size_m = {size_m}\n")
    f.write(f"resolution = {resolution}\n")
    f.write(f"wavelength = {wavelength}\n")
    f.write(f"siemens_radius = {siemens_radius}\n")
    f.write(f"num_spokes = {num_spokes}\n")
    f.write(f"vertical_shift = {vertical_shift}\n")
    f.write(f"S = {S}\n")
    f.write(f"K = {K}\n")
    f.write(f"f_c = {f_c}\n")
    f.write(f"normalize = {normalize}\n")
    f.write(f"Elapsed Simulation Time = {global_elapsed_time:.2f} seconds\n")









