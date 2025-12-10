# apparatus.py

import numpy as np
import matplotlib.pyplot as plt

import objects

# TODO fix
# target = objects.SiemensStar(radius=0.002, vertical_shift=0.0)
# target.plot(resolution=512)

# initalize wavefront
wavefront = objects.Wavefront(size_m=0.01, resolution=512, wavelength=670e-9)
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
wavefronts = spifi_mask.apply(wavefront, total_time=0.01, dt=0.0001, min_grating_period=0.0005) # make sure min_grating period is at least 10x pixel size
wavefronts[0].plot_wavefront(title="Wavefront after SPIFI Mask (first frame)")

# propagate wavefront at each time step to the next lens
for wf in wavefronts:
    wf.propagate(distance=0.2)
wavefronts[0].plot_wavefront(title="Wavefront after SPIFI Mask and Propagation (first frame)")

# apply second lens and propagate to object plane (imaging lens)
imaging_lens = objects.ConvergingLens(focal_length=0.1)
for wf in wavefronts:
    imaging_lens.apply(wf)
    wf.propagate(distance=0.2)
wavefronts[0].plot_wavefront(title="Wavefront after Imaging Lens (first frame)")

# wavefronts hit the target
target = objects.SiemensStar(radius=0.002)
target.plot(resolution=512)
for wf in wavefronts:
    target.apply(wf)




