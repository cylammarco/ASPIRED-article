from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle
from aspired import spectral_reduction
import numpy as np
from scipy.ndimage import rotate
from statsmodels.nonparametric.smoothers_lowess import lowess

fits_file = fits.open('ogg2m001-en06-20160111-0005-e00.fits.fz')[1]
data = fits_file.data
header = fits_file.header
data = rotate(data, 6)

red_spatial_mask = np.arange(180, 380)
red_spec_mask = np.arange(15, 1500)

twodspec = spectral_reduction.TwoDSpec(data,
                                       header=header,
                                       spatial_mask=red_spatial_mask,
                                       spec_mask=red_spec_mask,
                                       cosmicray=True,
                                       sigclip=3.,
                                       readnoise=3.5,
                                       gain=2.3,
                                       log_level='CRITICAL',
                                       log_file_name=None)

twodspec.ap_trace(nspec=1,
                  nwindow=30,
                  ap_faint=20,
                  trace_width=20,
                  shift_tol=50,
                  fit_deg=7,
                  display=False)

trace = np.array(twodspec.spectrum_list[0].trace)

# trace is at (spatial) pix 70 for (dispersion) pix 600
extraction_slice = Polygon(
    [i for i in zip(np.arange(len(trace)), trace - 30)] +
    [i for i in zip(np.arange(len(trace))[::-1], trace[::-1] + 30)],
    hatch='\\/...',
    edgecolor='C1',
    alpha=0.3,
    facecolor='none',
    label="Region for extraction")
# arrows
sky_arrow_1 = FancyArrowPatch((750, 35), (750, 50),
                              color='C2',
                              arrowstyle='<|-|>',
                              mutation_scale=8)
sky_arrow_2 = FancyArrowPatch((750, 85), (750, 100),
                              color='C2',
                              arrowstyle='<|-|>',
                              mutation_scale=8)
source_arrow = FancyArrowPatch((750, 55), (750, 80),
                               color='C3',
                               arrowstyle='<|-|>',
                               mutation_scale=8)

# boxes
box_1 = Rectangle((360, 50), 20, 70, edgecolor='black', facecolor='none', lw=1)
box_2 = Rectangle((1110, 40),
                  20,
                  70,
                  edgecolor='black',
                  facecolor='none',
                  lw=1)

slice_1 = twodspec.img[50:120, 370]
slice_2 = twodspec.img[40:110, 1120]

lowess_fit_1 = lowess(slice_1[25:46], np.arange(74, 95), frac=0.1)[:, 1]
lowess_fit_2 = lowess(slice_2[25:46], np.arange(64, 85), frac=0.1)[:, 1]

fig = plt.figure(1, figsize=(6, 6))
fig.clf()

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 2, 3)
ax3 = fig.add_subplot(2, 2, 4)

ax1.imshow(np.log10(twodspec.img),
           origin='lower',
           aspect='auto',
           vmin=np.nanpercentile(np.log10(twodspec.img), 10),
           vmax=np.nanpercentile(np.log10(twodspec.img), 97),
           cmap='gray_r')

# Extraction slice
ax1.add_patch(extraction_slice)
# Sky arrows
ax1.add_patch(sky_arrow_1)
ax1.add_patch(sky_arrow_2)
# Source arrow
ax1.add_patch(source_arrow)
# Boxes
ax1.add_patch(box_1)
ax1.add_patch(box_2)

# Sky
ax1.plot(trace + 30, c='C2', label="Region for sky estimation")
ax1.plot(trace + 20, c='C2')
ax1.plot(trace - 20, c='C2')
ax1.plot(trace - 30, c='C2')

# Source
ax1.plot(trace + 10, c='C3', label="Region for the source count")
ax1.plot(trace - 10, c='C3')
ax1.legend()
ax1.set_xlabel('Pixel (Dispersion Direction)')
ax1.set_ylabel('Pixel (Spatial Direction)')
ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_ylim(10, 190)

ax1.text(300, 40, 'Slice 1')
ax1.text(1050, 30, 'Slice 2')

vmin = min(slice_1) - 10
vmax = max(slice_1) * 1.05

ax2.plot(np.arange(50, 120), slice_1, color='black')
ax2.plot(np.arange(75, 96), lowess_fit_1 + 50, color='C0')
ax2.vlines(trace[370] - 1, vmin, vmax, ls=":", color='black')
ax2.vlines(trace[370] + 10 - 1, vmin, vmax, ls=":", color='C3')
ax2.vlines(trace[370] - 10 - 1, vmin, vmax, ls=":", color='C3')
ax2.vlines(trace[370] + 20 - 1, vmin, vmax, ls=":", color='C2')
ax2.vlines(trace[370] + 30 - 1, vmin, vmax, ls=":", color='C2')
ax2.vlines(trace[370] - 20 - 1, vmin, vmax, ls=":", color='C2')
ax2.vlines(trace[370] - 30 - 1, vmin, vmax, ls=":", color='C2')

ax3.plot(np.arange(40, 110), slice_2, color='black', label='Centroid (Trace)')
ax3.plot(np.arange(65, 86),
         lowess_fit_2 + 50,
         color='C0',
         label='LOWESS profile')
ax3.vlines(trace[1120], vmin, vmax, ls=":", color='black')
ax3.vlines(trace[1120] + 10,
           vmin,
           vmax,
           ls=":",
           color='C3',
           label='Source region')
ax3.vlines(trace[1120] - 10, vmin, vmax, ls=":", color='C3')
ax3.vlines(trace[1120] + 20,
           vmin,
           vmax,
           ls=":",
           color='C2',
           label='Sky region')
ax3.vlines(trace[1120] + 30, vmin, vmax, ls=":", color='C2')
ax3.vlines(trace[1120] - 20, vmin, vmax, ls=":", color='C2')
ax3.vlines(trace[1120] - 30, vmin, vmax, ls=":", color='C2')

ax2.set_xlim(45, 123)
ax3.set_xlim(36, 110)

ax2.set_ylim(vmin, vmax)
ax3.set_ylim(vmin, vmax)
ax3.set_yticks([])

ax2.text(48, 1100, 'Slice 1')
ax3.text(40, 1100, 'Slice 2')

ax2.set_ylabel('Electron count')
ax2.set_xlabel('Pixel (Spatial Direction)')
ax3.set_xlabel('Pixel (Spatial Direction)')

ax3.legend()

plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.05)
plt.savefig('fig_03_extraction_profile.jpg')
plt.savefig('fig_03_extraction_profile.pdf')
