from astropy.io import fits
from aspired import spectral_reduction
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage
import copy
import numpy as np

fits_file = fits.open('ogg2m001-en06-20160111-0005-e00.fits.fz')[1]
data = fits_file.data
header = fits_file.header

red_spatial_mask = np.arange(0, 330)
red_spec_mask = np.arange(0, 1500)

upsample_factor = 10

red = spectral_reduction.TwoDSpec(data,
                                  header=header,
                                  spatial_mask=red_spatial_mask,
                                  spec_mask=red_spec_mask,
                                  cosmicray=True,
                                  sigclip=3.,
                                  readnoise=3.5,
                                  gain=2.3,
                                  log_level='CRITICAL',
                                  log_file_name=None)

red.ap_trace(nspec=1,
             ap_faint=10,
             trace_width=25,
             shift_tol=50,
             fit_deg=7,
             display=False)

red.compute_rectification(upsample_factor=upsample_factor)

img_tmp = ndimage.zoom(red.img, zoom=upsample_factor)

y_tmp = ndimage.zoom(np.array(red.spectrum_list[0].trace),
                     zoom=upsample_factor)

for i in range(1, red.spec_size * upsample_factor):
    shift_i = int(
        np.round((y_tmp[i] - y_tmp[len(y_tmp) // 2]) * upsample_factor))
    img_column_i = img_tmp[:, i]
    img_tmp[:, i] = np.roll(img_column_i, -shift_i)

img_tmp = ndimage.zoom(img_tmp, zoom=1. / upsample_factor)

fig = plt.figure(1, figsize=(6, 6))
fig.clf()

ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

ax1.imshow(np.log10(red.img),
           vmin=np.nanpercentile(np.log10(red.img), 10),
           vmax=np.nanpercentile(np.log10(red.img), 99),
           origin='lower',
           aspect='auto')
ax2.imshow(np.log10(img_tmp),
           vmin=np.nanpercentile(np.log10(red.img), 10),
           vmax=np.nanpercentile(np.log10(red.img), 99),
           origin='lower',
           aspect='auto')
ax3.imshow(np.log10(red.img_rectified),
           vmin=np.nanpercentile(np.log10(red.img), 10),
           vmax=np.nanpercentile(np.log10(red.img), 99),
           origin='lower',
           aspect='auto')

ax1.vlines(200, 0, 250, lw=1, ls=":", color='white')
ax2.vlines(200, 0, 250, lw=1, ls=":", color='white')
ax3.vlines(200, 0, 250, lw=1, ls=":", color='white')

ax1.set_ylim(70, 230)
ax2.set_ylim(70, 150)
ax3.set_ylim(70, 150)

ax1.set_xticks([])
ax2.set_xticks([])

ax2.set_ylabel('Pixel (Spatial)')
ax3.set_xlabel('Pixel (Dispersion)')

plt.tight_layout()
plt.subplots_adjust(hspace=0)

fig.savefig('fig_02_rectification.jpg')
