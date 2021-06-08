from aspired import spectral_reduction
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import copy
import numpy as np

data = fits.open('v_s_20180810_27_1_0_2.fits.gz')[0].data

twodspec = spectral_reduction.TwoDSpec(data, cosmicray=True, readnoise=5.7, gain=2.6)

twodspec.ap_trace(nspec=1, nwindow=10, display=False)

# Tophat
twodspec.ap_extract(apwidth=10, skywidth=5, skysep=0, skydeg=1, optimal=False)
count_tophat = copy.deepcopy(twodspec.spectrum_list[0].count)
img_residual_tophat = copy.deepcopy(twodspec.img_residual)

# Horne86
twodspec.ap_extract(apwidth=10,
                    skywidth=5,
                    skysep=0,
                    skydeg=1,
                    optimal=True,
                    algorithm='horne86')
count_horne86 = copy.deepcopy(twodspec.spectrum_list[0].count)
img_residual_horne86 = copy.deepcopy(twodspec.img_residual)

# Marsh89
twodspec.ap_extract(apwidth=10,
                    skywidth=5,
                    skysep=0,
                    skydeg=1,
                    optimal=True,
                    algorithm='marsh89',
                    pord=4,
                    nreject=0,
                    qmode='fast-linear')
count_marsh89 = copy.deepcopy(twodspec.spectrum_list[0].count)
img_residual_marsh89 = copy.deepcopy(twodspec.img_residual)


fig = plt.figure(1, figsize=(6, 6))
fig.clf()

gs = GridSpec(5, 1, figure=fig)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1:3])
ax3 = fig.add_subplot(gs[3:])

ax1.imshow(data, origin='lower', aspect='auto')
ax1.plot(np.array(twodspec.spectrum_list[0].trace)-0.5, lw=1, ls=':', color='black')
ax1.set_xticks([])
ax1.set_ylim(120, 145)
ax1.set_ylabel('Pixel (Spatial)')

ax2.plot(count_tophat, color='red', label='Tophat')
ax2.plot(count_horne86, color='green', label='Horne86')
ax2.plot(count_marsh89, color='blue', label='Marsh89')

ax2.set_ylabel(r'e$^{-}$ count')
ax2.set_ylim(0, 45000)
ax2.legend(framealpha=0.9)

ax3.plot(np.array(count_horne86) / np.array(count_tophat), color='red', label='Horne86 / Tophat')
ax3.plot(np.array(count_horne86) / np.array(count_marsh89), color='blue', label='Horne86 / Marsh89')
ax3.legend(loc='upper left', framealpha=0.9)
ax3.set_ylim(0.995, 1.055)
ax3.set_xlabel('Pixel (Dispersion)')
ax3.set_ylabel(r'e$^-$ count Ratio (Spatial)')

ax1.set_xlim(0, 1024)
ax2.set_xlim(0, 1024)
ax3.set_xlim(0, 1024)

fig.tight_layout()
fig.subplots_adjust(hspace=0)
fig.savefig('fig_03_extraction_compared.jpg')

fig2 = plt.figure(1, figsize=(6, 6))
fig2.clf()

ax1 = fig2.add_subplot(3, 1, 1)
ax2 = fig2.add_subplot(3, 1, 2)
ax3 = fig2.add_subplot(3, 1, 3)

ax1.imshow(img_residual_tophat, origin='lower', aspect='auto')
ax2.imshow(img_residual_horne86, origin='lower', aspect='auto')
ax3.imshow(img_residual_marsh89, origin='lower', aspect='auto')

ax1.set_ylim(120, 145)
ax2.set_ylim(120, 145)
ax3.set_ylim(120, 145)

ax1.set_xticks([])
ax2.set_xticks([])

fig.tight_layout()
fig.subplots_adjust(hspace=0)
