from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import rotate
from scipy.signal import correlate
import numpy as np

data = fits.open('v_s_20180810_27_1_0_2.fits.gz')[0].data
data = rotate(data, 2.15)

fig1 = plt.figure(1, figsize=(6, 6))
fig1.clf()

# top
ax1 = fig1.add_subplot(2, 1, 1)
# bottom left
ax2 = fig1.add_subplot(2, 2, 3)
# bottom right
ax3 = fig1.add_subplot(2, 2, 4)

ax1.imshow(data, origin='lower', aspect='auto')

rect_2 = Rectangle((200, 140),
                   100,
                   40,
                   lw=3,
                   edgecolor='orange',
                   facecolor='none')
rect_3 = Rectangle((250, 140),
                   100,
                   40,
                   lw=2,
                   edgecolor='green',
                   facecolor='none')
rect_4 = Rectangle((300, 140),
                   100,
                   40,
                   lw=1,
                   edgecolor='blue',
                   facecolor='none')

ax1.add_patch(rect_2)
ax1.add_patch(rect_3)
ax1.add_patch(rect_4)

ax1.xaxis.tick_top()
ax1.xaxis.set_label_position('top')
ax1.set_xlim(100, 500)
ax1.set_xlabel('Pixel (Dispersion)')
ax1.set_ylim(139, 181)
ax1.set_ylabel('Pixel (Spatial)')

line2 = np.sum(data[140:180, 200:300], axis=1)
line3 = np.sum(data[140:180, 250:350], axis=1)
line4 = np.sum(data[140:180, 350:400], axis=1)

line2 /= max(line2)
line3 /= max(line3)
line4 /= max(line4)

ax2.plot(range(140, 180), line2, color='orange')
ax2.plot(range(140, 180), line3, color='green')
ax2.plot(range(140, 180), line4, color='blue')
ax2.vlines(np.argmax(line2) + 140,
           0,
           max(line2) * 1.05,
           ls=":",
           color='orange')
ax2.vlines(np.argmax(line3) + 140, 0, max(line3) * 1.05, ls=":", color='green')
ax2.vlines(np.argmax(line4) + 140, 0, max(line4) * 1.05, ls=":", color='blue')
ax2.set_xlim(151, 166)
ax2.set_ylim(-0.01, max(line2) * 1.05)

ax2.set_yticks([])

ax2.set_ylabel(r'Normalised e$^{-}$ count')
ax2.set_xlabel('Pixel (Spatial)')

cor1 = correlate(line3, line2, mode='same')
cor2 = correlate(line4, line3, mode='same')

ax3.plot(range(-int(len(line2) / 2), int(len(line2) / 2)),
         cor1,
         color='green',
         label='Rel. to Orange')
ax3.plot(range(-int(len(line2) / 2), int(len(line2) / 2)),
         cor2,
         color='blue',
         label='Rel. to Green')
ax3.vlines(range(-int(len(line2) / 2), int(len(line2) / 2))[np.argmax(cor1)],
           0,
           max(cor1) * 1.05,
           ls=":",
           color='green')
ax3.vlines(range(-int(len(line2) / 2), int(len(line2) / 2))[np.argmax(cor2)],
           0,
           max(cor1) * 1.05,
           ls=":",
           color='blue')

ax3.set_ylabel('Cross-correlated value')
ax3.set_xlabel('Pixel shift')
ax3.set_xlim(-12, 6)
ax3.set_ylim(0, max(cor1) * 1.05)
ax3.set_yticks([])
ax3.legend()

fig1.tight_layout()
fig1.savefig('fig_01_tracing.jpg')
