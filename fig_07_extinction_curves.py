from aspired import spectral_reduction
from matplotlib import pyplot as plt
import numpy as np

wave = np.arange(3000, 10000)
onedspec = spectral_reduction.OneDSpec(logger_name=None)

onedspec.set_atmospheric_extinction('orm')
orm_ext = onedspec.extinction_func(wave)

onedspec.set_atmospheric_extinction('mk')
mk_ext = onedspec.extinction_func(wave)

onedspec.set_atmospheric_extinction('cp')
cp_ext = onedspec.extinction_func(wave)

onedspec.set_atmospheric_extinction('ls')
ls_ext = onedspec.extinction_func(wave)

plt.figure(7, figsize=(6, 6))
plt.clf()
plt.plot(wave, orm_ext, label='Roque de los Muchachos Observatory (2420 m)')
plt.plot(wave, mk_ext, label='Mauna Kea Observatories (4205 m)')
plt.plot(wave, cp_ext, label='Cerro Paranal Observatory (2635 m)')
plt.plot(wave, ls_ext, label='La Silla Observatory (2400 m)')

plt.xlim(3000, 10000)
plt.ylim(0, 1)

plt.xlabel(r'Wavelength ($\mathrm{\AA}$)')
plt.ylabel('Extinction ( mag / airmass )')

plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig('fig_07_extinction_curves.jpg')
plt.savefig('fig_07_extinction_curves.pdf')
