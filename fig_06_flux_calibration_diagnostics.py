from aspired import spectral_reduction
from astropy.io import fits
from matplotlib import pyplot as plt
import copy
import numpy as np

atlas = [
    4193.5, 4385.77, 4500.98, 4524.68, 4582.75, 4624.28, 4671.23, 4697.02,
    4734.15, 4807.02, 4921.48, 5028.28, 5618.88, 5823.89, 5893.29, 5934.17,
    6182.42, 6318.06, 6472.841, 6595.56, 6668.92, 6728.01, 6827.32, 6976.18,
    7119.60, 7257.9, 7393.8, 7584.68, 7642.02, 7740.31, 7802.65, 7887.40,
    7967.34, 8057.258
]
element = ['Xe'] * len(atlas)

standard_fits = fits.open('v_s_20180810_27_1_0_2.fits.gz')[0]
science_fits = fits.open('v_e_20180810_12_1_0_2.fits.gz')[0]
arc = fits.open('v_a_20180810_28_1_0_1.fits.gz')[0]

standard_twodspec = spectral_reduction.TwoDSpec(standard_fits.data,
                                                standard_fits.header,
                                                cosmicray=True)

standard_twodspec.ap_trace(nspec=1, nwindow=10, display=False)

science_twodspec = spectral_reduction.TwoDSpec(science_fits.data,
                                               science_fits.header,
                                               cosmicray=True)

science_twodspec.ap_trace(nspec=1, nwindow=10, display=False)

# Standard
standard_twodspec.add_arc(arc)
standard_twodspec.extract_arc_spec()
standard_twodspec.ap_extract(apwidth=10,
                             skywidth=5,
                             skysep=0,
                             skydeg=1,
                             optimal=True,
                             algorithm='horne86')
standard_count = copy.deepcopy(standard_twodspec.spectrum_list[0].count)

# Science
science_twodspec.add_arc(arc)
science_twodspec.extract_arc_spec()
science_twodspec.ap_extract(apwidth=10,
                            skywidth=5,
                            skysep=0,
                            skydeg=1,
                            optimal=True,
                            algorithm='horne86')
science_count = copy.deepcopy(science_twodspec.spectrum_list[0].count)

onedspec = spectral_reduction.OneDSpec()
onedspec.from_twodspec(science_twodspec, stype='science')
onedspec.from_twodspec(standard_twodspec, stype='standard')

# Find the peaks of the arc
onedspec.find_arc_lines(prominence=2,
                        distance=5,
                        refine_window_width=3)

onedspec.initialise_calibrator()
onedspec.set_hough_properties(range_tolerance=500.,
                              xbins=100,
                              ybins=100,
                              min_wavelength=3800.,
                              max_wavelength=8200.)
onedspec.set_ransac_properties(sample_size=10,
                               top_n_candidate=10,
                               filter_close=True,
                               ransac_tolerance=10.)
onedspec.add_user_atlas(elements=element,
                        wavelengths=atlas,
                        constrain_poly=True)
onedspec.do_hough_transform()

# Solve for the pixel-to-wavelength solution
onedspec.fit(max_tries=2000)

onedspec.apply_wavelength_calibration()

onedspec.load_standard('hilt102')
onedspec.get_sensitivity()
onedspec.apply_flux_calibration()

sensitivity = np.array(onedspec.science_spectrum_list[0].sensitivity)
wave = np.array(onedspec.standard_spectrum_list[0].wave)
count = np.array(onedspec.standard_spectrum_list[0].count)

wave_standard = np.array(onedspec.standard_spectrum_list[0].wave_resampled)
flux_standard = np.array(onedspec.standard_spectrum_list[0].flux_resampled)

wave_literature = np.array(onedspec.standard_spectrum_list[0].wave_literature)
flux_literature = np.array(onedspec.standard_spectrum_list[0].flux_literature)

mask = wave > 4000.

fig = plt.figure(1, figsize=(8, 6))
fig.clf()

ax1 = fig.add_subplot(2, 1, 1)
ax2 = ax1.twinx()
ax3 = fig.add_subplot(2, 1, 2)

lns1 = ax1.plot(wave[mask], count[mask], label=r"Observed e$^-$ Count")
lns2 = ax1.plot(wave_literature,
                flux_literature * 1e15,
                label=r"Flux (Literature) $\times 10^{15}$")

lns3 = ax2.plot(wave[mask],
                sensitivity[mask],
                label='Sensitivity',
                color='black')
ax2.set_ylim(np.nanpercentile(sensitivity[mask], [0, 99.9]) * 0.9)
ax2.set_yscale('log')

lns4 = ax3.plot(wave_literature,
                flux_literature,
                color='C1',
                label=r"Literature Flux")
lns5 = ax3.plot(wave_standard,
                flux_standard,
                color='C2',
                label='Calibrated Flux')

ax1.set_xlim(3900, 8100)
ax2.set_xlim(3900, 8100)
ax3.set_xlim(3900, 8100)

ax3.set_ylim(5e-14, 3.1e-13)

lns = lns1 + lns2 + lns3
labs = [ln.get_label() for ln in lns]
ax2.legend(lns, labs)

lns_b = lns4 + lns5
labs_b = [ln.get_label() for ln in lns_b]
ax3.legend(lns_b, labs_b)

ax1.set_xticks([])
ax2.set_xticks([])

ax3.set_xlabel(r'Wavelength (A)')

ax1.set_ylabel(r'Electron Count (e$^-$)')
ax2.set_ylabel(r'Flux ( erg / s / cm$^2$ / A) / e$^-$ Count')
ax3.set_ylabel(r'Flux ( erg / s / cm$^2$ / A)')

fig.tight_layout()
fig.subplots_adjust(hspace=0)
plt.savefig('fig_06_flux_calibration_diagnostics.jpg')
plt.savefig('fig_06_flux_calibration_diagnostics.pdf')
