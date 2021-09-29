from aspired import spectral_reduction
from astropy.io import fits

fits_file = fits.open('v_s_20180810_27_1_0_2.fits.gz')[0]
data = fits_file.data
arc = fits.open('v_a_20180810_28_1_0_1.fits.gz')[0]

temperature = fits_file.header['REFTEMP']
pressure = fits_file.header['REFPRES'] * 100.
relative_humidity = fits_file.header['REFHUMID']

atlas = [
    4193.5, 4385.77, 4500.98, 4524.68, 4582.75, 4624.28, 4671.23, 4697.02,
    4734.15, 4807.02, 4921.48, 5028.28, 5618.88, 5823.89, 5893.29, 5934.17,
    6182.42, 6318.06, 6472.841, 6595.56, 6668.92, 6728.01, 6827.32, 6976.18,
    7119.60, 7257.9, 7393.8, 7584.68, 7642.02, 7740.31, 7802.65, 7887.40,
    7967.34, 8057.258
]
element = ['Xe'] * len(atlas)

twodspec = spectral_reduction.TwoDSpec(data, fits_file.header, cosmicray=True)

twodspec.ap_trace(nspec=1, nwindow=10, display=False)

twodspec.add_arc(arc)
twodspec.extract_arc_spec()

onedspec = spectral_reduction.OneDSpec()
onedspec.from_twodspec(twodspec, stype='science')

# Find the peaks of the arc
onedspec.find_arc_lines(prominence=2,
                        distance=5,
                        refine_window_width=3,
                        display=True,
                        stype='science')

onedspec.initialise_calibrator(stype='science')
onedspec.set_hough_properties(range_tolerance=500.,
                              xbins=100,
                              ybins=100,
                              min_wavelength=3800.,
                              max_wavelength=8200.,
                              stype='science')
onedspec.set_ransac_properties(sample_size=10,
                               top_n_candidate=10,
                               filter_close=True,
                               ransac_tolerance=10.,
                               stype='science')
onedspec.add_user_atlas(elements=element,
                        wavelengths=atlas,
                        pressure=pressure,
                        temperature=temperature,
                        relative_humidity=relative_humidity,
                        constrain_poly=True,
                        stype='science')
onedspec.do_hough_transform(stype='science')

# Solve for the pixel-to-wavelength solution
onedspec.fit(max_tries=2000, stype='science', display=True)

# Note, saved a screenshot instead of saving natively.
