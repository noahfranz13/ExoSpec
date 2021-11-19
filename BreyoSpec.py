import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits


class BreyoSpec():

    def __init__(self, filepaths):

        '''
        filepaths [list] : list of paths to the fits files with flux data
        '''

        normFluxes = []
        normWaves = []

        if type(filepaths) == str:
            filepaths = [filepaths]

        for path in filepaths:

            hdu = fits.open(path)[0]
            flux = hdu.data
            hdr = hdu.header

            normWave, normFlux = self.norm(flux, hdr)

            whereVisible = np.where((normWave > 4000) * (normWave < 7000))[0]

            normWaves.append(normWave[whereVisible])
            normFluxes.append(normFlux[whereVisible])

        self.flux = np.array(normFluxes)
        self.wave = np.array(normWaves)

    def norm(self, inFlux, header):
        '''
        Function to normalize the demetra output data. Uses specutils to fit the
        spectrum and then flatten it

        returns : wavelength array [Angstroms], Flux array [units of input]
        '''

        from specutils import Spectrum1D, SpectralRegion
        from astropy.wcs import WCS
        from astropy.modeling import models, fitting
        from specutils.fitting import fit_generic_continuum
        from astropy import units as u

        # create specutils Spectrum1D Object
        wcsData = WCS(header)

        spec = Spectrum1D(flux=inFlux * u.Jy, wcs=wcsData)

        # get y-continuum fit for spectra
        g1Fit = fit_generic_continuum(spec, exclude_regions=[SpectralRegion(3700 * u.AA, 4000 * u.AA), SpectralRegion(4825 * u.AA, 4885 * u.AA), SpectralRegion(6520 * u.AA, 6540 * u.AA)])
        yCont = g1Fit(spec.spectral_axis)

        normSpec = spec.flux / yCont

        print(len(normSpec), len(spec.spectral_axis))

        return np.array(spec.spectral_axis), np.array(normSpec)

    def plot(self, ax=None, **kwargs):

        if not ax:
            fig, ax = plt.subplots(figsize=(16,6))

        ax.plot(self.wave[0], self.flux[0], **kwargs)
        ax.set_xlabel('Wavelength [$\AA$]', fontsize=16)
        ax.set_ylabel('Flux', fontsize=16)
        ax.grid()
