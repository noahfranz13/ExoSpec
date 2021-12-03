import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from speclite import resample


class BreyoSpec():

    def __init__(self, filepaths=None, inFlux=None, inHdr=None):

        '''
        filepaths [list] : list of paths to the fits files with flux data
        inFlux [list] :  list of flux arrays, if filepaths isn't provided, must
                         provide this and inHdr
        inHdr [list] : headers corresponding to inFlux
        '''

        # make sure each is a list
        if filepaths and type(filepaths) == str:
            filepaths = [filepaths]

        if inFlux and np.array(inFlux).ndim == 1:
            inFlux = [inFlux]

        if inHdr and np.array(inHdr).ndim == 1:
            inHdr = [inHdr]

        # extract flux info from inputs
        if filepaths:

            fluxes = []
            hdrs = []

            for path in filepaths:

                hdu = fits.open(path)[0]
                fluxes.append(hdu.data)
                hdrs.append(hdu.header)

        elif influx and inHdr:

            fluxes = inFlux
            hdrs = inHdr

        else:
            raise Exception("Please provide either filepaths or inFlux AND inHdr")

        # normalize the flux array using specutils
        normFluxes = []
        normWaves = []

        for flux, hdr in zip(fluxes, hdrs):

            normWave, normFlux = self.norm(flux, hdr)

            whereVisible = np.where((normWave > 4000) * (normWave < 7000))[0]

            normWaves.append(normWave[whereVisible])
            normFluxes.append(normFlux[whereVisible])

        # assign normalized wave and fluxes to instance variables
        if len(normFluxes) == 1:
            self.flux = np.array(normFluxes, dtype=float)
            self.wave = np.array(normWaves, dtype=float)
        else:
            self.flux = np.array(normFluxes, dtype=object)
            self.wave = np.array(normWaves, dtype=object)

        self.resample()

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

        # convert from ADU to e- for ATIK 460EX CCD
        normSpecElec = np.array(normSpec) * 0.27 # 0.27 e-/ADU

        return np.array(spec.spectral_axis), normSpecElec

    def resample(self, dw=1):

        resampledWave = np.arange(4000+dw,6999,dw)
        resampledFlux = []

        for wave, flux in zip(self.wave, self.flux):

            data = np.ones(len(wave), [('wlen', float), ('flux', float)])
            data['wlen'] = wave
            data['flux'] = flux

            outFlux = resample(data, wave, resampledWave, 'flux')

            resampledFlux.append(outFlux)

        avgFlux = np.mean(np.array(resampledFlux)['flux'], axis=0)

        self.wave = resampledWave
        self.flux = avgFlux

    def plot(self, ax=None, **kwargs):

        if not ax:
            fig, ax = plt.subplots(figsize=(16,6))

        ax.plot(self.wave, self.flux, **kwargs)
        ax.set_xlabel('Wavelength [$\AA$]', fontsize=16)
        ax.set_ylabel('Flux', fontsize=16)
        ax.grid()
