import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits


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

        return np.array(spec.spectral_axis), np.array(normSpec)

    def plot(self, ax=None, **kwargs):

        n = self.flux.shape[0]

        if not ax:
            fig, ax = plt.subplots(figsize=(16,6))

        for ii in range(n):

            ax.plot(self.wave[ii], self.flux[ii], **kwargs)
            ax.set_xlabel('Wavelength [$\AA$]', fontsize=16)
            ax.set_ylabel('Flux', fontsize=16)
            ax.grid()
