import numpy as np
import matplotlib.pyplot as plt

class BreyoSpec():

    def __init__(self, flux):

        '''
        flux [array]       : array of flux, can be n-dimensional
        wavelength [array] : 1 dimensional array of wavelengths
        '''

        if len(flux.shape) == 1:
            flux = [flux]

        normFluxes = []
        for f in flux:
            normWave, normFlux = self.norm()
            normFluxes.append(normFlux)

        self.flux = np.array(normFluxes)
        self.wave = np.array(normWave)

    def norm(inFlux):
        '''
        Function to normalize the demetra output data. Uses specutils to fit the
        spectrum and then flatten it

        returns : wavelength array [Angstroms], Flux array [units of input]
        '''

        from specutils import Spectrum1D, SpectralRegion
        import astropy.wcs as fitswcs
        from astropy.modeling import models, fitting
        from specutils.fitting import fit_generic_continuum
        from astropy import units as u

        # create specutils Spectrum1D Object
        wcsData = fitswcs.WCS(header={'CDELT1': header['CDELT1'], 'CRVAL1': header['CRVAL1'],
                               'CUNIT1': header['CUNIT1'], 'CTYPE1': header['CTYPE1'],
                               'CRPIX1': header['CRPIX1']})

        spec = Spectrum1D(flux=inFlux * u.Jy, wcs=wcsData)

        # get y-continuum fit for spectra
        g1Fit = fit_generic_continuum(spec, exclude_regions=[SpectralRegion(3700 * u.AA, 4000 * u.AA), SpectralRegion(4825 * u.AA, 4885 * u.AA), SpectralRegion(6520 * u.AA, 6540 * u.AA)])
        yCont = g1Fit(spec.spectral_axis)

        normSpec = spec / yCont

        return normSpec.spectral_axis, normSpec.flux
