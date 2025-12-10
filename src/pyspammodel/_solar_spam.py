import numpy as np
import xarray as xr
import pyspammodel._misc as _m


class SolarSpam:
    '''
    Solar-SPAM model class.
    '''
    def __init__(self):
        self._dataset = _m.get_solar_spam_coeffs()

    def _get_f107(self, f107):
        try:
            if isinstance(f107, float) or isinstance(f107, int):
                return np.array([f107 ** 2, f107, 1], dtype=np.float64).reshape(1, 3)
            return np.vstack([np.array([x ** 2, x, 1]) for x in f107], dtype=np.float64)
        except TypeError:
            raise TypeError('Only int, float or array-like object types are allowed')

    def _check_types(self, f107):
        if isinstance(f107, (float, int, np.integer, list, np.ndarray)):
            if isinstance(f107, (list, np.ndarray)):
                if not all([isinstance(x, (float, int, np.integer,)) for x in f107]):
                    raise TypeError(
                        f'Only float and int types are allowed in array.')
        else:
            raise TypeError(f'Only float, int, list and np.ndarray types are allowed. f107 was {type(f107)}')
        return True

    def _predict(self, matrix_a, vector_x):
        h = 6.62607015e-34
        c = 299792458
        l = np.array(np.arange(0.5, 190.5, 1) * 1e-9).reshape((190,1))

        res = np.dot(matrix_a, vector_x) / (h*c / l)
        return res

    def get_spectral_bands(self, f107):
        if self._check_types(f107):
            F107 = self._get_f107(f107)

        coeffs = np.vstack((np.array(self._dataset['P1'], dtype=np.float64),
                            np.array(self._dataset['P2'], dtype=np.float64),
                            np.array(self._dataset['P3'], dtype=np.float64))).T

        res = self._predict(coeffs, F107.T)
        return xr.Dataset(data_vars={'euv_flux_spectra': (('band_center', 'F107'), res),
                                     'lband': ('band_number', self._dataset['lband'].data),
                                     'uband': ('band_number', self._dataset['uband'].data)},
                          coords={'F107': F107[:, 1],
                                  'band_center': self._dataset['center'].data,
                                  'band_number': np.arange(190)},
                          attrs={'model name': 'Solar-SPAM',
                                 'F10.7 units': '10^-22 · W · m^-2 · Hz^-1',
                                 'spectra units': 'm^-2 · s^-1 · nm^-1',
                                 'wavelength units': 'nm',
                                 'euv_flux_spectra': 'modeled EUV solar photon flux spectra',
                                 'lband': 'lower boundary of wavelength interval',
                                 'uband': 'upper boundary of wavelength interval'})

    def get_spectra(self, f107):
        return self.get_spectral_bands(f107)

    def predict(self, f107):
        return self.get_spectral_bands(f107)
