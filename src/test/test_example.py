import numpy as np
import pandas as pd
from pyspammodel._solar_spam import SolarSpam
from pyspammodel._aero_spam import AeroSpam
from pyspammodel._misc import calc_diff_energy_flux


# Solar-SPAM model testing function
def test_solarspam():
    data = pd.read_csv('solarspam_ref.csv').to_numpy()
    solarspam = SolarSpam()
    f107 = np.arange(65,220,10)
    spectrum = calc_diff_energy_flux(solarspam.get_spectral_bands(f107)['euv_flux_spectra'].to_numpy())
    res = abs(data - spectrum) / data * 100
    assert (res < 1e-2).all() == True


# # Aero-SPAM model testing function
def test_aerospam():
    data = pd.read_csv('aerospam_ref.csv').to_numpy()
    aerospam = AeroSpam()
    f107 = np.arange(65,220,10)
    spectrum = aerospam.predict(f107)['euv_flux_spectra'].to_numpy()
    res = abs(data - spectrum) / data * 100
    assert (res < 1e-7).all() == True




