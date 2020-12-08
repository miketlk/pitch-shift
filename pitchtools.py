import math
import numpy as np
from scipy import signal


def pitch_shift_linear(sig, ratio):
    """Shifts pitch using linear interpolator"""
    n_res = round(len(sig) / ratio)
    res = np.zeros(n_res, np.float32)
    for ires in range(len(res)):
        isig = int(ires * ratio)
        x = ires * ratio - isig
        if isig + 1 < len(sig):
            # Linear interpolation
            res[ires] = sig[isig] + x * (sig[isig + 1] - sig[isig])
        elif isig < len(sig):
            # Fallback to nearest value if we are at the boundary
            res[ires] = sig[isig]
        else:
            res[ires] = 0.0
    return res.astype(np.float32)


def pitch_shift_ovs2_linear(sig, ratio, upsample=True):
    """Shifts pitch using linear interpolator and 2x oversampling."""
    if upsample:
        sig = signal.resample_poly(sig, 2, 1)
    n_res = round(len(sig) / ratio)
    res = np.zeros(n_res, np.float32)
    for ires in range(len(res)):
        isig = int(ires * ratio)
        x = ires * ratio - isig
        if isig + 1 < len(sig):
            res[ires] = sig[isig] + x * (sig[isig + 1] - sig[isig])
        elif isig < len(sig):
            # Fallback to nearest value if we are at the boundary
            res[ires] = sig[isig]
        else:
            res[ires] = 0.0
    return signal.resample_poly(res, 1, 2).astype(np.float32)


def pitch_shift_ovs2_poly_6p5o(sig, ratio, upsample=True):
    """Shifts pitch using 6-point 5-order polynomial interpolator and 2x
    oversampling.
    """
    # The polynomial is taken from the publication "Polynomial Interpolators
    # for High-Quality Resampling of Oversampled Audio" by Olli Niemitalo, 2001

    if upsample:
        sig = signal.resample_poly(sig, 2, 1)
    n_res = round(len(sig) / ratio)
    res = np.zeros(n_res, np.float32)
    for ires in range(len(res)):
        isig = int(ires * ratio)
        x = ires * ratio - isig
        if 2 <= isig < len(sig) - 3:
            # Polynomial interpolation
            z = x - 1/2.0
            even1 = sig[isig + 1] + sig[isig]
            odd1 = sig[isig + 1] - sig[isig]
            even2 = sig[isig + 2] + sig[isig - 1]
            odd2 = sig[isig + 2] - sig[isig - 1]
            even3 = sig[isig + 3] + sig[isig - 2]
            odd3 = sig[isig + 3] - sig[isig - 2]
            c0 = even1*0.40513396007145713 + even2 * \
                0.09251794438424393 + even3*0.00234806603570670
            c1 = odd1*0.28342806338906690 + odd2 * \
                0.21703277024054901 + odd3*0.01309294748731515
            c2 = even1*-0.191337682540351941 + even2 * \
                0.16187844487943592 + even3*0.02946017143111912
            c3 = odd1*-0.16471626190554542 + odd2*- \
                0.00154547203542499 + odd3*0.03399271444851909
            c4 = even1*0.03845798729588149 + even2*- \
                0.05712936104242644 + even3*0.01866750929921070
            c5 = odd1*0.04317950185225609 + odd2*- \
                0.01802814255926417 + odd3*0.00152170021558204
            res[ires] = ((((c5*z+c4)*z+c3)*z+c2)*z+c1)*z+c0
        if isig + 1 < len(sig):
            # Fallback to linear if we are too close to boundary
            res[ires] = sig[isig] + x * (sig[isig + 1] - sig[isig])
        elif isig < len(sig):
            # Fallback to nearest value if we are at the boundary
            res[ires] = sig[isig]
        else:
            res[ires] = 0.0
    return signal.resample_poly(res, 1, 2).astype(np.float32)

