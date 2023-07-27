import samplerate2
import pytest
import numpy as np


@pytest.mark.parametrize("errnum", list(range(1, 25)))
def test_resampling_error(errnum):
    if 0 < errnum < 24:
        with pytest.raises(samplerate2.ResamplingError):
            samplerate2._error_handler(errnum)
    elif errnum != 0:
        with pytest.raises(RuntimeError):
            samplerate2._error_handler(errnum)


def test_unknown_converter_type():
    with pytest.raises(ValueError):
        samplerate2._get_converter_type("super-downsampling")


def test_ndim_too_big():
    resampler = samplerate2.Resampler("sinc_fastest", 1)
    with pytest.raises(ValueError):
        # fails because the input has 3 dimensions
        resampler.process(np.zeros((16000, 2, 2), dtype=np.float32), 0.5)


def test_incorrect_channel_number():
    resampler = samplerate2.Resampler("sinc_fastest", 1)
    with pytest.raises(ValueError):
        # fails because we defined the converter for 1 channel
        resampler.process(np.zeros((16000, 2), dtype=np.float32), 0.5)
