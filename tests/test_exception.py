import numpy as np
import pytest

import samplerate2


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


def test_resample_zero_channel_input():
    data = np.zeros((16000, 0), dtype=np.float32)
    # does not produce an error, the output shape is (0, 0)
    # because the number of samples converted is zero
    samplerate2.resample(data, 0.5, "sinc_fastest")


def test_resample_zero_len_input():
    data = np.zeros((0, 1), dtype=np.float32)
    # does not produce an error, the output shape is (0, 1)
    # same as the input
    samplerate2.resample(data, 0.5, "sinc_fastest")


def test_resample_ndim_too_big():
    data = np.zeros((16000, 1, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        # fails because the input has 3 dimensions
        samplerate2.resample(data, 0.5, "sinc_fastest")


def test_resampler_ndim_too_big():
    data = np.zeros((16000, 1, 1), dtype=np.float32)
    resampler = samplerate2.Resampler("sinc_fastest", 1)
    with pytest.raises(ValueError):
        # fails because the input has 3 dimensions
        resampler.process(data, 0.5)


def test_resampler_incorrect_channel_number():
    data = np.zeros((16000, 2), dtype=np.float32)
    resampler = samplerate2.Resampler("sinc_fastest", 1)
    with pytest.raises(ValueError):
        # fails because we defined the converter for 1 channel
        resampler.process(data, 0.5)


def test_callback_resampler_ndim_too_big():
    data = np.zeros((16000, 1, 1), dtype=np.float32)

    def producer():
        yield data
        while True:
            yield None

    callback = lambda p=producer(): next(p)

    cb_resampler = samplerate2.CallbackResampler(callback, 0.5, "sinc_fastest", 1)
    with pytest.raises(ValueError):
        # fails because the input has 3 dimensions
        cb_resampler.read(len(data))


def test_callback_resampler_incorrect_channel_number():
    data = np.zeros((16000, 2), dtype=np.float32)

    def producer():
        yield data
        while True:
            yield None

    callback = lambda p=producer(): next(p)

    cb_resampler = samplerate2.CallbackResampler(callback, 0.5, "sinc_fastest", 1)
    with pytest.raises(ValueError):
        # fails because we defined the converter for 1 channel
        cb_resampler.read(len(data))
