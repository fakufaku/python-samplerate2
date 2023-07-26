import samplerate2
import pytest


@pytest.mark.parametrize("errnum", list(range(1, 25)))
def test_resampling_error(errnum):
    if 0 < errnum < 24:
        with pytest.raises(samplerate2.ResamplingError):
            samplerate2._error_handler(errnum)
    elif errnum != 0:
        with pytest.raises(RuntimeError):
            samplerate2._error_handler(errnum)
