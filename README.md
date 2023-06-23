# resampack

This is a thin wrapper around the [libsamplerate](https://github.com/libsndfile/libsamplerate) library.

The library is statically linked which I hope should resolve some portability issues of [samplerate](https://github.com/tuxu/python-samplerate/).
Compared to [resampy](https://github.com/bmcfee/resampy), it only has minimal dependencies on numpy.
