# samplerate2

A thin wrapper around the [libsamplerate](https://github.com/libsndfile/libsamplerate) library.
This is a drop-in replacement for the [samplerate](https://github.com/tuxu/python-samplerate/) package.

The library is statically linked which I hope should resolve some portability issues of [samplerate](https://github.com/tuxu/python-samplerate/).
Compared to [resampy](https://github.com/bmcfee/resampy), samplerate2 only depends on numpy.
