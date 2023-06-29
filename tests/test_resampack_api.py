import numpy as np
import pytest

import resampack as samplerate


@pytest.fixture(scope="module", params=[1, 2])
def data(request):
    num_channels = request.param
    periods = np.linspace(0, 10, 1000)
    input_data = [
        np.sin(2 * np.pi * periods + i * np.pi / 2) for i in range(num_channels)
    ]
    return (
        (num_channels, input_data[0])
        if num_channels == 1
        else (num_channels, np.transpose(input_data))
    )


@pytest.fixture(params=[0, 1, 2, 3, 4])
def converter_type(request):
    return request.param


def test_simple(data, converter_type, ratio=2.0):
    _, input_data = data
    samplerate.resample(input_data, ratio, converter_type)


def test_process(data, converter_type, ratio=2.0):
    num_channels, input_data = data
    src = samplerate.Resampler(converter_type, num_channels)
    src.process(input_data, ratio)


def test_match(data, converter_type, ratio=2.0):
    num_channels, input_data = data
    output_simple = samplerate.resample(input_data, ratio, converter_type)
    resampler = samplerate.Resampler(converter_type, channels=num_channels)
    output_full = resampler.process(input_data, ratio, end_of_input=True)
    assert np.allclose(output_simple, output_full)


@pytest.mark.parametrize(
    "input_obj,expected_type",
    [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        ("sinc_best", 0),
        ("sinc_medium", 1),
        ("sinc_fastest", 2),
        ("zero_order_hold", 3),
        ("linear", 4),
        (samplerate.ConverterType.sinc_best, 0),
        (samplerate.ConverterType.sinc_medium, 1),
        (samplerate.ConverterType.sinc_fastest, 2),
        (samplerate.ConverterType.zero_order_hold, 3),
        (samplerate.ConverterType.linear, 4),
    ],
)
def test_converter_type(input_obj, expected_type):
    ret = samplerate._get_converter_type(input_obj)
    assert ret == expected_type
