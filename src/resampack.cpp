/*
 * Python bindings for libsamplerate
 * Copyright (C) 2023  Robin Scheibler
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * You should have received a copy of the MIT License along with this program.
 * If not, see <https://opensource.org/licenses/MIT>.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samplerate.h>

#include <cmath>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

class Resampler {
 private:
  SRC_STATE *_state = nullptr;
  int _err_num = 0;

 public:
  int _converter_type = 0;
  int _channels = 0;

 private:
  void _process_error(int) {}

 public:
  Resampler(int converter_type, int channels)
      : _converter_type(converter_type), _channels(channels) {
    _state = src_new(_converter_type, _channels, &_err_num);
    if (_err_num != 0) std::runtime_error(src_strerror(_err_num));
  }

  // copy constructor
  Resampler(const Resampler &r)
      : _converter_type(r._converter_type), _channels(r._channels) {
    _state = src_clone(r._state, &_err_num);
    if (_err_num != 0) std::runtime_error(src_strerror(_err_num));
  }

  ~Resampler() { src_delete(_state); }

  py::array_t<float, py::array::c_style> process(
      py::array_t<float, py::array::c_style | py::array::forcecast> input,
      double sr_ratio, bool end_of_input) {
    // accessors for the arrays
    py::buffer_info inbuf = input.request();

    // set the number of channels
    int channels = 1;
    if (inbuf.ndim == 2)
      channels = inbuf.shape[1];
    else if (inbuf.ndim > 2)
      throw std::runtime_error("Input array should have at most 2 dimensions");

    if (channels != _channels)
      throw std::runtime_error("Invalid number of channels in input data.");

    const size_t new_size = size_t(std::ceil(inbuf.shape[0] * sr_ratio));

    // allocate output array
    std::vector<size_t> out_shape{new_size};
    if (inbuf.ndim == 2) out_shape.push_back(size_t(channels));
    auto output = py::array_t<float, py::array::c_style>(out_shape);
    py::buffer_info outbuf = output.request();

    // libsamplerate struct
    SRC_DATA src_data = {
        static_cast<float *>(inbuf.ptr),   // data_in
        static_cast<float *>(outbuf.ptr),  // data_out
        inbuf.shape[0],                    // input_frames
        long(new_size),                    // output_frames
        0,             // input_frames_used, filled by libsamplerate
        0,             // output_frames_gen, filled by libsamplerate
        end_of_input,  // end_of_input, not used by src_simple ?
        sr_ratio       // src_ratio, sampling rate conversion ratio
    };

    int ret_code = src_process(_state, &src_data);

    if (ret_code != 0) std::runtime_error(src_strerror(ret_code));

    // create a shorter view of the array
    if ((size_t)src_data.output_frames_gen < new_size) {
      out_shape[0] = src_data.output_frames_gen;
      output = py::array_t<float, py::array::c_style>(
          out_shape, static_cast<float *>(outbuf.ptr));
    }

    return output;
  }

  void set_ratio(double new_ratio) {
    int ret_code = src_set_ratio(_state, new_ratio);
    if (ret_code != 0) std::runtime_error(src_strerror(ret_code));
  }

  void reset() {
    int ret_code = src_reset(_state);
    if (ret_code != 0) std::runtime_error(src_strerror(ret_code));
  }

  Resampler clone() const { return Resampler(*this); }
};

py::array_t<float, py::array::c_style> resample_impl(
    py::array_t<float, py::array::c_style | py::array::forcecast> input,
    double sr_ratio, int converter_type = SRC_SINC_BEST_QUALITY) {
  // input array has shape (n_samples, n_channels)

  // accessors for the arrays
  py::buffer_info inbuf = input.request();

  // set the number of channels
  int channels = 1;
  if (inbuf.ndim == 2)
    channels = inbuf.shape[1];
  else if (inbuf.ndim > 2)
    throw std::runtime_error("Input array should have at most 2 dimensions");

  const size_t new_size = size_t(std::ceil(inbuf.shape[0] * sr_ratio));

  // allocate output array
  std::vector<size_t> out_shape{new_size};
  if (inbuf.ndim == 2) out_shape.push_back(size_t(channels));
  auto output = py::array_t<float, py::array::c_style>(out_shape);
  py::buffer_info outbuf = output.request();

  // libsamplerate struct
  SRC_DATA src_data = {
      static_cast<float *>(inbuf.ptr),   // data_in
      static_cast<float *>(outbuf.ptr),  // data_out
      inbuf.shape[0],                    // input_frames
      long(new_size),                    // output_frames
      0,        // input_frames_used, filled by libsamplerate
      0,        // output_frames_gen, filled by libsamplerate
      0,        // end_of_input, not used by src_simple ?
      sr_ratio  // src_ratio, sampling rate conversion ratio
  };

  int ret_code = src_simple(&src_data, converter_type, channels);

  if (ret_code != 0) std::runtime_error(src_strerror(ret_code));

  // create a shorter view of the array
  if ((size_t)src_data.output_frames_gen < new_size) {
    out_shape[0] = src_data.output_frames_gen;
    output = py::array_t<float, py::array::c_style>(
        out_shape, static_cast<float *>(outbuf.ptr));
  }

  return output;
}

PYBIND11_MODULE(resampack, m) {
  m.doc() =
      "A simple python wrapper library around libsamplerate";  // optional
                                                               // module
                                                               // docstring

  m.def("resample", &resample_impl, "Resample function", "input"_a, "ratio"_a,
        "converter_type"_a = int(SRC_SINC_BEST_QUALITY));

  py::class_<Resampler>(m, "Resampler")
      .def(py::init<int, int>(), "converter_type"_a = 0, "channels"_a = 1)
      .def(py::init<Resampler>())
      .def("process", &Resampler::process, "Process a block of data", "input"_a,
           "ratio"_a, "end_of_input"_a = false)
      .def("reset", &Resampler::reset, "Reset the resampling process")
      .def("set_ratio", &Resampler::set_ratio, "Change the sampling ratio")
      .def("clone", &Resampler::clone, "Create a copy of the resampler object")
      .def_readonly("converter_type", &Resampler::_converter_type,
                    "The converter type")
      .def_readonly("channels", &Resampler::_channels,
                    "The number of channels");

  m.attr("SINC_BEST") = int(SRC_SINC_BEST_QUALITY);
  m.attr("SINC_MEDIUM") = int(SRC_SINC_MEDIUM_QUALITY);
  m.attr("SINC_FASTEST") = int(SRC_SINC_FASTEST);
  m.attr("ZERO_ORDER_HOLD") = int(SRC_ZERO_ORDER_HOLD);
  m.attr("LINEAR") = int(SRC_LINEAR);
}
