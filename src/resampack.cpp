/*
 * Python bindings for resampack
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

py::array_t<float, py::array::c_style> resample_impl(
    py::array_t<float, py::array::c_style | py::array::forcecast> input,
    int orig_freq, int new_freq, int converter_type = SRC_SINC_BEST_QUALITY) {
  // input array has shape (n_samples, n_channels)

  // accessors for the arrays
  py::buffer_info inbuf = input.request();

  // set the number of channels
  int channels = 1;
  if (inbuf.ndim == 2) channels = inbuf.shape[1];
  else if (inbuf.ndim > 2)
    throw std::runtime_error("Input array should have at most 2 dimensions");

  //
  double sr_ratio = double(new_freq) / double(orig_freq);
  const size_t new_size = size_t(std::ceil(inbuf.shape[0] * sr_ratio));

  // allocate output array
  std::vector<size_t> out_shape{new_size};
  if (inbuf.ndim == 2)
    out_shape.push_back(size_t(channels));
  auto output = py::array_t<float, py::array::c_style>(out_shape);
  py::buffer_info outbuf = output.request();

  // libsamplerate struct
  SRC_DATA src_data = {
      static_cast<float *>(inbuf.ptr),   // data_in
      static_cast<float *>(outbuf.ptr),  // data_out
      inbuf.shape[0],                    // input_frames
      long(new_size),                          // output_frames
      0,        // input_frames_used, filled by libsamplerate
      0,        // output_frames_gen, filled by libsamplerate
      0,        // end_of_input, not used by src_simple ?
      sr_ratio  // src_ratio, sampling rate conversion ratio
  };

  int ret_code = src_simple(&src_data, converter_type, channels);

  if (ret_code != 0) std::runtime_error(src_strerror(ret_code));

  return output;
}

PYBIND11_MODULE(resampack, m) {
  m.doc() =
      "A simple wrapper around libsamplerate";  // optional module docstring

  m.def("resample", &resample_impl, "Resample function");
}
