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

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <samplerate.h>

#include <cmath>
#include <iostream>
#include <string>
#include <typeinfo>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;

using callback_t =
    std::function<py::array_t<float, py::array::c_style | py::array::forcecast>(
        void)>;
using np_array_f32 =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

enum ConverterType {
  sinc_best = 0,
  sinc_medium,
  sinc_fastest,
  zero_order_hold,
  linear
};

long the_callback_func(void *cb_data, float **data);

int get_converter_type(const py::object &obj) {
  if (py::isinstance<py::str>(obj)) {
    py::str py_s = obj;
    std::string s = static_cast<std::string>(py_s);
    if (s.compare("sinc_best") == 0) {
      return 0;
    } else if (s.compare("sinc_medium") == 0) {
      return 1;
    } else if (s.compare("sinc_fastest") == 0) {
      return 2;
    } else if (s.compare("zero_order_hold") == 0) {
      return 3;
    } else if (s.compare("linear") == 0) {
      return 4;
    }
  } else if (py::isinstance<py::int_>(obj)) {
    py::int_ val = obj;
    return static_cast<int>(val);
  } else if (py::isinstance<ConverterType>(obj)) {
    py::int_ c = obj.attr("value");
    return static_cast<int>(c);
  }

  std::runtime_error("Unsupported converter type");
  return -1;
}

class Resampler {
 private:
  SRC_STATE *_state = nullptr;
  int _err_num = 0;

 public:
  int _converter_type = 0;
  int _channels = 0;

 private:
  void _process_error() {
    if (_err_num != 0) std::runtime_error(src_strerror(_err_num));
  }

 public:
  Resampler(const py::object &converter_type, int channels)
      : _converter_type(get_converter_type(converter_type)),
        _channels(channels) {
    _state = src_new(_converter_type, _channels, &_err_num);
  }

  // copy constructor
  Resampler(const Resampler &r)
      : _converter_type(r._converter_type), _channels(r._channels) {
    _state = src_clone(r._state, &_err_num);
    _process_error();
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

    _err_num = src_process(_state, &src_data);
    _process_error();

    // create a shorter view of the array
    if ((size_t)src_data.output_frames_gen < new_size) {
      out_shape[0] = src_data.output_frames_gen;
      output = py::array_t<float, py::array::c_style>(
          out_shape, static_cast<float *>(outbuf.ptr));
    }

    return output;
  }

  void set_ratio(double new_ratio) {
    _err_num = src_set_ratio(_state, new_ratio);
    _process_error();
  }

  void reset() {
    _err_num = src_reset(_state);
    _process_error();
  }

  Resampler clone() const { return Resampler(*this); }
};

class CallbackResampler {
 private:
  SRC_STATE *_state = nullptr;
  int _err_num = 0;
  callback_t _callback = nullptr;
  np_array_f32 _current_buffer;
  size_t _buffer_ndim = 0;

 public:
  double _ratio = 0.0;
  int _converter_type = 0;
  size_t _channels = 0;

 private:
  void _process_error() {
    if (_err_num != 0) std::runtime_error(src_strerror(_err_num));
  }

 public:
  CallbackResampler(const callback_t &callback_func, double ratio,
                    const py::object &converter_type, size_t channels)
      : _callback(callback_func),
        _ratio(ratio),
        _converter_type(get_converter_type(converter_type)),
        _channels(channels) {
    _state = src_callback_new(the_callback_func, _converter_type, (int)_channels,
                              &_err_num, static_cast<void *>(this));
  }

  // copy constructor
  CallbackResampler(const CallbackResampler &r)
      : _callback(r._callback),
        _ratio(r._ratio),
        _converter_type(r._converter_type),
        _channels(r._channels) {
    _state = src_clone(r._state, &_err_num);
    _process_error();
  }

  ~CallbackResampler() { src_delete(_state); }

  void set_buffer(const np_array_f32 &new_buf) { _current_buffer = new_buf; }
  size_t get_channels() { return _channels; }

  np_array_f32 callback(void) {
    auto input = _callback();

    auto inbuf = input.request();
    if (_buffer_ndim == 0) _buffer_ndim = inbuf.ndim;

    _current_buffer = input;
    return input;
  }

  py::array_t<float, py::array::c_style> read(size_t frames) {
    // allocate output array
    std::vector<size_t> out_shape{frames, _channels};
    auto output = py::array_t<float, py::array::c_style>(out_shape);
    py::buffer_info outbuf = output.request();

    // read from the callback
    size_t output_frames_gen = src_callback_read(
        _state, _ratio, (long)frames, static_cast<float *>(outbuf.ptr));

    // check error status
    if (output_frames_gen == 0) {
      _err_num = src_error(_state);
      _process_error();
    }

    // if there is only one channel and the input array had only on dimension
    // we also output a 1D array
    if (_channels == 1 and _buffer_ndim == 1) {
      out_shape.pop_back();
      output = py::array_t<float, py::array::c_style>(
          out_shape, static_cast<float *>(outbuf.ptr));
    }

    // create a shorter view of the array
    if (output_frames_gen < frames) {
      out_shape[0] = output_frames_gen;
      output = py::array_t<float, py::array::c_style>(
          out_shape, static_cast<float *>(outbuf.ptr));
    }

    return output;
  }

  void set_starting_ratio(double new_ratio) {
    _err_num = src_set_ratio(_state, new_ratio);
    _process_error();
    _ratio = new_ratio;
  }

  void reset() {
    _err_num = src_reset(_state);
    _process_error();
  }

  CallbackResampler clone() const { return CallbackResampler(*this); }
};

long the_callback_func(void *cb_data, float **data) {
  CallbackResampler *cb = static_cast<CallbackResampler *>(cb_data);
  int cb_channels = cb->get_channels();

  // get the data as a numpy array
  auto input = cb->callback();

  // accessors for the arrays
  py::buffer_info inbuf = input.request();

  // end of stream is signaled by a None, which is cast to a ndarray with ndim == 0
  if (inbuf.ndim == 0)
    return 0;

  // set the number of channels
  int channels = 1;
  if (inbuf.ndim == 2)
    channels = inbuf.shape[1];
  else if (inbuf.ndim > 2)
    throw std::runtime_error("Input array should have at most 2 dimensions");

  if (channels != cb_channels)
    throw std::runtime_error("Invalid number of channels in input data.");

  *data = static_cast<float *>(inbuf.ptr);

  return (long)inbuf.shape[0];
}

py::array_t<float, py::array::c_style> resample_impl(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &input,
    double sr_ratio, const py::object &converter_type) {
  // input array has shape (n_samples, n_channels)
  int converter_type_int = get_converter_type(converter_type);

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

  int ret_code = src_simple(&src_data, converter_type_int, channels);

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

  // give access to this function for testing
  m.def("_get_converter_type", &get_converter_type,
        "Convert python object to integer of converter tpe or raise an error "
        "if illegal");

  m.def("resample", &resample_impl, "Resample function", "input"_a, "ratio"_a,
        "converter_type"_a = int(SRC_SINC_BEST_QUALITY));

  py::class_<Resampler>(m, "Resampler")
      .def(py::init<const py::object &, int>(), "converter_type"_a = 0,
           "channels"_a = 1)
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

  py::class_<CallbackResampler>(m, "CallbackResampler")
      .def(py::init<const callback_t &, double, const py::object &, int>(),
           "callback"_a, "ratio"_a, "converter_type"_a = 0, "channels"_a = 1)
      .def(py::init<CallbackResampler>())
      .def("read", &CallbackResampler::read,
           "Read a block of data from the callback", "num_frames"_a)
      .def("reset", &CallbackResampler::reset, "Reset the resampling process")
      .def("set_starting_ratio", &CallbackResampler::set_starting_ratio,
           "Change the sampling ratio")
      .def("clone", &CallbackResampler::clone,
           "Create a copy of the resampler object")
      .def_readonly("converter_type", &CallbackResampler::_converter_type,
                    "The converter type")
      .def_readonly("channels", &CallbackResampler::_channels,
                    "The number of channels");

  py::enum_<ConverterType>(m, "ConverterType")
      .value("sinc_best", sinc_best)
      .value("sinc_medium", sinc_medium)
      .value("sinc_fastest", sinc_fastest)
      .value("zero_order_hold", zero_order_hold)
      .value("linear", linear)
      .export_values();
}
