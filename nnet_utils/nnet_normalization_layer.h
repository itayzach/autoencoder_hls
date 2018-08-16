//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_NORMALIZATION_LAYER_H_
#define NNET_NORMALIZATION_LAYER_H_

#include "nnet_common.h"
#include "hls_stream.h"
//#include <math.h>
#include <hls_math.h>
namespace nnet {

struct norm_layer_config
{
	// Internal data type definitions
	typedef float norm_t;

    // Layer Sizes
    static const unsigned n = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    // partitioning arrays cyclically to go with roll factors?
};

template<class data_T, class res_T, typename CONFIG_T>
void normalization_layer(
    data_T    data[CONFIG_T::n],
    res_T     res[CONFIG_T::n])
{
	typename CONFIG_T::norm_t data_square[CONFIG_T::n];
	typename CONFIG_T::norm_t squares_sum;
	typename CONFIG_T::norm_t sqrt_res;
	const typename CONFIG_T::norm_t sqrt2 = 1.41421;
	typename CONFIG_T::norm_t div_res[CONFIG_T::n];

    if (CONFIG_T::io_type == io_parallel){
        // For parallel inputs:
        //   - completely partition arrays -- target fabric
        //   - if we have an unroll factor, limit number of multipliers
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
        #pragma HLS ARRAY_PARTITION variable=data_square complete
        #pragma HLS ARRAY_PARTITION variable=div_res complete


    } else if (CONFIG_T::io_type == io_serial){
        #pragma HLS DATAFLOW
    }

    // Square inputs
    Square: for(int ii = 0; ii < CONFIG_T::n; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        data_square[ii] = data[ii] * data[ii];
    }

    // Sum of squares
    squares_sum = 0;
	SquaresSum: for(int ii = 0; ii < CONFIG_T::n; ii++) {
		if (CONFIG_T::io_type == io_serial){
			#pragma HLS UNROLL
		}
		squares_sum += data_square[ii];
	}

	// Square root
	sqrt_res = hls::sqrt(squares_sum);

	// Divide by square root
	Divide: for(int ii = 0; ii < CONFIG_T::n; ii++) {
		if (CONFIG_T::io_type == io_serial){
			#pragma HLS UNROLL
		}
		div_res[ii] = sqrt2 * (data[ii] / sqrt_res);
	}

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n; ires++){
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        res[ires] = (res_T) (div_res[ires]);
    }    
}

}

#endif
