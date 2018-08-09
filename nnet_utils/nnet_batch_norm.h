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

#ifndef NNET_BATCH_NORM_LAYER_H_
#define NNET_BATCH_NORM_LAYER_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

struct bn_layer_config
{
    // Internal data type definitions
	typedef float mean_t;
	typedef float inv_sigma_t;
	typedef float gamma_t;
	typedef float beta_t;
    typedef float bn_t;

    // Layer Sizes
    static const unsigned n = 10;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
    // partitioning arrays cyclically to go with roll factors?
};

template<class data_T, class res_T, typename CONFIG_T>
void batch_norm_layer(
    data_T    data[CONFIG_T::n],
    res_T     res[CONFIG_T::n],
    typename CONFIG_T::mean_t mean[CONFIG_T::n],
	typename CONFIG_T::inv_sigma_t inv_sigma[CONFIG_T::n],
	typename CONFIG_T::gamma_t gamma[CONFIG_T::n],
	typename CONFIG_T::beta_t beta[CONFIG_T::n])
{
	typename CONFIG_T::bn_t mean_sub[CONFIG_T::n];
	typename CONFIG_T::bn_t mult_inv_sigma[CONFIG_T::n];
	typename CONFIG_T::bn_t mult_gamma[CONFIG_T::n];
	typename CONFIG_T::bn_t add_beta[CONFIG_T::n];
    //typename CONFIG_T::accum_t mult[CONFIG_T::n][CONFIG_T::n];
    //typename CONFIG_T::accum_t acc[CONFIG_T::n];

    // Use a function_instantiate in case it helps to explicitly optimize unchanging weights/biases
    #pragma HLS function_instantiate variable=mean,inv_sigma,gamma,beta

    if (CONFIG_T::io_type == io_parallel){
        // For parallel inputs:
        //   - completely partition arrays -- target fabric
        //   - if we have an unroll factor, limit number of multipliers
        #pragma HLS PIPELINE II=CONFIG_T::reuse_factor

        // #pragma HLS ARRAY_PARTITION variable=weights complete // remove this line for now, it breaks compression sometimes
        #pragma HLS ARRAY_PARTITION variable=mean complete
        #pragma HLS ARRAY_PARTITION variable=inv_sigma complete
        #pragma HLS ARRAY_PARTITION variable=gamma complete
		#pragma HLS ARRAY_PARTITION variable=beta complete


    } else if (CONFIG_T::io_type == io_serial){
        #pragma HLS DATAFLOW
    }

    // Subtract mean
    SubMean: for(int ii = 0; ii < CONFIG_T::n; ii++) {
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        mean_sub[ii] = data[ii] - mean[ii];

    }

    // Multiply by 1/sigma
	MultInvSigma: for(int ii = 0; ii < CONFIG_T::n; ii++) {
		if (CONFIG_T::io_type == io_serial){
			#pragma HLS UNROLL
		}
		mult_inv_sigma[ii] = mean_sub[ii] * inv_sigma[ii];

	}

	// Multiply by gamma
	MultGamma: for(int ii = 0; ii < CONFIG_T::n; ii++) {
		if (CONFIG_T::io_type == io_serial){
			#pragma HLS UNROLL
		}
		mult_gamma[ii] = mult_inv_sigma[ii] * gamma[ii];

	}

	// Add beta
	AddBeta: for(int ii = 0; ii < CONFIG_T::n; ii++) {
		if (CONFIG_T::io_type == io_serial){
			#pragma HLS UNROLL
		}
		add_beta[ii] = mult_gamma[ii] * beta[ii];

	}

    // Cast to "res_t" type
    Result: for(int ires = 0; ires < CONFIG_T::n; ires++){
        if (CONFIG_T::io_type == io_serial){
            #pragma HLS UNROLL
        }
        res[ires] = (res_T) (add_beta[ires]);
    }    
}

}

#endif
