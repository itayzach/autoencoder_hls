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
#include <iostream>

#include "parameters.h"
#include "autoencoder.h"

#include "nnet_layer.h"
//#include "nnet_conv.h"
#include "nnet_activation.h"

//hls-fpga-machine-learning insert weights
#include "weights/w1.h"
#include "weights/b1.h"
#include "weights/w2.h"
#include "weights/b2.h"

void encoder(
		  input_t data[M],
		  result_t res[M],
		  unsigned short &const_size_in,
		  unsigned short &const_size_out)
{

	//hls-fpga-machine-learning insert IO
	#pragma HLS ARRAY_RESHAPE variable=data complete dim=0
	#pragma HLS ARRAY_RESHAPE variable=res complete dim=0
	#pragma HLS INTERFACE ap_vld port=data,res

	#pragma HLS PIPELINE

    const_size_in   = M;
    const_size_out  = M;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************
    //Dense
	result_t logits1[M];
	#pragma HLS ARRAY_PARTITION variable=logits1 complete dim=0
	nnet::compute_layer<input_t, result_t, config1>(data, res, w1, b1);

	//Softmax
//	nnet::softmax<result_t, result_t, softmax_config2>(logits2, res);



}
