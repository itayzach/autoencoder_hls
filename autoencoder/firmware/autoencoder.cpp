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
#include "weights/enc_w1.h"
#include "weights/enc_b1.h"
#include "weights/enc_w2.h"
#include "weights/enc_b2.h"
#include "weights/enc_bn3.h"

#include "weights/dec_w1.h"
#include "weights/dec_b1.h"
#include "weights/dec_w2.h"
#include "weights/dec_b2.h"
// ========================================================================
// encoder
// ========================================================================
void encoder(
  input_t data[M],
  result_t res[n_channel],
  unsigned short &const_size_in,
  unsigned short &const_size_out)
{

	//hls-fpga-machine-learning insert IO
	#pragma HLS ARRAY_RESHAPE variable=data complete dim=0
	#pragma HLS ARRAY_RESHAPE variable=res complete dim=0
	#pragma HLS INTERFACE ap_vld port=data,res

	#pragma HLS PIPELINE

    const_size_in   = M;
    const_size_out  = n_channel;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************
    // Dense
	result_t logits1[M];
	#pragma HLS ARRAY_PARTITION variable=logits1 complete dim=0
	nnet::compute_layer<input_t, result_t, enc_config1>(data, logits1, enc_w1, enc_b1);

	// ReLU
	input_t layer1_relu_out[M];
	#pragma HLS ARRAY_PARTITION variable=layer1_relu_out complete dim=0
	nnet::relu<input_t, input_t, enc_relu_config1>(logits1, layer1_relu_out);

	// Dense
	result_t logits2[n_channel];
	#pragma HLS ARRAY_PARTITION variable=logits2 complete dim=0
	nnet::compute_layer<input_t, result_t, enc_config2>(layer1_relu_out, logits2, enc_w2, enc_b2);

	// BatchNorm
	nnet::batch_norm_layer<input_t, result_t, enc_bn_config3>(logits2, res, enc_mean, enc_inv_sigma, enc_gamma, enc_beta);


}

// ========================================================================
// decoder
// ========================================================================
void decoder(
  input_t data[n_channel],
  result_t res[M],
  unsigned short &const_size_in,
  unsigned short &const_size_out)
{

	//hls-fpga-machine-learning insert IO
	#pragma HLS ARRAY_RESHAPE variable=data complete dim=0
	#pragma HLS ARRAY_RESHAPE variable=res complete dim=0
	#pragma HLS INTERFACE ap_vld port=data,res

	#pragma HLS PIPELINE

    const_size_in   = n_channel;
    const_size_out  = M;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************
    // Dense
	result_t logits1[n_channel];
	#pragma HLS ARRAY_PARTITION variable=logits1 complete dim=0
	nnet::compute_layer<input_t, result_t, dec_config1>(data, logits1, dec_w1, dec_b1);

	// ReLU
	input_t layer1_relu_out[n_channel];
	#pragma HLS ARRAY_PARTITION variable=layer1_relu_out complete dim=0
	nnet::relu<input_t, input_t, dec_relu_config1>(logits1, layer1_relu_out);

	// Dense
	result_t logits2[M];
	#pragma HLS ARRAY_PARTITION variable=logits2 complete dim=0
	nnet::compute_layer<input_t, result_t, dec_config2>(layer1_relu_out, logits2, dec_w2, dec_b2);


	// Softmax
	 nnet::softmax<result_t, result_t, dec_softmax_config2>(logits2, res);

}
