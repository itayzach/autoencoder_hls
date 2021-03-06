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

// standard and hls libraries
#include <iostream>
#include <hls_dsp.h>
#include "ap_axi_sdata.h"

// local libraries
#include "parameters.h"
#include "encoder.h"

// nnet libraries
#include "nnet_layer.h"
#include "nnet_activation.h"
#include "nnet_normalization_layer.h"

// weights
#include "weights/enc_weights.h"

// ========================================================================
// encoder
// ========================================================================
void encoder(input_t data[M_in], result_t res[n_channel]) {

	//hls-fpga-machine-learning insert IO
#pragma HLS ARRAY_RESHAPE variable=data complete dim=0
#pragma HLS ARRAY_RESHAPE variable=res complete dim=0
	//#pragma HLS INTERFACE axis port=data,res

#pragma HLS PIPELINE II=100

	unsigned short const const_size_in = M_in;
	unsigned short const const_size_out = n_channel;

	// ****************************************
	// NETWORK INSTANTIATION
	// ****************************************
	// Dense
	result_t logits1[M_in];
#pragma HLS ARRAY_PARTITION variable=logits1 complete dim=0
	nnet::compute_layer<input_t, result_t, enc_config1>(data, logits1, enc_w1,
			enc_b1);

	// ReLU
	input_t layer1_relu_out[M_in];
#pragma HLS ARRAY_PARTITION variable=layer1_relu_out complete dim=0
	nnet::relu<input_t, input_t, enc_relu_config1>(logits1, layer1_relu_out);

	// Dense
	result_t logits2[n_channel];
#pragma HLS ARRAY_PARTITION variable=logits2 complete dim=0
	nnet::compute_layer<input_t, result_t, enc_config2>(layer1_relu_out,
			logits2, enc_w2, enc_b2);

	// Normalize
	nnet::normalization_layer<result_t, result_t, enc_norm_config3>(logits2,
			res);

}


void encoder_top(
		hls::stream<axis_input_t> &axis_enc_data_in,
		hls::stream<axis_result_t> &axis_enc_data_out
		) {

#pragma HLS INTERFACE axis port=axis_enc_data_in
#pragma HLS INTERFACE axis port=axis_enc_data_out
#pragma HLS INTERFACE ap_ctrl_none port=return

//#pragma HLS PIPELINE II=1

	axis_input_t axis_enc_data_in_item[M_in];
	axis_result_t axis_enc_data_out_item;
	input_t enc_data_in[M_in];
	result_t enc_data_out[n_channel];


	for (int i = 0; i < M_in; i++) {
		axis_enc_data_in_item[i] = axis_enc_data_in.read();
		enc_data_in[i] = axis_enc_data_in_item[i].data;
	}

	encoder(enc_data_in, enc_data_out);

	for (int i = 0; i < n_channel; i++) {
		bool is_last = 0;

		axis_enc_data_out_item.data = enc_data_out[i];

		if (axis_enc_data_in_item[3].last) {
			is_last = 1;
		}

		axis_enc_data_out_item.last = is_last && (i == n_channel - 1);
		axis_enc_data_out.write(axis_enc_data_out_item);
		if (axis_enc_data_out_item.last) {
			break;
		}
	}


//	for (int i = 0; i < n_channel; i++) {
//		axis_enc_data_out_item.data = enc_data_out[i];
//		axis_enc_data_out_item.keep = axis_enc_data_in_item[i].keep;
//		axis_enc_data_out_item.last = axis_enc_data_in_item[i].last;
//		axis_enc_data_out.write(axis_enc_data_out_item);
//		if (axis_enc_data_out_item.last) {
//			break;
//		}
//	}

}

/////////////////////////////////////////////////////////////////////////////////
// only mult
/////////////////////////////////////////////////////////////////////////////////

//void encoder_decoder(
//  hls::stream<axis_input_t> &enc_data_in,
//  //result_t enc_data_out[n_channel],
//  //input_t dec_data_in[n_channel],
//  hls::stream<axis_result_t> &dec_data_out)
//{
//#pragma HLS INTERFACE axis port=enc_data_in
//#pragma HLS INTERFACE axis port=dec_data_out
////#pragma HLS INTERFACE axis register both latency=2 port=enc_data_in
////#pragma HLS INTERFACE axis register both latency=2 port=dec_data_out
//
//#pragma HLS INTERFACE ap_ctrl_none port=return
////#pragma HLS ARRAY_RESHAPE variable=enc_data_in complete dim=0
////#pragma HLS ARRAY_RESHAPE variable=enc_data_out complete dim=0
////#pragma HLS ARRAY_RESHAPE variable=dec_data_in complete dim=0
////#pragma HLS ARRAY_RESHAPE variable=dec_data_out complete dim=0
//
//#pragma HLS PIPELINE II=100
//
//	axis_input_t enc_data_in_tmp;
//	axis_result_t dec_data_out_tmp;
//
////	for(int i = 0; i < M_in; i++){
//	while(!enc_data_in.empty()) {
//		enc_data_in_tmp = enc_data_in.read();
//		result_t res;
//#pragma HLS RESOURCE variable=res core=MulnS
//		res = enc_data_in_tmp.data * enc_data_in_tmp.data;
//		dec_data_out_tmp.data = res;
//		dec_data_out_tmp.keep = enc_data_in_tmp.keep;
//		dec_data_out_tmp.last = enc_data_in_tmp.last;
//		dec_data_out.write(dec_data_out_tmp);
//		if (dec_data_out_tmp.last) {
//			break;
//		}
//    }
//
//}

