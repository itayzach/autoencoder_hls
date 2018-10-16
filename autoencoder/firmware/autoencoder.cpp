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
#include "autoencoder.h"

// nnet libraries
#include "nnet_layer.h"
#include "nnet_activation.h"
#include "nnet_normalization_layer.h"

// weights
#include "weights/enc_weights.h"
#include "weights/dec_weights.h"


// ========================================================================
// encoder
// ========================================================================
void encoder(
    input_t data[M_in],
    result_t res[n_channel])
{

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=data complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=res complete dim=0
    //#pragma HLS INTERFACE axis port=data,res

    #pragma HLS PIPELINE II=100

	unsigned short const const_size_in   = M_in;
	unsigned short const const_size_out  = n_channel;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************
    // Dense
    result_t logits1[M_in];
    #pragma HLS ARRAY_PARTITION variable=logits1 complete dim=0
    nnet::compute_layer<input_t, result_t, enc_config1>(data, logits1, enc_w1, enc_b1);

     // ReLU
     input_t layer1_relu_out[M_in];
     #pragma HLS ARRAY_PARTITION variable=layer1_relu_out complete dim=0
     nnet::relu<input_t, input_t, enc_relu_config1>(logits1, layer1_relu_out);

    // Dense
    result_t logits2[n_channel];
    #pragma HLS ARRAY_PARTITION variable=logits2 complete dim=0
    nnet::compute_layer<input_t, result_t, enc_config2>(layer1_relu_out, logits2, enc_w2, enc_b2);

    // Normalize
    nnet::normalization_layer<result_t, result_t, enc_norm_config3>(logits2, res);

}

// ========================================================================
// decoder
// ========================================================================
void decoder(
    input_t data[n_channel],
    result_t res[M_in])
{

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=data complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=res complete dim=0
    //#pragma HLS INTERFACE axis port=data,res

    #pragma HLS PIPELINE II=100

    unsigned short const const_size_in   = n_channel;
    unsigned short const const_size_out  = M_in;

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************
    // Dense
    result_t logits1[M_in];
    #pragma HLS ARRAY_PARTITION variable=logits1 complete dim=0
    nnet::compute_layer<input_t, result_t, dec_config1>(data, logits1, dec_w1, dec_b1);

    // ReLU
    input_t layer1_relu_out[M_in];
    #pragma HLS ARRAY_PARTITION variable=layer1_relu_out complete dim=0
    nnet::relu<input_t, input_t, dec_relu_config1>(logits1, layer1_relu_out);

    // Dense
    result_t logits2[M_in];
    #pragma HLS ARRAY_PARTITION variable=logits2 complete dim=0
    nnet::compute_layer<input_t, result_t, dec_config2>(layer1_relu_out, logits2, dec_w2, dec_b2);

    // Softmax
    result_t logits3[M_in];
    nnet::softmax<result_t, result_t, dec_softmax_config2>(logits2, logits3);

    // Argmax
    result_t max_val = logits3[0];
    result_t max_idx = 0;
    argmax: for(int ii = 1; ii < const_size_out; ii++) {
        if (logits3[ii] > max_val) {
            max_idx = ii;
            max_val = logits3[ii];
        }
    }

    reset_res: for(int ii = 0; ii < const_size_out; ii++) {
    	if (ii == max_idx) {
    		res[max_idx] = 1.0;
    	} else {
    		res[ii] = 0.0;
    	}

    }



}


// ========================================================================
// encoder_decoder
// ========================================================================
//void encoder_decoder(
//  input_t enc_data_in[M_in],
//  //result_t enc_data_out[n_channel],
//  //input_t dec_data_in[n_channel],
//  result_t dec_data_out[M_in])
//{
//#pragma HLS INTERFACE axis register both depth=4 latency=2 port=enc_data_in
//#pragma HLS INTERFACE axis register both depth=4 latency=2 port=dec_data_out
//#pragma HLS INTERFACE ap_ctrl_none port=return
////#pragma HLS ARRAY_RESHAPE variable=enc_data_in complete dim=0
////#pragma HLS ARRAY_RESHAPE variable=enc_data_out complete dim=0
////#pragma HLS ARRAY_RESHAPE variable=dec_data_in complete dim=0
////#pragma HLS ARRAY_RESHAPE variable=dec_data_out complete dim=0
//
//#pragma HLS PIPELINE II=100
//
//#pragma HLS RESOURCE variable=enc_data_out core=FIFO latency=2
//#pragma HLS RESOURCE variable=dec_data_in core=FIFO latency=2
//	result_t enc_data_out[n_channel];
//	input_t dec_data_in[n_channel];
//
//    encoder(enc_data_in, enc_data_out);
//
//    for (int i = 0; i < n_channel; i++) {
//    	dec_data_in[i] = enc_data_out[i];
//    }
//
//    decoder(dec_data_in, dec_data_out);
//}

void encoder_decoder(
  hls::stream<axis_input_t> &enc_data_in,
  //result_t enc_data_out[n_channel],
  //input_t dec_data_in[n_channel],
  hls::stream<axis_result_t> &dec_data_out)
{
//#pragma HLS INTERFACE axis port=enc_data_in
//#pragma HLS INTERFACE axis port=dec_data_out
#pragma HLS INTERFACE axis register both latency=2 port=enc_data_in
#pragma HLS INTERFACE axis register both latency=2 port=dec_data_out

#pragma HLS INTERFACE ap_ctrl_none port=return
//#pragma HLS ARRAY_RESHAPE variable=enc_data_in complete dim=0
//#pragma HLS ARRAY_RESHAPE variable=enc_data_out complete dim=0
//#pragma HLS ARRAY_RESHAPE variable=dec_data_in complete dim=0
//#pragma HLS ARRAY_RESHAPE variable=dec_data_out complete dim=0

#pragma HLS PIPELINE II=100

#pragma HLS RESOURCE variable=enc_data_out core=FIFO latency=2
#pragma HLS RESOURCE variable=dec_data_in core=FIFO latency=2
	axis_input_t enc_data_in_tmp[M_in];
	axis_result_t dec_data_out_tmp;
	input_t enc_data_in_reg[M_in];
	result_t dec_data_out_reg[M_in];

	for(int i = 0; i < M_in; i++){
		enc_data_in_tmp[i] = enc_data_in.read();
		enc_data_in_reg[i] = enc_data_in_tmp[i].data;
	}

	result_t enc_data_out[n_channel];
	input_t dec_data_in[n_channel];

    encoder(enc_data_in_reg, enc_data_out);

    for (int i = 0; i < n_channel; i++) {
    	dec_data_in[i] = enc_data_out[i];
    }

    decoder(dec_data_in, dec_data_out_reg);

    for(int i = 0; i < M_in; i++) {
    	dec_data_out_tmp.data = dec_data_out_reg[i];
    	dec_data_out_tmp.keep = enc_data_in_tmp[i].keep;
		dec_data_out_tmp.strb = enc_data_in_tmp[i].strb;
		dec_data_out_tmp.user = enc_data_in_tmp[i].user;
		dec_data_out_tmp.last = enc_data_in_tmp[i].last;
		dec_data_out_tmp.id   = enc_data_in_tmp[i].id;
		dec_data_out_tmp.dest = enc_data_in_tmp[i].dest;
    	dec_data_out.write(dec_data_out_tmp);
    }

}
