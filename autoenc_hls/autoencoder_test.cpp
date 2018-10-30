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
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hls_dsp.h>
#include <random>
#include <cmath>

#include "ap_axi_sdata.h"

#include "firmware/parameters.h"
#include "firmware/autoencoder.h"
#include "nnet_helpers.h"
#include "firmware/weights/expected.h"

#define SEED 5
#define NUM_SIMULATIONS 1
#define NUM_SIGNALS 10000
//#define DEBUG_SOFTMAX

const char separator = ' ';
const int fieldWidth = 15;


int single_print_and_check_results(result_t* result, result_t* expected, int size, const float allowed_precent_diff, int do_print) {
	int err_cnt = 0;
	for (int i = 0; i < size; i++) {
	        float diff = 0.0;
	        if (expected[i] == 0.0) { // dividing by 0 is bad
	            diff = (float) result[i];
	        } else {
	            diff = 100.0 * ((float) result[i] - (float) expected[i]) / (float) expected[i];
	        }
	        if (do_print) {
				std::cout << std::setw(fieldWidth) << result[i];
				std::cout << std::setw(fieldWidth) << expected[i];
				std::cout << std::setw(fieldWidth) << diff << "%";
	        }
	        if (abs(diff) > allowed_precent_diff) {
	            err_cnt++;
	            if (do_print)
	            	std::cout << " << ERROR";
	        }
	        if (do_print)
	        	std::cout << std::endl;
	    }
	return err_cnt;
}

int txrx_data_print_and_check_results(
		int simIdx,
		unsigned int* tx_data_rec, unsigned int* rx_data_rec) {
    int err_cnt = 0;
    for (int sigIdx = 0; sigIdx < NUM_SIGNALS; sigIdx++) {
    	//std::cout << std::setw(fieldWidth) << tx_data_rec[sigIdx];
		//std::cout << std::setw(fieldWidth) << rx_data_rec[sigIdx];
		if (tx_data_rec[sigIdx] != rx_data_rec[sigIdx]) {
			err_cnt++;
			//std::cout << " << ERROR";
		}
		//std::cout << std::endl;
    }

    return err_cnt;
}


int elaborated_print_and_check_results(
		int simIdx,
		result_t* dec_data_out_rec, result_t* dec_expected, const float dec_allowed_precent_diff, int do_print) {
    int err_cnt = 0;

    std::cout << "**************************************************************************" << std::endl;
    std::cout << "* Simulation #"        << simIdx << std::endl;
    std::cout << "* allowed dec diff : " << dec_allowed_precent_diff << " %" << std::endl;
    std::cout << "**************************************************************************" << std::endl;
    std::cout << std::setw(fieldWidth) << "result";
    std::cout << std::setw(fieldWidth) << "expected";
    std::cout << std::setw(fieldWidth) << "diff";
    std::cout << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;

    for (int sigIdx = 0; sigIdx < NUM_SIGNALS; sigIdx++) {
    	int sig_err;
    	if (do_print) {
    		std::cout << "Signal:" << std::endl;
    	}
		if (do_print) {
			std::cout << "--------------------------------------------------------------------------" << std::endl;
			std::cout << "RX (decoder):" << std::endl;
		}
		sig_err = single_print_and_check_results(&dec_data_out_rec[sigIdx*M_in], &dec_expected[sigIdx*M_in], M_in, dec_allowed_precent_diff, do_print);
		if (sig_err && do_print) {
			std::cout << " << ERROR @ idx" << sigIdx << std::endl;
		}
		err_cnt += sig_err;
		if (do_print) {
			std::cout << "==========================================================================" << std::endl;
		}
    }

    return err_cnt;
}

int main(int argc, char **argv) {

    // ========================================================================
    // TX setup
    // ========================================================================
	unsigned short enc_size_in, enc_size_out;
    // for the following input: TX[0] val : {0.0, 0.0, 0.0, 1.0} = 3
    // expected w/o normalization  : [3.9336898 3.453742]
    // expected with normalization : [1.0562046 0.9404423]
    //input_t  enc_data_in[M_in]  = {0.0, 0.0, 0.0, 1.0};
    //result_t enc_expected[n_channel] = {1.0557002,  0.94100845};

	hls::stream<axis_input_t>  enc_data_in;
    result_t enc_data_out[n_channel];
    for (int i = 0; i < M_in; i++) {
        enc_data_out[i] = 0;
    }

    // ========================================================================
    // RX setup
    // ========================================================================
	unsigned short dec_size_in, dec_size_out;
	//input_t dec_data_in[n_channel] = {1.0557002,  0.94100845};
    //result_t dec_expected[M_in] = {0.7573132, 0.0377044, 0.1024912, 0.1024912}; // withsoftmax
    //result_t dec_expected[M_in] = {0.0,         3.0197535,  0.32067692, 0.7918129 }; // first layer+relu
    //result_t dec_expected[M_in] = {7.7156507e-04, 5.8024080e-04, 2.0423336e-03, 9.9660587e-01};
    //result_t dec_expected[M_in] = {0.0, 0.0, 0.0, 1.0};

	input_t dec_data_in[n_channel];
	hls::stream<axis_result_t> dec_data_out;

    // ========================================================================
    // Simulation setup
    // ========================================================================
    const float enc_allowed_precent_diff = 0.001;
	const float noise_allowed_precent_diff = 200;
	const float dec_allowed_precent_diff = 0.0;

    // ========================================================================
	// Run simulation
	// ========================================================================
	float EbNo_dB_array[NUM_SIMULATIONS];
	unsigned int err_cnt_array[NUM_SIMULATIONS];
	t_snr snr = 15.5;
	for (int simIdx = 0; simIdx < NUM_SIMULATIONS; simIdx++, snr+=1) {
		std::cout << "snr = " << snr << std::endl;
		int sim_err_cnt = 0;

		// Record arrays
		unsigned int tx_data_rec[NUM_SIGNALS];
		result_t enc_data_out_rec[n_channel*NUM_SIGNALS];
		result_t enc_expected_rec[n_channel*NUM_SIGNALS];
		result_t dec_data_in_rec[n_channel*NUM_SIGNALS];
		result_t dec_data_out_rec[M_in*NUM_SIGNALS];
		result_t dec_expected_rec[M_in*NUM_SIGNALS];
		unsigned int rx_data_rec[NUM_SIGNALS] = {0};
		unsigned int tx_data;

		// Run for each possible signal
		for (int sigIdx = 0; sigIdx < NUM_SIGNALS; sigIdx++) {
			// Generate random data
			// tx_data = 1;
			// tx_data = sigIdx % M_in;
			tx_data = rand () % M_in;

			// Set enc data in
			for (int i = 0; i < M_in; i++) {
				axis_input_t enc_data_in_tmp;

				if (i == tx_data) {
					enc_data_in_tmp.data = 1;
				} else {
					enc_data_in_tmp.data = 0;
				}
				enc_data_in_tmp.keep = 0xF;
				enc_data_in_tmp.user = sigIdx;
				enc_data_in_tmp.last = (sigIdx == NUM_SIGNALS - 1) && (i == M_in - 1);

				enc_data_in << enc_data_in_tmp;
			}

			int awgn_en = 0;

			// ================================================================
			// Top function
			// ================================================================
			encoder_decoder(enc_data_in,
							dec_data_out,
							snr,
							awgn_en);

			// Add to record arrays
			tx_data_rec[sigIdx] = tx_data;

			for (int i = 0; i < n_channel; i ++) {
				enc_data_out_rec[sigIdx*n_channel + i] = enc_data_out[i];
				dec_data_in_rec[sigIdx*n_channel + i] = dec_data_in[i];
				enc_expected_rec[sigIdx*n_channel + i] = enc_expected[tx_data*n_channel + i];
			}
			for (int i = 0; i < M_in; i++) {
				axis_result_t dec_data_out_tmp;
				dec_data_out_tmp = dec_data_out.read();
				dec_data_out_rec[sigIdx*M_in + i] = dec_data_out_tmp.data;
				dec_expected_rec[sigIdx*M_in + i] = dec_expected[tx_data*M_in + i];
				rx_data_rec[sigIdx] += (unsigned int)dec_data_out_rec[sigIdx*M_in + i] * i;
			}
		}

		// ====================================================================
		// Print and check results
		// ====================================================================
		err_cnt_array[simIdx] = txrx_data_print_and_check_results(simIdx, tx_data_rec, rx_data_rec);
		int do_print = 0;
		std::cout << "Sim #" << simIdx << ": SNR = " << snr << "\t" << (float)err_cnt_array[simIdx]/(float)NUM_SIGNALS << std::endl;
//		sim_err_cnt = elaborated_print_and_check_results(
//			simIdx,
//			dec_data_out_rec, dec_expected_rec, dec_allowed_precent_diff, do_print);
//		std::cout << "errors count = " << sim_err_cnt << std::endl;
	}

    return 0;
}
