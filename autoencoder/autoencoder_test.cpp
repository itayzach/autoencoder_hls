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
#define NUM_SIGNALS 4
#define LOOPBACK_MODE 1

const char separator = ' ';
const int fieldWidth = 15;


float randn(float mu, float sigma) {
//	float U1, U2, W, mult;
//	static float X1, X2;
//	static int call = 0;
//	if (call == 1) {
//		call = !call;
//		return (mu + sigma * (float) X2);
//	}
//
//	do {
//		U1 = -1 + ((float) rand() / RAND_MAX) * 2;
//		U2 = -1 + ((float) rand() / RAND_MAX) * 2;
//		W = pow(U1, 2) + pow(U2, 2);
//	} while (W >= 1 || W == 0);
//
//	mult = sqrt((-2 * log(W)) / W);
//	X1 = U1 * mult;
//	X2 = U2 * mult;
//
//	call = !call;
//
//	return (mu + sigma * (float) X1);

	std::random_device rd{};
	std::mt19937 gen{rd()};

	// values near the mean are the most likely
	// standard deviation affects the dispersion of generated values from the mean
	std::normal_distribution<> d{mu,sigma};
	return d(gen);

}

int single_print_and_check_results(result_t* result, result_t* expected, int size, const float allowed_precent_diff) {
	int err_cnt = 0;
	for (int i = 0; i < size; i++) {
	        float diff = 0.0;
	        if (expected[i] == 0.0) { // diving by 0 is bad
	            diff = (float) result[i];
	        } else {
	            diff = 100.0 * ((float) result[i] - (float) expected[i]) / (float) expected[i];
	        }
	        std::cout << std::setw(fieldWidth) << result[i];
	        std::cout << std::setw(fieldWidth) << expected[i];
	        std::cout << std::setw(fieldWidth) << diff << "%";
	        if (abs(diff) > allowed_precent_diff) {
	            err_cnt++;
	            std::cout << " << ERROR";
	        }
	        std::cout << std::endl;
	    }
	return err_cnt;
}

int txrx_data_print_and_check_results(
		int simIdx, float EbNo_dB, float noise_std,
		unsigned int* tx_data_rec, unsigned int* rx_data_rec) {
    int err_cnt = 0;
//	std::cout << "**************************************************************************" << std::endl;
//	std::cout << "* Simulation #"        << simIdx << std::endl;
//	std::cout << "* Eb/No            : " << EbNo_dB << "[dB]" << std::endl;
//    std::cout << "* noise std        : " << noise_std << std::endl;
//	std::cout << "**************************************************************************" << std::endl;
//	std::cout << std::setw(fieldWidth) << "result";
//	std::cout << std::setw(fieldWidth) << "expected";
//	std::cout << std::endl;
//	std::cout << "--------------------------------------------------------------------------" << std::endl;
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
		int simIdx, float EbNo_dB, float noise_std,
		result_t* enc_data_out_rec, result_t* enc_expected, const float enc_allowed_precent_diff,
		result_t* dec_data_in_rec,  result_t* noise,        const float noise_allowed_precent_diff,
		result_t* dec_data_out_rec, result_t* dec_expected, const float dec_allowed_precent_diff) {
    int err_cnt = 0;

    std::cout << "**************************************************************************" << std::endl;
    std::cout << "* Simulation #"        << simIdx << std::endl;
    std::cout << "* Eb/No            : " << EbNo_dB << "[dB]" << std::endl;
    std::cout << "* noise std        : " << noise_std << std::endl;
    std::cout << "* allowed enc diff : " << enc_allowed_precent_diff << " %" << std::endl;
    std::cout << "* allowed dec diff : " << dec_allowed_precent_diff << " %" << std::endl;
    std::cout << "**************************************************************************" << std::endl;
    std::cout << std::setw(fieldWidth) << "result";
    std::cout << std::setw(fieldWidth) << "expected";
    std::cout << std::setw(fieldWidth) << "diff";
    std::cout << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;

    for (int sigIdx = 0; sigIdx < NUM_SIGNALS; sigIdx++) {
    	std::cout << "Signal:" << std::endl;
		err_cnt += single_print_and_check_results(&dec_expected[sigIdx*M_in], &dec_expected[sigIdx*M_in], M_in, 0.0);
		std::cout << "--------------------------------------------------------------------------" << std::endl;
		std::cout << "TX (encoder):" << std::endl;
		err_cnt += single_print_and_check_results(&enc_data_out_rec[sigIdx*n_channel], &enc_expected[sigIdx*n_channel], n_channel, enc_allowed_precent_diff);
		std::cout << "--------------------------------------------------------------------------" << std::endl;
		std::cout << "AWGN:" << std::endl;
		err_cnt += single_print_and_check_results(&dec_data_in_rec[sigIdx*n_channel], &noise[sigIdx*n_channel], n_channel, noise_allowed_precent_diff);
		std::cout << "--------------------------------------------------------------------------" << std::endl;
		std::cout << "RX (decoder):" << std::endl;
		err_cnt += single_print_and_check_results(&dec_data_out_rec[sigIdx*M_in], &dec_expected[sigIdx*M_in], M_in, dec_allowed_precent_diff);
		std::cout << "==========================================================================" << std::endl;
    }
//    std::cout << "--------------------------------------------------------------------------" << std::endl;
//    std::cout << "total of " << err_cnt << " errors" << std::endl;
//    std::cout << "--------------------------------------------------------------------------" << std::endl;
    std::cout << std::endl << std::endl;

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
    // Noise setup
    // ========================================================================
	float R = log2(M_in) / n_channel;
    ap_uint<8> lfsr_seed = SEED;
	static hls::awgn<8> my_awgn(lfsr_seed);
	//ap_int<8> noise[100];
	ap_ufixed<32,8> snr = 7.0;
	for (int i = 0; i < 100; i++) {
		//my_awgn(snr, noise[i]);
		//std::cout << "noise[" << i << "] = " << noise[i] << std::endl;
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
//    for (int i = 0; i < M_in; i++) {
//        dec_data_out[i].data = 0;
//    }

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
	std::random_device rd{};
	std::mt19937 gen{rd()};
	std::normal_distribution<> norm_dist{0.0, 1.0};
	for (int simIdx = 0; simIdx < NUM_SIMULATIONS; simIdx++) {
		int sim_err_cnt = 0;
		// Generate random noise
		float EbNo_dB = 7.0 + 1.5*simIdx; // [dB]
		EbNo_dB_array[simIdx] = EbNo_dB;
		float EbNo = pow(10.0, EbNo_dB/10.0);
		float noise_std = sqrt(1/(2*R*EbNo));

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
			tx_data = sigIdx % M_in;
//			tx_data = rand () % M_in;
			// Reset enc data in
			for (int i = 0; i < M_in; i++) {
				axis_input_t enc_data_in_tmp;

				if (i == tx_data) {
					enc_data_in_tmp.data = 1;
				} else {
					enc_data_in_tmp.data = 0;
				}
				enc_data_in_tmp.keep = 15;
				enc_data_in_tmp.last = (i == M_in - 1);

				enc_data_in << enc_data_in_tmp;
			}

			if (LOOPBACK_MODE == 0) {
				// TX
				//encoder(enc_data_in, enc_data_out);

				// AWGN
				for (int i = 0; i < n_channel; i++) {
					input_t noise = noise_std * norm_dist(gen);
					dec_data_in[i] = enc_data_out[i] + noise;
				}
				// RX
				//decoder(dec_data_in, dec_data_out);

				// Add to record arrays
				tx_data_rec[sigIdx] = tx_data;

//				for (int i = 0; i < n_channel; i ++) {
//					enc_data_out_rec[sigIdx*n_channel + i] = enc_data_out[i];
//					dec_data_in_rec[sigIdx*n_channel + i] = dec_data_in[i];
//					enc_expected_rec[sigIdx*n_channel + i] = enc_expected[tx_data*n_channel + i];
//				}
//				for (int i = 0; i < M_in; i++) {
//					dec_data_out_rec[sigIdx*M_in + i] = dec_data_out.read();
//					dec_expected_rec[sigIdx*M_in + i] = dec_expected[tx_data*M_in + i];
//					rx_data_rec[sigIdx] += (unsigned int)dec_data_out_rec[sigIdx*M_in + i] * i;
//				}

			} else {
				int bypass = 0;
				encoder_decoder(enc_data_in,
								dec_data_out);

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

		}
		// Print and check results
		err_cnt_array[simIdx] = txrx_data_print_and_check_results(simIdx, EbNo_dB, noise_std, tx_data_rec, rx_data_rec);
		std::cout << "Sim #" << simIdx << ": Eb/No = " << EbNo_dB_array[simIdx] << "\t" << (float)err_cnt_array[simIdx]/(float)NUM_SIGNALS << std::endl;
		sim_err_cnt = elaborated_print_and_check_results(
			simIdx, EbNo_dB, noise_std,
			enc_data_out_rec, enc_expected_rec, enc_allowed_precent_diff,
			dec_data_in_rec,  enc_data_out_rec, noise_allowed_precent_diff,
			dec_data_out_rec, dec_expected_rec, dec_allowed_precent_diff);
	}

    return 0;
}
