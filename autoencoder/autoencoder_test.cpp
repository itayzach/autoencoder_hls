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

#include "firmware/parameters.h"
#include "firmware/autoencoder.h"
#include "nnet_helpers.h"

int print_and_check_results(const char* enc_dec_str, result_t* result, result_t* expected, const float allowed_precent_diff) {
    int err_cnt = 0;
    const char separator = ' ';
    const int fieldWidth = 15;
    int size = (strcmp(enc_dec_str, "ENCODER") == 0)     ? n_channel :
    		   (strcmp(enc_dec_str, "ENCODER_DBG") == 0) ? n_channel :
               (strcmp(enc_dec_str, "DECODER") == 0)     ? M_in : -1;
    assert(size > 0);
    std::cout << "*************************************" << std::endl;
    std::cout << "* " << enc_dec_str << std::endl;
    std::cout << "* allowed diff [%] : " << allowed_precent_diff << std::endl;
    std::cout << "*************************************" << std::endl;
    std::cout << std::setw(fieldWidth) << "result";
    std::cout << std::setw(fieldWidth) << "expected";
    std::cout << std::setw(fieldWidth) << "diff [%]";
    std::cout << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    for (int i = 0; i < size; i++) {
        float diff = 0.0;
        if (expected[i] == 0.0) { // diving by 0 is bad
            diff = (float) result[i];
        } else {
            diff = 100.0 * ((float) result[i] - (float) expected[i]) / (float) expected[i];
        }
        std::cout << std::setw(fieldWidth) << result[i];
        std::cout << std::setw(fieldWidth) << expected[i];
        std::cout << std::setw(fieldWidth) << diff;
        if (abs(diff) > allowed_precent_diff) {
			err_cnt++;
			std::cout << " << ERROR";
		}
        std::cout << std::endl;


    }
    std::cout << "-------------------------------------" << std::endl;
    std::cout << "total of " << err_cnt << " errors" << std::endl;
    std::cout << "-------------------------------------" << std::endl;

    return err_cnt;
}

int main(int argc, char **argv) {

	// ========================================================================
	// TX
	// ========================================================================
    // for the following input: TX[0] val : {0.0, 0.0, 0.0, 1.0} = 3
	// expected w/o normalization  : [3.9336898 3.453742]
    // expected with normalization : [1.0562046 0.9404423]
    input_t  enc_data_in[M_in]  = {0.0, 0.0, 0.0, 1.0};
    result_t enc_expected[n_channel] = {3.9336898, 3.453742};

    result_t enc_result[M_in];
    for (int i = 0; i < M_in; i++) {
        enc_result[i] = 0;
    }

    unsigned short enc_size_in, enc_size_out;
    encoder(enc_data_in, enc_result, enc_size_in, enc_size_out);

    // print and check results
    const float enc_allowed_precent_diff = 0.1;
    int enc_err_cnt = print_and_check_results("ENCODER_DBG", enc_result, enc_expected, enc_allowed_precent_diff);

    // ========================================================================
    // Noise
    // ========================================================================

    // ========================================================================
    // RX
    // ========================================================================
    // input_t  dec_data_in[n_channel]  = { 5.0, 6.0 };
	// result_t dec_expected[M_in] = { 6.0, 3.0, 4.0, 4.0 }; // without softmax
    input_t dec_data_in[n_channel]  = { 0.8473452, -0.8654846 };
	//result_t dec_expected[M_in] = {0.7573132, 0.0377044, 0.1024912, 0.1024912}; // withsoftmax
    //result_t dec_expected[M_in] = {0.0,         3.0197535,  0.32067692, 0.7918129 }; // first layer+relu
    //result_t dec_expected[M_in] = {7.7156507e-04, 5.8024080e-04, 2.0423336e-03, 9.9660587e-01};
    result_t dec_expected[M_in] = {0.0, 0.0, 0.0, 1.0};

	result_t dec_result[M_in];
	for (int i = 0; i < M_in; i++) {
		dec_result[i] = 0;
	}

    unsigned short dec_size_in, dec_size_out;
    decoder(dec_data_in, dec_result, dec_size_in, dec_size_out);

    // print and check results
    const float dec_allowed_precent_diff = 0.0;
	int dec_err_cnt = print_and_check_results("DECODER", dec_result, dec_expected, dec_allowed_precent_diff);

    return enc_err_cnt + dec_err_cnt;
}
