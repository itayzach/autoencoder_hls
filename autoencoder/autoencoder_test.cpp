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

int print_and_check_results(const char* enc_dec_str, result_t* result, result_t* expected) {
	int err_cnt = 0;
	const char separator    = ' ';
	const int fieldWidth    = 10;
	int size = (strcmp(enc_dec_str, "ENCODER") == 0) ? n :
			   (strcmp(enc_dec_str, "DECODER") == 0) ? M : -1;
	std::cout << "*************************************" << std::endl;
	std::cout << "************** " << enc_dec_str << " **************" << std::endl;
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
		std::cout << std::endl;

		if (abs(diff) > 0.5) {
			err_cnt++;
		}
	}
	std::cout << "-------------------------------------" << std::endl;
	std::cout << "total of " << err_cnt << " errors" << std::endl;
	std::cout << "-------------------------------------" << std::endl;

	return err_cnt;
}

int main(int argc, char **argv) {

	//hls-fpga-machine-learning insert data
	input_t  enc_data_in[M]  = { 1.0, 12.8764, -13.0, 4.0 };
//	result_t expected[M] = { 2.0, 13.8764, 0.0, 5.0 };
	result_t enc_expected[n] = { 14.8764, 6.0 };

	result_t enc_result[M];
	for (int i = 0; i < M; i++) {
		enc_result[i] = 0;
	}

	// ========================================================================
	// TX
	// ========================================================================
	unsigned short enc_size_in, enc_size_out;
	encoder(enc_data_in, enc_result, enc_size_in, enc_size_out);

	// print results
	int enc_err_cnt = print_and_check_results("ENCODER", enc_result, enc_expected);

	// ========================================================================
	// Noise
	// ========================================================================

	// ========================================================================
	// RX
	// ========================================================================




	return enc_err_cnt;
}
