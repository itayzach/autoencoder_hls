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

int main(int argc, char **argv) {

	//hls-fpga-machine-learning insert data
	int err_cnt = 0;
	input_t  data_in[M]  = { 1.0, 2.0, 13.0, 4.0 };
	result_t expected[M] = { 2.0, 3.0, 14.0, 5.0 };

	result_t result[M];
	for (int i = 0; i < M; i++) {
		result[i] = 0;
	}

	// TX
	unsigned short size_in, size_out;
	encoder(data_in, result, size_in, size_out);

	// Noise

	// RX


	// print results
	const char separator    = ' ';
	const int fieldWidth    = 10;
	std::cout << "-------------------------------------" << std::endl;
	std::cout << std::setw(fieldWidth) << "result";
	std::cout << std::setw(fieldWidth) << "expected";
	std::cout << std::setw(fieldWidth) << "diff [%]";
	std::cout << std::endl;
	std::cout << "-------------------------------------" << std::endl;

	for (int i = 0; i < M; i++) {
		float diff = 100.0 * ((float) result[i] - (float) expected[i]) / (float) expected[i];
		std::cout << std::setw(fieldWidth) << result[i];
		std::cout << std::setw(fieldWidth) << expected[i];
		std::cout << std::setw(fieldWidth) << diff;
		std::cout << std::endl;

		if (diff > 0.5) {
			err_cnt++;
		}
	}
	std::cout << "-------------------------------------" << std::endl;
	std::cout << "total of " << err_cnt << " errors" << std::endl;
	std::cout << "-------------------------------------" << std::endl;

	return err_cnt;
}
