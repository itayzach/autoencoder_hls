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

#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

#include "parameters.h"


// Prototype of top level function for C-synthesis
void encoder(
    input_t data[M_in],
    result_t res[n_channel]);

void decoder(
    input_t data[n_channel],
    result_t res[M_in]);

void encoder_decoder(
    input_t enc_data_in[M_in],
    //result_t enc_data_out[n_channel],
    //input_t dec_data_in[n_channel],
    result_t dec_data_out[M_in]);

#endif
