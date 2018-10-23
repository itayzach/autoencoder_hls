#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_layer.h"
#include "nnet_conv.h"
#include "nnet_activation.h"
#include "nnet_common.h"
#include "nnet_normalization_layer.h"

#define M_in 4
//#define k 2 // log2(M_in)
#define n_channel 2

//hls-fpga-machine-learning insert numbers
typedef ap_fixed<32,8> accum_default_t;
typedef ap_fixed<32,8> weight_default_t;
typedef ap_fixed<32,8> bias_default_t;

typedef ap_fixed<32,8> norm_default_t;

typedef ap_fixed<32,8> input_t;
typedef ap_fixed<32,8> result_t;

typedef struct {
	input_t           data;
	ap_uint<(32+7)/8> keep;
	ap_uint<1>        last;
} axis_input_t;

typedef struct {
	result_t          data;
	ap_uint<(32+7)/8> keep;
	ap_uint<1>        last;
} axis_result_t;

// ========================================================================
// Encoder parameters
// ========================================================================
struct enc_config1 : nnet::layer_config {
    static const unsigned n_in = M_in;
    static const unsigned n_out = M_in;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
};

struct enc_relu_config1 : nnet::activ_config {
    static const unsigned n_in = M_in;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct enc_config2 : nnet::layer_config {
    static const unsigned n_in = M_in;
    static const unsigned n_out = n_channel;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
};

struct enc_norm_config3 : nnet::norm_layer_config {
    static const unsigned n = n_channel;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 10;
    typedef norm_default_t norm_t;
};

// ========================================================================
// Decoder parameters
// ========================================================================
struct dec_config1 : nnet::layer_config {
    static const unsigned n_in = n_channel;
    static const unsigned n_out = M_in;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
};

struct dec_relu_config1 : nnet::activ_config {
    static const unsigned n_in = M_in;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};

struct dec_config2 : nnet::layer_config {
    static const unsigned n_in = M_in;
    static const unsigned n_out = M_in;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    typedef accum_default_t accum_t;
    typedef bias_default_t bias_t;
    typedef weight_default_t weight_t;
};


struct dec_softmax_config2 : nnet::activ_config {
    static const unsigned n_in = M_in;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
};
#endif
