// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.2.1
// Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module relu (
        ap_ready,
        data_0_V_read,
        data_1_V_read,
        data_2_V_read,
        data_3_V_read,
        ap_return_0,
        ap_return_1,
        ap_return_2,
        ap_return_3
);


output   ap_ready;
input  [31:0] data_0_V_read;
input  [31:0] data_1_V_read;
input  [31:0] data_2_V_read;
input  [31:0] data_3_V_read;
output  [31:0] ap_return_0;
output  [31:0] ap_return_1;
output  [31:0] ap_return_2;
output  [31:0] ap_return_3;

wire   [0:0] tmp_s_fu_54_p2;
wire   [30:0] tmp_208_fu_60_p1;
wire   [30:0] res_0_V_write_assig_fu_64_p3;
wire   [0:0] tmp_97_1_fu_76_p2;
wire   [30:0] tmp_209_fu_82_p1;
wire   [30:0] res_1_V_write_assig_fu_86_p3;
wire   [0:0] tmp_97_2_fu_98_p2;
wire   [30:0] tmp_210_fu_104_p1;
wire   [30:0] res_2_V_write_assig_fu_108_p3;
wire   [0:0] tmp_97_3_fu_120_p2;
wire   [30:0] tmp_211_fu_126_p1;
wire   [30:0] res_3_V_write_assig_fu_130_p3;
wire   [31:0] res_0_V_write_assig_3_fu_72_p1;
wire   [31:0] res_1_V_write_assig_3_fu_94_p1;
wire   [31:0] res_2_V_write_assig_3_fu_116_p1;
wire   [31:0] res_3_V_write_assig_3_fu_138_p1;

assign ap_ready = 1'b1;

assign ap_return_0 = res_0_V_write_assig_3_fu_72_p1;

assign ap_return_1 = res_1_V_write_assig_3_fu_94_p1;

assign ap_return_2 = res_2_V_write_assig_3_fu_116_p1;

assign ap_return_3 = res_3_V_write_assig_3_fu_138_p1;

assign res_0_V_write_assig_3_fu_72_p1 = res_0_V_write_assig_fu_64_p3;

assign res_0_V_write_assig_fu_64_p3 = ((tmp_s_fu_54_p2[0:0] === 1'b1) ? tmp_208_fu_60_p1 : 31'd0);

assign res_1_V_write_assig_3_fu_94_p1 = res_1_V_write_assig_fu_86_p3;

assign res_1_V_write_assig_fu_86_p3 = ((tmp_97_1_fu_76_p2[0:0] === 1'b1) ? tmp_209_fu_82_p1 : 31'd0);

assign res_2_V_write_assig_3_fu_116_p1 = res_2_V_write_assig_fu_108_p3;

assign res_2_V_write_assig_fu_108_p3 = ((tmp_97_2_fu_98_p2[0:0] === 1'b1) ? tmp_210_fu_104_p1 : 31'd0);

assign res_3_V_write_assig_3_fu_138_p1 = res_3_V_write_assig_fu_130_p3;

assign res_3_V_write_assig_fu_130_p3 = ((tmp_97_3_fu_120_p2[0:0] === 1'b1) ? tmp_211_fu_126_p1 : 31'd0);

assign tmp_208_fu_60_p1 = data_0_V_read[30:0];

assign tmp_209_fu_82_p1 = data_1_V_read[30:0];

assign tmp_210_fu_104_p1 = data_2_V_read[30:0];

assign tmp_211_fu_126_p1 = data_3_V_read[30:0];

assign tmp_97_1_fu_76_p2 = (($signed(data_1_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign tmp_97_2_fu_98_p2 = (($signed(data_2_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign tmp_97_3_fu_120_p2 = (($signed(data_3_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

assign tmp_s_fu_54_p2 = (($signed(data_0_V_read) > $signed(32'd0)) ? 1'b1 : 1'b0);

endmodule //relu
