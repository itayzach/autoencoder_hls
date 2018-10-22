// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.2
// Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module compute_layer_0_0_0 (
        ap_clk,
        ap_rst,
        data_0_V_read,
        data_1_V_read,
        data_2_V_read,
        data_3_V_read,
        ap_return_0,
        ap_return_1,
        ap_ce
);


input   ap_clk;
input   ap_rst;
input  [31:0] data_0_V_read;
input  [31:0] data_1_V_read;
input  [31:0] data_2_V_read;
input  [31:0] data_3_V_read;
output  [31:0] ap_return_0;
output  [31:0] ap_return_1;
input   ap_ce;

reg   [31:0] tmp_115_reg_571;
wire    ap_block_state1_pp0_stage0_iter0;
wire    ap_block_state2_pp0_stage0_iter1;
wire    ap_block_pp0_stage0_11001;
reg   [31:0] tmp_123_0_1_reg_576;
reg   [31:0] tmp_123_1_reg_581;
reg   [31:0] tmp_123_1_1_reg_586;
reg   [31:0] tmp_123_2_reg_591;
reg   [26:0] tmp_101_reg_596;
reg   [31:0] tmp_123_3_reg_601;
reg   [31:0] tmp_123_3_1_reg_606;
wire  signed [31:0] p_Val2_1_1_fu_86_p0;
wire  signed [55:0] OP1_V_1_cast_fu_433_p1;
wire    ap_block_pp0_stage0;
wire  signed [31:0] p_Val2_3_1_fu_87_p0;
wire  signed [55:0] OP1_V_3_cast_fu_489_p1;
wire  signed [31:0] p_Val2_2_1_fu_88_p0;
wire  signed [31:0] p_Val2_3_fu_89_p0;
wire  signed [31:0] p_Val2_0_1_fu_90_p0;
wire  signed [55:0] OP1_V_cast_fu_407_p1;
wire  signed [31:0] p_Val2_2_fu_91_p0;
wire  signed [31:0] p_Val2_1_fu_92_p0;
wire  signed [31:0] p_Val2_s_fu_93_p0;
wire   [55:0] p_Val2_s_fu_93_p2;
wire   [55:0] p_Val2_0_1_fu_90_p2;
wire   [55:0] p_Val2_1_fu_92_p2;
wire   [55:0] p_Val2_1_1_fu_86_p2;
wire  signed [31:0] OP1_V_2_cast7_fu_459_p0;
wire  signed [31:0] OP1_V_2_cast_fu_464_p0;
wire   [55:0] p_Val2_2_fu_91_p2;
wire   [50:0] p_Val2_2_1_fu_88_p2;
wire   [55:0] p_Val2_3_fu_89_p2;
wire   [55:0] p_Val2_3_1_fu_87_p2;
wire   [31:0] tmp3_fu_522_p2;
wire   [31:0] tmp2_fu_527_p2;
wire   [31:0] tmp1_fu_518_p2;
wire   [31:0] tmp6_fu_542_p2;
wire  signed [31:0] tmp_102_fu_515_p1;
wire   [31:0] tmp5_fu_547_p2;
wire   [31:0] tmp4_fu_538_p2;
wire   [31:0] res_0_V_write_assig_fu_532_p2;
wire   [31:0] res_1_V_write_assig_fu_553_p2;

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_ce))) begin
        tmp_101_reg_596 <= {{p_Val2_2_1_fu_88_p2[50:24]}};
        tmp_115_reg_571 <= {{p_Val2_s_fu_93_p2[55:24]}};
        tmp_123_0_1_reg_576 <= {{p_Val2_0_1_fu_90_p2[55:24]}};
        tmp_123_1_1_reg_586 <= {{p_Val2_1_1_fu_86_p2[55:24]}};
        tmp_123_1_reg_581 <= {{p_Val2_1_fu_92_p2[55:24]}};
        tmp_123_2_reg_591 <= {{p_Val2_2_fu_91_p2[55:24]}};
        tmp_123_3_1_reg_606 <= {{p_Val2_3_1_fu_87_p2[55:24]}};
        tmp_123_3_reg_601 <= {{p_Val2_3_fu_89_p2[55:24]}};
    end
end

assign OP1_V_1_cast_fu_433_p1 = $signed(data_1_V_read);

assign OP1_V_2_cast7_fu_459_p0 = data_2_V_read;

assign OP1_V_2_cast_fu_464_p0 = data_2_V_read;

assign OP1_V_3_cast_fu_489_p1 = $signed(data_3_V_read);

assign OP1_V_cast_fu_407_p1 = $signed(data_0_V_read);

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage0_11001 = ~(1'b1 == 1'b1);

assign ap_block_state1_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state2_pp0_stage0_iter1 = ~(1'b1 == 1'b1);

assign ap_return_0 = res_0_V_write_assig_fu_532_p2;

assign ap_return_1 = res_1_V_write_assig_fu_553_p2;

assign p_Val2_0_1_fu_90_p0 = OP1_V_cast_fu_407_p1;

assign p_Val2_0_1_fu_90_p2 = ($signed(p_Val2_0_1_fu_90_p0) * $signed('h1000285));

assign p_Val2_1_1_fu_86_p0 = OP1_V_1_cast_fu_433_p1;

assign p_Val2_1_1_fu_86_p2 = ($signed(p_Val2_1_1_fu_86_p0) * $signed(-56'hC2FCAB));

assign p_Val2_1_fu_92_p0 = OP1_V_1_cast_fu_433_p1;

assign p_Val2_1_fu_92_p2 = ($signed(p_Val2_1_fu_92_p0) * $signed(-56'hEF69D3));

assign p_Val2_2_1_fu_88_p0 = OP1_V_2_cast7_fu_459_p0;

assign p_Val2_2_1_fu_88_p2 = ($signed(p_Val2_2_1_fu_88_p0) * $signed('h7545F));

assign p_Val2_2_fu_91_p0 = OP1_V_2_cast_fu_464_p0;

assign p_Val2_2_fu_91_p2 = ($signed(p_Val2_2_fu_91_p0) * $signed('hBE5370));

assign p_Val2_3_1_fu_87_p0 = OP1_V_3_cast_fu_489_p1;

assign p_Val2_3_1_fu_87_p2 = ($signed(p_Val2_3_1_fu_87_p0) * $signed('h18A9781));

assign p_Val2_3_fu_89_p0 = OP1_V_3_cast_fu_489_p1;

assign p_Val2_3_fu_89_p2 = ($signed(p_Val2_3_fu_89_p0) * $signed(-56'hD643A3));

assign p_Val2_s_fu_93_p0 = OP1_V_cast_fu_407_p1;

assign p_Val2_s_fu_93_p2 = ($signed(p_Val2_s_fu_93_p0) * $signed('h156C271));

assign res_0_V_write_assig_fu_532_p2 = (tmp2_fu_527_p2 + tmp1_fu_518_p2);

assign res_1_V_write_assig_fu_553_p2 = (tmp5_fu_547_p2 + tmp4_fu_538_p2);

assign tmp1_fu_518_p2 = (tmp_115_reg_571 + tmp_123_1_reg_581);

assign tmp2_fu_527_p2 = (tmp3_fu_522_p2 + tmp_123_2_reg_591);

assign tmp3_fu_522_p2 = (tmp_123_3_reg_601 + 32'd981664);

assign tmp4_fu_538_p2 = (tmp_123_0_1_reg_576 + tmp_123_1_1_reg_586);

assign tmp5_fu_547_p2 = ($signed(tmp6_fu_542_p2) + $signed(tmp_102_fu_515_p1));

assign tmp6_fu_542_p2 = ($signed(tmp_123_3_1_reg_606) + $signed(32'd4293718549));

assign tmp_102_fu_515_p1 = $signed(tmp_101_reg_596);

endmodule //compute_layer_0_0_0
