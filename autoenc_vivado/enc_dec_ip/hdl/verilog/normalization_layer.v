// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.2.1
// Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module normalization_layer (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        data_0_V_read,
        data_1_V_read,
        ap_return
);

parameter    ap_ST_fsm_pp0_stage0 = 10'd1;
parameter    ap_ST_fsm_pp0_stage1 = 10'd2;
parameter    ap_ST_fsm_pp0_stage2 = 10'd4;
parameter    ap_ST_fsm_pp0_stage3 = 10'd8;
parameter    ap_ST_fsm_pp0_stage4 = 10'd16;
parameter    ap_ST_fsm_pp0_stage5 = 10'd32;
parameter    ap_ST_fsm_pp0_stage6 = 10'd64;
parameter    ap_ST_fsm_pp0_stage7 = 10'd128;
parameter    ap_ST_fsm_pp0_stage8 = 10'd256;
parameter    ap_ST_fsm_pp0_stage9 = 10'd512;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [31:0] data_0_V_read;
input  [31:0] data_1_V_read;
output  [63:0] ap_return;

reg ap_done;
reg ap_idle;
reg ap_ready;

(* fsm_encoding = "none" *) reg   [9:0] ap_CS_fsm;
wire    ap_CS_fsm_pp0_stage0;
reg    ap_enable_reg_pp0_iter0;
wire    ap_block_pp0_stage0;
reg    ap_enable_reg_pp0_iter1;
reg    ap_enable_reg_pp0_iter2;
reg    ap_enable_reg_pp0_iter3;
reg    ap_enable_reg_pp0_iter4;
reg    ap_enable_reg_pp0_iter5;
reg    ap_enable_reg_pp0_iter6;
reg    ap_enable_reg_pp0_iter7;
reg    ap_enable_reg_pp0_iter8;
reg    ap_idle_pp0;
wire    ap_CS_fsm_pp0_stage9;
wire    ap_block_state10_pp0_stage9_iter0;
wire    ap_block_state20_pp0_stage9_iter1;
wire    ap_block_state30_pp0_stage9_iter2;
wire    ap_block_state40_pp0_stage9_iter3;
wire    ap_block_state50_pp0_stage9_iter4;
wire    ap_block_state60_pp0_stage9_iter5;
wire    ap_block_state70_pp0_stage9_iter6;
wire    ap_block_state80_pp0_stage9_iter7;
wire    ap_block_pp0_stage9_11001;
reg  signed [31:0] data_0_V_read_4_reg_140;
reg    ap_block_state1_pp0_stage0_iter0;
wire    ap_block_state11_pp0_stage0_iter1;
wire    ap_block_state21_pp0_stage0_iter2;
wire    ap_block_state31_pp0_stage0_iter3;
wire    ap_block_state41_pp0_stage0_iter4;
wire    ap_block_state51_pp0_stage0_iter5;
wire    ap_block_state61_pp0_stage0_iter6;
wire    ap_block_state71_pp0_stage0_iter7;
wire    ap_block_state81_pp0_stage0_iter8;
reg    ap_block_pp0_stage0_11001;
reg  signed [31:0] data_0_V_read_4_reg_140_pp0_iter1_reg;
reg  signed [31:0] data_0_V_read_4_reg_140_pp0_iter2_reg;
reg   [31:0] data_square_0_V_reg_145;
reg  signed [31:0] data_1_V_read_4_reg_150;
wire    ap_CS_fsm_pp0_stage1;
wire    ap_block_state2_pp0_stage1_iter0;
wire    ap_block_state12_pp0_stage1_iter1;
wire    ap_block_state22_pp0_stage1_iter2;
wire    ap_block_state32_pp0_stage1_iter3;
wire    ap_block_state42_pp0_stage1_iter4;
wire    ap_block_state52_pp0_stage1_iter5;
wire    ap_block_state62_pp0_stage1_iter6;
wire    ap_block_state72_pp0_stage1_iter7;
wire    ap_block_state82_pp0_stage1_iter8;
wire    ap_block_pp0_stage1_11001;
reg  signed [31:0] data_1_V_read_4_reg_150_pp0_iter1_reg;
reg  signed [31:0] data_1_V_read_4_reg_150_pp0_iter2_reg;
reg   [31:0] data_square_1_V_reg_155;
wire   [31:0] p_Val2_66_1_fu_93_p2;
reg   [31:0] p_Val2_66_1_reg_160;
wire    ap_CS_fsm_pp0_stage2;
wire    ap_block_state3_pp0_stage2_iter0;
wire    ap_block_state13_pp0_stage2_iter1;
wire    ap_block_state23_pp0_stage2_iter2;
wire    ap_block_state33_pp0_stage2_iter3;
wire    ap_block_state43_pp0_stage2_iter4;
wire    ap_block_state53_pp0_stage2_iter5;
wire    ap_block_state63_pp0_stage2_iter6;
wire    ap_block_state73_pp0_stage2_iter7;
wire    ap_block_state83_pp0_stage2_iter8;
wire    ap_block_pp0_stage2_11001;
wire   [27:0] grp_sqrt_fixed_32_8_s_fu_48_ap_return;
reg   [27:0] sqrt_res_V_reg_165;
wire    ap_CS_fsm_pp0_stage3;
wire    ap_block_state4_pp0_stage3_iter0;
wire    ap_block_state14_pp0_stage3_iter1;
wire    ap_block_state24_pp0_stage3_iter2;
wire    ap_block_state34_pp0_stage3_iter3;
wire    ap_block_state44_pp0_stage3_iter4;
wire    ap_block_state54_pp0_stage3_iter5;
wire    ap_block_state64_pp0_stage3_iter6;
wire    ap_block_state74_pp0_stage3_iter7;
wire    ap_block_state84_pp0_stage3_iter8;
wire    ap_block_pp0_stage3_11001;
wire   [55:0] tmp_s_fu_97_p1;
reg   [55:0] tmp_s_reg_170;
wire    ap_CS_fsm_pp0_stage4;
wire    ap_block_state5_pp0_stage4_iter0;
wire    ap_block_state15_pp0_stage4_iter1;
wire    ap_block_state25_pp0_stage4_iter2;
wire    ap_block_state35_pp0_stage4_iter3;
wire    ap_block_state45_pp0_stage4_iter4;
wire    ap_block_state55_pp0_stage4_iter5;
wire    ap_block_state65_pp0_stage4_iter6;
wire    ap_block_state75_pp0_stage4_iter7;
wire    ap_block_state85_pp0_stage4_iter8;
wire    ap_block_pp0_stage4_11001;
wire    ap_CS_fsm_pp0_stage5;
wire    ap_block_state6_pp0_stage5_iter0;
wire    ap_block_state16_pp0_stage5_iter1;
wire    ap_block_state26_pp0_stage5_iter2;
wire    ap_block_state36_pp0_stage5_iter3;
wire    ap_block_state46_pp0_stage5_iter4;
wire    ap_block_state56_pp0_stage5_iter5;
wire    ap_block_state66_pp0_stage5_iter6;
wire    ap_block_state76_pp0_stage5_iter7;
wire    ap_block_pp0_stage5_11001;
wire   [31:0] grp_fu_107_p2;
reg   [31:0] tmp_96_reg_186;
reg    ap_enable_reg_pp0_iter0_reg;
wire    ap_block_pp0_stage9_subdone;
wire    ap_block_pp0_stage4_subdone;
reg   [31:0] ap_port_reg_data_1_V_read;
reg    grp_sqrt_fixed_32_8_s_fu_48_ap_ce;
wire    ap_block_state7_pp0_stage6_iter0_ignore_call12;
wire    ap_block_state17_pp0_stage6_iter1_ignore_call12;
wire    ap_block_state27_pp0_stage6_iter2_ignore_call12;
wire    ap_block_state37_pp0_stage6_iter3_ignore_call12;
wire    ap_block_state47_pp0_stage6_iter4_ignore_call12;
wire    ap_block_state57_pp0_stage6_iter5_ignore_call12;
wire    ap_block_state67_pp0_stage6_iter6_ignore_call12;
wire    ap_block_state77_pp0_stage6_iter7_ignore_call12;
wire    ap_block_pp0_stage6_11001;
wire    ap_block_state8_pp0_stage7_iter0_ignore_call12;
wire    ap_block_state18_pp0_stage7_iter1_ignore_call12;
wire    ap_block_state28_pp0_stage7_iter2_ignore_call12;
wire    ap_block_state38_pp0_stage7_iter3_ignore_call12;
wire    ap_block_state48_pp0_stage7_iter4_ignore_call12;
wire    ap_block_state58_pp0_stage7_iter5_ignore_call12;
wire    ap_block_state68_pp0_stage7_iter6_ignore_call12;
wire    ap_block_state78_pp0_stage7_iter7_ignore_call12;
wire    ap_block_pp0_stage7_11001;
wire    ap_block_state9_pp0_stage8_iter0_ignore_call12;
wire    ap_block_state19_pp0_stage8_iter1_ignore_call12;
wire    ap_block_state29_pp0_stage8_iter2_ignore_call12;
wire    ap_block_state39_pp0_stage8_iter3_ignore_call12;
wire    ap_block_state49_pp0_stage8_iter4_ignore_call12;
wire    ap_block_state59_pp0_stage8_iter5_ignore_call12;
wire    ap_block_state69_pp0_stage8_iter6_ignore_call12;
wire    ap_block_state79_pp0_stage8_iter7_ignore_call12;
wire    ap_block_pp0_stage8_11001;
wire    ap_CS_fsm_pp0_stage6;
wire    ap_CS_fsm_pp0_stage7;
wire    ap_CS_fsm_pp0_stage8;
wire    ap_block_pp0_stage3;
wire  signed [31:0] OP1_V_3_cast_fu_53_p0;
wire  signed [31:0] p_Val2_s_fu_57_p0;
wire  signed [55:0] OP1_V_3_cast_fu_53_p1;
wire  signed [31:0] p_Val2_s_fu_57_p1;
wire   [55:0] p_Val2_s_fu_57_p2;
wire  signed [31:0] OP1_V_3_1_cast_fu_73_p0;
wire    ap_block_pp0_stage1;
wire  signed [31:0] p_Val2_1_fu_77_p0;
wire  signed [55:0] OP1_V_3_1_cast_fu_73_p1;
wire  signed [31:0] p_Val2_1_fu_77_p1;
wire   [55:0] p_Val2_1_fu_77_p2;
wire    ap_block_pp0_stage2;
wire    ap_block_pp0_stage4;
wire   [55:0] grp_fu_107_p0;
wire   [28:0] grp_fu_107_p1;
wire    ap_block_pp0_stage5;
wire   [55:0] grp_fu_120_p0;
wire   [28:0] grp_fu_120_p1;
wire   [31:0] grp_fu_120_p2;
wire   [31:0] div_res_1_V_fu_128_p1;
wire   [31:0] div_res_0_V_fu_125_p1;
reg   [9:0] ap_NS_fsm;
reg    ap_block_pp0_stage0_subdone;
reg    ap_idle_pp0_1to8;
wire    ap_block_pp0_stage1_subdone;
wire    ap_block_pp0_stage2_subdone;
wire    ap_block_pp0_stage3_subdone;
reg    ap_idle_pp0_0to7;
reg    ap_reset_idle_pp0;
wire    ap_block_pp0_stage5_subdone;
wire    ap_block_state7_pp0_stage6_iter0;
wire    ap_block_state17_pp0_stage6_iter1;
wire    ap_block_state27_pp0_stage6_iter2;
wire    ap_block_state37_pp0_stage6_iter3;
wire    ap_block_state47_pp0_stage6_iter4;
wire    ap_block_state57_pp0_stage6_iter5;
wire    ap_block_state67_pp0_stage6_iter6;
wire    ap_block_state77_pp0_stage6_iter7;
wire    ap_block_pp0_stage6_subdone;
wire    ap_block_state8_pp0_stage7_iter0;
wire    ap_block_state18_pp0_stage7_iter1;
wire    ap_block_state28_pp0_stage7_iter2;
wire    ap_block_state38_pp0_stage7_iter3;
wire    ap_block_state48_pp0_stage7_iter4;
wire    ap_block_state58_pp0_stage7_iter5;
wire    ap_block_state68_pp0_stage7_iter6;
wire    ap_block_state78_pp0_stage7_iter7;
wire    ap_block_pp0_stage7_subdone;
wire    ap_block_state9_pp0_stage8_iter0;
wire    ap_block_state19_pp0_stage8_iter1;
wire    ap_block_state29_pp0_stage8_iter2;
wire    ap_block_state39_pp0_stage8_iter3;
wire    ap_block_state49_pp0_stage8_iter4;
wire    ap_block_state59_pp0_stage8_iter5;
wire    ap_block_state69_pp0_stage8_iter6;
wire    ap_block_state79_pp0_stage8_iter7;
wire    ap_block_pp0_stage8_subdone;
wire    ap_enable_pp0;
wire   [55:0] grp_fu_107_p10;

// power-on initialization
initial begin
#0 ap_CS_fsm = 10'd1;
#0 ap_enable_reg_pp0_iter1 = 1'b0;
#0 ap_enable_reg_pp0_iter2 = 1'b0;
#0 ap_enable_reg_pp0_iter3 = 1'b0;
#0 ap_enable_reg_pp0_iter4 = 1'b0;
#0 ap_enable_reg_pp0_iter5 = 1'b0;
#0 ap_enable_reg_pp0_iter6 = 1'b0;
#0 ap_enable_reg_pp0_iter7 = 1'b0;
#0 ap_enable_reg_pp0_iter8 = 1'b0;
#0 ap_enable_reg_pp0_iter0_reg = 1'b0;
end

sqrt_fixed_32_8_s grp_sqrt_fixed_32_8_s_fu_48(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .x_V(p_Val2_66_1_reg_160),
    .ap_return(grp_sqrt_fixed_32_8_s_fu_48_ap_return),
    .ap_ce(grp_sqrt_fixed_32_8_s_fu_48_ap_ce)
);

encoder_decoder_sbkb #(
    .ID( 1 ),
    .NUM_STAGE( 60 ),
    .din0_WIDTH( 56 ),
    .din1_WIDTH( 29 ),
    .dout_WIDTH( 32 ))
encoder_decoder_sbkb_U11(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(grp_fu_107_p0),
    .din1(grp_fu_107_p1),
    .ce(1'b1),
    .dout(grp_fu_107_p2)
);

encoder_decoder_sbkb #(
    .ID( 1 ),
    .NUM_STAGE( 60 ),
    .din0_WIDTH( 56 ),
    .din1_WIDTH( 29 ),
    .dout_WIDTH( 32 ))
encoder_decoder_sbkb_U12(
    .clk(ap_clk),
    .reset(ap_rst),
    .din0(grp_fu_120_p0),
    .din1(grp_fu_120_p1),
    .ce(1'b1),
    .dout(grp_fu_120_p2)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_pp0_stage0;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter0_reg <= 1'b0;
    end else begin
        if ((1'b1 == ap_CS_fsm_pp0_stage0)) begin
            ap_enable_reg_pp0_iter0_reg <= ap_start;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter1 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage9_subdone) & (1'b1 == ap_CS_fsm_pp0_stage9))) begin
            ap_enable_reg_pp0_iter1 <= ap_enable_reg_pp0_iter0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter2 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage9_subdone) & (1'b1 == ap_CS_fsm_pp0_stage9))) begin
            ap_enable_reg_pp0_iter2 <= ap_enable_reg_pp0_iter1;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter3 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage9_subdone) & (1'b1 == ap_CS_fsm_pp0_stage9))) begin
            ap_enable_reg_pp0_iter3 <= ap_enable_reg_pp0_iter2;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter4 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage9_subdone) & (1'b1 == ap_CS_fsm_pp0_stage9))) begin
            ap_enable_reg_pp0_iter4 <= ap_enable_reg_pp0_iter3;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter5 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage9_subdone) & (1'b1 == ap_CS_fsm_pp0_stage9))) begin
            ap_enable_reg_pp0_iter5 <= ap_enable_reg_pp0_iter4;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter6 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage9_subdone) & (1'b1 == ap_CS_fsm_pp0_stage9))) begin
            ap_enable_reg_pp0_iter6 <= ap_enable_reg_pp0_iter5;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter7 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage9_subdone) & (1'b1 == ap_CS_fsm_pp0_stage9))) begin
            ap_enable_reg_pp0_iter7 <= ap_enable_reg_pp0_iter6;
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_enable_reg_pp0_iter8 <= 1'b0;
    end else begin
        if (((1'b0 == ap_block_pp0_stage9_subdone) & (1'b1 == ap_CS_fsm_pp0_stage9))) begin
            ap_enable_reg_pp0_iter8 <= ap_enable_reg_pp0_iter7;
        end else if (((ap_enable_reg_pp0_iter7 == 1'b0) & (1'b0 == ap_block_pp0_stage4_subdone) & (1'b1 == ap_CS_fsm_pp0_stage4))) begin
            ap_enable_reg_pp0_iter8 <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        ap_port_reg_data_1_V_read <= data_1_V_read;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001))) begin
        data_0_V_read_4_reg_140 <= data_0_V_read;
        data_0_V_read_4_reg_140_pp0_iter1_reg <= data_0_V_read_4_reg_140;
        data_0_V_read_4_reg_140_pp0_iter2_reg <= data_0_V_read_4_reg_140_pp0_iter1_reg;
        data_square_0_V_reg_145 <= {{p_Val2_s_fu_57_p2[55:24]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_11001))) begin
        data_1_V_read_4_reg_150 <= ap_port_reg_data_1_V_read;
        data_1_V_read_4_reg_150_pp0_iter1_reg <= data_1_V_read_4_reg_150;
        data_1_V_read_4_reg_150_pp0_iter2_reg <= data_1_V_read_4_reg_150_pp0_iter1_reg;
        data_square_1_V_reg_155 <= {{p_Val2_1_fu_77_p2[55:24]}};
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001))) begin
        p_Val2_66_1_reg_160 <= p_Val2_66_1_fu_93_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage3) & (1'b0 == ap_block_pp0_stage3_11001))) begin
        sqrt_res_V_reg_165 <= grp_sqrt_fixed_32_8_s_fu_48_ap_return;
        tmp_96_reg_186 <= grp_fu_107_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_pp0_stage4) & (1'b0 == ap_block_pp0_stage4_11001))) begin
        tmp_s_reg_170[27 : 0] <= tmp_s_fu_97_p1[27 : 0];
    end
end

always @ (*) begin
    if ((((ap_enable_reg_pp0_iter8 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage4) & (1'b0 == ap_block_pp0_stage4_11001)) | ((1'b0 == ap_block_pp0_stage0) & (ap_start == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0)))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_pp0_stage0)) begin
        ap_enable_reg_pp0_iter0 = ap_start;
    end else begin
        ap_enable_reg_pp0_iter0 = ap_enable_reg_pp0_iter0_reg;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (ap_idle_pp0 == 1'b1) & (1'b1 == ap_CS_fsm_pp0_stage0))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter8 == 1'b0) & (ap_enable_reg_pp0_iter7 == 1'b0) & (ap_enable_reg_pp0_iter6 == 1'b0) & (ap_enable_reg_pp0_iter5 == 1'b0) & (ap_enable_reg_pp0_iter4 == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
        ap_idle_pp0 = 1'b1;
    end else begin
        ap_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter7 == 1'b0) & (ap_enable_reg_pp0_iter6 == 1'b0) & (ap_enable_reg_pp0_iter5 == 1'b0) & (ap_enable_reg_pp0_iter4 == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b0))) begin
        ap_idle_pp0_0to7 = 1'b1;
    end else begin
        ap_idle_pp0_0to7 = 1'b0;
    end
end

always @ (*) begin
    if (((ap_enable_reg_pp0_iter8 == 1'b0) & (ap_enable_reg_pp0_iter7 == 1'b0) & (ap_enable_reg_pp0_iter6 == 1'b0) & (ap_enable_reg_pp0_iter5 == 1'b0) & (ap_enable_reg_pp0_iter4 == 1'b0) & (ap_enable_reg_pp0_iter3 == 1'b0) & (ap_enable_reg_pp0_iter2 == 1'b0) & (ap_enable_reg_pp0_iter1 == 1'b0))) begin
        ap_idle_pp0_1to8 = 1'b1;
    end else begin
        ap_idle_pp0_1to8 = 1'b0;
    end
end

always @ (*) begin
    if (((1'b0 == ap_block_pp0_stage9_11001) & (1'b1 == ap_CS_fsm_pp0_stage9) & (ap_enable_reg_pp0_iter0 == 1'b1))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (ap_idle_pp0_0to7 == 1'b1))) begin
        ap_reset_idle_pp0 = 1'b1;
    end else begin
        ap_reset_idle_pp0 = 1'b0;
    end
end

always @ (*) begin
    if ((((1'b0 == ap_block_pp0_stage9_11001) & (1'b1 == ap_CS_fsm_pp0_stage9)) | ((1'b0 == ap_block_pp0_stage8_11001) & (1'b1 == ap_CS_fsm_pp0_stage8)) | ((1'b0 == ap_block_pp0_stage7_11001) & (1'b1 == ap_CS_fsm_pp0_stage7)) | ((1'b0 == ap_block_pp0_stage6_11001) & (1'b1 == ap_CS_fsm_pp0_stage6)) | ((1'b1 == ap_CS_fsm_pp0_stage0) & (1'b0 == ap_block_pp0_stage0_11001)) | ((1'b0 == ap_block_pp0_stage5_11001) & (1'b1 == ap_CS_fsm_pp0_stage5)) | ((1'b1 == ap_CS_fsm_pp0_stage4) & (1'b0 == ap_block_pp0_stage4_11001)) | ((1'b1 == ap_CS_fsm_pp0_stage3) & (1'b0 == ap_block_pp0_stage3_11001)) | ((1'b1 == ap_CS_fsm_pp0_stage2) & (1'b0 == ap_block_pp0_stage2_11001)) | ((1'b1 == ap_CS_fsm_pp0_stage1) & (1'b0 == ap_block_pp0_stage1_11001)))) begin
        grp_sqrt_fixed_32_8_s_fu_48_ap_ce = 1'b1;
    end else begin
        grp_sqrt_fixed_32_8_s_fu_48_ap_ce = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_pp0_stage0 : begin
            if ((~((ap_start == 1'b0) & (ap_idle_pp0_1to8 == 1'b1)) & (1'b0 == ap_block_pp0_stage0_subdone))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end
        end
        ap_ST_fsm_pp0_stage1 : begin
            if ((1'b0 == ap_block_pp0_stage1_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage1;
            end
        end
        ap_ST_fsm_pp0_stage2 : begin
            if ((1'b0 == ap_block_pp0_stage2_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage2;
            end
        end
        ap_ST_fsm_pp0_stage3 : begin
            if ((1'b0 == ap_block_pp0_stage3_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage3;
            end
        end
        ap_ST_fsm_pp0_stage4 : begin
            if (((ap_reset_idle_pp0 == 1'b0) & (1'b0 == ap_block_pp0_stage4_subdone))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage5;
            end else if (((1'b0 == ap_block_pp0_stage4_subdone) & (ap_reset_idle_pp0 == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage4;
            end
        end
        ap_ST_fsm_pp0_stage5 : begin
            if ((1'b0 == ap_block_pp0_stage5_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage6;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage5;
            end
        end
        ap_ST_fsm_pp0_stage6 : begin
            if ((1'b0 == ap_block_pp0_stage6_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage7;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage6;
            end
        end
        ap_ST_fsm_pp0_stage7 : begin
            if ((1'b0 == ap_block_pp0_stage7_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage8;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage7;
            end
        end
        ap_ST_fsm_pp0_stage8 : begin
            if ((1'b0 == ap_block_pp0_stage8_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage9;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage8;
            end
        end
        ap_ST_fsm_pp0_stage9 : begin
            if ((1'b0 == ap_block_pp0_stage9_subdone)) begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage0;
            end else begin
                ap_NS_fsm = ap_ST_fsm_pp0_stage9;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign OP1_V_3_1_cast_fu_73_p0 = ap_port_reg_data_1_V_read;

assign OP1_V_3_1_cast_fu_73_p1 = OP1_V_3_1_cast_fu_73_p0;

assign OP1_V_3_cast_fu_53_p0 = data_0_V_read;

assign OP1_V_3_cast_fu_53_p1 = OP1_V_3_cast_fu_53_p0;

assign ap_CS_fsm_pp0_stage0 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_pp0_stage1 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_pp0_stage2 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_pp0_stage3 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_pp0_stage4 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_pp0_stage5 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_pp0_stage6 = ap_CS_fsm[32'd6];

assign ap_CS_fsm_pp0_stage7 = ap_CS_fsm[32'd7];

assign ap_CS_fsm_pp0_stage8 = ap_CS_fsm[32'd8];

assign ap_CS_fsm_pp0_stage9 = ap_CS_fsm[32'd9];

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_pp0_stage0_11001 = ((ap_start == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b1));
end

always @ (*) begin
    ap_block_pp0_stage0_subdone = ((ap_start == 1'b0) & (ap_enable_reg_pp0_iter0 == 1'b1));
end

assign ap_block_pp0_stage1 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage1_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage1_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage2 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage2_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage2_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage3 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage3_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage3_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage4 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage4_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage4_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage5 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage5_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage5_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage6_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage6_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage7_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage7_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage8_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage8_subdone = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage9_11001 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage9_subdone = ~(1'b1 == 1'b1);

assign ap_block_state10_pp0_stage9_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state11_pp0_stage0_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state12_pp0_stage1_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state13_pp0_stage2_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state14_pp0_stage3_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state15_pp0_stage4_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state16_pp0_stage5_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state17_pp0_stage6_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state17_pp0_stage6_iter1_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state18_pp0_stage7_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state18_pp0_stage7_iter1_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state19_pp0_stage8_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state19_pp0_stage8_iter1_ignore_call12 = ~(1'b1 == 1'b1);

always @ (*) begin
    ap_block_state1_pp0_stage0_iter0 = (ap_start == 1'b0);
end

assign ap_block_state20_pp0_stage9_iter1 = ~(1'b1 == 1'b1);

assign ap_block_state21_pp0_stage0_iter2 = ~(1'b1 == 1'b1);

assign ap_block_state22_pp0_stage1_iter2 = ~(1'b1 == 1'b1);

assign ap_block_state23_pp0_stage2_iter2 = ~(1'b1 == 1'b1);

assign ap_block_state24_pp0_stage3_iter2 = ~(1'b1 == 1'b1);

assign ap_block_state25_pp0_stage4_iter2 = ~(1'b1 == 1'b1);

assign ap_block_state26_pp0_stage5_iter2 = ~(1'b1 == 1'b1);

assign ap_block_state27_pp0_stage6_iter2 = ~(1'b1 == 1'b1);

assign ap_block_state27_pp0_stage6_iter2_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state28_pp0_stage7_iter2 = ~(1'b1 == 1'b1);

assign ap_block_state28_pp0_stage7_iter2_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state29_pp0_stage8_iter2 = ~(1'b1 == 1'b1);

assign ap_block_state29_pp0_stage8_iter2_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state2_pp0_stage1_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state30_pp0_stage9_iter2 = ~(1'b1 == 1'b1);

assign ap_block_state31_pp0_stage0_iter3 = ~(1'b1 == 1'b1);

assign ap_block_state32_pp0_stage1_iter3 = ~(1'b1 == 1'b1);

assign ap_block_state33_pp0_stage2_iter3 = ~(1'b1 == 1'b1);

assign ap_block_state34_pp0_stage3_iter3 = ~(1'b1 == 1'b1);

assign ap_block_state35_pp0_stage4_iter3 = ~(1'b1 == 1'b1);

assign ap_block_state36_pp0_stage5_iter3 = ~(1'b1 == 1'b1);

assign ap_block_state37_pp0_stage6_iter3 = ~(1'b1 == 1'b1);

assign ap_block_state37_pp0_stage6_iter3_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state38_pp0_stage7_iter3 = ~(1'b1 == 1'b1);

assign ap_block_state38_pp0_stage7_iter3_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state39_pp0_stage8_iter3 = ~(1'b1 == 1'b1);

assign ap_block_state39_pp0_stage8_iter3_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state3_pp0_stage2_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state40_pp0_stage9_iter3 = ~(1'b1 == 1'b1);

assign ap_block_state41_pp0_stage0_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state42_pp0_stage1_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state43_pp0_stage2_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state44_pp0_stage3_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state45_pp0_stage4_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state46_pp0_stage5_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state47_pp0_stage6_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state47_pp0_stage6_iter4_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state48_pp0_stage7_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state48_pp0_stage7_iter4_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state49_pp0_stage8_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state49_pp0_stage8_iter4_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state4_pp0_stage3_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state50_pp0_stage9_iter4 = ~(1'b1 == 1'b1);

assign ap_block_state51_pp0_stage0_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state52_pp0_stage1_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state53_pp0_stage2_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state54_pp0_stage3_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state55_pp0_stage4_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state56_pp0_stage5_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state57_pp0_stage6_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state57_pp0_stage6_iter5_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state58_pp0_stage7_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state58_pp0_stage7_iter5_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state59_pp0_stage8_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state59_pp0_stage8_iter5_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state5_pp0_stage4_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state60_pp0_stage9_iter5 = ~(1'b1 == 1'b1);

assign ap_block_state61_pp0_stage0_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state62_pp0_stage1_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state63_pp0_stage2_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state64_pp0_stage3_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state65_pp0_stage4_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state66_pp0_stage5_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state67_pp0_stage6_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state67_pp0_stage6_iter6_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state68_pp0_stage7_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state68_pp0_stage7_iter6_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state69_pp0_stage8_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state69_pp0_stage8_iter6_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state6_pp0_stage5_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state70_pp0_stage9_iter6 = ~(1'b1 == 1'b1);

assign ap_block_state71_pp0_stage0_iter7 = ~(1'b1 == 1'b1);

assign ap_block_state72_pp0_stage1_iter7 = ~(1'b1 == 1'b1);

assign ap_block_state73_pp0_stage2_iter7 = ~(1'b1 == 1'b1);

assign ap_block_state74_pp0_stage3_iter7 = ~(1'b1 == 1'b1);

assign ap_block_state75_pp0_stage4_iter7 = ~(1'b1 == 1'b1);

assign ap_block_state76_pp0_stage5_iter7 = ~(1'b1 == 1'b1);

assign ap_block_state77_pp0_stage6_iter7 = ~(1'b1 == 1'b1);

assign ap_block_state77_pp0_stage6_iter7_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state78_pp0_stage7_iter7 = ~(1'b1 == 1'b1);

assign ap_block_state78_pp0_stage7_iter7_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state79_pp0_stage8_iter7 = ~(1'b1 == 1'b1);

assign ap_block_state79_pp0_stage8_iter7_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state7_pp0_stage6_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state7_pp0_stage6_iter0_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state80_pp0_stage9_iter7 = ~(1'b1 == 1'b1);

assign ap_block_state81_pp0_stage0_iter8 = ~(1'b1 == 1'b1);

assign ap_block_state82_pp0_stage1_iter8 = ~(1'b1 == 1'b1);

assign ap_block_state83_pp0_stage2_iter8 = ~(1'b1 == 1'b1);

assign ap_block_state84_pp0_stage3_iter8 = ~(1'b1 == 1'b1);

assign ap_block_state85_pp0_stage4_iter8 = ~(1'b1 == 1'b1);

assign ap_block_state8_pp0_stage7_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state8_pp0_stage7_iter0_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_block_state9_pp0_stage8_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state9_pp0_stage8_iter0_ignore_call12 = ~(1'b1 == 1'b1);

assign ap_enable_pp0 = (ap_idle_pp0 ^ 1'b1);

assign ap_return = {{div_res_1_V_fu_128_p1}, {div_res_0_V_fu_125_p1}};

assign div_res_0_V_fu_125_p1 = tmp_96_reg_186[31:0];

assign div_res_1_V_fu_128_p1 = grp_fu_120_p2[31:0];

assign grp_fu_107_p0 = {{data_0_V_read_4_reg_140_pp0_iter2_reg}, {24'd0}};

assign grp_fu_107_p1 = grp_fu_107_p10;

assign grp_fu_107_p10 = sqrt_res_V_reg_165;

assign grp_fu_120_p0 = {{data_1_V_read_4_reg_150_pp0_iter2_reg}, {24'd0}};

assign grp_fu_120_p1 = tmp_s_reg_170;

assign p_Val2_1_fu_77_p0 = OP1_V_3_1_cast_fu_73_p1;

assign p_Val2_1_fu_77_p1 = OP1_V_3_1_cast_fu_73_p1;

assign p_Val2_1_fu_77_p2 = ($signed(p_Val2_1_fu_77_p0) * $signed(p_Val2_1_fu_77_p1));

assign p_Val2_66_1_fu_93_p2 = (data_square_1_V_reg_155 + data_square_0_V_reg_145);

assign p_Val2_s_fu_57_p0 = OP1_V_3_cast_fu_53_p1;

assign p_Val2_s_fu_57_p1 = OP1_V_3_cast_fu_53_p1;

assign p_Val2_s_fu_57_p2 = ($signed(p_Val2_s_fu_57_p0) * $signed(p_Val2_s_fu_57_p1));

assign tmp_s_fu_97_p1 = sqrt_res_V_reg_165;

always @ (posedge ap_clk) begin
    tmp_s_reg_170[55:28] <= 28'b0000000000000000000000000000;
end

endmodule //normalization_layer
