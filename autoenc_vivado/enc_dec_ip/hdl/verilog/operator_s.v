// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.2.1
// Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module operator_s (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        snr_V_read,
        ap_return
);

parameter    ap_ST_fsm_state1 = 10'd1;
parameter    ap_ST_fsm_state2 = 10'd2;
parameter    ap_ST_fsm_state3 = 10'd4;
parameter    ap_ST_fsm_state4 = 10'd8;
parameter    ap_ST_fsm_state5 = 10'd16;
parameter    ap_ST_fsm_state6 = 10'd32;
parameter    ap_ST_fsm_state7 = 10'd64;
parameter    ap_ST_fsm_state8 = 10'd128;
parameter    ap_ST_fsm_state9 = 10'd256;
parameter    ap_ST_fsm_state10 = 10'd512;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [7:0] snr_V_read;
output  [31:0] ap_return;

reg ap_done;
reg ap_idle;
reg ap_ready;

(* fsm_encoding = "none" *) reg   [9:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg   [127:0] my_awgn_lfsr128_V;
wire   [8:0] coarseContents_address0;
reg    coarseContents_ce0;
wire   [16:0] coarseContents_q0;
wire   [8:0] gradientContents_address0;
reg    gradientContents_ce0;
wire   [12:0] gradientContents_q0;
wire   [7:0] scaleLookup_address0;
reg    scaleLookup_ce0;
wire   [16:0] scaleLookup_q0;
reg   [127:0] p_Val2_s_reg_1279;
wire   [2:0] i_1_fu_344_p2;
reg   [2:0] i_1_reg_1291;
wire    ap_CS_fsm_state2;
wire   [5:0] tmp_86_fu_366_p2;
reg   [5:0] tmp_86_reg_1296;
wire   [0:0] exitcond1_fu_338_p2;
wire   [1:0] tmp_217_fu_377_p1;
reg   [1:0] tmp_217_reg_1303;
wire   [127:0] r_V_9_fu_393_p2;
reg   [127:0] r_V_9_reg_1311;
reg   [0:0] tmp_218_reg_1317;
wire   [0:0] sel_tmp_fu_417_p2;
reg   [0:0] sel_tmp_reg_1326;
wire   [0:0] sel_tmp100_fu_423_p2;
reg   [0:0] sel_tmp100_reg_1334;
wire   [0:0] sel_tmp101_fu_429_p2;
reg   [0:0] sel_tmp101_reg_1341;
wire   [0:0] or_cond_fu_435_p2;
reg   [0:0] or_cond_reg_1352;
wire   [3:0] normStage_cast_fu_651_p1;
reg   [3:0] normStage_cast_reg_1363;
wire    ap_CS_fsm_state3;
wire   [2:0] normStage_1_fu_661_p2;
reg   [2:0] normStage_1_reg_1371;
wire   [0:0] exitcond_fu_655_p2;
wire   [2:0] op2_assign_2_fu_681_p2;
reg   [2:0] op2_assign_2_reg_1383;
wire   [9:0] tmp_219_fu_706_p1;
reg  signed [9:0] tmp_219_reg_1388;
reg   [16:0] coarseContents_load_reg_1406;
wire    ap_CS_fsm_state5;
reg   [12:0] gradientContents_loa_reg_1411;
wire   [0:0] sel_tmp41_fu_903_p2;
reg   [0:0] sel_tmp41_reg_1416;
wire  signed [22:0] r_V_12_fu_1240_p2;
reg  signed [22:0] r_V_12_reg_1423;
wire    ap_CS_fsm_state6;
wire   [28:0] noiseGen_3_V_fu_1012_p3;
wire    ap_CS_fsm_state7;
wire   [28:0] noiseGen_3_V_1_fu_1051_p3;
wire   [28:0] noiseGen_3_V_3_fu_1089_p3;
wire   [28:0] noiseGen_3_V_6_fu_1105_p3;
wire   [30:0] centralLimitNoise_V_fu_1149_p2;
reg   [30:0] centralLimitNoise_V_reg_1448;
wire    ap_CS_fsm_state8;
reg   [16:0] scale_V_reg_1453;
wire   [47:0] r_V_31_fu_1161_p2;
reg   [47:0] r_V_31_reg_1458;
wire    ap_CS_fsm_state9;
reg   [4:0] norm_V_address0;
reg    norm_V_ce0;
reg    norm_V_we0;
reg   [14:0] norm_V_d0;
wire   [14:0] norm_V_q0;
reg  signed [28:0] noiseGen_V_3_reg_239;
reg  signed [28:0] noiseGen_V_2_reg_251;
reg  signed [28:0] noiseGen_3_V_2_reg_263;
reg  signed [28:0] noiseGen_3_V_4_reg_275;
reg   [2:0] i_reg_287;
reg   [2:0] normStage_reg_298;
wire    ap_CS_fsm_state4;
wire   [63:0] tmp_86_cast_fu_372_p1;
wire   [63:0] tmp_47_fu_647_p1;
wire   [63:0] tmp_90_cast_fu_676_p1;
wire   [63:0] tmp_61_fu_709_p1;
wire   [63:0] tmp_92_cast_fu_828_p1;
wire   [0:0] phitmp4_fu_744_p2;
wire   [63:0] tmp_91_cast_fu_861_p1;
wire   [127:0] p_Result_s_fu_633_p3;
reg   [8:0] bramChapter_3_V_1_fu_140;
wire   [8:0] bramChapter_0_V_1_fu_489_p3;
wire   [8:0] bramChapter_3_V_9_fu_813_p3;
reg   [8:0] bramChapter_3_V_2_fu_144;
wire   [8:0] bramChapter_1_V_1_fu_481_p3;
wire   [8:0] bramChapter_3_V_8_fu_806_p3;
reg   [8:0] bramChapter_3_V_4_fu_148;
wire   [8:0] newSel3_fu_465_p3;
wire   [8:0] bramChapter_3_V_5_fu_792_p3;
reg   [8:0] bramChapter_3_V_fu_152;
wire   [8:0] newSel1_fu_449_p3;
wire   [8:0] bramChapter_3_V_3_fu_778_p3;
wire   [14:0] r_V_15_fu_870_p2;
wire   [4:0] tmp_85_fu_354_p3;
wire   [5:0] p_shl_cast_fu_362_p1;
wire   [5:0] tmp_51_cast_fu_350_p1;
wire   [6:0] op2_assign_fu_381_p3;
wire   [127:0] tmp_52_fu_389_p1;
wire   [8:0] newSel_fu_441_p3;
wire   [8:0] newSel2_fu_457_p3;
wire   [8:0] sel_tmp102_fu_473_p3;
wire   [29:0] tmp_78_fu_529_p4;
wire   [63:0] r_V_19_fu_538_p3;
wire   [63:0] lfsr1_V_fu_517_p4;
wire   [63:0] r_V_20_fu_546_p2;
wire   [28:0] r_V_s_fu_552_p4;
wire   [63:0] r_V_21_fu_562_p1;
wire   [63:0] r_V_22_fu_566_p2;
wire   [63:0] r_V_23_fu_572_p2;
wire   [5:0] tmp_214_fu_584_p1;
wire   [63:0] r_V_25_fu_587_p3;
wire   [63:0] lfsr2_V_fu_526_p1;
wire   [63:0] r_V_26_fu_595_p2;
wire   [50:0] r_V_6_fu_601_p4;
wire   [63:0] r_V_27_fu_611_p1;
wire   [63:0] r_V_28_fu_615_p2;
wire   [63:0] r_V_29_fu_621_p2;
wire   [63:0] r_V_24_fu_578_p2;
wire   [63:0] r_V_30_fu_627_p2;
wire   [5:0] tmp_64_cast_fu_667_p1;
wire   [5:0] tmp_90_fu_671_p2;
wire   [4:0] phitmp2_fu_687_p4;
wire   [8:0] grp_fu_321_p6;
wire   [8:0] tmp_58_fu_696_p1;
wire   [8:0] tmp_59_fu_700_p2;
wire   [3:0] op2_assign_2_cast_fu_715_p1;
wire   [3:0] r_V_14_fu_718_p2;
wire   [3:0] op2_assign_1_fu_724_p2;
wire   [14:0] op2_assign_1_cast_fu_730_p1;
wire   [14:0] tmp_68_fu_734_p2;
wire   [8:0] tmp_220_fu_740_p1;
wire   [3:0] op2_assign_3_fu_750_p2;
wire   [8:0] op2_assign_3_cast_fu_755_p1;
wire   [8:0] r_V_16_fu_759_p2;
wire   [8:0] bramChapter_3_V_10_fu_765_p2;
wire   [8:0] newSel15_fu_771_p3;
wire   [8:0] newSel16_fu_785_p3;
wire   [8:0] bramChapter_3_V_7_fu_799_p3;
wire   [5:0] tmp_77_cast_fu_820_p1;
wire   [5:0] tmp_92_fu_823_p2;
wire   [5:0] tmp_71_cast_fu_853_p1;
wire   [5:0] tmp_91_fu_856_p2;
wire   [14:0] tmp_72_cast_fu_866_p1;
wire   [0:0] sel_tmp36_fu_877_p2;
wire   [0:0] sel_tmp38_fu_882_p2;
wire   [0:0] sel_tmp40_fu_887_p2;
wire   [0:0] tmp89_fu_898_p2;
wire   [0:0] tmp88_fu_892_p2;
wire   [26:0] r_V_13_fu_918_p3;
wire  signed [28:0] tmp_62_fu_915_p1;
wire   [28:0] r_V_11_cast_fu_925_p1;
wire   [0:0] sel_tmp26_fu_941_p2;
wire   [0:0] sel_tmp47_fu_960_p2;
wire   [0:0] sel_tmp44_fu_956_p2;
wire   [28:0] noiseGen_0_V_fu_929_p2;
wire   [0:0] sel_tmp31_fu_951_p2;
wire   [0:0] sel_tmp27_fu_946_p2;
wire   [28:0] noiseGen_0_V_2_fu_935_p2;
wire   [0:0] or_cond2_fu_973_p2;
wire   [0:0] or_cond3_fu_986_p2;
wire   [28:0] newSel4_fu_979_p3;
wire   [28:0] newSel5_fu_991_p3;
wire   [0:0] or_cond4_fu_999_p2;
wire   [28:0] newSel6_fu_1004_p3;
wire   [28:0] newSel8_fu_1020_p3;
wire   [28:0] newSel9_fu_1028_p3;
wire   [28:0] newSel7_fu_1036_p3;
wire   [28:0] newSel10_fu_1043_p3;
wire   [28:0] newSel11_fu_1059_p3;
wire   [28:0] newSel12_fu_1067_p3;
wire   [28:0] newSel13_fu_1074_p3;
wire   [28:0] newSel14_fu_1081_p3;
wire   [0:0] sel_tmp51_fu_964_p2;
wire   [0:0] sel_tmp54_fu_969_p2;
wire   [28:0] noiseGen_3_V_5_fu_1097_p3;
wire  signed [29:0] p_8_cast_fu_1117_p1;
wire  signed [29:0] p_7_cast_fu_1113_p1;
wire   [29:0] tmp_fu_1129_p2;
wire  signed [29:0] tmp_cast_fu_1121_p1;
wire  signed [29:0] tmp_cast_58_fu_1125_p1;
wire   [29:0] tmp84_fu_1139_p2;
wire  signed [30:0] tmp184_cast_fu_1145_p1;
wire  signed [30:0] tmp183_cast_fu_1135_p1;
wire  signed [30:0] r_V_31_fu_1161_p0;
wire   [16:0] r_V_31_fu_1161_p1;
wire    ap_CS_fsm_state10;
wire   [48:0] tmp_48_fu_1167_p1;
wire   [48:0] r_V_fu_1170_p2;
wire   [3:0] tmp_216_fu_1186_p4;
wire   [34:0] roundedNoise_V_fu_1176_p4;
wire   [0:0] icmp_fu_1196_p2;
wire   [0:0] tmp_56_fu_1202_p2;
wire   [0:0] tmp_s_fu_1226_p2;
wire   [31:0] saturatedNoise_V_1_fu_1218_p3;
wire   [31:0] saturatedNoise_V_fu_1208_p4;
wire   [12:0] r_V_12_fu_1240_p0;
reg   [9:0] ap_NS_fsm;
wire   [22:0] r_V_12_fu_1240_p00;
wire   [47:0] r_V_31_fu_1161_p10;

// power-on initialization
initial begin
#0 ap_CS_fsm = 10'd1;
#0 my_awgn_lfsr128_V = 128'd1512366075204170930279365292653862640;
end

operator_s_coarsecud #(
    .DataWidth( 17 ),
    .AddressRange( 512 ),
    .AddressWidth( 9 ))
coarseContents_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(coarseContents_address0),
    .ce0(coarseContents_ce0),
    .q0(coarseContents_q0)
);

operator_s_gradiedEe #(
    .DataWidth( 13 ),
    .AddressRange( 512 ),
    .AddressWidth( 9 ))
gradientContents_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(gradientContents_address0),
    .ce0(gradientContents_ce0),
    .q0(gradientContents_q0)
);

operator_s_scaleLeOg #(
    .DataWidth( 17 ),
    .AddressRange( 256 ),
    .AddressWidth( 8 ))
scaleLookup_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(scaleLookup_address0),
    .ce0(scaleLookup_ce0),
    .q0(scaleLookup_q0)
);

operator_s_norm_V #(
    .DataWidth( 15 ),
    .AddressRange( 20 ),
    .AddressWidth( 5 ))
norm_V_U(
    .clk(ap_clk),
    .reset(ap_rst),
    .address0(norm_V_address0),
    .ce0(norm_V_ce0),
    .we0(norm_V_we0),
    .d0(norm_V_d0),
    .q0(norm_V_q0)
);

encoder_decoder_mfYi #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 9 ),
    .din1_WIDTH( 9 ),
    .din2_WIDTH( 9 ),
    .din3_WIDTH( 9 ),
    .din4_WIDTH( 2 ),
    .dout_WIDTH( 9 ))
encoder_decoder_mfYi_U17(
    .din0(bramChapter_3_V_1_fu_140),
    .din1(bramChapter_3_V_2_fu_144),
    .din2(bramChapter_3_V_4_fu_148),
    .din3(bramChapter_3_V_fu_152),
    .din4(tmp_217_reg_1303),
    .dout(grp_fu_321_p6)
);

encoder_decoder_mg8j #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 13 ),
    .din1_WIDTH( 10 ),
    .dout_WIDTH( 23 ))
encoder_decoder_mg8j_U18(
    .din0(r_V_12_fu_1240_p0),
    .din1(tmp_219_reg_1388),
    .dout(r_V_12_fu_1240_p2)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (((phitmp4_fu_744_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        bramChapter_3_V_1_fu_140 <= bramChapter_3_V_9_fu_813_p3;
    end else if (((exitcond1_fu_338_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        bramChapter_3_V_1_fu_140 <= bramChapter_0_V_1_fu_489_p3;
    end
end

always @ (posedge ap_clk) begin
    if (((phitmp4_fu_744_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        bramChapter_3_V_2_fu_144 <= bramChapter_3_V_8_fu_806_p3;
    end else if (((exitcond1_fu_338_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        bramChapter_3_V_2_fu_144 <= bramChapter_1_V_1_fu_481_p3;
    end
end

always @ (posedge ap_clk) begin
    if (((phitmp4_fu_744_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        bramChapter_3_V_4_fu_148 <= bramChapter_3_V_5_fu_792_p3;
    end else if (((exitcond1_fu_338_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        bramChapter_3_V_4_fu_148 <= newSel3_fu_465_p3;
    end
end

always @ (posedge ap_clk) begin
    if (((phitmp4_fu_744_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        bramChapter_3_V_fu_152 <= bramChapter_3_V_3_fu_778_p3;
    end else if (((exitcond1_fu_338_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        bramChapter_3_V_fu_152 <= newSel1_fu_449_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        i_reg_287 <= i_1_reg_1291;
    end else if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        i_reg_287 <= 3'd0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        normStage_reg_298 <= normStage_1_reg_1371;
    end else if (((exitcond1_fu_338_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        normStage_reg_298 <= 3'd0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state8)) begin
        centralLimitNoise_V_reg_1448 <= centralLimitNoise_V_fu_1149_p2;
        scale_V_reg_1453 <= scaleLookup_q0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        coarseContents_load_reg_1406 <= coarseContents_q0;
        gradientContents_loa_reg_1411 <= gradientContents_q0;
        sel_tmp41_reg_1416 <= sel_tmp41_fu_903_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        i_1_reg_1291 <= i_1_fu_344_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((exitcond1_fu_338_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state2))) begin
        my_awgn_lfsr128_V <= p_Result_s_fu_633_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        noiseGen_3_V_2_reg_263 <= noiseGen_3_V_3_fu_1089_p3;
        noiseGen_3_V_4_reg_275 <= noiseGen_3_V_6_fu_1105_p3;
        noiseGen_V_2_reg_251 <= noiseGen_3_V_1_fu_1051_p3;
        noiseGen_V_3_reg_239 <= noiseGen_3_V_fu_1012_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        normStage_1_reg_1371 <= normStage_1_fu_661_p2;
        normStage_cast_reg_1363[2 : 0] <= normStage_cast_fu_651_p1[2 : 0];
    end
end

always @ (posedge ap_clk) begin
    if (((exitcond_fu_655_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
        op2_assign_2_reg_1383 <= op2_assign_2_fu_681_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((exitcond1_fu_338_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        or_cond_reg_1352 <= or_cond_fu_435_p2;
        r_V_9_reg_1311 <= r_V_9_fu_393_p2;
        sel_tmp100_reg_1334 <= sel_tmp100_fu_423_p2;
        sel_tmp101_reg_1341 <= sel_tmp101_fu_429_p2;
        sel_tmp_reg_1326 <= sel_tmp_fu_417_p2;
        tmp_217_reg_1303 <= tmp_217_fu_377_p1;
        tmp_218_reg_1317 <= r_V_9_fu_393_p2[32'd31];
        tmp_86_reg_1296 <= tmp_86_fu_366_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        p_Val2_s_reg_1279 <= my_awgn_lfsr128_V;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        r_V_12_reg_1423 <= r_V_12_fu_1240_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state9)) begin
        r_V_31_reg_1458 <= r_V_31_fu_1161_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((exitcond_fu_655_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state3))) begin
        tmp_219_reg_1388 <= tmp_219_fu_706_p1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state10) | ((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1)))) begin
        ap_done = 1'b1;
    end else begin
        ap_done = 1'b0;
    end
end

always @ (*) begin
    if (((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1))) begin
        ap_idle = 1'b1;
    end else begin
        ap_idle = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state10)) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        coarseContents_ce0 = 1'b1;
    end else begin
        coarseContents_ce0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        gradientContents_ce0 = 1'b1;
    end else begin
        gradientContents_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((phitmp4_fu_744_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state4))) begin
        norm_V_address0 = tmp_91_cast_fu_861_p1;
    end else if (((phitmp4_fu_744_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        norm_V_address0 = tmp_92_cast_fu_828_p1;
    end else if ((1'b1 == ap_CS_fsm_state3)) begin
        norm_V_address0 = tmp_90_cast_fu_676_p1;
    end else if ((1'b1 == ap_CS_fsm_state2)) begin
        norm_V_address0 = tmp_86_cast_fu_372_p1;
    end else begin
        norm_V_address0 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2) | ((phitmp4_fu_744_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state4)) | ((phitmp4_fu_744_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4)))) begin
        norm_V_ce0 = 1'b1;
    end else begin
        norm_V_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((phitmp4_fu_744_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state4))) begin
        norm_V_d0 = r_V_15_fu_870_p2;
    end else if (((phitmp4_fu_744_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        norm_V_d0 = norm_V_q0;
    end else if ((1'b1 == ap_CS_fsm_state2)) begin
        norm_V_d0 = {{r_V_9_fu_393_p2[29:15]}};
    end else begin
        norm_V_d0 = 'bx;
    end
end

always @ (*) begin
    if ((((exitcond1_fu_338_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2)) | ((phitmp4_fu_744_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state4)) | ((phitmp4_fu_744_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4)))) begin
        norm_V_we0 = 1'b1;
    end else begin
        norm_V_we0 = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        scaleLookup_ce0 = 1'b1;
    end else begin
        scaleLookup_ce0 = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if (((exitcond1_fu_338_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end
        end
        ap_ST_fsm_state3 : begin
            if (((exitcond_fu_655_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state3))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state7;
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state2;
        end
        ap_ST_fsm_state8 : begin
            ap_NS_fsm = ap_ST_fsm_state9;
        end
        ap_ST_fsm_state9 : begin
            ap_NS_fsm = ap_ST_fsm_state10;
        end
        ap_ST_fsm_state10 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state10 = ap_CS_fsm[32'd9];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state7 = ap_CS_fsm[32'd6];

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd7];

assign ap_CS_fsm_state9 = ap_CS_fsm[32'd8];

assign ap_return = ((tmp_s_fu_1226_p2[0:0] === 1'b1) ? saturatedNoise_V_1_fu_1218_p3 : saturatedNoise_V_fu_1208_p4);

assign bramChapter_0_V_1_fu_489_p3 = ((sel_tmp101_fu_429_p2[0:0] === 1'b1) ? 9'd0 : bramChapter_3_V_1_fu_140);

assign bramChapter_1_V_1_fu_481_p3 = ((sel_tmp101_fu_429_p2[0:0] === 1'b1) ? bramChapter_3_V_2_fu_144 : sel_tmp102_fu_473_p3);

assign bramChapter_3_V_10_fu_765_p2 = (grp_fu_321_p6 + r_V_16_fu_759_p2);

assign bramChapter_3_V_3_fu_778_p3 = ((or_cond_reg_1352[0:0] === 1'b1) ? bramChapter_3_V_fu_152 : newSel15_fu_771_p3);

assign bramChapter_3_V_5_fu_792_p3 = ((or_cond_reg_1352[0:0] === 1'b1) ? bramChapter_3_V_4_fu_148 : newSel16_fu_785_p3);

assign bramChapter_3_V_7_fu_799_p3 = ((sel_tmp100_reg_1334[0:0] === 1'b1) ? bramChapter_3_V_10_fu_765_p2 : bramChapter_3_V_2_fu_144);

assign bramChapter_3_V_8_fu_806_p3 = ((sel_tmp101_reg_1341[0:0] === 1'b1) ? bramChapter_3_V_2_fu_144 : bramChapter_3_V_7_fu_799_p3);

assign bramChapter_3_V_9_fu_813_p3 = ((sel_tmp101_reg_1341[0:0] === 1'b1) ? bramChapter_3_V_10_fu_765_p2 : bramChapter_3_V_1_fu_140);

assign centralLimitNoise_V_fu_1149_p2 = ($signed(tmp184_cast_fu_1145_p1) + $signed(tmp183_cast_fu_1135_p1));

assign coarseContents_address0 = tmp_61_fu_709_p1;

assign exitcond1_fu_338_p2 = ((i_reg_287 == 3'd4) ? 1'b1 : 1'b0);

assign exitcond_fu_655_p2 = ((normStage_reg_298 == 3'd4) ? 1'b1 : 1'b0);

assign gradientContents_address0 = tmp_61_fu_709_p1;

assign i_1_fu_344_p2 = (i_reg_287 + 3'd1);

assign icmp_fu_1196_p2 = (($signed(tmp_216_fu_1186_p4) > $signed(4'd0)) ? 1'b1 : 1'b0);

assign lfsr1_V_fu_517_p4 = {{p_Val2_s_reg_1279[127:64]}};

assign lfsr2_V_fu_526_p1 = p_Val2_s_reg_1279[63:0];

assign newSel10_fu_1043_p3 = ((or_cond3_fu_986_p2[0:0] === 1'b1) ? noiseGen_V_2_reg_251 : newSel9_fu_1028_p3);

assign newSel11_fu_1059_p3 = ((sel_tmp47_fu_960_p2[0:0] === 1'b1) ? noiseGen_0_V_fu_929_p2 : noiseGen_3_V_2_reg_263);

assign newSel12_fu_1067_p3 = ((sel_tmp41_reg_1416[0:0] === 1'b1) ? noiseGen_3_V_2_reg_263 : noiseGen_0_V_2_fu_935_p2);

assign newSel13_fu_1074_p3 = ((sel_tmp101_reg_1341[0:0] === 1'b1) ? noiseGen_3_V_2_reg_263 : newSel11_fu_1059_p3);

assign newSel14_fu_1081_p3 = ((or_cond3_fu_986_p2[0:0] === 1'b1) ? newSel12_fu_1067_p3 : noiseGen_3_V_2_reg_263);

assign newSel15_fu_771_p3 = ((sel_tmp_reg_1326[0:0] === 1'b1) ? bramChapter_3_V_fu_152 : bramChapter_3_V_10_fu_765_p2);

assign newSel16_fu_785_p3 = ((sel_tmp_reg_1326[0:0] === 1'b1) ? bramChapter_3_V_10_fu_765_p2 : bramChapter_3_V_4_fu_148);

assign newSel1_fu_449_p3 = ((or_cond_fu_435_p2[0:0] === 1'b1) ? bramChapter_3_V_fu_152 : newSel_fu_441_p3);

assign newSel2_fu_457_p3 = ((sel_tmp_fu_417_p2[0:0] === 1'b1) ? 9'd0 : bramChapter_3_V_4_fu_148);

assign newSel3_fu_465_p3 = ((or_cond_fu_435_p2[0:0] === 1'b1) ? bramChapter_3_V_4_fu_148 : newSel2_fu_457_p3);

assign newSel4_fu_979_p3 = ((sel_tmp41_reg_1416[0:0] === 1'b1) ? noiseGen_0_V_fu_929_p2 : noiseGen_V_3_reg_239);

assign newSel5_fu_991_p3 = ((sel_tmp27_fu_946_p2[0:0] === 1'b1) ? noiseGen_V_3_reg_239 : noiseGen_0_V_2_fu_935_p2);

assign newSel6_fu_1004_p3 = ((or_cond3_fu_986_p2[0:0] === 1'b1) ? newSel4_fu_979_p3 : newSel5_fu_991_p3);

assign newSel7_fu_1036_p3 = ((sel_tmp101_reg_1341[0:0] === 1'b1) ? noiseGen_V_2_reg_251 : newSel8_fu_1020_p3);

assign newSel8_fu_1020_p3 = ((sel_tmp47_fu_960_p2[0:0] === 1'b1) ? noiseGen_V_2_reg_251 : noiseGen_0_V_fu_929_p2);

assign newSel9_fu_1028_p3 = ((sel_tmp27_fu_946_p2[0:0] === 1'b1) ? noiseGen_0_V_2_fu_935_p2 : noiseGen_V_2_reg_251);

assign newSel_fu_441_p3 = ((sel_tmp_fu_417_p2[0:0] === 1'b1) ? bramChapter_3_V_fu_152 : 9'd0);

assign noiseGen_0_V_2_fu_935_p2 = ($signed(r_V_11_cast_fu_925_p1) + $signed(tmp_62_fu_915_p1));

assign noiseGen_0_V_fu_929_p2 = ($signed(tmp_62_fu_915_p1) - $signed(r_V_11_cast_fu_925_p1));

assign noiseGen_3_V_1_fu_1051_p3 = ((or_cond4_fu_999_p2[0:0] === 1'b1) ? newSel7_fu_1036_p3 : newSel10_fu_1043_p3);

assign noiseGen_3_V_3_fu_1089_p3 = ((or_cond4_fu_999_p2[0:0] === 1'b1) ? newSel13_fu_1074_p3 : newSel14_fu_1081_p3);

assign noiseGen_3_V_5_fu_1097_p3 = ((sel_tmp51_fu_964_p2[0:0] === 1'b1) ? noiseGen_0_V_2_fu_935_p2 : noiseGen_3_V_4_reg_275);

assign noiseGen_3_V_6_fu_1105_p3 = ((sel_tmp54_fu_969_p2[0:0] === 1'b1) ? noiseGen_0_V_fu_929_p2 : noiseGen_3_V_5_fu_1097_p3);

assign noiseGen_3_V_fu_1012_p3 = ((or_cond4_fu_999_p2[0:0] === 1'b1) ? noiseGen_V_3_reg_239 : newSel6_fu_1004_p3);

assign normStage_1_fu_661_p2 = (normStage_reg_298 + 3'd1);

assign normStage_cast_fu_651_p1 = normStage_reg_298;

assign op2_assign_1_cast_fu_730_p1 = op2_assign_1_fu_724_p2;

assign op2_assign_1_fu_724_p2 = (r_V_14_fu_718_p2 ^ 4'd15);

assign op2_assign_2_cast_fu_715_p1 = op2_assign_2_reg_1383;

assign op2_assign_2_fu_681_p2 = (3'd3 - normStage_reg_298);

assign op2_assign_3_cast_fu_755_p1 = op2_assign_3_fu_750_p2;

assign op2_assign_3_fu_750_p2 = ($signed(4'd8) - $signed(normStage_cast_reg_1363));

assign op2_assign_fu_381_p3 = {{tmp_217_fu_377_p1}, {5'd0}};

assign or_cond2_fu_973_p2 = (sel_tmp47_fu_960_p2 | sel_tmp44_fu_956_p2);

assign or_cond3_fu_986_p2 = (sel_tmp41_reg_1416 | sel_tmp31_fu_951_p2);

assign or_cond4_fu_999_p2 = (sel_tmp101_reg_1341 | or_cond2_fu_973_p2);

assign or_cond_fu_435_p2 = (sel_tmp101_fu_429_p2 | sel_tmp100_fu_423_p2);

assign p_7_cast_fu_1113_p1 = noiseGen_3_V_4_reg_275;

assign p_8_cast_fu_1117_p1 = noiseGen_3_V_2_reg_263;

assign p_Result_s_fu_633_p3 = {{r_V_24_fu_578_p2}, {r_V_30_fu_627_p2}};

assign p_shl_cast_fu_362_p1 = tmp_85_fu_354_p3;

assign phitmp2_fu_687_p4 = {{r_V_9_reg_1311[14:10]}};

assign phitmp4_fu_744_p2 = ((tmp_220_fu_740_p1 == 9'd0) ? 1'b1 : 1'b0);

assign r_V_11_cast_fu_925_p1 = r_V_13_fu_918_p3;

assign r_V_12_fu_1240_p0 = r_V_12_fu_1240_p00;

assign r_V_12_fu_1240_p00 = gradientContents_loa_reg_1411;

assign r_V_13_fu_918_p3 = {{coarseContents_load_reg_1406}, {10'd0}};

assign r_V_14_fu_718_p2 = 4'd1 << op2_assign_2_cast_fu_715_p1;

assign r_V_15_fu_870_p2 = norm_V_q0 << tmp_72_cast_fu_866_p1;

assign r_V_16_fu_759_p2 = 9'd1 << op2_assign_3_cast_fu_755_p1;

assign r_V_19_fu_538_p3 = {{tmp_78_fu_529_p4}, {34'd0}};

assign r_V_20_fu_546_p2 = (r_V_19_fu_538_p3 ^ lfsr1_V_fu_517_p4);

assign r_V_21_fu_562_p1 = r_V_s_fu_552_p4;

assign r_V_22_fu_566_p2 = (r_V_21_fu_562_p1 ^ r_V_20_fu_546_p2);

assign r_V_23_fu_572_p2 = r_V_22_fu_566_p2 << 64'd1;

assign r_V_24_fu_578_p2 = (r_V_23_fu_572_p2 ^ r_V_22_fu_566_p2);

assign r_V_25_fu_587_p3 = {{tmp_214_fu_584_p1}, {58'd0}};

assign r_V_26_fu_595_p2 = (r_V_25_fu_587_p3 ^ lfsr2_V_fu_526_p1);

assign r_V_27_fu_611_p1 = r_V_6_fu_601_p4;

assign r_V_28_fu_615_p2 = (r_V_27_fu_611_p1 ^ r_V_26_fu_595_p2);

assign r_V_29_fu_621_p2 = r_V_28_fu_615_p2 << 64'd7;

assign r_V_30_fu_627_p2 = (r_V_29_fu_621_p2 ^ r_V_28_fu_615_p2);

assign r_V_31_fu_1161_p0 = centralLimitNoise_V_reg_1448;

assign r_V_31_fu_1161_p1 = r_V_31_fu_1161_p10;

assign r_V_31_fu_1161_p10 = scale_V_reg_1453;

assign r_V_31_fu_1161_p2 = ($signed(r_V_31_fu_1161_p0) * $signed({{1'b0}, {r_V_31_fu_1161_p1}}));

assign r_V_6_fu_601_p4 = {{r_V_26_fu_595_p2[63:13]}};

assign r_V_9_fu_393_p2 = p_Val2_s_reg_1279 >> tmp_52_fu_389_p1;

assign r_V_fu_1170_p2 = (49'd4096 + tmp_48_fu_1167_p1);

assign r_V_s_fu_552_p4 = {{r_V_20_fu_546_p2[63:35]}};

assign roundedNoise_V_fu_1176_p4 = {{r_V_fu_1170_p2[47:13]}};

assign saturatedNoise_V_1_fu_1218_p3 = ((icmp_fu_1196_p2[0:0] === 1'b1) ? 32'd2147483647 : 32'd2147483649);

assign saturatedNoise_V_fu_1208_p4 = {{r_V_fu_1170_p2[44:13]}};

assign scaleLookup_address0 = tmp_47_fu_647_p1;

assign sel_tmp100_fu_423_p2 = ((tmp_217_fu_377_p1 == 2'd1) ? 1'b1 : 1'b0);

assign sel_tmp101_fu_429_p2 = ((tmp_217_fu_377_p1 == 2'd0) ? 1'b1 : 1'b0);

assign sel_tmp102_fu_473_p3 = ((sel_tmp100_fu_423_p2[0:0] === 1'b1) ? 9'd0 : bramChapter_3_V_2_fu_144);

assign sel_tmp26_fu_941_p2 = (tmp_218_reg_1317 ^ 1'd1);

assign sel_tmp27_fu_946_p2 = (sel_tmp_reg_1326 & sel_tmp26_fu_941_p2);

assign sel_tmp31_fu_951_p2 = (sel_tmp26_fu_941_p2 & sel_tmp100_reg_1334);

assign sel_tmp36_fu_877_p2 = ((tmp_217_reg_1303 != 2'd0) ? 1'b1 : 1'b0);

assign sel_tmp38_fu_882_p2 = ((tmp_217_reg_1303 != 2'd1) ? 1'b1 : 1'b0);

assign sel_tmp40_fu_887_p2 = ((tmp_217_reg_1303 != 2'd2) ? 1'b1 : 1'b0);

assign sel_tmp41_fu_903_p2 = (tmp89_fu_898_p2 & tmp88_fu_892_p2);

assign sel_tmp44_fu_956_p2 = (tmp_218_reg_1317 & sel_tmp_reg_1326);

assign sel_tmp47_fu_960_p2 = (tmp_218_reg_1317 & sel_tmp100_reg_1334);

assign sel_tmp51_fu_964_p2 = (sel_tmp26_fu_941_p2 & sel_tmp101_reg_1341);

assign sel_tmp54_fu_969_p2 = (tmp_218_reg_1317 & sel_tmp101_reg_1341);

assign sel_tmp_fu_417_p2 = ((tmp_217_fu_377_p1 == 2'd2) ? 1'b1 : 1'b0);

assign tmp183_cast_fu_1135_p1 = $signed(tmp_fu_1129_p2);

assign tmp184_cast_fu_1145_p1 = $signed(tmp84_fu_1139_p2);

assign tmp84_fu_1139_p2 = ($signed(tmp_cast_fu_1121_p1) + $signed(tmp_cast_58_fu_1125_p1));

assign tmp88_fu_892_p2 = (sel_tmp38_fu_882_p2 & sel_tmp36_fu_877_p2);

assign tmp89_fu_898_p2 = (tmp_218_reg_1317 & sel_tmp40_fu_887_p2);

assign tmp_214_fu_584_p1 = p_Val2_s_reg_1279[5:0];

assign tmp_216_fu_1186_p4 = {{r_V_fu_1170_p2[47:44]}};

assign tmp_217_fu_377_p1 = i_reg_287[1:0];

assign tmp_219_fu_706_p1 = r_V_9_reg_1311[9:0];

assign tmp_220_fu_740_p1 = tmp_68_fu_734_p2[8:0];

assign tmp_47_fu_647_p1 = snr_V_read;

assign tmp_48_fu_1167_p1 = r_V_31_reg_1458;

assign tmp_51_cast_fu_350_p1 = i_reg_287;

assign tmp_52_fu_389_p1 = op2_assign_fu_381_p3;

assign tmp_56_fu_1202_p2 = (($signed(roundedNoise_V_fu_1176_p4) < $signed(35'd32212254721)) ? 1'b1 : 1'b0);

assign tmp_58_fu_696_p1 = phitmp2_fu_687_p4;

assign tmp_59_fu_700_p2 = (grp_fu_321_p6 + tmp_58_fu_696_p1);

assign tmp_61_fu_709_p1 = tmp_59_fu_700_p2;

assign tmp_62_fu_915_p1 = r_V_12_reg_1423;

assign tmp_64_cast_fu_667_p1 = normStage_reg_298;

assign tmp_68_fu_734_p2 = norm_V_q0 >> op2_assign_1_cast_fu_730_p1;

assign tmp_71_cast_fu_853_p1 = normStage_1_reg_1371;

assign tmp_72_cast_fu_866_p1 = r_V_14_fu_718_p2;

assign tmp_77_cast_fu_820_p1 = normStage_1_reg_1371;

assign tmp_78_fu_529_p4 = {{p_Val2_s_reg_1279[93:64]}};

assign tmp_85_fu_354_p3 = {{i_reg_287}, {2'd0}};

assign tmp_86_cast_fu_372_p1 = tmp_86_fu_366_p2;

assign tmp_86_fu_366_p2 = (p_shl_cast_fu_362_p1 + tmp_51_cast_fu_350_p1);

assign tmp_90_cast_fu_676_p1 = tmp_90_fu_671_p2;

assign tmp_90_fu_671_p2 = (tmp_86_reg_1296 + tmp_64_cast_fu_667_p1);

assign tmp_91_cast_fu_861_p1 = tmp_91_fu_856_p2;

assign tmp_91_fu_856_p2 = (tmp_86_reg_1296 + tmp_71_cast_fu_853_p1);

assign tmp_92_cast_fu_828_p1 = tmp_92_fu_823_p2;

assign tmp_92_fu_823_p2 = (tmp_86_reg_1296 + tmp_77_cast_fu_820_p1);

assign tmp_cast_58_fu_1125_p1 = noiseGen_V_3_reg_239;

assign tmp_cast_fu_1121_p1 = noiseGen_V_2_reg_251;

assign tmp_fu_1129_p2 = ($signed(p_8_cast_fu_1117_p1) + $signed(p_7_cast_fu_1113_p1));

assign tmp_s_fu_1226_p2 = (tmp_56_fu_1202_p2 | icmp_fu_1196_p2);

always @ (posedge ap_clk) begin
    normStage_cast_reg_1363[3] <= 1'b0;
end

endmodule //operator_s
