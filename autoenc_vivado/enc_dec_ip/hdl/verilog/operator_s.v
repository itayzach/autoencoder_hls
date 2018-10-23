// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.2
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
        awgn_32_lfsr128_V_read,
        snr_V_read,
        ap_return_0,
        ap_return_1
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
input  [127:0] awgn_32_lfsr128_V_read;
input  [7:0] snr_V_read;
output  [31:0] ap_return_0;
output  [127:0] ap_return_1;

reg ap_done;
reg ap_idle;
reg ap_ready;

(* fsm_encoding = "none" *) reg   [9:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire   [8:0] coarseContents_address0;
reg    coarseContents_ce0;
wire   [16:0] coarseContents_q0;
wire   [8:0] gradientContents_address0;
reg    gradientContents_ce0;
wire   [12:0] gradientContents_q0;
wire   [7:0] scaleLookup_address0;
reg    scaleLookup_ce0;
wire   [16:0] scaleLookup_q0;
wire   [2:0] i_1_fu_350_p2;
reg   [2:0] i_1_reg_1303;
wire    ap_CS_fsm_state2;
wire   [5:0] tmp_86_fu_372_p2;
reg   [5:0] tmp_86_reg_1308;
wire   [0:0] exitcond1_fu_344_p2;
wire   [1:0] tmp_217_fu_383_p1;
reg   [1:0] tmp_217_reg_1315;
wire   [127:0] r_V_9_fu_399_p2;
reg   [127:0] r_V_9_reg_1323;
reg   [0:0] tmp_218_reg_1329;
wire   [0:0] sel_tmp_fu_423_p2;
reg   [0:0] sel_tmp_reg_1338;
wire   [0:0] sel_tmp100_fu_429_p2;
reg   [0:0] sel_tmp100_reg_1346;
wire   [0:0] sel_tmp101_fu_435_p2;
reg   [0:0] sel_tmp101_reg_1353;
wire   [0:0] or_cond_fu_441_p2;
reg   [0:0] or_cond_reg_1364;
wire   [3:0] normStage_cast_fu_527_p1;
reg   [3:0] normStage_cast_reg_1375;
wire    ap_CS_fsm_state3;
wire   [2:0] normStage_1_fu_537_p2;
reg   [2:0] normStage_1_reg_1383;
wire   [0:0] exitcond_fu_531_p2;
wire   [2:0] op2_assign_2_fu_557_p2;
reg   [2:0] op2_assign_2_reg_1395;
wire   [9:0] tmp_219_fu_582_p1;
reg  signed [9:0] tmp_219_reg_1400;
reg   [16:0] coarseContents_load_reg_1418;
wire    ap_CS_fsm_state5;
reg   [12:0] gradientContents_loa_reg_1423;
wire   [0:0] sel_tmp41_fu_779_p2;
reg   [0:0] sel_tmp41_reg_1428;
wire  signed [22:0] r_V_12_fu_1252_p2;
reg  signed [22:0] r_V_12_reg_1435;
wire    ap_CS_fsm_state6;
wire   [28:0] noiseGen_3_V_fu_888_p3;
wire    ap_CS_fsm_state7;
wire   [28:0] noiseGen_3_V_1_fu_927_p3;
wire   [28:0] noiseGen_3_V_3_fu_965_p3;
wire   [28:0] noiseGen_3_V_6_fu_981_p3;
wire   [30:0] centralLimitNoise_V_fu_1025_p2;
reg   [30:0] centralLimitNoise_V_reg_1460;
wire    ap_CS_fsm_state8;
reg   [16:0] scale_V_reg_1465;
wire   [47:0] r_V_31_fu_1037_p2;
reg   [47:0] r_V_31_reg_1470;
wire    ap_CS_fsm_state9;
reg   [4:0] norm_V_address0;
reg    norm_V_ce0;
reg    norm_V_we0;
reg   [14:0] norm_V_d0;
wire   [14:0] norm_V_q0;
reg  signed [28:0] noiseGen_V_3_reg_249;
reg  signed [28:0] noiseGen_V_2_reg_261;
reg  signed [28:0] noiseGen_3_V_2_reg_273;
reg  signed [28:0] noiseGen_3_V_4_reg_285;
reg   [2:0] i_reg_297;
reg   [2:0] normStage_reg_308;
wire    ap_CS_fsm_state4;
wire   [63:0] tmp_86_cast_fu_378_p1;
wire   [63:0] tmp_47_fu_523_p1;
wire   [63:0] tmp_90_cast_fu_552_p1;
wire   [63:0] tmp_61_fu_585_p1;
wire   [63:0] tmp_92_cast_fu_704_p1;
wire   [0:0] phitmp4_fu_620_p2;
wire   [63:0] tmp_91_cast_fu_737_p1;
reg   [8:0] bramChapter_3_V_1_fu_144;
wire   [8:0] bramChapter_0_V_1_fu_495_p3;
wire   [8:0] bramChapter_3_V_9_fu_689_p3;
reg   [8:0] bramChapter_3_V_2_fu_148;
wire   [8:0] bramChapter_1_V_1_fu_487_p3;
wire   [8:0] bramChapter_3_V_8_fu_682_p3;
reg   [8:0] bramChapter_3_V_4_fu_152;
wire   [8:0] newSel3_fu_471_p3;
wire   [8:0] bramChapter_3_V_5_fu_668_p3;
reg   [8:0] bramChapter_3_V_fu_156;
wire   [8:0] newSel1_fu_455_p3;
wire   [8:0] bramChapter_3_V_3_fu_654_p3;
wire   [14:0] r_V_15_fu_746_p2;
wire   [4:0] tmp_85_fu_360_p3;
wire   [5:0] p_shl_cast_fu_368_p1;
wire   [5:0] tmp_51_cast_fu_356_p1;
wire   [6:0] op2_assign_fu_387_p3;
wire   [127:0] tmp_52_fu_395_p1;
wire   [8:0] newSel_fu_447_p3;
wire   [8:0] newSel2_fu_463_p3;
wire   [8:0] sel_tmp102_fu_479_p3;
wire   [5:0] tmp_64_cast_fu_543_p1;
wire   [5:0] tmp_90_fu_547_p2;
wire   [4:0] phitmp2_fu_563_p4;
wire   [8:0] grp_fu_331_p6;
wire   [8:0] tmp_58_fu_572_p1;
wire   [8:0] tmp_59_fu_576_p2;
wire   [3:0] op2_assign_2_cast_fu_591_p1;
wire   [3:0] r_V_14_fu_594_p2;
wire   [3:0] op2_assign_1_fu_600_p2;
wire   [14:0] op2_assign_1_cast_fu_606_p1;
wire   [14:0] tmp_68_fu_610_p2;
wire   [8:0] tmp_220_fu_616_p1;
wire   [3:0] op2_assign_3_fu_626_p2;
wire   [8:0] op2_assign_3_cast_fu_631_p1;
wire   [8:0] r_V_16_fu_635_p2;
wire   [8:0] bramChapter_3_V_10_fu_641_p2;
wire   [8:0] newSel15_fu_647_p3;
wire   [8:0] newSel16_fu_661_p3;
wire   [8:0] bramChapter_3_V_7_fu_675_p3;
wire   [5:0] tmp_77_cast_fu_696_p1;
wire   [5:0] tmp_92_fu_699_p2;
wire   [5:0] tmp_71_cast_fu_729_p1;
wire   [5:0] tmp_91_fu_732_p2;
wire   [14:0] tmp_72_cast_fu_742_p1;
wire   [0:0] sel_tmp36_fu_753_p2;
wire   [0:0] sel_tmp38_fu_758_p2;
wire   [0:0] sel_tmp40_fu_763_p2;
wire   [0:0] tmp89_fu_774_p2;
wire   [0:0] tmp88_fu_768_p2;
wire   [26:0] r_V_13_fu_794_p3;
wire  signed [28:0] tmp_62_fu_791_p1;
wire   [28:0] r_V_11_cast_fu_801_p1;
wire   [0:0] sel_tmp26_fu_817_p2;
wire   [0:0] sel_tmp47_fu_836_p2;
wire   [0:0] sel_tmp44_fu_832_p2;
wire   [28:0] noiseGen_0_V_fu_805_p2;
wire   [0:0] sel_tmp31_fu_827_p2;
wire   [0:0] sel_tmp27_fu_822_p2;
wire   [28:0] noiseGen_0_V_2_fu_811_p2;
wire   [0:0] or_cond2_fu_849_p2;
wire   [0:0] or_cond3_fu_862_p2;
wire   [28:0] newSel4_fu_855_p3;
wire   [28:0] newSel5_fu_867_p3;
wire   [0:0] or_cond4_fu_875_p2;
wire   [28:0] newSel6_fu_880_p3;
wire   [28:0] newSel8_fu_896_p3;
wire   [28:0] newSel9_fu_904_p3;
wire   [28:0] newSel7_fu_912_p3;
wire   [28:0] newSel10_fu_919_p3;
wire   [28:0] newSel11_fu_935_p3;
wire   [28:0] newSel12_fu_943_p3;
wire   [28:0] newSel13_fu_950_p3;
wire   [28:0] newSel14_fu_957_p3;
wire   [0:0] sel_tmp51_fu_840_p2;
wire   [0:0] sel_tmp54_fu_845_p2;
wire   [28:0] noiseGen_3_V_5_fu_973_p3;
wire  signed [29:0] p_8_cast_fu_993_p1;
wire  signed [29:0] p_7_cast_fu_989_p1;
wire   [29:0] tmp_fu_1005_p2;
wire  signed [29:0] tmp_cast_fu_997_p1;
wire  signed [29:0] tmp_cast_60_fu_1001_p1;
wire   [29:0] tmp84_fu_1015_p2;
wire  signed [30:0] tmp184_cast_fu_1021_p1;
wire  signed [30:0] tmp183_cast_fu_1011_p1;
wire  signed [30:0] r_V_31_fu_1037_p0;
wire   [16:0] r_V_31_fu_1037_p1;
wire    ap_CS_fsm_state10;
wire   [29:0] tmp_78_fu_1055_p4;
wire   [63:0] r_V_19_fu_1064_p3;
wire   [63:0] lfsr1_V_fu_1043_p4;
wire   [63:0] r_V_20_fu_1072_p2;
wire   [28:0] r_V_s_fu_1078_p4;
wire   [63:0] r_V_21_fu_1088_p1;
wire   [63:0] r_V_22_fu_1092_p2;
wire   [63:0] r_V_23_fu_1098_p2;
wire   [5:0] tmp_214_fu_1110_p1;
wire   [63:0] r_V_25_fu_1113_p3;
wire   [63:0] lfsr2_V_fu_1052_p1;
wire   [63:0] r_V_26_fu_1121_p2;
wire   [50:0] r_V_6_fu_1127_p4;
wire   [63:0] r_V_27_fu_1137_p1;
wire   [63:0] r_V_28_fu_1141_p2;
wire   [63:0] r_V_29_fu_1147_p2;
wire   [63:0] r_V_24_fu_1104_p2;
wire   [63:0] r_V_30_fu_1153_p2;
wire   [48:0] tmp_48_fu_1167_p1;
wire   [48:0] r_V_fu_1170_p2;
wire   [3:0] tmp_216_fu_1186_p4;
wire   [34:0] roundedNoise_V_fu_1176_p4;
wire   [0:0] icmp_fu_1196_p2;
wire   [0:0] tmp_56_fu_1202_p2;
wire   [0:0] tmp_s_fu_1226_p2;
wire   [31:0] saturatedNoise_V_1_fu_1218_p3;
wire   [31:0] saturatedNoise_V_fu_1208_p4;
wire   [31:0] ssdm_int_V_write_ass_fu_1232_p3;
wire   [127:0] p_Result_s_fu_1159_p3;
wire   [12:0] r_V_12_fu_1252_p0;
reg   [9:0] ap_NS_fsm;
wire   [22:0] r_V_12_fu_1252_p00;
wire   [47:0] r_V_31_fu_1037_p10;

// power-on initialization
initial begin
#0 ap_CS_fsm = 10'd1;
end

operator_s_coarsebkb #(
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

operator_s_gradiecud #(
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

operator_s_scaleLdEe #(
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

encoder_decoder_meOg #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 9 ),
    .din1_WIDTH( 9 ),
    .din2_WIDTH( 9 ),
    .din3_WIDTH( 9 ),
    .din4_WIDTH( 2 ),
    .dout_WIDTH( 9 ))
encoder_decoder_meOg_U1(
    .din0(bramChapter_3_V_1_fu_144),
    .din1(bramChapter_3_V_2_fu_148),
    .din2(bramChapter_3_V_4_fu_152),
    .din3(bramChapter_3_V_fu_156),
    .din4(tmp_217_reg_1315),
    .dout(grp_fu_331_p6)
);

encoder_decoder_mfYi #(
    .ID( 1 ),
    .NUM_STAGE( 1 ),
    .din0_WIDTH( 13 ),
    .din1_WIDTH( 10 ),
    .dout_WIDTH( 23 ))
encoder_decoder_mfYi_U2(
    .din0(r_V_12_fu_1252_p0),
    .din1(tmp_219_reg_1400),
    .dout(r_V_12_fu_1252_p2)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (((phitmp4_fu_620_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        bramChapter_3_V_1_fu_144 <= bramChapter_3_V_9_fu_689_p3;
    end else if (((exitcond1_fu_344_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        bramChapter_3_V_1_fu_144 <= bramChapter_0_V_1_fu_495_p3;
    end
end

always @ (posedge ap_clk) begin
    if (((phitmp4_fu_620_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        bramChapter_3_V_2_fu_148 <= bramChapter_3_V_8_fu_682_p3;
    end else if (((exitcond1_fu_344_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        bramChapter_3_V_2_fu_148 <= bramChapter_1_V_1_fu_487_p3;
    end
end

always @ (posedge ap_clk) begin
    if (((phitmp4_fu_620_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        bramChapter_3_V_4_fu_152 <= bramChapter_3_V_5_fu_668_p3;
    end else if (((exitcond1_fu_344_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        bramChapter_3_V_4_fu_152 <= newSel3_fu_471_p3;
    end
end

always @ (posedge ap_clk) begin
    if (((phitmp4_fu_620_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        bramChapter_3_V_fu_156 <= bramChapter_3_V_3_fu_654_p3;
    end else if (((exitcond1_fu_344_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        bramChapter_3_V_fu_156 <= newSel1_fu_455_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        i_reg_297 <= i_1_reg_1303;
    end else if (((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
        i_reg_297 <= 3'd0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state4)) begin
        normStage_reg_308 <= normStage_1_reg_1383;
    end else if (((exitcond1_fu_344_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        normStage_reg_308 <= 3'd0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state8)) begin
        centralLimitNoise_V_reg_1460 <= centralLimitNoise_V_fu_1025_p2;
        scale_V_reg_1465 <= scaleLookup_q0;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        coarseContents_load_reg_1418 <= coarseContents_q0;
        gradientContents_loa_reg_1423 <= gradientContents_q0;
        sel_tmp41_reg_1428 <= sel_tmp41_fu_779_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        i_1_reg_1303 <= i_1_fu_350_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state7)) begin
        noiseGen_3_V_2_reg_273 <= noiseGen_3_V_3_fu_965_p3;
        noiseGen_3_V_4_reg_285 <= noiseGen_3_V_6_fu_981_p3;
        noiseGen_V_2_reg_261 <= noiseGen_3_V_1_fu_927_p3;
        noiseGen_V_3_reg_249 <= noiseGen_3_V_fu_888_p3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        normStage_1_reg_1383 <= normStage_1_fu_537_p2;
        normStage_cast_reg_1375[2 : 0] <= normStage_cast_fu_527_p1[2 : 0];
    end
end

always @ (posedge ap_clk) begin
    if (((exitcond_fu_531_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state3))) begin
        op2_assign_2_reg_1395 <= op2_assign_2_fu_557_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((exitcond1_fu_344_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
        or_cond_reg_1364 <= or_cond_fu_441_p2;
        r_V_9_reg_1323 <= r_V_9_fu_399_p2;
        sel_tmp100_reg_1346 <= sel_tmp100_fu_429_p2;
        sel_tmp101_reg_1353 <= sel_tmp101_fu_435_p2;
        sel_tmp_reg_1338 <= sel_tmp_fu_423_p2;
        tmp_217_reg_1315 <= tmp_217_fu_383_p1;
        tmp_218_reg_1329 <= r_V_9_fu_399_p2[32'd31];
        tmp_86_reg_1308 <= tmp_86_fu_372_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state6)) begin
        r_V_12_reg_1435 <= r_V_12_fu_1252_p2;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state9)) begin
        r_V_31_reg_1470 <= r_V_31_fu_1037_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((exitcond_fu_531_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state3))) begin
        tmp_219_reg_1400 <= tmp_219_fu_582_p1;
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
    if (((phitmp4_fu_620_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state4))) begin
        norm_V_address0 = tmp_91_cast_fu_737_p1;
    end else if (((phitmp4_fu_620_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        norm_V_address0 = tmp_92_cast_fu_704_p1;
    end else if ((1'b1 == ap_CS_fsm_state3)) begin
        norm_V_address0 = tmp_90_cast_fu_552_p1;
    end else if ((1'b1 == ap_CS_fsm_state2)) begin
        norm_V_address0 = tmp_86_cast_fu_378_p1;
    end else begin
        norm_V_address0 = 'bx;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state3) | (1'b1 == ap_CS_fsm_state2) | ((phitmp4_fu_620_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state4)) | ((phitmp4_fu_620_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4)))) begin
        norm_V_ce0 = 1'b1;
    end else begin
        norm_V_ce0 = 1'b0;
    end
end

always @ (*) begin
    if (((phitmp4_fu_620_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state4))) begin
        norm_V_d0 = r_V_15_fu_746_p2;
    end else if (((phitmp4_fu_620_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4))) begin
        norm_V_d0 = norm_V_q0;
    end else if ((1'b1 == ap_CS_fsm_state2)) begin
        norm_V_d0 = {{r_V_9_fu_399_p2[29:15]}};
    end else begin
        norm_V_d0 = 'bx;
    end
end

always @ (*) begin
    if ((((exitcond1_fu_344_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2)) | ((phitmp4_fu_620_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state4)) | ((phitmp4_fu_620_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state4)))) begin
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
            if (((exitcond1_fu_344_p2 == 1'd0) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end
        end
        ap_ST_fsm_state3 : begin
            if (((exitcond_fu_531_p2 == 1'd1) & (1'b1 == ap_CS_fsm_state3))) begin
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

assign ap_return_0 = ssdm_int_V_write_ass_fu_1232_p3;

assign ap_return_1 = p_Result_s_fu_1159_p3;

assign bramChapter_0_V_1_fu_495_p3 = ((sel_tmp101_fu_435_p2[0:0] === 1'b1) ? 9'd0 : bramChapter_3_V_1_fu_144);

assign bramChapter_1_V_1_fu_487_p3 = ((sel_tmp101_fu_435_p2[0:0] === 1'b1) ? bramChapter_3_V_2_fu_148 : sel_tmp102_fu_479_p3);

assign bramChapter_3_V_10_fu_641_p2 = (grp_fu_331_p6 + r_V_16_fu_635_p2);

assign bramChapter_3_V_3_fu_654_p3 = ((or_cond_reg_1364[0:0] === 1'b1) ? bramChapter_3_V_fu_156 : newSel15_fu_647_p3);

assign bramChapter_3_V_5_fu_668_p3 = ((or_cond_reg_1364[0:0] === 1'b1) ? bramChapter_3_V_4_fu_152 : newSel16_fu_661_p3);

assign bramChapter_3_V_7_fu_675_p3 = ((sel_tmp100_reg_1346[0:0] === 1'b1) ? bramChapter_3_V_10_fu_641_p2 : bramChapter_3_V_2_fu_148);

assign bramChapter_3_V_8_fu_682_p3 = ((sel_tmp101_reg_1353[0:0] === 1'b1) ? bramChapter_3_V_2_fu_148 : bramChapter_3_V_7_fu_675_p3);

assign bramChapter_3_V_9_fu_689_p3 = ((sel_tmp101_reg_1353[0:0] === 1'b1) ? bramChapter_3_V_10_fu_641_p2 : bramChapter_3_V_1_fu_144);

assign centralLimitNoise_V_fu_1025_p2 = ($signed(tmp184_cast_fu_1021_p1) + $signed(tmp183_cast_fu_1011_p1));

assign coarseContents_address0 = tmp_61_fu_585_p1;

assign exitcond1_fu_344_p2 = ((i_reg_297 == 3'd4) ? 1'b1 : 1'b0);

assign exitcond_fu_531_p2 = ((normStage_reg_308 == 3'd4) ? 1'b1 : 1'b0);

assign gradientContents_address0 = tmp_61_fu_585_p1;

assign i_1_fu_350_p2 = (i_reg_297 + 3'd1);

assign icmp_fu_1196_p2 = (($signed(tmp_216_fu_1186_p4) > $signed(4'd0)) ? 1'b1 : 1'b0);

assign lfsr1_V_fu_1043_p4 = {{awgn_32_lfsr128_V_read[127:64]}};

assign lfsr2_V_fu_1052_p1 = awgn_32_lfsr128_V_read[63:0];

assign newSel10_fu_919_p3 = ((or_cond3_fu_862_p2[0:0] === 1'b1) ? noiseGen_V_2_reg_261 : newSel9_fu_904_p3);

assign newSel11_fu_935_p3 = ((sel_tmp47_fu_836_p2[0:0] === 1'b1) ? noiseGen_0_V_fu_805_p2 : noiseGen_3_V_2_reg_273);

assign newSel12_fu_943_p3 = ((sel_tmp41_reg_1428[0:0] === 1'b1) ? noiseGen_3_V_2_reg_273 : noiseGen_0_V_2_fu_811_p2);

assign newSel13_fu_950_p3 = ((sel_tmp101_reg_1353[0:0] === 1'b1) ? noiseGen_3_V_2_reg_273 : newSel11_fu_935_p3);

assign newSel14_fu_957_p3 = ((or_cond3_fu_862_p2[0:0] === 1'b1) ? newSel12_fu_943_p3 : noiseGen_3_V_2_reg_273);

assign newSel15_fu_647_p3 = ((sel_tmp_reg_1338[0:0] === 1'b1) ? bramChapter_3_V_fu_156 : bramChapter_3_V_10_fu_641_p2);

assign newSel16_fu_661_p3 = ((sel_tmp_reg_1338[0:0] === 1'b1) ? bramChapter_3_V_10_fu_641_p2 : bramChapter_3_V_4_fu_152);

assign newSel1_fu_455_p3 = ((or_cond_fu_441_p2[0:0] === 1'b1) ? bramChapter_3_V_fu_156 : newSel_fu_447_p3);

assign newSel2_fu_463_p3 = ((sel_tmp_fu_423_p2[0:0] === 1'b1) ? 9'd0 : bramChapter_3_V_4_fu_152);

assign newSel3_fu_471_p3 = ((or_cond_fu_441_p2[0:0] === 1'b1) ? bramChapter_3_V_4_fu_152 : newSel2_fu_463_p3);

assign newSel4_fu_855_p3 = ((sel_tmp41_reg_1428[0:0] === 1'b1) ? noiseGen_0_V_fu_805_p2 : noiseGen_V_3_reg_249);

assign newSel5_fu_867_p3 = ((sel_tmp27_fu_822_p2[0:0] === 1'b1) ? noiseGen_V_3_reg_249 : noiseGen_0_V_2_fu_811_p2);

assign newSel6_fu_880_p3 = ((or_cond3_fu_862_p2[0:0] === 1'b1) ? newSel4_fu_855_p3 : newSel5_fu_867_p3);

assign newSel7_fu_912_p3 = ((sel_tmp101_reg_1353[0:0] === 1'b1) ? noiseGen_V_2_reg_261 : newSel8_fu_896_p3);

assign newSel8_fu_896_p3 = ((sel_tmp47_fu_836_p2[0:0] === 1'b1) ? noiseGen_V_2_reg_261 : noiseGen_0_V_fu_805_p2);

assign newSel9_fu_904_p3 = ((sel_tmp27_fu_822_p2[0:0] === 1'b1) ? noiseGen_0_V_2_fu_811_p2 : noiseGen_V_2_reg_261);

assign newSel_fu_447_p3 = ((sel_tmp_fu_423_p2[0:0] === 1'b1) ? bramChapter_3_V_fu_156 : 9'd0);

assign noiseGen_0_V_2_fu_811_p2 = ($signed(r_V_11_cast_fu_801_p1) + $signed(tmp_62_fu_791_p1));

assign noiseGen_0_V_fu_805_p2 = ($signed(tmp_62_fu_791_p1) - $signed(r_V_11_cast_fu_801_p1));

assign noiseGen_3_V_1_fu_927_p3 = ((or_cond4_fu_875_p2[0:0] === 1'b1) ? newSel7_fu_912_p3 : newSel10_fu_919_p3);

assign noiseGen_3_V_3_fu_965_p3 = ((or_cond4_fu_875_p2[0:0] === 1'b1) ? newSel13_fu_950_p3 : newSel14_fu_957_p3);

assign noiseGen_3_V_5_fu_973_p3 = ((sel_tmp51_fu_840_p2[0:0] === 1'b1) ? noiseGen_0_V_2_fu_811_p2 : noiseGen_3_V_4_reg_285);

assign noiseGen_3_V_6_fu_981_p3 = ((sel_tmp54_fu_845_p2[0:0] === 1'b1) ? noiseGen_0_V_fu_805_p2 : noiseGen_3_V_5_fu_973_p3);

assign noiseGen_3_V_fu_888_p3 = ((or_cond4_fu_875_p2[0:0] === 1'b1) ? noiseGen_V_3_reg_249 : newSel6_fu_880_p3);

assign normStage_1_fu_537_p2 = (normStage_reg_308 + 3'd1);

assign normStage_cast_fu_527_p1 = normStage_reg_308;

assign op2_assign_1_cast_fu_606_p1 = op2_assign_1_fu_600_p2;

assign op2_assign_1_fu_600_p2 = (r_V_14_fu_594_p2 ^ 4'd15);

assign op2_assign_2_cast_fu_591_p1 = op2_assign_2_reg_1395;

assign op2_assign_2_fu_557_p2 = (3'd3 - normStage_reg_308);

assign op2_assign_3_cast_fu_631_p1 = op2_assign_3_fu_626_p2;

assign op2_assign_3_fu_626_p2 = ($signed(4'd8) - $signed(normStage_cast_reg_1375));

assign op2_assign_fu_387_p3 = {{tmp_217_fu_383_p1}, {5'd0}};

assign or_cond2_fu_849_p2 = (sel_tmp47_fu_836_p2 | sel_tmp44_fu_832_p2);

assign or_cond3_fu_862_p2 = (sel_tmp41_reg_1428 | sel_tmp31_fu_827_p2);

assign or_cond4_fu_875_p2 = (sel_tmp101_reg_1353 | or_cond2_fu_849_p2);

assign or_cond_fu_441_p2 = (sel_tmp101_fu_435_p2 | sel_tmp100_fu_429_p2);

assign p_7_cast_fu_989_p1 = noiseGen_3_V_4_reg_285;

assign p_8_cast_fu_993_p1 = noiseGen_3_V_2_reg_273;

assign p_Result_s_fu_1159_p3 = {{r_V_24_fu_1104_p2}, {r_V_30_fu_1153_p2}};

assign p_shl_cast_fu_368_p1 = tmp_85_fu_360_p3;

assign phitmp2_fu_563_p4 = {{r_V_9_reg_1323[14:10]}};

assign phitmp4_fu_620_p2 = ((tmp_220_fu_616_p1 == 9'd0) ? 1'b1 : 1'b0);

assign r_V_11_cast_fu_801_p1 = r_V_13_fu_794_p3;

assign r_V_12_fu_1252_p0 = r_V_12_fu_1252_p00;

assign r_V_12_fu_1252_p00 = gradientContents_loa_reg_1423;

assign r_V_13_fu_794_p3 = {{coarseContents_load_reg_1418}, {10'd0}};

assign r_V_14_fu_594_p2 = 4'd1 << op2_assign_2_cast_fu_591_p1;

assign r_V_15_fu_746_p2 = norm_V_q0 << tmp_72_cast_fu_742_p1;

assign r_V_16_fu_635_p2 = 9'd1 << op2_assign_3_cast_fu_631_p1;

assign r_V_19_fu_1064_p3 = {{tmp_78_fu_1055_p4}, {34'd0}};

assign r_V_20_fu_1072_p2 = (r_V_19_fu_1064_p3 ^ lfsr1_V_fu_1043_p4);

assign r_V_21_fu_1088_p1 = r_V_s_fu_1078_p4;

assign r_V_22_fu_1092_p2 = (r_V_21_fu_1088_p1 ^ r_V_20_fu_1072_p2);

assign r_V_23_fu_1098_p2 = r_V_22_fu_1092_p2 << 64'd1;

assign r_V_24_fu_1104_p2 = (r_V_23_fu_1098_p2 ^ r_V_22_fu_1092_p2);

assign r_V_25_fu_1113_p3 = {{tmp_214_fu_1110_p1}, {58'd0}};

assign r_V_26_fu_1121_p2 = (r_V_25_fu_1113_p3 ^ lfsr2_V_fu_1052_p1);

assign r_V_27_fu_1137_p1 = r_V_6_fu_1127_p4;

assign r_V_28_fu_1141_p2 = (r_V_27_fu_1137_p1 ^ r_V_26_fu_1121_p2);

assign r_V_29_fu_1147_p2 = r_V_28_fu_1141_p2 << 64'd7;

assign r_V_30_fu_1153_p2 = (r_V_29_fu_1147_p2 ^ r_V_28_fu_1141_p2);

assign r_V_31_fu_1037_p0 = centralLimitNoise_V_reg_1460;

assign r_V_31_fu_1037_p1 = r_V_31_fu_1037_p10;

assign r_V_31_fu_1037_p10 = scale_V_reg_1465;

assign r_V_31_fu_1037_p2 = ($signed(r_V_31_fu_1037_p0) * $signed({{1'b0}, {r_V_31_fu_1037_p1}}));

assign r_V_6_fu_1127_p4 = {{r_V_26_fu_1121_p2[63:13]}};

assign r_V_9_fu_399_p2 = awgn_32_lfsr128_V_read >> tmp_52_fu_395_p1;

assign r_V_fu_1170_p2 = (49'd4096 + tmp_48_fu_1167_p1);

assign r_V_s_fu_1078_p4 = {{r_V_20_fu_1072_p2[63:35]}};

assign roundedNoise_V_fu_1176_p4 = {{r_V_fu_1170_p2[47:13]}};

assign saturatedNoise_V_1_fu_1218_p3 = ((icmp_fu_1196_p2[0:0] === 1'b1) ? 32'd2147483647 : 32'd2147483649);

assign saturatedNoise_V_fu_1208_p4 = {{r_V_fu_1170_p2[44:13]}};

assign scaleLookup_address0 = tmp_47_fu_523_p1;

assign sel_tmp100_fu_429_p2 = ((tmp_217_fu_383_p1 == 2'd1) ? 1'b1 : 1'b0);

assign sel_tmp101_fu_435_p2 = ((tmp_217_fu_383_p1 == 2'd0) ? 1'b1 : 1'b0);

assign sel_tmp102_fu_479_p3 = ((sel_tmp100_fu_429_p2[0:0] === 1'b1) ? 9'd0 : bramChapter_3_V_2_fu_148);

assign sel_tmp26_fu_817_p2 = (tmp_218_reg_1329 ^ 1'd1);

assign sel_tmp27_fu_822_p2 = (sel_tmp_reg_1338 & sel_tmp26_fu_817_p2);

assign sel_tmp31_fu_827_p2 = (sel_tmp26_fu_817_p2 & sel_tmp100_reg_1346);

assign sel_tmp36_fu_753_p2 = ((tmp_217_reg_1315 != 2'd0) ? 1'b1 : 1'b0);

assign sel_tmp38_fu_758_p2 = ((tmp_217_reg_1315 != 2'd1) ? 1'b1 : 1'b0);

assign sel_tmp40_fu_763_p2 = ((tmp_217_reg_1315 != 2'd2) ? 1'b1 : 1'b0);

assign sel_tmp41_fu_779_p2 = (tmp89_fu_774_p2 & tmp88_fu_768_p2);

assign sel_tmp44_fu_832_p2 = (tmp_218_reg_1329 & sel_tmp_reg_1338);

assign sel_tmp47_fu_836_p2 = (tmp_218_reg_1329 & sel_tmp100_reg_1346);

assign sel_tmp51_fu_840_p2 = (sel_tmp26_fu_817_p2 & sel_tmp101_reg_1353);

assign sel_tmp54_fu_845_p2 = (tmp_218_reg_1329 & sel_tmp101_reg_1353);

assign sel_tmp_fu_423_p2 = ((tmp_217_fu_383_p1 == 2'd2) ? 1'b1 : 1'b0);

assign ssdm_int_V_write_ass_fu_1232_p3 = ((tmp_s_fu_1226_p2[0:0] === 1'b1) ? saturatedNoise_V_1_fu_1218_p3 : saturatedNoise_V_fu_1208_p4);

assign tmp183_cast_fu_1011_p1 = $signed(tmp_fu_1005_p2);

assign tmp184_cast_fu_1021_p1 = $signed(tmp84_fu_1015_p2);

assign tmp84_fu_1015_p2 = ($signed(tmp_cast_fu_997_p1) + $signed(tmp_cast_60_fu_1001_p1));

assign tmp88_fu_768_p2 = (sel_tmp38_fu_758_p2 & sel_tmp36_fu_753_p2);

assign tmp89_fu_774_p2 = (tmp_218_reg_1329 & sel_tmp40_fu_763_p2);

assign tmp_214_fu_1110_p1 = awgn_32_lfsr128_V_read[5:0];

assign tmp_216_fu_1186_p4 = {{r_V_fu_1170_p2[47:44]}};

assign tmp_217_fu_383_p1 = i_reg_297[1:0];

assign tmp_219_fu_582_p1 = r_V_9_reg_1323[9:0];

assign tmp_220_fu_616_p1 = tmp_68_fu_610_p2[8:0];

assign tmp_47_fu_523_p1 = snr_V_read;

assign tmp_48_fu_1167_p1 = r_V_31_reg_1470;

assign tmp_51_cast_fu_356_p1 = i_reg_297;

assign tmp_52_fu_395_p1 = op2_assign_fu_387_p3;

assign tmp_56_fu_1202_p2 = (($signed(roundedNoise_V_fu_1176_p4) < $signed(35'd32212254721)) ? 1'b1 : 1'b0);

assign tmp_58_fu_572_p1 = phitmp2_fu_563_p4;

assign tmp_59_fu_576_p2 = (grp_fu_331_p6 + tmp_58_fu_572_p1);

assign tmp_61_fu_585_p1 = tmp_59_fu_576_p2;

assign tmp_62_fu_791_p1 = r_V_12_reg_1435;

assign tmp_64_cast_fu_543_p1 = normStage_reg_308;

assign tmp_68_fu_610_p2 = norm_V_q0 >> op2_assign_1_cast_fu_606_p1;

assign tmp_71_cast_fu_729_p1 = normStage_1_reg_1383;

assign tmp_72_cast_fu_742_p1 = r_V_14_fu_594_p2;

assign tmp_77_cast_fu_696_p1 = normStage_1_reg_1383;

assign tmp_78_fu_1055_p4 = {{awgn_32_lfsr128_V_read[93:64]}};

assign tmp_85_fu_360_p3 = {{i_reg_297}, {2'd0}};

assign tmp_86_cast_fu_378_p1 = tmp_86_fu_372_p2;

assign tmp_86_fu_372_p2 = (p_shl_cast_fu_368_p1 + tmp_51_cast_fu_356_p1);

assign tmp_90_cast_fu_552_p1 = tmp_90_fu_547_p2;

assign tmp_90_fu_547_p2 = (tmp_86_reg_1308 + tmp_64_cast_fu_543_p1);

assign tmp_91_cast_fu_737_p1 = tmp_91_fu_732_p2;

assign tmp_91_fu_732_p2 = (tmp_86_reg_1308 + tmp_71_cast_fu_729_p1);

assign tmp_92_cast_fu_704_p1 = tmp_92_fu_699_p2;

assign tmp_92_fu_699_p2 = (tmp_86_reg_1308 + tmp_77_cast_fu_696_p1);

assign tmp_cast_60_fu_1001_p1 = noiseGen_V_3_reg_249;

assign tmp_cast_fu_997_p1 = noiseGen_V_2_reg_261;

assign tmp_fu_1005_p2 = ($signed(p_8_cast_fu_993_p1) + $signed(p_7_cast_fu_989_p1));

assign tmp_s_fu_1226_p2 = (tmp_56_fu_1202_p2 | icmp_fu_1196_p2);

always @ (posedge ap_clk) begin
    normStage_cast_reg_1375[3] <= 1'b0;
end

endmodule //operator_s
