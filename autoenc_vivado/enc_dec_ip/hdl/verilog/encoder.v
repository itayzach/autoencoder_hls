// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.2
// Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module encoder (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        data_V_read,
        ap_return
);

parameter    ap_ST_fsm_state1 = 91'd1;
parameter    ap_ST_fsm_state2 = 91'd2;
parameter    ap_ST_fsm_state3 = 91'd4;
parameter    ap_ST_fsm_state4 = 91'd8;
parameter    ap_ST_fsm_state5 = 91'd16;
parameter    ap_ST_fsm_state6 = 91'd32;
parameter    ap_ST_fsm_state7 = 91'd64;
parameter    ap_ST_fsm_state8 = 91'd128;
parameter    ap_ST_fsm_state9 = 91'd256;
parameter    ap_ST_fsm_state10 = 91'd512;
parameter    ap_ST_fsm_state11 = 91'd1024;
parameter    ap_ST_fsm_state12 = 91'd2048;
parameter    ap_ST_fsm_state13 = 91'd4096;
parameter    ap_ST_fsm_state14 = 91'd8192;
parameter    ap_ST_fsm_state15 = 91'd16384;
parameter    ap_ST_fsm_state16 = 91'd32768;
parameter    ap_ST_fsm_state17 = 91'd65536;
parameter    ap_ST_fsm_state18 = 91'd131072;
parameter    ap_ST_fsm_state19 = 91'd262144;
parameter    ap_ST_fsm_state20 = 91'd524288;
parameter    ap_ST_fsm_state21 = 91'd1048576;
parameter    ap_ST_fsm_state22 = 91'd2097152;
parameter    ap_ST_fsm_state23 = 91'd4194304;
parameter    ap_ST_fsm_state24 = 91'd8388608;
parameter    ap_ST_fsm_state25 = 91'd16777216;
parameter    ap_ST_fsm_state26 = 91'd33554432;
parameter    ap_ST_fsm_state27 = 91'd67108864;
parameter    ap_ST_fsm_state28 = 91'd134217728;
parameter    ap_ST_fsm_state29 = 91'd268435456;
parameter    ap_ST_fsm_state30 = 91'd536870912;
parameter    ap_ST_fsm_state31 = 91'd1073741824;
parameter    ap_ST_fsm_state32 = 91'd2147483648;
parameter    ap_ST_fsm_state33 = 91'd4294967296;
parameter    ap_ST_fsm_state34 = 91'd8589934592;
parameter    ap_ST_fsm_state35 = 91'd17179869184;
parameter    ap_ST_fsm_state36 = 91'd34359738368;
parameter    ap_ST_fsm_state37 = 91'd68719476736;
parameter    ap_ST_fsm_state38 = 91'd137438953472;
parameter    ap_ST_fsm_state39 = 91'd274877906944;
parameter    ap_ST_fsm_state40 = 91'd549755813888;
parameter    ap_ST_fsm_state41 = 91'd1099511627776;
parameter    ap_ST_fsm_state42 = 91'd2199023255552;
parameter    ap_ST_fsm_state43 = 91'd4398046511104;
parameter    ap_ST_fsm_state44 = 91'd8796093022208;
parameter    ap_ST_fsm_state45 = 91'd17592186044416;
parameter    ap_ST_fsm_state46 = 91'd35184372088832;
parameter    ap_ST_fsm_state47 = 91'd70368744177664;
parameter    ap_ST_fsm_state48 = 91'd140737488355328;
parameter    ap_ST_fsm_state49 = 91'd281474976710656;
parameter    ap_ST_fsm_state50 = 91'd562949953421312;
parameter    ap_ST_fsm_state51 = 91'd1125899906842624;
parameter    ap_ST_fsm_state52 = 91'd2251799813685248;
parameter    ap_ST_fsm_state53 = 91'd4503599627370496;
parameter    ap_ST_fsm_state54 = 91'd9007199254740992;
parameter    ap_ST_fsm_state55 = 91'd18014398509481984;
parameter    ap_ST_fsm_state56 = 91'd36028797018963968;
parameter    ap_ST_fsm_state57 = 91'd72057594037927936;
parameter    ap_ST_fsm_state58 = 91'd144115188075855872;
parameter    ap_ST_fsm_state59 = 91'd288230376151711744;
parameter    ap_ST_fsm_state60 = 91'd576460752303423488;
parameter    ap_ST_fsm_state61 = 91'd1152921504606846976;
parameter    ap_ST_fsm_state62 = 91'd2305843009213693952;
parameter    ap_ST_fsm_state63 = 91'd4611686018427387904;
parameter    ap_ST_fsm_state64 = 91'd9223372036854775808;
parameter    ap_ST_fsm_state65 = 91'd18446744073709551616;
parameter    ap_ST_fsm_state66 = 91'd36893488147419103232;
parameter    ap_ST_fsm_state67 = 91'd73786976294838206464;
parameter    ap_ST_fsm_state68 = 91'd147573952589676412928;
parameter    ap_ST_fsm_state69 = 91'd295147905179352825856;
parameter    ap_ST_fsm_state70 = 91'd590295810358705651712;
parameter    ap_ST_fsm_state71 = 91'd1180591620717411303424;
parameter    ap_ST_fsm_state72 = 91'd2361183241434822606848;
parameter    ap_ST_fsm_state73 = 91'd4722366482869645213696;
parameter    ap_ST_fsm_state74 = 91'd9444732965739290427392;
parameter    ap_ST_fsm_state75 = 91'd18889465931478580854784;
parameter    ap_ST_fsm_state76 = 91'd37778931862957161709568;
parameter    ap_ST_fsm_state77 = 91'd75557863725914323419136;
parameter    ap_ST_fsm_state78 = 91'd151115727451828646838272;
parameter    ap_ST_fsm_state79 = 91'd302231454903657293676544;
parameter    ap_ST_fsm_state80 = 91'd604462909807314587353088;
parameter    ap_ST_fsm_state81 = 91'd1208925819614629174706176;
parameter    ap_ST_fsm_state82 = 91'd2417851639229258349412352;
parameter    ap_ST_fsm_state83 = 91'd4835703278458516698824704;
parameter    ap_ST_fsm_state84 = 91'd9671406556917033397649408;
parameter    ap_ST_fsm_state85 = 91'd19342813113834066795298816;
parameter    ap_ST_fsm_state86 = 91'd38685626227668133590597632;
parameter    ap_ST_fsm_state87 = 91'd77371252455336267181195264;
parameter    ap_ST_fsm_state88 = 91'd154742504910672534362390528;
parameter    ap_ST_fsm_state89 = 91'd309485009821345068724781056;
parameter    ap_ST_fsm_state90 = 91'd618970019642690137449562112;
parameter    ap_ST_fsm_state91 = 91'd1237940039285380274899124224;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [127:0] data_V_read;
output  [63:0] ap_return;

reg ap_done;
reg ap_idle;
reg ap_ready;

(* fsm_encoding = "none" *) reg   [90:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
reg   [31:0] logits1_0_V_reg_96;
wire    ap_CS_fsm_state2;
reg   [31:0] logits1_1_V_reg_101;
reg   [31:0] logits1_2_V_reg_106;
reg   [31:0] logits1_3_V_reg_111;
reg   [31:0] layer1_relu_out_0_V_reg_116;
wire    ap_CS_fsm_state3;
reg   [31:0] layer1_relu_out_1_V_reg_121;
reg   [31:0] layer1_relu_out_2_V_reg_126;
reg   [31:0] layer1_relu_out_3_V_reg_131;
reg   [31:0] logits2_0_V_reg_136;
wire    ap_CS_fsm_state5;
reg   [31:0] logits2_1_V_reg_141;
wire    grp_normalization_layer_fu_28_ap_start;
wire    grp_normalization_layer_fu_28_ap_done;
wire    grp_normalization_layer_fu_28_ap_idle;
wire    grp_normalization_layer_fu_28_ap_ready;
wire   [63:0] grp_normalization_layer_fu_28_ap_return;
wire   [31:0] grp_compute_layer_0_0_0_s_fu_34_ap_return_0;
wire   [31:0] grp_compute_layer_0_0_0_s_fu_34_ap_return_1;
wire   [31:0] grp_compute_layer_0_0_0_s_fu_34_ap_return_2;
wire   [31:0] grp_compute_layer_0_0_0_s_fu_34_ap_return_3;
reg    grp_compute_layer_0_0_0_s_fu_34_ap_ce;
wire   [31:0] grp_compute_layer_0_0_0_fu_40_ap_return_0;
wire   [31:0] grp_compute_layer_0_0_0_fu_40_ap_return_1;
reg    grp_compute_layer_0_0_0_fu_40_ap_ce;
wire    ap_CS_fsm_state4;
wire    call_ret1_relu_fu_48_ap_ready;
wire   [31:0] call_ret1_relu_fu_48_ap_return_0;
wire   [31:0] call_ret1_relu_fu_48_ap_return_1;
wire   [31:0] call_ret1_relu_fu_48_ap_return_2;
wire   [31:0] call_ret1_relu_fu_48_ap_return_3;
reg    grp_normalization_layer_fu_28_ap_start_reg;
reg   [90:0] ap_NS_fsm;
wire    ap_NS_fsm_state6;
wire    ap_CS_fsm_state6;
wire    ap_CS_fsm_state91;

// power-on initialization
initial begin
#0 ap_CS_fsm = 91'd1;
#0 grp_normalization_layer_fu_28_ap_start_reg = 1'b0;
end

normalization_layer grp_normalization_layer_fu_28(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_normalization_layer_fu_28_ap_start),
    .ap_done(grp_normalization_layer_fu_28_ap_done),
    .ap_idle(grp_normalization_layer_fu_28_ap_idle),
    .ap_ready(grp_normalization_layer_fu_28_ap_ready),
    .data_0_V_read(logits2_0_V_reg_136),
    .data_1_V_read(logits2_1_V_reg_141),
    .ap_return(grp_normalization_layer_fu_28_ap_return)
);

compute_layer_0_0_0_s grp_compute_layer_0_0_0_s_fu_34(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .data_V_read(data_V_read),
    .ap_return_0(grp_compute_layer_0_0_0_s_fu_34_ap_return_0),
    .ap_return_1(grp_compute_layer_0_0_0_s_fu_34_ap_return_1),
    .ap_return_2(grp_compute_layer_0_0_0_s_fu_34_ap_return_2),
    .ap_return_3(grp_compute_layer_0_0_0_s_fu_34_ap_return_3),
    .ap_ce(grp_compute_layer_0_0_0_s_fu_34_ap_ce)
);

compute_layer_0_0_0 grp_compute_layer_0_0_0_fu_40(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .data_0_V_read(layer1_relu_out_0_V_reg_116),
    .data_1_V_read(layer1_relu_out_1_V_reg_121),
    .data_2_V_read(layer1_relu_out_2_V_reg_126),
    .data_3_V_read(layer1_relu_out_3_V_reg_131),
    .ap_return_0(grp_compute_layer_0_0_0_fu_40_ap_return_0),
    .ap_return_1(grp_compute_layer_0_0_0_fu_40_ap_return_1),
    .ap_ce(grp_compute_layer_0_0_0_fu_40_ap_ce)
);

relu call_ret1_relu_fu_48(
    .ap_ready(call_ret1_relu_fu_48_ap_ready),
    .data_0_V_read(logits1_0_V_reg_96),
    .data_1_V_read(logits1_1_V_reg_101),
    .data_2_V_read(logits1_2_V_reg_106),
    .data_3_V_read(logits1_3_V_reg_111),
    .ap_return_0(call_ret1_relu_fu_48_ap_return_0),
    .ap_return_1(call_ret1_relu_fu_48_ap_return_1),
    .ap_return_2(call_ret1_relu_fu_48_ap_return_2),
    .ap_return_3(call_ret1_relu_fu_48_ap_return_3)
);

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        ap_CS_fsm <= ap_ST_fsm_state1;
    end else begin
        ap_CS_fsm <= ap_NS_fsm;
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_normalization_layer_fu_28_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b1 == ap_NS_fsm_state6) & (1'b1 == ap_CS_fsm_state5))) begin
            grp_normalization_layer_fu_28_ap_start_reg <= 1'b1;
        end else if ((grp_normalization_layer_fu_28_ap_ready == 1'b1)) begin
            grp_normalization_layer_fu_28_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state3)) begin
        layer1_relu_out_0_V_reg_116 <= call_ret1_relu_fu_48_ap_return_0;
        layer1_relu_out_1_V_reg_121 <= call_ret1_relu_fu_48_ap_return_1;
        layer1_relu_out_2_V_reg_126 <= call_ret1_relu_fu_48_ap_return_2;
        layer1_relu_out_3_V_reg_131 <= call_ret1_relu_fu_48_ap_return_3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state2)) begin
        logits1_0_V_reg_96 <= grp_compute_layer_0_0_0_s_fu_34_ap_return_0;
        logits1_1_V_reg_101 <= grp_compute_layer_0_0_0_s_fu_34_ap_return_1;
        logits1_2_V_reg_106 <= grp_compute_layer_0_0_0_s_fu_34_ap_return_2;
        logits1_3_V_reg_111 <= grp_compute_layer_0_0_0_s_fu_34_ap_return_3;
    end
end

always @ (posedge ap_clk) begin
    if ((1'b1 == ap_CS_fsm_state5)) begin
        logits2_0_V_reg_136 <= grp_compute_layer_0_0_0_fu_40_ap_return_0;
        logits2_1_V_reg_141 <= grp_compute_layer_0_0_0_fu_40_ap_return_1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state91) | ((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1)))) begin
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
    if ((1'b1 == ap_CS_fsm_state91)) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state4) | (1'b1 == ap_CS_fsm_state5))) begin
        grp_compute_layer_0_0_0_fu_40_ap_ce = 1'b1;
    end else begin
        grp_compute_layer_0_0_0_fu_40_ap_ce = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_CS_fsm_state2) | ((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1)))) begin
        grp_compute_layer_0_0_0_s_fu_34_ap_ce = 1'b1;
    end else begin
        grp_compute_layer_0_0_0_s_fu_34_ap_ce = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((ap_start == 1'b1) & (1'b1 == ap_CS_fsm_state1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            ap_NS_fsm = ap_ST_fsm_state3;
        end
        ap_ST_fsm_state3 : begin
            ap_NS_fsm = ap_ST_fsm_state4;
        end
        ap_ST_fsm_state4 : begin
            ap_NS_fsm = ap_ST_fsm_state5;
        end
        ap_ST_fsm_state5 : begin
            ap_NS_fsm = ap_ST_fsm_state6;
        end
        ap_ST_fsm_state6 : begin
            ap_NS_fsm = ap_ST_fsm_state7;
        end
        ap_ST_fsm_state7 : begin
            ap_NS_fsm = ap_ST_fsm_state8;
        end
        ap_ST_fsm_state8 : begin
            ap_NS_fsm = ap_ST_fsm_state9;
        end
        ap_ST_fsm_state9 : begin
            ap_NS_fsm = ap_ST_fsm_state10;
        end
        ap_ST_fsm_state10 : begin
            ap_NS_fsm = ap_ST_fsm_state11;
        end
        ap_ST_fsm_state11 : begin
            ap_NS_fsm = ap_ST_fsm_state12;
        end
        ap_ST_fsm_state12 : begin
            ap_NS_fsm = ap_ST_fsm_state13;
        end
        ap_ST_fsm_state13 : begin
            ap_NS_fsm = ap_ST_fsm_state14;
        end
        ap_ST_fsm_state14 : begin
            ap_NS_fsm = ap_ST_fsm_state15;
        end
        ap_ST_fsm_state15 : begin
            ap_NS_fsm = ap_ST_fsm_state16;
        end
        ap_ST_fsm_state16 : begin
            ap_NS_fsm = ap_ST_fsm_state17;
        end
        ap_ST_fsm_state17 : begin
            ap_NS_fsm = ap_ST_fsm_state18;
        end
        ap_ST_fsm_state18 : begin
            ap_NS_fsm = ap_ST_fsm_state19;
        end
        ap_ST_fsm_state19 : begin
            ap_NS_fsm = ap_ST_fsm_state20;
        end
        ap_ST_fsm_state20 : begin
            ap_NS_fsm = ap_ST_fsm_state21;
        end
        ap_ST_fsm_state21 : begin
            ap_NS_fsm = ap_ST_fsm_state22;
        end
        ap_ST_fsm_state22 : begin
            ap_NS_fsm = ap_ST_fsm_state23;
        end
        ap_ST_fsm_state23 : begin
            ap_NS_fsm = ap_ST_fsm_state24;
        end
        ap_ST_fsm_state24 : begin
            ap_NS_fsm = ap_ST_fsm_state25;
        end
        ap_ST_fsm_state25 : begin
            ap_NS_fsm = ap_ST_fsm_state26;
        end
        ap_ST_fsm_state26 : begin
            ap_NS_fsm = ap_ST_fsm_state27;
        end
        ap_ST_fsm_state27 : begin
            ap_NS_fsm = ap_ST_fsm_state28;
        end
        ap_ST_fsm_state28 : begin
            ap_NS_fsm = ap_ST_fsm_state29;
        end
        ap_ST_fsm_state29 : begin
            ap_NS_fsm = ap_ST_fsm_state30;
        end
        ap_ST_fsm_state30 : begin
            ap_NS_fsm = ap_ST_fsm_state31;
        end
        ap_ST_fsm_state31 : begin
            ap_NS_fsm = ap_ST_fsm_state32;
        end
        ap_ST_fsm_state32 : begin
            ap_NS_fsm = ap_ST_fsm_state33;
        end
        ap_ST_fsm_state33 : begin
            ap_NS_fsm = ap_ST_fsm_state34;
        end
        ap_ST_fsm_state34 : begin
            ap_NS_fsm = ap_ST_fsm_state35;
        end
        ap_ST_fsm_state35 : begin
            ap_NS_fsm = ap_ST_fsm_state36;
        end
        ap_ST_fsm_state36 : begin
            ap_NS_fsm = ap_ST_fsm_state37;
        end
        ap_ST_fsm_state37 : begin
            ap_NS_fsm = ap_ST_fsm_state38;
        end
        ap_ST_fsm_state38 : begin
            ap_NS_fsm = ap_ST_fsm_state39;
        end
        ap_ST_fsm_state39 : begin
            ap_NS_fsm = ap_ST_fsm_state40;
        end
        ap_ST_fsm_state40 : begin
            ap_NS_fsm = ap_ST_fsm_state41;
        end
        ap_ST_fsm_state41 : begin
            ap_NS_fsm = ap_ST_fsm_state42;
        end
        ap_ST_fsm_state42 : begin
            ap_NS_fsm = ap_ST_fsm_state43;
        end
        ap_ST_fsm_state43 : begin
            ap_NS_fsm = ap_ST_fsm_state44;
        end
        ap_ST_fsm_state44 : begin
            ap_NS_fsm = ap_ST_fsm_state45;
        end
        ap_ST_fsm_state45 : begin
            ap_NS_fsm = ap_ST_fsm_state46;
        end
        ap_ST_fsm_state46 : begin
            ap_NS_fsm = ap_ST_fsm_state47;
        end
        ap_ST_fsm_state47 : begin
            ap_NS_fsm = ap_ST_fsm_state48;
        end
        ap_ST_fsm_state48 : begin
            ap_NS_fsm = ap_ST_fsm_state49;
        end
        ap_ST_fsm_state49 : begin
            ap_NS_fsm = ap_ST_fsm_state50;
        end
        ap_ST_fsm_state50 : begin
            ap_NS_fsm = ap_ST_fsm_state51;
        end
        ap_ST_fsm_state51 : begin
            ap_NS_fsm = ap_ST_fsm_state52;
        end
        ap_ST_fsm_state52 : begin
            ap_NS_fsm = ap_ST_fsm_state53;
        end
        ap_ST_fsm_state53 : begin
            ap_NS_fsm = ap_ST_fsm_state54;
        end
        ap_ST_fsm_state54 : begin
            ap_NS_fsm = ap_ST_fsm_state55;
        end
        ap_ST_fsm_state55 : begin
            ap_NS_fsm = ap_ST_fsm_state56;
        end
        ap_ST_fsm_state56 : begin
            ap_NS_fsm = ap_ST_fsm_state57;
        end
        ap_ST_fsm_state57 : begin
            ap_NS_fsm = ap_ST_fsm_state58;
        end
        ap_ST_fsm_state58 : begin
            ap_NS_fsm = ap_ST_fsm_state59;
        end
        ap_ST_fsm_state59 : begin
            ap_NS_fsm = ap_ST_fsm_state60;
        end
        ap_ST_fsm_state60 : begin
            ap_NS_fsm = ap_ST_fsm_state61;
        end
        ap_ST_fsm_state61 : begin
            ap_NS_fsm = ap_ST_fsm_state62;
        end
        ap_ST_fsm_state62 : begin
            ap_NS_fsm = ap_ST_fsm_state63;
        end
        ap_ST_fsm_state63 : begin
            ap_NS_fsm = ap_ST_fsm_state64;
        end
        ap_ST_fsm_state64 : begin
            ap_NS_fsm = ap_ST_fsm_state65;
        end
        ap_ST_fsm_state65 : begin
            ap_NS_fsm = ap_ST_fsm_state66;
        end
        ap_ST_fsm_state66 : begin
            ap_NS_fsm = ap_ST_fsm_state67;
        end
        ap_ST_fsm_state67 : begin
            ap_NS_fsm = ap_ST_fsm_state68;
        end
        ap_ST_fsm_state68 : begin
            ap_NS_fsm = ap_ST_fsm_state69;
        end
        ap_ST_fsm_state69 : begin
            ap_NS_fsm = ap_ST_fsm_state70;
        end
        ap_ST_fsm_state70 : begin
            ap_NS_fsm = ap_ST_fsm_state71;
        end
        ap_ST_fsm_state71 : begin
            ap_NS_fsm = ap_ST_fsm_state72;
        end
        ap_ST_fsm_state72 : begin
            ap_NS_fsm = ap_ST_fsm_state73;
        end
        ap_ST_fsm_state73 : begin
            ap_NS_fsm = ap_ST_fsm_state74;
        end
        ap_ST_fsm_state74 : begin
            ap_NS_fsm = ap_ST_fsm_state75;
        end
        ap_ST_fsm_state75 : begin
            ap_NS_fsm = ap_ST_fsm_state76;
        end
        ap_ST_fsm_state76 : begin
            ap_NS_fsm = ap_ST_fsm_state77;
        end
        ap_ST_fsm_state77 : begin
            ap_NS_fsm = ap_ST_fsm_state78;
        end
        ap_ST_fsm_state78 : begin
            ap_NS_fsm = ap_ST_fsm_state79;
        end
        ap_ST_fsm_state79 : begin
            ap_NS_fsm = ap_ST_fsm_state80;
        end
        ap_ST_fsm_state80 : begin
            ap_NS_fsm = ap_ST_fsm_state81;
        end
        ap_ST_fsm_state81 : begin
            ap_NS_fsm = ap_ST_fsm_state82;
        end
        ap_ST_fsm_state82 : begin
            ap_NS_fsm = ap_ST_fsm_state83;
        end
        ap_ST_fsm_state83 : begin
            ap_NS_fsm = ap_ST_fsm_state84;
        end
        ap_ST_fsm_state84 : begin
            ap_NS_fsm = ap_ST_fsm_state85;
        end
        ap_ST_fsm_state85 : begin
            ap_NS_fsm = ap_ST_fsm_state86;
        end
        ap_ST_fsm_state86 : begin
            ap_NS_fsm = ap_ST_fsm_state87;
        end
        ap_ST_fsm_state87 : begin
            ap_NS_fsm = ap_ST_fsm_state88;
        end
        ap_ST_fsm_state88 : begin
            ap_NS_fsm = ap_ST_fsm_state89;
        end
        ap_ST_fsm_state89 : begin
            ap_NS_fsm = ap_ST_fsm_state90;
        end
        ap_ST_fsm_state90 : begin
            ap_NS_fsm = ap_ST_fsm_state91;
        end
        ap_ST_fsm_state91 : begin
            ap_NS_fsm = ap_ST_fsm_state1;
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state91 = ap_CS_fsm[32'd90];

assign ap_NS_fsm_state6 = ap_NS_fsm[32'd5];

assign ap_return = grp_normalization_layer_fu_28_ap_return;

assign grp_normalization_layer_fu_28_ap_start = grp_normalization_layer_fu_28_ap_start_reg;

endmodule //encoder
