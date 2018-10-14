// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.2
// Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module decoder (
        ap_clk,
        ap_rst,
        ap_start,
        ap_done,
        ap_idle,
        ap_ready,
        data_V_read,
        res_V_TDATA,
        res_V_TVALID,
        res_V_TREADY,
        res_V_TDATA_blk_n,
        ap_ce
);

parameter    ap_ST_fsm_state1 = 12'd1;
parameter    ap_ST_fsm_state2 = 12'd2;
parameter    ap_ST_fsm_state3 = 12'd4;
parameter    ap_ST_fsm_state4 = 12'd8;
parameter    ap_ST_fsm_state5 = 12'd16;
parameter    ap_ST_fsm_state6 = 12'd32;
parameter    ap_ST_fsm_state7 = 12'd64;
parameter    ap_ST_fsm_state8 = 12'd128;
parameter    ap_ST_fsm_state9 = 12'd256;
parameter    ap_ST_fsm_state10 = 12'd512;
parameter    ap_ST_fsm_state11 = 12'd1024;
parameter    ap_ST_fsm_state12 = 12'd2048;

input   ap_clk;
input   ap_rst;
input   ap_start;
output   ap_done;
output   ap_idle;
output   ap_ready;
input  [63:0] data_V_read;
output  [127:0] res_V_TDATA;
output   res_V_TVALID;
input   res_V_TREADY;
output   res_V_TDATA_blk_n;
input   ap_ce;

reg ap_done;
reg ap_idle;
reg ap_ready;
reg res_V_TVALID;
reg res_V_TDATA_blk_n;

(* fsm_encoding = "none" *) reg   [11:0] ap_CS_fsm;
wire    ap_CS_fsm_state1;
wire    ap_CS_fsm_state12;
reg   [31:0] layer1_relu_out_0_V_reg_296;
wire    ap_CS_fsm_state2;
reg   [31:0] layer1_relu_out_1_V_reg_301;
reg   [31:0] layer1_relu_out_2_V_reg_306;
reg   [31:0] layer1_relu_out_3_V_reg_311;
reg   [31:0] logits2_0_V_reg_316;
wire    ap_CS_fsm_state4;
reg   [31:0] logits2_1_V_reg_321;
reg   [31:0] logits2_2_V_reg_326;
reg   [31:0] logits2_3_V_reg_331;
wire   [31:0] grp_softmax_fu_73_ap_return_0;
wire   [31:0] grp_softmax_fu_73_ap_return_1;
wire   [31:0] grp_softmax_fu_73_ap_return_2;
wire   [31:0] grp_softmax_fu_73_ap_return_3;
reg   [31:0] call_ret5_reg_336_3;
wire    ap_CS_fsm_state10;
reg   [31:0] logits3_2_V_reg_341;
wire   [0:0] tmp_s_fu_171_p2;
reg   [0:0] tmp_s_reg_347;
wire   [31:0] logits3_V_1_0_logits_fu_177_p3;
reg   [31:0] logits3_V_1_0_logits_reg_352;
wire   [0:0] tmp_196_1_fu_188_p2;
reg   [0:0] tmp_196_1_reg_358;
wire    ap_CS_fsm_state11;
wire   [0:0] tmp_196_2_fu_198_p2;
reg   [0:0] tmp_196_2_reg_363;
wire    grp_softmax_fu_73_ap_start;
wire    grp_softmax_fu_73_ap_done;
wire    grp_softmax_fu_73_ap_idle;
wire    grp_softmax_fu_73_ap_ready;
reg    grp_softmax_fu_73_ap_ce;
wire    ap_CS_fsm_state5;
wire    ap_CS_fsm_state6;
wire    ap_CS_fsm_state7;
wire    ap_CS_fsm_state8;
wire    ap_CS_fsm_state9;
wire   [31:0] grp_compute_layer_0_0_0_1_fu_85_ap_return_0;
wire   [31:0] grp_compute_layer_0_0_0_1_fu_85_ap_return_1;
wire   [31:0] grp_compute_layer_0_0_0_1_fu_85_ap_return_2;
wire   [31:0] grp_compute_layer_0_0_0_1_fu_85_ap_return_3;
reg    grp_compute_layer_0_0_0_1_fu_85_ap_ce;
wire    ap_CS_fsm_state3;
wire   [31:0] grp_compute_layer_0_0_0_2_fu_93_ap_return_0;
wire   [31:0] grp_compute_layer_0_0_0_2_fu_93_ap_return_1;
wire   [31:0] grp_compute_layer_0_0_0_2_fu_93_ap_return_2;
wire   [31:0] grp_compute_layer_0_0_0_2_fu_93_ap_return_3;
reg    grp_compute_layer_0_0_0_2_fu_93_ap_ce;
wire    call_ret3_relu_1_fu_99_ap_ready;
wire   [31:0] call_ret3_relu_1_fu_99_ap_return_0;
wire   [31:0] call_ret3_relu_1_fu_99_ap_return_1;
wire   [31:0] call_ret3_relu_1_fu_99_ap_return_2;
wire   [31:0] call_ret3_relu_1_fu_99_ap_return_3;
reg    grp_softmax_fu_73_ap_start_reg;
reg   [11:0] ap_NS_fsm;
wire    ap_NS_fsm_state5;
reg    ap_reg_ioackin_res_V_TREADY;
reg    ap_sig_ioackin_res_V_TREADY;
wire   [31:0] logits3_V_2_0_logits_fu_192_p3;
wire   [0:0] tmp_fu_218_p2;
wire   [25:0] p_s_fu_211_p3;
wire   [25:0] p_cast_cast_fu_204_p3;
wire   [25:0] p_s_49_fu_222_p3;
wire   [0:0] tmp_91_fu_230_p2;
wire   [0:0] tmp_199_1_fu_244_p2;
wire   [0:0] tmp_199_2_fu_258_p2;
wire   [24:0] tmp_92_fu_272_p3;
wire   [31:0] tmp_116_cast_fu_264_p3;
wire   [31:0] tmp_115_cast_fu_250_p3;
wire   [31:0] tmp_114_cast_fu_236_p3;
wire   [120:0] tmp_214_fu_279_p5;

// power-on initialization
initial begin
#0 ap_CS_fsm = 12'd1;
#0 grp_softmax_fu_73_ap_start_reg = 1'b0;
#0 ap_reg_ioackin_res_V_TREADY = 1'b0;
end

softmax grp_softmax_fu_73(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .ap_start(grp_softmax_fu_73_ap_start),
    .ap_done(grp_softmax_fu_73_ap_done),
    .ap_idle(grp_softmax_fu_73_ap_idle),
    .ap_ready(grp_softmax_fu_73_ap_ready),
    .ap_ce(grp_softmax_fu_73_ap_ce),
    .data_0_V_read(logits2_0_V_reg_316),
    .data_1_V_read(logits2_1_V_reg_321),
    .data_2_V_read(logits2_2_V_reg_326),
    .data_3_V_read(logits2_3_V_reg_331),
    .ap_return_0(grp_softmax_fu_73_ap_return_0),
    .ap_return_1(grp_softmax_fu_73_ap_return_1),
    .ap_return_2(grp_softmax_fu_73_ap_return_2),
    .ap_return_3(grp_softmax_fu_73_ap_return_3)
);

compute_layer_0_0_0_1 grp_compute_layer_0_0_0_1_fu_85(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .data_0_V_read(layer1_relu_out_0_V_reg_296),
    .data_1_V_read(layer1_relu_out_1_V_reg_301),
    .data_2_V_read(layer1_relu_out_2_V_reg_306),
    .data_3_V_read(layer1_relu_out_3_V_reg_311),
    .ap_return_0(grp_compute_layer_0_0_0_1_fu_85_ap_return_0),
    .ap_return_1(grp_compute_layer_0_0_0_1_fu_85_ap_return_1),
    .ap_return_2(grp_compute_layer_0_0_0_1_fu_85_ap_return_2),
    .ap_return_3(grp_compute_layer_0_0_0_1_fu_85_ap_return_3),
    .ap_ce(grp_compute_layer_0_0_0_1_fu_85_ap_ce)
);

compute_layer_0_0_0_2 grp_compute_layer_0_0_0_2_fu_93(
    .ap_clk(ap_clk),
    .ap_rst(ap_rst),
    .data_V_read(data_V_read),
    .ap_return_0(grp_compute_layer_0_0_0_2_fu_93_ap_return_0),
    .ap_return_1(grp_compute_layer_0_0_0_2_fu_93_ap_return_1),
    .ap_return_2(grp_compute_layer_0_0_0_2_fu_93_ap_return_2),
    .ap_return_3(grp_compute_layer_0_0_0_2_fu_93_ap_return_3),
    .ap_ce(grp_compute_layer_0_0_0_2_fu_93_ap_ce)
);

relu_1 call_ret3_relu_1_fu_99(
    .ap_ready(call_ret3_relu_1_fu_99_ap_ready),
    .data_0_V_read(grp_compute_layer_0_0_0_2_fu_93_ap_return_0),
    .data_1_V_read(grp_compute_layer_0_0_0_2_fu_93_ap_return_1),
    .data_2_V_read(grp_compute_layer_0_0_0_2_fu_93_ap_return_2),
    .data_3_V_read(grp_compute_layer_0_0_0_2_fu_93_ap_return_3),
    .ap_return_0(call_ret3_relu_1_fu_99_ap_return_0),
    .ap_return_1(call_ret3_relu_1_fu_99_ap_return_1),
    .ap_return_2(call_ret3_relu_1_fu_99_ap_return_2),
    .ap_return_3(call_ret3_relu_1_fu_99_ap_return_3)
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
        ap_reg_ioackin_res_V_TREADY <= 1'b0;
    end else begin
        if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state12))) begin
            if ((ap_sig_ioackin_res_V_TREADY == 1'b1)) begin
                ap_reg_ioackin_res_V_TREADY <= 1'b0;
            end else if ((res_V_TREADY == 1'b1)) begin
                ap_reg_ioackin_res_V_TREADY <= 1'b1;
            end
        end
    end
end

always @ (posedge ap_clk) begin
    if (ap_rst == 1'b1) begin
        grp_softmax_fu_73_ap_start_reg <= 1'b0;
    end else begin
        if (((1'b1 == ap_CS_fsm_state4) & (1'b1 == ap_NS_fsm_state5))) begin
            grp_softmax_fu_73_ap_start_reg <= 1'b1;
        end else if ((grp_softmax_fu_73_ap_ready == 1'b1)) begin
            grp_softmax_fu_73_ap_start_reg <= 1'b0;
        end
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state10))) begin
        call_ret5_reg_336_3 <= grp_softmax_fu_73_ap_return_3;
        logits3_2_V_reg_341 <= grp_softmax_fu_73_ap_return_2;
        logits3_V_1_0_logits_reg_352 <= logits3_V_1_0_logits_fu_177_p3;
        tmp_s_reg_347 <= tmp_s_fu_171_p2;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state2))) begin
        layer1_relu_out_0_V_reg_296 <= call_ret3_relu_1_fu_99_ap_return_0;
        layer1_relu_out_1_V_reg_301 <= call_ret3_relu_1_fu_99_ap_return_1;
        layer1_relu_out_2_V_reg_306 <= call_ret3_relu_1_fu_99_ap_return_2;
        layer1_relu_out_3_V_reg_311 <= call_ret3_relu_1_fu_99_ap_return_3;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state4))) begin
        logits2_0_V_reg_316 <= grp_compute_layer_0_0_0_1_fu_85_ap_return_0;
        logits2_1_V_reg_321 <= grp_compute_layer_0_0_0_1_fu_85_ap_return_1;
        logits2_2_V_reg_326 <= grp_compute_layer_0_0_0_1_fu_85_ap_return_2;
        logits2_3_V_reg_331 <= grp_compute_layer_0_0_0_1_fu_85_ap_return_3;
    end
end

always @ (posedge ap_clk) begin
    if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state11))) begin
        tmp_196_1_reg_358 <= tmp_196_1_fu_188_p2;
        tmp_196_2_reg_363 <= tmp_196_2_fu_198_p2;
    end
end

always @ (*) begin
    if ((((1'b1 == ap_ce) & (ap_sig_ioackin_res_V_TREADY == 1'b1) & (1'b1 == ap_CS_fsm_state12)) | ((ap_start == 1'b0) & (1'b1 == ap_CS_fsm_state1)))) begin
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
    if (((1'b1 == ap_ce) & (ap_sig_ioackin_res_V_TREADY == 1'b1) & (1'b1 == ap_CS_fsm_state12))) begin
        ap_ready = 1'b1;
    end else begin
        ap_ready = 1'b0;
    end
end

always @ (*) begin
    if ((ap_reg_ioackin_res_V_TREADY == 1'b0)) begin
        ap_sig_ioackin_res_V_TREADY = res_V_TREADY;
    end else begin
        ap_sig_ioackin_res_V_TREADY = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_ce) & ((1'b1 == ap_CS_fsm_state4) | (1'b1 == ap_CS_fsm_state3)))) begin
        grp_compute_layer_0_0_0_1_fu_85_ap_ce = 1'b1;
    end else begin
        grp_compute_layer_0_0_0_1_fu_85_ap_ce = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_ce) & ((1'b1 == ap_CS_fsm_state2) | ((1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))))) begin
        grp_compute_layer_0_0_0_2_fu_93_ap_ce = 1'b1;
    end else begin
        grp_compute_layer_0_0_0_2_fu_93_ap_ce = 1'b0;
    end
end

always @ (*) begin
    if (((1'b1 == ap_ce) & ((1'b1 == ap_CS_fsm_state10) | (1'b1 == ap_CS_fsm_state9) | (1'b1 == ap_CS_fsm_state8) | (1'b1 == ap_CS_fsm_state7) | (1'b1 == ap_CS_fsm_state6) | (1'b1 == ap_CS_fsm_state5)))) begin
        grp_softmax_fu_73_ap_ce = 1'b1;
    end else begin
        grp_softmax_fu_73_ap_ce = 1'b0;
    end
end

always @ (*) begin
    if ((1'b1 == ap_CS_fsm_state12)) begin
        res_V_TDATA_blk_n = res_V_TREADY;
    end else begin
        res_V_TDATA_blk_n = 1'b1;
    end
end

always @ (*) begin
    if (((1'b1 == ap_ce) & (ap_reg_ioackin_res_V_TREADY == 1'b0) & (1'b1 == ap_CS_fsm_state12))) begin
        res_V_TVALID = 1'b1;
    end else begin
        res_V_TVALID = 1'b0;
    end
end

always @ (*) begin
    case (ap_CS_fsm)
        ap_ST_fsm_state1 : begin
            if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state1) & (ap_start == 1'b1))) begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end
        end
        ap_ST_fsm_state2 : begin
            if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state2))) begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state2;
            end
        end
        ap_ST_fsm_state3 : begin
            if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state3))) begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state3;
            end
        end
        ap_ST_fsm_state4 : begin
            if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state4))) begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state4;
            end
        end
        ap_ST_fsm_state5 : begin
            if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state5))) begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state5;
            end
        end
        ap_ST_fsm_state6 : begin
            if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state6))) begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state6;
            end
        end
        ap_ST_fsm_state7 : begin
            if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state7))) begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state7;
            end
        end
        ap_ST_fsm_state8 : begin
            if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state8))) begin
                ap_NS_fsm = ap_ST_fsm_state9;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state8;
            end
        end
        ap_ST_fsm_state9 : begin
            if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state9))) begin
                ap_NS_fsm = ap_ST_fsm_state10;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state9;
            end
        end
        ap_ST_fsm_state10 : begin
            if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state10))) begin
                ap_NS_fsm = ap_ST_fsm_state11;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state10;
            end
        end
        ap_ST_fsm_state11 : begin
            if (((1'b1 == ap_ce) & (1'b1 == ap_CS_fsm_state11))) begin
                ap_NS_fsm = ap_ST_fsm_state12;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state11;
            end
        end
        ap_ST_fsm_state12 : begin
            if (((1'b1 == ap_ce) & (ap_sig_ioackin_res_V_TREADY == 1'b1) & (1'b1 == ap_CS_fsm_state12))) begin
                ap_NS_fsm = ap_ST_fsm_state1;
            end else begin
                ap_NS_fsm = ap_ST_fsm_state12;
            end
        end
        default : begin
            ap_NS_fsm = 'bx;
        end
    endcase
end

assign ap_CS_fsm_state1 = ap_CS_fsm[32'd0];

assign ap_CS_fsm_state10 = ap_CS_fsm[32'd9];

assign ap_CS_fsm_state11 = ap_CS_fsm[32'd10];

assign ap_CS_fsm_state12 = ap_CS_fsm[32'd11];

assign ap_CS_fsm_state2 = ap_CS_fsm[32'd1];

assign ap_CS_fsm_state3 = ap_CS_fsm[32'd2];

assign ap_CS_fsm_state4 = ap_CS_fsm[32'd3];

assign ap_CS_fsm_state5 = ap_CS_fsm[32'd4];

assign ap_CS_fsm_state6 = ap_CS_fsm[32'd5];

assign ap_CS_fsm_state7 = ap_CS_fsm[32'd6];

assign ap_CS_fsm_state8 = ap_CS_fsm[32'd7];

assign ap_CS_fsm_state9 = ap_CS_fsm[32'd8];

assign ap_NS_fsm_state5 = ap_NS_fsm[32'd4];

assign grp_softmax_fu_73_ap_start = grp_softmax_fu_73_ap_start_reg;

assign logits3_V_1_0_logits_fu_177_p3 = ((tmp_s_fu_171_p2[0:0] === 1'b1) ? grp_softmax_fu_73_ap_return_1 : grp_softmax_fu_73_ap_return_0);

assign logits3_V_2_0_logits_fu_192_p3 = ((tmp_196_1_fu_188_p2[0:0] === 1'b1) ? logits3_2_V_reg_341 : logits3_V_1_0_logits_reg_352);

assign p_cast_cast_fu_204_p3 = ((tmp_s_reg_347[0:0] === 1'b1) ? 26'd16777216 : 26'd0);

assign p_s_49_fu_222_p3 = ((tmp_fu_218_p2[0:0] === 1'b1) ? p_s_fu_211_p3 : p_cast_cast_fu_204_p3);

assign p_s_fu_211_p3 = ((tmp_196_2_reg_363[0:0] === 1'b1) ? 26'd50331648 : 26'd33554432);

assign res_V_TDATA = tmp_214_fu_279_p5;

assign tmp_114_cast_fu_236_p3 = ((tmp_91_fu_230_p2[0:0] === 1'b1) ? 32'd16777216 : 32'd0);

assign tmp_115_cast_fu_250_p3 = ((tmp_199_1_fu_244_p2[0:0] === 1'b1) ? 32'd16777216 : 32'd0);

assign tmp_116_cast_fu_264_p3 = ((tmp_199_2_fu_258_p2[0:0] === 1'b1) ? 32'd16777216 : 32'd0);

assign tmp_196_1_fu_188_p2 = (($signed(logits3_2_V_reg_341) > $signed(logits3_V_1_0_logits_reg_352)) ? 1'b1 : 1'b0);

assign tmp_196_2_fu_198_p2 = (($signed(call_ret5_reg_336_3) > $signed(logits3_V_2_0_logits_fu_192_p3)) ? 1'b1 : 1'b0);

assign tmp_199_1_fu_244_p2 = ((p_s_49_fu_222_p3 == 26'd16777216) ? 1'b1 : 1'b0);

assign tmp_199_2_fu_258_p2 = ((p_s_49_fu_222_p3 == 26'd33554432) ? 1'b1 : 1'b0);

assign tmp_214_fu_279_p5 = {{{{tmp_92_fu_272_p3}, {tmp_116_cast_fu_264_p3}}, {tmp_115_cast_fu_250_p3}}, {tmp_114_cast_fu_236_p3}};

assign tmp_91_fu_230_p2 = ((p_s_49_fu_222_p3 == 26'd0) ? 1'b1 : 1'b0);

assign tmp_92_fu_272_p3 = ((tmp_196_2_reg_363[0:0] === 1'b1) ? 25'd16777216 : 25'd0);

assign tmp_fu_218_p2 = (tmp_196_2_reg_363 | tmp_196_1_reg_358);

assign tmp_s_fu_171_p2 = (($signed(grp_softmax_fu_73_ap_return_1) > $signed(grp_softmax_fu_73_ap_return_0)) ? 1'b1 : 1'b0);

endmodule //decoder
