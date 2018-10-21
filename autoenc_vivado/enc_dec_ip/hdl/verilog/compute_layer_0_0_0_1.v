// ==============================================================
// RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.2.1
// Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
// 
// ===========================================================

`timescale 1 ns / 1 ps 

module compute_layer_0_0_0_1 (
        ap_clk,
        ap_rst,
        data_0_V_read,
        data_1_V_read,
        data_2_V_read,
        data_3_V_read,
        ap_return_0,
        ap_return_1,
        ap_return_2,
        ap_return_3,
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
output  [31:0] ap_return_2;
output  [31:0] ap_return_3;
input   ap_ce;

reg   [31:0] tmp_113_reg_1571;
wire    ap_block_state1_pp0_stage0_iter0;
wire    ap_block_state2_pp0_stage0_iter1;
wire    ap_block_pp0_stage0_11001;
reg   [26:0] tmp_114_reg_1576;
reg   [31:0] tmp_153_0_2_reg_1581;
reg   [31:0] tmp_153_0_3_reg_1586;
reg   [31:0] tmp_153_1_reg_1591;
reg   [31:0] tmp_153_1_1_reg_1596;
reg   [31:0] tmp_153_1_2_reg_1601;
reg   [31:0] tmp_153_1_3_reg_1606;
reg   [31:0] tmp_153_2_reg_1611;
reg   [31:0] tmp_153_2_1_reg_1616;
reg   [26:0] tmp_s_reg_1621;
reg   [31:0] tmp_153_2_3_reg_1626;
reg   [31:0] tmp_153_3_reg_1631;
reg   [31:0] tmp_153_3_1_reg_1636;
reg   [31:0] tmp_153_3_2_reg_1641;
reg   [31:0] tmp_153_3_3_reg_1646;
wire  signed [31:0] p_Val2_3_3_fu_106_p0;
wire  signed [55:0] OP1_V_3_cast_fu_1411_p1;
wire    ap_block_pp0_stage0;
wire  signed [31:0] p_Val2_s_fu_107_p0;
wire  signed [55:0] OP1_V_cast_fu_1264_p1;
wire  signed [31:0] p_Val2_2_1_fu_108_p0;
wire  signed [55:0] OP1_V_2_cast_fu_1364_p1;
wire  signed [31:0] p_Val2_1_fu_109_p0;
wire  signed [55:0] OP1_V_1_cast_fu_1311_p1;
wire  signed [31:0] p_Val2_1_1_fu_110_p0;
wire  signed [31:0] p_Val2_0_1_fu_111_p0;
wire  signed [31:0] p_Val2_1_2_fu_112_p0;
wire  signed [31:0] p_Val2_2_3_fu_113_p0;
wire  signed [31:0] p_Val2_0_3_fu_114_p0;
wire  signed [31:0] p_Val2_2_fu_115_p0;
wire  signed [31:0] p_Val2_2_2_fu_116_p0;
wire  signed [31:0] p_Val2_3_fu_117_p0;
wire  signed [31:0] p_Val2_1_3_fu_118_p0;
wire  signed [31:0] p_Val2_3_1_fu_119_p0;
wire  signed [31:0] p_Val2_3_2_fu_120_p0;
wire  signed [31:0] p_Val2_0_2_fu_121_p0;
wire  signed [31:0] OP1_V_cast3_fu_1259_p0;
wire  signed [31:0] OP1_V_cast_fu_1264_p0;
wire   [55:0] p_Val2_s_fu_107_p2;
wire   [50:0] p_Val2_0_1_fu_111_p2;
wire   [55:0] p_Val2_0_2_fu_121_p2;
wire   [55:0] p_Val2_0_3_fu_114_p2;
wire   [55:0] p_Val2_1_fu_109_p2;
wire   [55:0] p_Val2_1_1_fu_110_p2;
wire   [55:0] p_Val2_1_2_fu_112_p2;
wire   [55:0] p_Val2_1_3_fu_118_p2;
wire  signed [31:0] OP1_V_2_cast2_fu_1359_p0;
wire  signed [31:0] OP1_V_2_cast_fu_1364_p0;
wire   [55:0] p_Val2_2_fu_115_p2;
wire   [55:0] p_Val2_2_1_fu_108_p2;
wire   [50:0] p_Val2_2_2_fu_116_p2;
wire   [55:0] p_Val2_2_3_fu_113_p2;
wire   [55:0] p_Val2_3_fu_117_p2;
wire   [55:0] p_Val2_3_1_fu_119_p2;
wire   [55:0] p_Val2_3_2_fu_120_p2;
wire   [55:0] p_Val2_3_3_fu_106_p2;
wire   [31:0] tmp3_fu_1469_p2;
wire   [31:0] tmp2_fu_1474_p2;
wire   [31:0] tmp1_fu_1465_p2;
wire  signed [31:0] tmp_115_fu_1459_p1;
wire   [31:0] tmp6_fu_1490_p2;
wire   [31:0] tmp5_fu_1495_p2;
wire   [31:0] tmp4_fu_1485_p2;
wire   [31:0] tmp9_fu_1510_p2;
wire  signed [31:0] tmp_116_fu_1462_p1;
wire   [31:0] tmp8_fu_1515_p2;
wire   [31:0] tmp7_fu_1506_p2;
wire   [31:0] tmp12_fu_1531_p2;
wire   [31:0] tmp11_fu_1536_p2;
wire   [31:0] tmp10_fu_1527_p2;
wire   [31:0] res_0_V_write_assig_fu_1479_p2;
wire   [31:0] res_1_V_write_assig_fu_1500_p2;
wire   [31:0] res_2_V_write_assig_fu_1521_p2;
wire   [31:0] res_3_V_write_assig_fu_1541_p2;

always @ (posedge ap_clk) begin
    if (((1'b0 == ap_block_pp0_stage0_11001) & (1'b1 == ap_ce))) begin
        tmp_113_reg_1571 <= {{p_Val2_s_fu_107_p2[55:24]}};
        tmp_114_reg_1576 <= {{p_Val2_0_1_fu_111_p2[50:24]}};
        tmp_153_0_2_reg_1581 <= {{p_Val2_0_2_fu_121_p2[55:24]}};
        tmp_153_0_3_reg_1586 <= {{p_Val2_0_3_fu_114_p2[55:24]}};
        tmp_153_1_1_reg_1596 <= {{p_Val2_1_1_fu_110_p2[55:24]}};
        tmp_153_1_2_reg_1601 <= {{p_Val2_1_2_fu_112_p2[55:24]}};
        tmp_153_1_3_reg_1606 <= {{p_Val2_1_3_fu_118_p2[55:24]}};
        tmp_153_1_reg_1591 <= {{p_Val2_1_fu_109_p2[55:24]}};
        tmp_153_2_1_reg_1616 <= {{p_Val2_2_1_fu_108_p2[55:24]}};
        tmp_153_2_3_reg_1626 <= {{p_Val2_2_3_fu_113_p2[55:24]}};
        tmp_153_2_reg_1611 <= {{p_Val2_2_fu_115_p2[55:24]}};
        tmp_153_3_1_reg_1636 <= {{p_Val2_3_1_fu_119_p2[55:24]}};
        tmp_153_3_2_reg_1641 <= {{p_Val2_3_2_fu_120_p2[55:24]}};
        tmp_153_3_3_reg_1646 <= {{p_Val2_3_3_fu_106_p2[55:24]}};
        tmp_153_3_reg_1631 <= {{p_Val2_3_fu_117_p2[55:24]}};
        tmp_s_reg_1621 <= {{p_Val2_2_2_fu_116_p2[50:24]}};
    end
end

assign OP1_V_1_cast_fu_1311_p1 = $signed(data_1_V_read);

assign OP1_V_2_cast2_fu_1359_p0 = data_2_V_read;

assign OP1_V_2_cast_fu_1364_p0 = data_2_V_read;

assign OP1_V_2_cast_fu_1364_p1 = OP1_V_2_cast_fu_1364_p0;

assign OP1_V_3_cast_fu_1411_p1 = $signed(data_3_V_read);

assign OP1_V_cast3_fu_1259_p0 = data_0_V_read;

assign OP1_V_cast_fu_1264_p0 = data_0_V_read;

assign OP1_V_cast_fu_1264_p1 = OP1_V_cast_fu_1264_p0;

assign ap_block_pp0_stage0 = ~(1'b1 == 1'b1);

assign ap_block_pp0_stage0_11001 = ~(1'b1 == 1'b1);

assign ap_block_state1_pp0_stage0_iter0 = ~(1'b1 == 1'b1);

assign ap_block_state2_pp0_stage0_iter1 = ~(1'b1 == 1'b1);

assign ap_return_0 = res_0_V_write_assig_fu_1479_p2;

assign ap_return_1 = res_1_V_write_assig_fu_1500_p2;

assign ap_return_2 = res_2_V_write_assig_fu_1521_p2;

assign ap_return_3 = res_3_V_write_assig_fu_1541_p2;

assign p_Val2_0_1_fu_111_p0 = OP1_V_cast3_fu_1259_p0;

assign p_Val2_0_1_fu_111_p2 = ($signed(p_Val2_0_1_fu_111_p0) * $signed('h242B3));

assign p_Val2_0_2_fu_121_p0 = OP1_V_cast_fu_1264_p1;

assign p_Val2_0_2_fu_121_p2 = ($signed(p_Val2_0_2_fu_121_p0) * $signed(-56'h12D7674));

assign p_Val2_0_3_fu_114_p0 = OP1_V_cast_fu_1264_p1;

assign p_Val2_0_3_fu_114_p2 = ($signed(p_Val2_0_3_fu_114_p0) * $signed('h1D4FD32));

assign p_Val2_1_1_fu_110_p0 = OP1_V_1_cast_fu_1311_p1;

assign p_Val2_1_1_fu_110_p2 = ($signed(p_Val2_1_1_fu_110_p0) * $signed(-56'h38189C8));

assign p_Val2_1_2_fu_112_p0 = OP1_V_1_cast_fu_1311_p1;

assign p_Val2_1_2_fu_112_p2 = ($signed(p_Val2_1_2_fu_112_p0) * $signed('h1E690C3));

assign p_Val2_1_3_fu_118_p0 = OP1_V_1_cast_fu_1311_p1;

assign p_Val2_1_3_fu_118_p2 = ($signed(p_Val2_1_3_fu_118_p0) * $signed(-56'h1F26550));

assign p_Val2_1_fu_109_p0 = OP1_V_1_cast_fu_1311_p1;

assign p_Val2_1_fu_109_p2 = ($signed(p_Val2_1_fu_109_p0) * $signed('hC8FD27));

assign p_Val2_2_1_fu_108_p0 = OP1_V_2_cast_fu_1364_p1;

assign p_Val2_2_1_fu_108_p2 = ($signed(p_Val2_2_1_fu_108_p0) * $signed('h11A736D));

assign p_Val2_2_2_fu_116_p0 = OP1_V_2_cast2_fu_1359_p0;

assign p_Val2_2_2_fu_116_p2 = ($signed(p_Val2_2_2_fu_116_p0) * $signed('h209B0));

assign p_Val2_2_3_fu_113_p0 = OP1_V_2_cast_fu_1364_p1;

assign p_Val2_2_3_fu_113_p2 = ($signed(p_Val2_2_3_fu_113_p0) * $signed(-56'h236478C));

assign p_Val2_2_fu_115_p0 = OP1_V_2_cast_fu_1364_p1;

assign p_Val2_2_fu_115_p2 = ($signed(p_Val2_2_fu_115_p0) * $signed(-56'h12BE6BD));

assign p_Val2_3_1_fu_119_p0 = OP1_V_3_cast_fu_1411_p1;

assign p_Val2_3_1_fu_119_p2 = ($signed(p_Val2_3_1_fu_119_p0) * $signed('h1C58863));

assign p_Val2_3_2_fu_120_p0 = OP1_V_3_cast_fu_1411_p1;

assign p_Val2_3_2_fu_120_p2 = ($signed(p_Val2_3_2_fu_120_p0) * $signed(-56'h2C3A179));

assign p_Val2_3_3_fu_106_p0 = OP1_V_3_cast_fu_1411_p1;

assign p_Val2_3_3_fu_106_p2 = ($signed(p_Val2_3_3_fu_106_p0) * $signed(-56'h248C57D));

assign p_Val2_3_fu_117_p0 = OP1_V_3_cast_fu_1411_p1;

assign p_Val2_3_fu_117_p2 = ($signed(p_Val2_3_fu_117_p0) * $signed('h26B80DF));

assign p_Val2_s_fu_107_p0 = OP1_V_cast_fu_1264_p1;

assign p_Val2_s_fu_107_p2 = ($signed(p_Val2_s_fu_107_p0) * $signed(-56'h2DFB3EC));

assign res_0_V_write_assig_fu_1479_p2 = (tmp2_fu_1474_p2 + tmp1_fu_1465_p2);

assign res_1_V_write_assig_fu_1500_p2 = (tmp5_fu_1495_p2 + tmp4_fu_1485_p2);

assign res_2_V_write_assig_fu_1521_p2 = (tmp8_fu_1515_p2 + tmp7_fu_1506_p2);

assign res_3_V_write_assig_fu_1541_p2 = (tmp11_fu_1536_p2 + tmp10_fu_1527_p2);

assign tmp10_fu_1527_p2 = (tmp_153_0_3_reg_1586 + tmp_153_1_3_reg_1606);

assign tmp11_fu_1536_p2 = (tmp12_fu_1531_p2 + tmp_153_2_3_reg_1626);

assign tmp12_fu_1531_p2 = (tmp_153_3_3_reg_1646 + 32'd12842469);

assign tmp1_fu_1465_p2 = (tmp_113_reg_1571 + tmp_153_1_reg_1591);

assign tmp2_fu_1474_p2 = (tmp3_fu_1469_p2 + tmp_153_2_reg_1611);

assign tmp3_fu_1469_p2 = ($signed(tmp_153_3_reg_1631) + $signed(32'd4262452247));

assign tmp4_fu_1485_p2 = ($signed(tmp_115_fu_1459_p1) + $signed(tmp_153_1_1_reg_1596));

assign tmp5_fu_1495_p2 = (tmp6_fu_1490_p2 + tmp_153_2_1_reg_1616);

assign tmp6_fu_1490_p2 = ($signed(tmp_153_3_1_reg_1636) + $signed(32'd4285771868));

assign tmp7_fu_1506_p2 = (tmp_153_0_2_reg_1581 + tmp_153_1_2_reg_1601);

assign tmp8_fu_1515_p2 = ($signed(tmp9_fu_1510_p2) + $signed(tmp_116_fu_1462_p1));

assign tmp9_fu_1510_p2 = (tmp_153_3_2_reg_1641 + 32'd25879662);

assign tmp_115_fu_1459_p1 = $signed(tmp_114_reg_1576);

assign tmp_116_fu_1462_p1 = $signed(tmp_s_reg_1621);

endmodule //compute_layer_0_0_0_1
