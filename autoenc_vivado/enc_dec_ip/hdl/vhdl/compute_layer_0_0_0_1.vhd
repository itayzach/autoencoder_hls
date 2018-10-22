-- ==============================================================
-- RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
-- Version: 2018.2
-- Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
-- 
-- ===========================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity compute_layer_0_0_0_1 is
port (
    ap_clk : IN STD_LOGIC;
    ap_rst : IN STD_LOGIC;
    data_0_V_read : IN STD_LOGIC_VECTOR (31 downto 0);
    data_1_V_read : IN STD_LOGIC_VECTOR (31 downto 0);
    data_2_V_read : IN STD_LOGIC_VECTOR (31 downto 0);
    data_3_V_read : IN STD_LOGIC_VECTOR (31 downto 0);
    ap_return_0 : OUT STD_LOGIC_VECTOR (31 downto 0);
    ap_return_1 : OUT STD_LOGIC_VECTOR (31 downto 0);
    ap_return_2 : OUT STD_LOGIC_VECTOR (31 downto 0);
    ap_return_3 : OUT STD_LOGIC_VECTOR (31 downto 0);
    ap_ce : IN STD_LOGIC );
end;


architecture behav of compute_layer_0_0_0_1 is 
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_boolean_1 : BOOLEAN := true;
    constant ap_const_boolean_0 : BOOLEAN := false;
    constant ap_const_lv56_FFFFFFFDB73A83 : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111101101101110011101010000011";
    constant ap_const_lv56_FFFFFFFD204C14 : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111101001000000100110000010100";
    constant ap_const_lv56_11A736D : STD_LOGIC_VECTOR (55 downto 0) := "00000000000000000000000000000001000110100111001101101101";
    constant ap_const_lv56_C8FD27 : STD_LOGIC_VECTOR (55 downto 0) := "00000000000000000000000000000000110010001111110100100111";
    constant ap_const_lv56_FFFFFFFC7E7638 : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111100011111100111011000111000";
    constant ap_const_lv51_242B3 : STD_LOGIC_VECTOR (50 downto 0) := "000000000000000000000000000000000100100001010110011";
    constant ap_const_lv56_1E690C3 : STD_LOGIC_VECTOR (55 downto 0) := "00000000000000000000000000000001111001101001000011000011";
    constant ap_const_lv56_FFFFFFFDC9B874 : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111101110010011011100001110100";
    constant ap_const_lv56_1D4FD32 : STD_LOGIC_VECTOR (55 downto 0) := "00000000000000000000000000000001110101001111110100110010";
    constant ap_const_lv56_FFFFFFFED41943 : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111110110101000001100101000011";
    constant ap_const_lv51_209B0 : STD_LOGIC_VECTOR (50 downto 0) := "000000000000000000000000000000000100000100110110000";
    constant ap_const_lv56_26B80DF : STD_LOGIC_VECTOR (55 downto 0) := "00000000000000000000000000000010011010111000000011011111";
    constant ap_const_lv56_FFFFFFFE0D9AB0 : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111110000011011001101010110000";
    constant ap_const_lv56_1C58863 : STD_LOGIC_VECTOR (55 downto 0) := "00000000000000000000000000000001110001011000100001100011";
    constant ap_const_lv56_FFFFFFFD3C5E87 : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111101001111000101111010000111";
    constant ap_const_lv56_FFFFFFFED2898C : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111110110100101000100110001100";
    constant ap_const_lv32_18 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000011000";
    constant ap_const_lv32_37 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000110111";
    constant ap_const_lv32_32 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000110010";
    constant ap_const_lv32_FE0FDC17 : STD_LOGIC_VECTOR (31 downto 0) := "11111110000011111101110000010111";
    constant ap_const_lv32_FF73B05C : STD_LOGIC_VECTOR (31 downto 0) := "11111111011100111011000001011100";
    constant ap_const_lv32_18AE46E : STD_LOGIC_VECTOR (31 downto 0) := "00000001100010101110010001101110";
    constant ap_const_lv32_C3F5E5 : STD_LOGIC_VECTOR (31 downto 0) := "00000000110000111111010111100101";
    constant ap_const_logic_0 : STD_LOGIC := '0';

    signal tmp_108_reg_1571 : STD_LOGIC_VECTOR (31 downto 0);
    signal ap_block_state1_pp0_stage0_iter0 : BOOLEAN;
    signal ap_block_state2_pp0_stage0_iter1 : BOOLEAN;
    signal ap_block_pp0_stage0_11001 : BOOLEAN;
    signal tmp_s_reg_1576 : STD_LOGIC_VECTOR (26 downto 0);
    signal tmp_153_0_2_reg_1581 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_153_0_3_reg_1586 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_153_1_reg_1591 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_153_1_1_reg_1596 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_153_1_2_reg_1601 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_153_1_3_reg_1606 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_153_2_reg_1611 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_153_2_1_reg_1616 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_110_reg_1621 : STD_LOGIC_VECTOR (26 downto 0);
    signal tmp_153_2_3_reg_1626 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_153_3_reg_1631 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_153_3_1_reg_1636 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_153_3_2_reg_1641 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_153_3_3_reg_1646 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_3_3_fu_106_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal OP1_V_3_cast_fu_1411_p1 : STD_LOGIC_VECTOR (55 downto 0);
    signal ap_block_pp0_stage0 : BOOLEAN;
    signal p_Val2_s_fu_107_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal OP1_V_cast_fu_1264_p1 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_2_1_fu_108_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal OP1_V_2_cast_fu_1364_p1 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_1_fu_109_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal OP1_V_1_cast_fu_1311_p1 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_1_1_fu_110_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_0_1_fu_111_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_1_2_fu_112_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_2_3_fu_113_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_0_3_fu_114_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_2_fu_115_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_2_2_fu_116_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_3_fu_117_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_1_3_fu_118_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_3_1_fu_119_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_3_2_fu_120_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_0_2_fu_121_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal OP1_V_cast3_fu_1259_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal OP1_V_cast_fu_1264_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_s_fu_107_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_0_1_fu_111_p2 : STD_LOGIC_VECTOR (50 downto 0);
    signal p_Val2_0_2_fu_121_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_0_3_fu_114_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_1_fu_109_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_1_1_fu_110_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_1_2_fu_112_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_1_3_fu_118_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal OP1_V_2_cast2_fu_1359_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal OP1_V_2_cast_fu_1364_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_2_fu_115_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_2_1_fu_108_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_2_2_fu_116_p2 : STD_LOGIC_VECTOR (50 downto 0);
    signal p_Val2_2_3_fu_113_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_3_fu_117_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_3_1_fu_119_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_3_2_fu_120_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_3_3_fu_106_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal tmp3_fu_1469_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp2_fu_1474_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp1_fu_1465_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_109_fu_1459_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp6_fu_1490_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp5_fu_1495_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp4_fu_1485_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp9_fu_1510_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_111_fu_1462_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp8_fu_1515_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp7_fu_1506_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp12_fu_1531_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp11_fu_1536_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp10_fu_1527_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal res_0_V_write_assig_fu_1479_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal res_1_V_write_assig_fu_1500_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal res_2_V_write_assig_fu_1521_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal res_3_V_write_assig_fu_1541_p2 : STD_LOGIC_VECTOR (31 downto 0);


begin



    process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if (((ap_const_boolean_0 = ap_block_pp0_stage0_11001) and (ap_const_logic_1 = ap_ce))) then
                tmp_108_reg_1571 <= p_Val2_s_fu_107_p2(55 downto 24);
                tmp_110_reg_1621 <= p_Val2_2_2_fu_116_p2(50 downto 24);
                tmp_153_0_2_reg_1581 <= p_Val2_0_2_fu_121_p2(55 downto 24);
                tmp_153_0_3_reg_1586 <= p_Val2_0_3_fu_114_p2(55 downto 24);
                tmp_153_1_1_reg_1596 <= p_Val2_1_1_fu_110_p2(55 downto 24);
                tmp_153_1_2_reg_1601 <= p_Val2_1_2_fu_112_p2(55 downto 24);
                tmp_153_1_3_reg_1606 <= p_Val2_1_3_fu_118_p2(55 downto 24);
                tmp_153_1_reg_1591 <= p_Val2_1_fu_109_p2(55 downto 24);
                tmp_153_2_1_reg_1616 <= p_Val2_2_1_fu_108_p2(55 downto 24);
                tmp_153_2_3_reg_1626 <= p_Val2_2_3_fu_113_p2(55 downto 24);
                tmp_153_2_reg_1611 <= p_Val2_2_fu_115_p2(55 downto 24);
                tmp_153_3_1_reg_1636 <= p_Val2_3_1_fu_119_p2(55 downto 24);
                tmp_153_3_2_reg_1641 <= p_Val2_3_2_fu_120_p2(55 downto 24);
                tmp_153_3_3_reg_1646 <= p_Val2_3_3_fu_106_p2(55 downto 24);
                tmp_153_3_reg_1631 <= p_Val2_3_fu_117_p2(55 downto 24);
                tmp_s_reg_1576 <= p_Val2_0_1_fu_111_p2(50 downto 24);
            end if;
        end if;
    end process;
        OP1_V_1_cast_fu_1311_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(data_1_V_read),56));

    OP1_V_2_cast2_fu_1359_p0 <= data_2_V_read;
    OP1_V_2_cast_fu_1364_p0 <= data_2_V_read;
        OP1_V_2_cast_fu_1364_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(OP1_V_2_cast_fu_1364_p0),56));

        OP1_V_3_cast_fu_1411_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(data_3_V_read),56));

    OP1_V_cast3_fu_1259_p0 <= data_0_V_read;
    OP1_V_cast_fu_1264_p0 <= data_0_V_read;
        OP1_V_cast_fu_1264_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(OP1_V_cast_fu_1264_p0),56));

        ap_block_pp0_stage0 <= not((ap_const_boolean_1 = ap_const_boolean_1));
        ap_block_pp0_stage0_11001 <= not((ap_const_boolean_1 = ap_const_boolean_1));
        ap_block_state1_pp0_stage0_iter0 <= not((ap_const_boolean_1 = ap_const_boolean_1));
        ap_block_state2_pp0_stage0_iter1 <= not((ap_const_boolean_1 = ap_const_boolean_1));
    ap_return_0 <= res_0_V_write_assig_fu_1479_p2;
    ap_return_1 <= res_1_V_write_assig_fu_1500_p2;
    ap_return_2 <= res_2_V_write_assig_fu_1521_p2;
    ap_return_3 <= res_3_V_write_assig_fu_1541_p2;
    p_Val2_0_1_fu_111_p0 <= OP1_V_cast3_fu_1259_p0;
    p_Val2_0_1_fu_111_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_0_1_fu_111_p0) * signed('0' &ap_const_lv51_242B3))), 51));
    p_Val2_0_2_fu_121_p0 <= OP1_V_cast_fu_1264_p1(32 - 1 downto 0);
    p_Val2_0_2_fu_121_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_0_2_fu_121_p0) * signed(ap_const_lv56_FFFFFFFED2898C))), 56));
    p_Val2_0_3_fu_114_p0 <= OP1_V_cast_fu_1264_p1(32 - 1 downto 0);
    p_Val2_0_3_fu_114_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_0_3_fu_114_p0) * signed('0' &ap_const_lv56_1D4FD32))), 56));
    p_Val2_1_1_fu_110_p0 <= OP1_V_1_cast_fu_1311_p1(32 - 1 downto 0);
    p_Val2_1_1_fu_110_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_1_1_fu_110_p0) * signed(ap_const_lv56_FFFFFFFC7E7638))), 56));
    p_Val2_1_2_fu_112_p0 <= OP1_V_1_cast_fu_1311_p1(32 - 1 downto 0);
    p_Val2_1_2_fu_112_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_1_2_fu_112_p0) * signed('0' &ap_const_lv56_1E690C3))), 56));
    p_Val2_1_3_fu_118_p0 <= OP1_V_1_cast_fu_1311_p1(32 - 1 downto 0);
    p_Val2_1_3_fu_118_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_1_3_fu_118_p0) * signed(ap_const_lv56_FFFFFFFE0D9AB0))), 56));
    p_Val2_1_fu_109_p0 <= OP1_V_1_cast_fu_1311_p1(32 - 1 downto 0);
    p_Val2_1_fu_109_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_1_fu_109_p0) * signed('0' &ap_const_lv56_C8FD27))), 56));
    p_Val2_2_1_fu_108_p0 <= OP1_V_2_cast_fu_1364_p1(32 - 1 downto 0);
    p_Val2_2_1_fu_108_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_2_1_fu_108_p0) * signed('0' &ap_const_lv56_11A736D))), 56));
    p_Val2_2_2_fu_116_p0 <= OP1_V_2_cast2_fu_1359_p0;
    p_Val2_2_2_fu_116_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_2_2_fu_116_p0) * signed('0' &ap_const_lv51_209B0))), 51));
    p_Val2_2_3_fu_113_p0 <= OP1_V_2_cast_fu_1364_p1(32 - 1 downto 0);
    p_Val2_2_3_fu_113_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_2_3_fu_113_p0) * signed(ap_const_lv56_FFFFFFFDC9B874))), 56));
    p_Val2_2_fu_115_p0 <= OP1_V_2_cast_fu_1364_p1(32 - 1 downto 0);
    p_Val2_2_fu_115_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_2_fu_115_p0) * signed(ap_const_lv56_FFFFFFFED41943))), 56));
    p_Val2_3_1_fu_119_p0 <= OP1_V_3_cast_fu_1411_p1(32 - 1 downto 0);
    p_Val2_3_1_fu_119_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_3_1_fu_119_p0) * signed('0' &ap_const_lv56_1C58863))), 56));
    p_Val2_3_2_fu_120_p0 <= OP1_V_3_cast_fu_1411_p1(32 - 1 downto 0);
    p_Val2_3_2_fu_120_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_3_2_fu_120_p0) * signed(ap_const_lv56_FFFFFFFD3C5E87))), 56));
    p_Val2_3_3_fu_106_p0 <= OP1_V_3_cast_fu_1411_p1(32 - 1 downto 0);
    p_Val2_3_3_fu_106_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_3_3_fu_106_p0) * signed(ap_const_lv56_FFFFFFFDB73A83))), 56));
    p_Val2_3_fu_117_p0 <= OP1_V_3_cast_fu_1411_p1(32 - 1 downto 0);
    p_Val2_3_fu_117_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_3_fu_117_p0) * signed('0' &ap_const_lv56_26B80DF))), 56));
    p_Val2_s_fu_107_p0 <= OP1_V_cast_fu_1264_p1(32 - 1 downto 0);
    p_Val2_s_fu_107_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_s_fu_107_p0) * signed(ap_const_lv56_FFFFFFFD204C14))), 56));
    res_0_V_write_assig_fu_1479_p2 <= std_logic_vector(unsigned(tmp2_fu_1474_p2) + unsigned(tmp1_fu_1465_p2));
    res_1_V_write_assig_fu_1500_p2 <= std_logic_vector(unsigned(tmp5_fu_1495_p2) + unsigned(tmp4_fu_1485_p2));
    res_2_V_write_assig_fu_1521_p2 <= std_logic_vector(unsigned(tmp8_fu_1515_p2) + unsigned(tmp7_fu_1506_p2));
    res_3_V_write_assig_fu_1541_p2 <= std_logic_vector(unsigned(tmp11_fu_1536_p2) + unsigned(tmp10_fu_1527_p2));
    tmp10_fu_1527_p2 <= std_logic_vector(unsigned(tmp_153_0_3_reg_1586) + unsigned(tmp_153_1_3_reg_1606));
    tmp11_fu_1536_p2 <= std_logic_vector(unsigned(tmp12_fu_1531_p2) + unsigned(tmp_153_2_3_reg_1626));
    tmp12_fu_1531_p2 <= std_logic_vector(unsigned(tmp_153_3_3_reg_1646) + unsigned(ap_const_lv32_C3F5E5));
    tmp1_fu_1465_p2 <= std_logic_vector(unsigned(tmp_108_reg_1571) + unsigned(tmp_153_1_reg_1591));
    tmp2_fu_1474_p2 <= std_logic_vector(unsigned(tmp3_fu_1469_p2) + unsigned(tmp_153_2_reg_1611));
    tmp3_fu_1469_p2 <= std_logic_vector(unsigned(tmp_153_3_reg_1631) + unsigned(ap_const_lv32_FE0FDC17));
    tmp4_fu_1485_p2 <= std_logic_vector(signed(tmp_109_fu_1459_p1) + signed(tmp_153_1_1_reg_1596));
    tmp5_fu_1495_p2 <= std_logic_vector(unsigned(tmp6_fu_1490_p2) + unsigned(tmp_153_2_1_reg_1616));
    tmp6_fu_1490_p2 <= std_logic_vector(unsigned(tmp_153_3_1_reg_1636) + unsigned(ap_const_lv32_FF73B05C));
    tmp7_fu_1506_p2 <= std_logic_vector(unsigned(tmp_153_0_2_reg_1581) + unsigned(tmp_153_1_2_reg_1601));
    tmp8_fu_1515_p2 <= std_logic_vector(unsigned(tmp9_fu_1510_p2) + unsigned(tmp_111_fu_1462_p1));
    tmp9_fu_1510_p2 <= std_logic_vector(unsigned(tmp_153_3_2_reg_1641) + unsigned(ap_const_lv32_18AE46E));
        tmp_109_fu_1459_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(tmp_s_reg_1576),32));

        tmp_111_fu_1462_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(tmp_110_reg_1621),32));

end behav;
