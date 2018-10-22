-- ==============================================================
-- RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
-- Version: 2018.2
-- Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
-- 
-- ===========================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity relu_1 is
port (
    ap_ready : OUT STD_LOGIC;
    data_0_V_read : IN STD_LOGIC_VECTOR (31 downto 0);
    data_1_V_read : IN STD_LOGIC_VECTOR (31 downto 0);
    data_2_V_read : IN STD_LOGIC_VECTOR (31 downto 0);
    data_3_V_read : IN STD_LOGIC_VECTOR (31 downto 0);
    ap_return_0 : OUT STD_LOGIC_VECTOR (31 downto 0);
    ap_return_1 : OUT STD_LOGIC_VECTOR (31 downto 0);
    ap_return_2 : OUT STD_LOGIC_VECTOR (31 downto 0);
    ap_return_3 : OUT STD_LOGIC_VECTOR (31 downto 0) );
end;


architecture behav of relu_1 is 
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_boolean_1 : BOOLEAN := true;
    constant ap_const_lv32_0 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000000";
    constant ap_const_lv31_0 : STD_LOGIC_VECTOR (30 downto 0) := "0000000000000000000000000000000";
    constant ap_const_logic_0 : STD_LOGIC := '0';

    signal tmp_s_fu_54_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_204_fu_60_p1 : STD_LOGIC_VECTOR (30 downto 0);
    signal res_0_V_write_assig_fu_64_p3 : STD_LOGIC_VECTOR (30 downto 0);
    signal tmp_98_1_fu_76_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_205_fu_82_p1 : STD_LOGIC_VECTOR (30 downto 0);
    signal res_1_V_write_assig_fu_86_p3 : STD_LOGIC_VECTOR (30 downto 0);
    signal tmp_98_2_fu_98_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_206_fu_104_p1 : STD_LOGIC_VECTOR (30 downto 0);
    signal res_2_V_write_assig_fu_108_p3 : STD_LOGIC_VECTOR (30 downto 0);
    signal tmp_98_3_fu_120_p2 : STD_LOGIC_VECTOR (0 downto 0);
    signal tmp_207_fu_126_p1 : STD_LOGIC_VECTOR (30 downto 0);
    signal res_3_V_write_assig_fu_130_p3 : STD_LOGIC_VECTOR (30 downto 0);
    signal res_0_V_write_assig_2_fu_72_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal res_1_V_write_assig_2_fu_94_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal res_2_V_write_assig_2_fu_116_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal res_3_V_write_assig_2_fu_138_p1 : STD_LOGIC_VECTOR (31 downto 0);


begin



    ap_ready <= ap_const_logic_1;
    ap_return_0 <= res_0_V_write_assig_2_fu_72_p1;
    ap_return_1 <= res_1_V_write_assig_2_fu_94_p1;
    ap_return_2 <= res_2_V_write_assig_2_fu_116_p1;
    ap_return_3 <= res_3_V_write_assig_2_fu_138_p1;
    res_0_V_write_assig_2_fu_72_p1 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(res_0_V_write_assig_fu_64_p3),32));
    res_0_V_write_assig_fu_64_p3 <= 
        tmp_204_fu_60_p1 when (tmp_s_fu_54_p2(0) = '1') else 
        ap_const_lv31_0;
    res_1_V_write_assig_2_fu_94_p1 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(res_1_V_write_assig_fu_86_p3),32));
    res_1_V_write_assig_fu_86_p3 <= 
        tmp_205_fu_82_p1 when (tmp_98_1_fu_76_p2(0) = '1') else 
        ap_const_lv31_0;
    res_2_V_write_assig_2_fu_116_p1 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(res_2_V_write_assig_fu_108_p3),32));
    res_2_V_write_assig_fu_108_p3 <= 
        tmp_206_fu_104_p1 when (tmp_98_2_fu_98_p2(0) = '1') else 
        ap_const_lv31_0;
    res_3_V_write_assig_2_fu_138_p1 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(res_3_V_write_assig_fu_130_p3),32));
    res_3_V_write_assig_fu_130_p3 <= 
        tmp_207_fu_126_p1 when (tmp_98_3_fu_120_p2(0) = '1') else 
        ap_const_lv31_0;
    tmp_204_fu_60_p1 <= data_0_V_read(31 - 1 downto 0);
    tmp_205_fu_82_p1 <= data_1_V_read(31 - 1 downto 0);
    tmp_206_fu_104_p1 <= data_2_V_read(31 - 1 downto 0);
    tmp_207_fu_126_p1 <= data_3_V_read(31 - 1 downto 0);
    tmp_98_1_fu_76_p2 <= "1" when (signed(data_1_V_read) > signed(ap_const_lv32_0)) else "0";
    tmp_98_2_fu_98_p2 <= "1" when (signed(data_2_V_read) > signed(ap_const_lv32_0)) else "0";
    tmp_98_3_fu_120_p2 <= "1" when (signed(data_3_V_read) > signed(ap_const_lv32_0)) else "0";
    tmp_s_fu_54_p2 <= "1" when (signed(data_0_V_read) > signed(ap_const_lv32_0)) else "0";
end behav;
