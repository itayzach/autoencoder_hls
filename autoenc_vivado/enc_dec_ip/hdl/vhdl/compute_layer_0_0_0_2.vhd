-- ==============================================================
-- RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
-- Version: 2018.2.1
-- Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
-- 
-- ===========================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity compute_layer_0_0_0_2 is
port (
    ap_clk : IN STD_LOGIC;
    ap_rst : IN STD_LOGIC;
    data_V_read : IN STD_LOGIC_VECTOR (63 downto 0);
    ap_return_0 : OUT STD_LOGIC_VECTOR (31 downto 0);
    ap_return_1 : OUT STD_LOGIC_VECTOR (31 downto 0);
    ap_return_2 : OUT STD_LOGIC_VECTOR (31 downto 0);
    ap_return_3 : OUT STD_LOGIC_VECTOR (31 downto 0);
    ap_ce : IN STD_LOGIC );
end;


architecture behav of compute_layer_0_0_0_2 is 
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_boolean_1 : BOOLEAN := true;
    constant ap_const_boolean_0 : BOOLEAN := false;
    constant ap_const_lv56_FFFFFFFF3DE627 : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111111001111011110011000100111";
    constant ap_const_lv56_FFFFFFFBE2ABEF : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111011111000101010101111101111";
    constant ap_const_lv56_A41F8F : STD_LOGIC_VECTOR (55 downto 0) := "00000000000000000000000000000000101001000001111110001111";
    constant ap_const_lv56_FFFFFFFDC5F54F : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111101110001011111010101001111";
    constant ap_const_lv56_FFFFFFFCB4AE8F : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111100101101001010111010001111";
    constant ap_const_lv56_8ED3B8 : STD_LOGIC_VECTOR (55 downto 0) := "00000000000000000000000000000000100011101101001110111000";
    constant ap_const_lv56_1AC7A32 : STD_LOGIC_VECTOR (55 downto 0) := "00000000000000000000000000000001101011000111101000110010";
    constant ap_const_lv56_29CEA2F : STD_LOGIC_VECTOR (55 downto 0) := "00000000000000000000000000000010100111001110101000101111";
    constant ap_const_lv32_18 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000011000";
    constant ap_const_lv32_37 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000110111";
    constant ap_const_lv32_20 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000100000";
    constant ap_const_lv32_3F : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000111111";
    constant ap_const_lv32_1A6C3B7 : STD_LOGIC_VECTOR (31 downto 0) := "00000001101001101100001110110111";
    constant ap_const_lv32_1617724 : STD_LOGIC_VECTOR (31 downto 0) := "00000001011000010111011100100100";
    constant ap_const_lv32_FFBEBFA9 : STD_LOGIC_VECTOR (31 downto 0) := "11111111101111101011111110101001";
    constant ap_const_lv32_165AA24 : STD_LOGIC_VECTOR (31 downto 0) := "00000001011001011010101000100100";
    constant ap_const_logic_0 : STD_LOGIC := '0';

    signal tmp_s_reg_563 : STD_LOGIC_VECTOR (31 downto 0);
    signal ap_block_state1_pp0_stage0_iter0 : BOOLEAN;
    signal ap_block_state2_pp0_stage0_iter1 : BOOLEAN;
    signal ap_block_pp0_stage0_11001 : BOOLEAN;
    signal tmp_168_0_1_reg_568 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_168_0_2_reg_573 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_168_0_3_reg_578 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_168_1_reg_583 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_168_1_1_reg_588 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_168_1_2_reg_593 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_168_1_3_reg_598 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_0_3_fu_68_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal OP1_V_cast_fu_393_p1 : STD_LOGIC_VECTOR (55 downto 0);
    signal ap_block_pp0_stage0 : BOOLEAN;
    signal p_Val2_1_3_fu_69_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal OP1_V_1_cast_fu_451_p1 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_1_1_fu_70_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_1_2_fu_71_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_0_1_fu_72_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_1_fu_73_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_0_2_fu_74_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_s_fu_75_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_223_fu_389_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_s_fu_75_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_0_1_fu_72_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_0_2_fu_74_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_0_3_fu_68_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal tmp_93_fu_441_p4 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_1_fu_73_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_1_1_fu_70_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_1_2_fu_71_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_1_3_fu_69_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal tmp1_fu_499_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp2_fu_509_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp3_fu_519_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp4_fu_529_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal res_0_V_write_assig_fu_504_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal res_1_V_write_assig_fu_514_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal res_2_V_write_assig_fu_524_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal res_3_V_write_assig_fu_534_p2 : STD_LOGIC_VECTOR (31 downto 0);


begin



    process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if (((ap_const_boolean_0 = ap_block_pp0_stage0_11001) and (ap_const_logic_1 = ap_ce))) then
                tmp_168_0_1_reg_568 <= p_Val2_0_1_fu_72_p2(55 downto 24);
                tmp_168_0_2_reg_573 <= p_Val2_0_2_fu_74_p2(55 downto 24);
                tmp_168_0_3_reg_578 <= p_Val2_0_3_fu_68_p2(55 downto 24);
                tmp_168_1_1_reg_588 <= p_Val2_1_1_fu_70_p2(55 downto 24);
                tmp_168_1_2_reg_593 <= p_Val2_1_2_fu_71_p2(55 downto 24);
                tmp_168_1_3_reg_598 <= p_Val2_1_3_fu_69_p2(55 downto 24);
                tmp_168_1_reg_583 <= p_Val2_1_fu_73_p2(55 downto 24);
                tmp_s_reg_563 <= p_Val2_s_fu_75_p2(55 downto 24);
            end if;
        end if;
    end process;
        OP1_V_1_cast_fu_451_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(tmp_93_fu_441_p4),56));

        OP1_V_cast_fu_393_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(tmp_223_fu_389_p1),56));

        ap_block_pp0_stage0 <= not((ap_const_boolean_1 = ap_const_boolean_1));
        ap_block_pp0_stage0_11001 <= not((ap_const_boolean_1 = ap_const_boolean_1));
        ap_block_state1_pp0_stage0_iter0 <= not((ap_const_boolean_1 = ap_const_boolean_1));
        ap_block_state2_pp0_stage0_iter1 <= not((ap_const_boolean_1 = ap_const_boolean_1));
    ap_return_0 <= res_0_V_write_assig_fu_504_p2;
    ap_return_1 <= res_1_V_write_assig_fu_514_p2;
    ap_return_2 <= res_2_V_write_assig_fu_524_p2;
    ap_return_3 <= res_3_V_write_assig_fu_534_p2;
    p_Val2_0_1_fu_72_p1 <= OP1_V_cast_fu_393_p1(32 - 1 downto 0);
    p_Val2_0_1_fu_72_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(ap_const_lv56_FFFFFFFCB4AE8F) * signed(p_Val2_0_1_fu_72_p1))), 56));
    p_Val2_0_2_fu_74_p1 <= OP1_V_cast_fu_393_p1(32 - 1 downto 0);
    p_Val2_0_2_fu_74_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed('0' &ap_const_lv56_1AC7A32) * signed(p_Val2_0_2_fu_74_p1))), 56));
    p_Val2_0_3_fu_68_p1 <= OP1_V_cast_fu_393_p1(32 - 1 downto 0);
    p_Val2_0_3_fu_68_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(ap_const_lv56_FFFFFFFF3DE627) * signed(p_Val2_0_3_fu_68_p1))), 56));
    p_Val2_1_1_fu_70_p1 <= OP1_V_1_cast_fu_451_p1(32 - 1 downto 0);
    p_Val2_1_1_fu_70_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed('0' &ap_const_lv56_A41F8F) * signed(p_Val2_1_1_fu_70_p1))), 56));
    p_Val2_1_2_fu_71_p1 <= OP1_V_1_cast_fu_451_p1(32 - 1 downto 0);
    p_Val2_1_2_fu_71_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(ap_const_lv56_FFFFFFFDC5F54F) * signed(p_Val2_1_2_fu_71_p1))), 56));
    p_Val2_1_3_fu_69_p1 <= OP1_V_1_cast_fu_451_p1(32 - 1 downto 0);
    p_Val2_1_3_fu_69_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(ap_const_lv56_FFFFFFFBE2ABEF) * signed(p_Val2_1_3_fu_69_p1))), 56));
    p_Val2_1_fu_73_p1 <= OP1_V_1_cast_fu_451_p1(32 - 1 downto 0);
    p_Val2_1_fu_73_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed('0' &ap_const_lv56_8ED3B8) * signed(p_Val2_1_fu_73_p1))), 56));
    p_Val2_s_fu_75_p1 <= OP1_V_cast_fu_393_p1(32 - 1 downto 0);
    p_Val2_s_fu_75_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed('0' &ap_const_lv56_29CEA2F) * signed(p_Val2_s_fu_75_p1))), 56));
    res_0_V_write_assig_fu_504_p2 <= std_logic_vector(unsigned(tmp_s_reg_563) + unsigned(tmp1_fu_499_p2));
    res_1_V_write_assig_fu_514_p2 <= std_logic_vector(unsigned(tmp_168_0_1_reg_568) + unsigned(tmp2_fu_509_p2));
    res_2_V_write_assig_fu_524_p2 <= std_logic_vector(unsigned(tmp_168_0_2_reg_573) + unsigned(tmp3_fu_519_p2));
    res_3_V_write_assig_fu_534_p2 <= std_logic_vector(unsigned(tmp_168_0_3_reg_578) + unsigned(tmp4_fu_529_p2));
    tmp1_fu_499_p2 <= std_logic_vector(unsigned(ap_const_lv32_1A6C3B7) + unsigned(tmp_168_1_reg_583));
    tmp2_fu_509_p2 <= std_logic_vector(unsigned(ap_const_lv32_1617724) + unsigned(tmp_168_1_1_reg_588));
    tmp3_fu_519_p2 <= std_logic_vector(signed(ap_const_lv32_FFBEBFA9) + signed(tmp_168_1_2_reg_593));
    tmp4_fu_529_p2 <= std_logic_vector(unsigned(ap_const_lv32_165AA24) + unsigned(tmp_168_1_3_reg_598));
    tmp_223_fu_389_p1 <= data_V_read(32 - 1 downto 0);
    tmp_93_fu_441_p4 <= data_V_read(63 downto 32);
end behav;
