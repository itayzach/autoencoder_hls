-- ==============================================================
-- RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
-- Version: 2018.2.1
-- Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
-- 
-- ===========================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity compute_layer_0_0_0 is
port (
    ap_clk : IN STD_LOGIC;
    ap_rst : IN STD_LOGIC;
    data_0_V_read : IN STD_LOGIC_VECTOR (31 downto 0);
    data_1_V_read : IN STD_LOGIC_VECTOR (31 downto 0);
    data_2_V_read : IN STD_LOGIC_VECTOR (31 downto 0);
    data_3_V_read : IN STD_LOGIC_VECTOR (31 downto 0);
    ap_return_0 : OUT STD_LOGIC_VECTOR (31 downto 0);
    ap_return_1 : OUT STD_LOGIC_VECTOR (31 downto 0);
    ap_ce : IN STD_LOGIC );
end;


architecture behav of compute_layer_0_0_0 is 
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_boolean_1 : BOOLEAN := true;
    constant ap_const_boolean_0 : BOOLEAN := false;
    constant ap_const_lv56_FFFFFFFF3D0355 : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111111001111010000001101010101";
    constant ap_const_lv56_18A9781 : STD_LOGIC_VECTOR (55 downto 0) := "00000000000000000000000000000001100010101001011110000001";
    constant ap_const_lv51_7545F : STD_LOGIC_VECTOR (50 downto 0) := "000000000000000000000000000000001110101010001011111";
    constant ap_const_lv56_FFFFFFFF29BC5D : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111111001010011011110001011101";
    constant ap_const_lv56_1000285 : STD_LOGIC_VECTOR (55 downto 0) := "00000000000000000000000000000001000000000000001010000101";
    constant ap_const_lv56_BE5370 : STD_LOGIC_VECTOR (55 downto 0) := "00000000000000000000000000000000101111100101001101110000";
    constant ap_const_lv56_FFFFFFFF10962D : STD_LOGIC_VECTOR (55 downto 0) := "11111111111111111111111111111111000100001001011000101101";
    constant ap_const_lv56_156C271 : STD_LOGIC_VECTOR (55 downto 0) := "00000000000000000000000000000001010101101100001001110001";
    constant ap_const_lv32_18 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000011000";
    constant ap_const_lv32_37 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000110111";
    constant ap_const_lv32_32 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000110010";
    constant ap_const_lv32_EFAA0 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000011101111101010100000";
    constant ap_const_lv32_FFECF215 : STD_LOGIC_VECTOR (31 downto 0) := "11111111111011001111001000010101";
    constant ap_const_logic_0 : STD_LOGIC := '0';

    signal tmp_115_reg_571 : STD_LOGIC_VECTOR (31 downto 0);
    signal ap_block_state1_pp0_stage0_iter0 : BOOLEAN;
    signal ap_block_state2_pp0_stage0_iter1 : BOOLEAN;
    signal ap_block_pp0_stage0_11001 : BOOLEAN;
    signal tmp_123_0_1_reg_576 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_123_1_reg_581 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_123_1_1_reg_586 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_123_2_reg_591 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_101_reg_596 : STD_LOGIC_VECTOR (26 downto 0);
    signal tmp_123_3_reg_601 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_123_3_1_reg_606 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_1_1_fu_86_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal OP1_V_1_cast_fu_433_p1 : STD_LOGIC_VECTOR (55 downto 0);
    signal ap_block_pp0_stage0 : BOOLEAN;
    signal p_Val2_3_1_fu_87_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal OP1_V_3_cast_fu_489_p1 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_2_1_fu_88_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_3_fu_89_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_0_1_fu_90_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal OP1_V_cast_fu_407_p1 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_2_fu_91_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_1_fu_92_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_s_fu_93_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_s_fu_93_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_0_1_fu_90_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_1_fu_92_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_1_1_fu_86_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal OP1_V_2_cast7_fu_459_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal OP1_V_2_cast_fu_464_p0 : STD_LOGIC_VECTOR (31 downto 0);
    signal p_Val2_2_fu_91_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_2_1_fu_88_p2 : STD_LOGIC_VECTOR (50 downto 0);
    signal p_Val2_3_fu_89_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal p_Val2_3_1_fu_87_p2 : STD_LOGIC_VECTOR (55 downto 0);
    signal tmp3_fu_522_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp2_fu_527_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp1_fu_518_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp6_fu_542_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp_102_fu_515_p1 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp5_fu_547_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal tmp4_fu_538_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal res_0_V_write_assig_fu_532_p2 : STD_LOGIC_VECTOR (31 downto 0);
    signal res_1_V_write_assig_fu_553_p2 : STD_LOGIC_VECTOR (31 downto 0);


begin



    process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if (((ap_const_boolean_0 = ap_block_pp0_stage0_11001) and (ap_const_logic_1 = ap_ce))) then
                tmp_101_reg_596 <= p_Val2_2_1_fu_88_p2(50 downto 24);
                tmp_115_reg_571 <= p_Val2_s_fu_93_p2(55 downto 24);
                tmp_123_0_1_reg_576 <= p_Val2_0_1_fu_90_p2(55 downto 24);
                tmp_123_1_1_reg_586 <= p_Val2_1_1_fu_86_p2(55 downto 24);
                tmp_123_1_reg_581 <= p_Val2_1_fu_92_p2(55 downto 24);
                tmp_123_2_reg_591 <= p_Val2_2_fu_91_p2(55 downto 24);
                tmp_123_3_1_reg_606 <= p_Val2_3_1_fu_87_p2(55 downto 24);
                tmp_123_3_reg_601 <= p_Val2_3_fu_89_p2(55 downto 24);
            end if;
        end if;
    end process;
        OP1_V_1_cast_fu_433_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(data_1_V_read),56));

    OP1_V_2_cast7_fu_459_p0 <= data_2_V_read;
    OP1_V_2_cast_fu_464_p0 <= data_2_V_read;
        OP1_V_3_cast_fu_489_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(data_3_V_read),56));

        OP1_V_cast_fu_407_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(data_0_V_read),56));

        ap_block_pp0_stage0 <= not((ap_const_boolean_1 = ap_const_boolean_1));
        ap_block_pp0_stage0_11001 <= not((ap_const_boolean_1 = ap_const_boolean_1));
        ap_block_state1_pp0_stage0_iter0 <= not((ap_const_boolean_1 = ap_const_boolean_1));
        ap_block_state2_pp0_stage0_iter1 <= not((ap_const_boolean_1 = ap_const_boolean_1));
    ap_return_0 <= res_0_V_write_assig_fu_532_p2;
    ap_return_1 <= res_1_V_write_assig_fu_553_p2;
    p_Val2_0_1_fu_90_p0 <= OP1_V_cast_fu_407_p1(32 - 1 downto 0);
    p_Val2_0_1_fu_90_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_0_1_fu_90_p0) * signed('0' &ap_const_lv56_1000285))), 56));
    p_Val2_1_1_fu_86_p0 <= OP1_V_1_cast_fu_433_p1(32 - 1 downto 0);
    p_Val2_1_1_fu_86_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_1_1_fu_86_p0) * signed(ap_const_lv56_FFFFFFFF3D0355))), 56));
    p_Val2_1_fu_92_p0 <= OP1_V_1_cast_fu_433_p1(32 - 1 downto 0);
    p_Val2_1_fu_92_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_1_fu_92_p0) * signed(ap_const_lv56_FFFFFFFF10962D))), 56));
    p_Val2_2_1_fu_88_p0 <= OP1_V_2_cast7_fu_459_p0;
    p_Val2_2_1_fu_88_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_2_1_fu_88_p0) * signed('0' &ap_const_lv51_7545F))), 51));
    p_Val2_2_fu_91_p0 <= OP1_V_2_cast_fu_464_p0;
    p_Val2_2_fu_91_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_2_fu_91_p0) * signed('0' &ap_const_lv56_BE5370))), 56));
    p_Val2_3_1_fu_87_p0 <= OP1_V_3_cast_fu_489_p1(32 - 1 downto 0);
    p_Val2_3_1_fu_87_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_3_1_fu_87_p0) * signed('0' &ap_const_lv56_18A9781))), 56));
    p_Val2_3_fu_89_p0 <= OP1_V_3_cast_fu_489_p1(32 - 1 downto 0);
    p_Val2_3_fu_89_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_3_fu_89_p0) * signed(ap_const_lv56_FFFFFFFF29BC5D))), 56));
    p_Val2_s_fu_93_p0 <= OP1_V_cast_fu_407_p1(32 - 1 downto 0);
    p_Val2_s_fu_93_p2 <= std_logic_vector(IEEE.numeric_std.resize(unsigned(std_logic_vector(signed(p_Val2_s_fu_93_p0) * signed('0' &ap_const_lv56_156C271))), 56));
    res_0_V_write_assig_fu_532_p2 <= std_logic_vector(unsigned(tmp2_fu_527_p2) + unsigned(tmp1_fu_518_p2));
    res_1_V_write_assig_fu_553_p2 <= std_logic_vector(unsigned(tmp5_fu_547_p2) + unsigned(tmp4_fu_538_p2));
    tmp1_fu_518_p2 <= std_logic_vector(unsigned(tmp_115_reg_571) + unsigned(tmp_123_1_reg_581));
    tmp2_fu_527_p2 <= std_logic_vector(unsigned(tmp3_fu_522_p2) + unsigned(tmp_123_2_reg_591));
    tmp3_fu_522_p2 <= std_logic_vector(unsigned(tmp_123_3_reg_601) + unsigned(ap_const_lv32_EFAA0));
    tmp4_fu_538_p2 <= std_logic_vector(unsigned(tmp_123_0_1_reg_576) + unsigned(tmp_123_1_1_reg_586));
    tmp5_fu_547_p2 <= std_logic_vector(unsigned(tmp6_fu_542_p2) + unsigned(tmp_102_fu_515_p1));
    tmp6_fu_542_p2 <= std_logic_vector(unsigned(tmp_123_3_1_reg_606) + unsigned(ap_const_lv32_FFECF215));
        tmp_102_fu_515_p1 <= std_logic_vector(IEEE.numeric_std.resize(signed(tmp_101_reg_596),32));

end behav;
