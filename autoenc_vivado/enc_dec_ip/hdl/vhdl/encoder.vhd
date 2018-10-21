-- ==============================================================
-- RTL generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
-- Version: 2018.2.1
-- Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
-- 
-- ===========================================================

library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

entity encoder is
port (
    ap_clk : IN STD_LOGIC;
    ap_rst : IN STD_LOGIC;
    ap_start : IN STD_LOGIC;
    ap_done : OUT STD_LOGIC;
    ap_idle : OUT STD_LOGIC;
    ap_ready : OUT STD_LOGIC;
    data_V_read : IN STD_LOGIC_VECTOR (127 downto 0);
    ap_return : OUT STD_LOGIC_VECTOR (63 downto 0) );
end;


architecture behav of encoder is 
    constant ap_const_logic_1 : STD_LOGIC := '1';
    constant ap_const_logic_0 : STD_LOGIC := '0';
    constant ap_ST_fsm_state1 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001";
    constant ap_ST_fsm_state2 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010";
    constant ap_ST_fsm_state3 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100";
    constant ap_ST_fsm_state4 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000";
    constant ap_ST_fsm_state5 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000";
    constant ap_ST_fsm_state6 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000";
    constant ap_ST_fsm_state7 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000";
    constant ap_ST_fsm_state8 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000";
    constant ap_ST_fsm_state9 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000";
    constant ap_ST_fsm_state10 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000";
    constant ap_ST_fsm_state11 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000";
    constant ap_ST_fsm_state12 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000";
    constant ap_ST_fsm_state13 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000";
    constant ap_ST_fsm_state14 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000";
    constant ap_ST_fsm_state15 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000";
    constant ap_ST_fsm_state16 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000";
    constant ap_ST_fsm_state17 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000";
    constant ap_ST_fsm_state18 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000";
    constant ap_ST_fsm_state19 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000";
    constant ap_ST_fsm_state20 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000";
    constant ap_ST_fsm_state21 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000";
    constant ap_ST_fsm_state22 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000";
    constant ap_ST_fsm_state23 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000";
    constant ap_ST_fsm_state24 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000";
    constant ap_ST_fsm_state25 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000";
    constant ap_ST_fsm_state26 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000";
    constant ap_ST_fsm_state27 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000";
    constant ap_ST_fsm_state28 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000";
    constant ap_ST_fsm_state29 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000";
    constant ap_ST_fsm_state30 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000";
    constant ap_ST_fsm_state31 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000";
    constant ap_ST_fsm_state32 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000";
    constant ap_ST_fsm_state33 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000";
    constant ap_ST_fsm_state34 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000";
    constant ap_ST_fsm_state35 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000";
    constant ap_ST_fsm_state36 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000";
    constant ap_ST_fsm_state37 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000";
    constant ap_ST_fsm_state38 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000";
    constant ap_ST_fsm_state39 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000";
    constant ap_ST_fsm_state40 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000";
    constant ap_ST_fsm_state41 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000";
    constant ap_ST_fsm_state42 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000";
    constant ap_ST_fsm_state43 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state44 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state45 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state46 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state47 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state48 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state49 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state50 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state51 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state52 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state53 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state54 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state55 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state56 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state57 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state58 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state59 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state60 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state61 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state62 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state63 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state64 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state65 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state66 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state67 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state68 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state69 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state70 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state71 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state72 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state73 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state74 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state75 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state76 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state77 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state78 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state79 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state80 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state81 : STD_LOGIC_VECTOR (90 downto 0) := "0000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state82 : STD_LOGIC_VECTOR (90 downto 0) := "0000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state83 : STD_LOGIC_VECTOR (90 downto 0) := "0000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state84 : STD_LOGIC_VECTOR (90 downto 0) := "0000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state85 : STD_LOGIC_VECTOR (90 downto 0) := "0000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state86 : STD_LOGIC_VECTOR (90 downto 0) := "0000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state87 : STD_LOGIC_VECTOR (90 downto 0) := "0000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state88 : STD_LOGIC_VECTOR (90 downto 0) := "0001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state89 : STD_LOGIC_VECTOR (90 downto 0) := "0010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state90 : STD_LOGIC_VECTOR (90 downto 0) := "0100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_ST_fsm_state91 : STD_LOGIC_VECTOR (90 downto 0) := "1000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000";
    constant ap_const_lv32_0 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000000";
    constant ap_const_lv32_1 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000001";
    constant ap_const_lv32_2 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000010";
    constant ap_const_lv32_4 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000100";
    constant ap_const_lv32_3 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000011";
    constant ap_const_lv32_5 : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000000000101";
    constant ap_const_lv32_5A : STD_LOGIC_VECTOR (31 downto 0) := "00000000000000000000000001011010";
    constant ap_const_boolean_1 : BOOLEAN := true;

    signal ap_CS_fsm : STD_LOGIC_VECTOR (90 downto 0) := "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001";
    attribute fsm_encoding : string;
    attribute fsm_encoding of ap_CS_fsm : signal is "none";
    signal ap_CS_fsm_state1 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state1 : signal is "none";
    signal logits1_0_V_reg_96 : STD_LOGIC_VECTOR (31 downto 0);
    signal ap_CS_fsm_state2 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state2 : signal is "none";
    signal logits1_1_V_reg_101 : STD_LOGIC_VECTOR (31 downto 0);
    signal logits1_2_V_reg_106 : STD_LOGIC_VECTOR (31 downto 0);
    signal logits1_3_V_reg_111 : STD_LOGIC_VECTOR (31 downto 0);
    signal layer1_relu_out_0_V_reg_116 : STD_LOGIC_VECTOR (31 downto 0);
    signal ap_CS_fsm_state3 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state3 : signal is "none";
    signal layer1_relu_out_1_V_reg_121 : STD_LOGIC_VECTOR (31 downto 0);
    signal layer1_relu_out_2_V_reg_126 : STD_LOGIC_VECTOR (31 downto 0);
    signal layer1_relu_out_3_V_reg_131 : STD_LOGIC_VECTOR (31 downto 0);
    signal logits2_0_V_reg_136 : STD_LOGIC_VECTOR (31 downto 0);
    signal ap_CS_fsm_state5 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state5 : signal is "none";
    signal logits2_1_V_reg_141 : STD_LOGIC_VECTOR (31 downto 0);
    signal grp_normalization_layer_fu_28_ap_start : STD_LOGIC;
    signal grp_normalization_layer_fu_28_ap_done : STD_LOGIC;
    signal grp_normalization_layer_fu_28_ap_idle : STD_LOGIC;
    signal grp_normalization_layer_fu_28_ap_ready : STD_LOGIC;
    signal grp_normalization_layer_fu_28_ap_return : STD_LOGIC_VECTOR (63 downto 0);
    signal grp_compute_layer_0_0_0_s_fu_34_ap_return_0 : STD_LOGIC_VECTOR (31 downto 0);
    signal grp_compute_layer_0_0_0_s_fu_34_ap_return_1 : STD_LOGIC_VECTOR (31 downto 0);
    signal grp_compute_layer_0_0_0_s_fu_34_ap_return_2 : STD_LOGIC_VECTOR (31 downto 0);
    signal grp_compute_layer_0_0_0_s_fu_34_ap_return_3 : STD_LOGIC_VECTOR (31 downto 0);
    signal grp_compute_layer_0_0_0_s_fu_34_ap_ce : STD_LOGIC;
    signal grp_compute_layer_0_0_0_fu_40_ap_return_0 : STD_LOGIC_VECTOR (31 downto 0);
    signal grp_compute_layer_0_0_0_fu_40_ap_return_1 : STD_LOGIC_VECTOR (31 downto 0);
    signal grp_compute_layer_0_0_0_fu_40_ap_ce : STD_LOGIC;
    signal ap_CS_fsm_state4 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state4 : signal is "none";
    signal call_ret1_relu_fu_48_ap_ready : STD_LOGIC;
    signal call_ret1_relu_fu_48_ap_return_0 : STD_LOGIC_VECTOR (31 downto 0);
    signal call_ret1_relu_fu_48_ap_return_1 : STD_LOGIC_VECTOR (31 downto 0);
    signal call_ret1_relu_fu_48_ap_return_2 : STD_LOGIC_VECTOR (31 downto 0);
    signal call_ret1_relu_fu_48_ap_return_3 : STD_LOGIC_VECTOR (31 downto 0);
    signal grp_normalization_layer_fu_28_ap_start_reg : STD_LOGIC := '0';
    signal ap_NS_fsm : STD_LOGIC_VECTOR (90 downto 0);
    signal ap_NS_fsm_state6 : STD_LOGIC;
    signal ap_CS_fsm_state6 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state6 : signal is "none";
    signal ap_CS_fsm_state91 : STD_LOGIC;
    attribute fsm_encoding of ap_CS_fsm_state91 : signal is "none";

    component normalization_layer IS
    port (
        ap_clk : IN STD_LOGIC;
        ap_rst : IN STD_LOGIC;
        ap_start : IN STD_LOGIC;
        ap_done : OUT STD_LOGIC;
        ap_idle : OUT STD_LOGIC;
        ap_ready : OUT STD_LOGIC;
        data_0_V_read : IN STD_LOGIC_VECTOR (31 downto 0);
        data_1_V_read : IN STD_LOGIC_VECTOR (31 downto 0);
        ap_return : OUT STD_LOGIC_VECTOR (63 downto 0) );
    end component;


    component compute_layer_0_0_0_s IS
    port (
        ap_clk : IN STD_LOGIC;
        ap_rst : IN STD_LOGIC;
        data_V_read : IN STD_LOGIC_VECTOR (127 downto 0);
        ap_return_0 : OUT STD_LOGIC_VECTOR (31 downto 0);
        ap_return_1 : OUT STD_LOGIC_VECTOR (31 downto 0);
        ap_return_2 : OUT STD_LOGIC_VECTOR (31 downto 0);
        ap_return_3 : OUT STD_LOGIC_VECTOR (31 downto 0);
        ap_ce : IN STD_LOGIC );
    end component;


    component compute_layer_0_0_0 IS
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
    end component;


    component relu IS
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
    end component;



begin
    grp_normalization_layer_fu_28 : component normalization_layer
    port map (
        ap_clk => ap_clk,
        ap_rst => ap_rst,
        ap_start => grp_normalization_layer_fu_28_ap_start,
        ap_done => grp_normalization_layer_fu_28_ap_done,
        ap_idle => grp_normalization_layer_fu_28_ap_idle,
        ap_ready => grp_normalization_layer_fu_28_ap_ready,
        data_0_V_read => logits2_0_V_reg_136,
        data_1_V_read => logits2_1_V_reg_141,
        ap_return => grp_normalization_layer_fu_28_ap_return);

    grp_compute_layer_0_0_0_s_fu_34 : component compute_layer_0_0_0_s
    port map (
        ap_clk => ap_clk,
        ap_rst => ap_rst,
        data_V_read => data_V_read,
        ap_return_0 => grp_compute_layer_0_0_0_s_fu_34_ap_return_0,
        ap_return_1 => grp_compute_layer_0_0_0_s_fu_34_ap_return_1,
        ap_return_2 => grp_compute_layer_0_0_0_s_fu_34_ap_return_2,
        ap_return_3 => grp_compute_layer_0_0_0_s_fu_34_ap_return_3,
        ap_ce => grp_compute_layer_0_0_0_s_fu_34_ap_ce);

    grp_compute_layer_0_0_0_fu_40 : component compute_layer_0_0_0
    port map (
        ap_clk => ap_clk,
        ap_rst => ap_rst,
        data_0_V_read => layer1_relu_out_0_V_reg_116,
        data_1_V_read => layer1_relu_out_1_V_reg_121,
        data_2_V_read => layer1_relu_out_2_V_reg_126,
        data_3_V_read => layer1_relu_out_3_V_reg_131,
        ap_return_0 => grp_compute_layer_0_0_0_fu_40_ap_return_0,
        ap_return_1 => grp_compute_layer_0_0_0_fu_40_ap_return_1,
        ap_ce => grp_compute_layer_0_0_0_fu_40_ap_ce);

    call_ret1_relu_fu_48 : component relu
    port map (
        ap_ready => call_ret1_relu_fu_48_ap_ready,
        data_0_V_read => logits1_0_V_reg_96,
        data_1_V_read => logits1_1_V_reg_101,
        data_2_V_read => logits1_2_V_reg_106,
        data_3_V_read => logits1_3_V_reg_111,
        ap_return_0 => call_ret1_relu_fu_48_ap_return_0,
        ap_return_1 => call_ret1_relu_fu_48_ap_return_1,
        ap_return_2 => call_ret1_relu_fu_48_ap_return_2,
        ap_return_3 => call_ret1_relu_fu_48_ap_return_3);





    ap_CS_fsm_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                ap_CS_fsm <= ap_ST_fsm_state1;
            else
                ap_CS_fsm <= ap_NS_fsm;
            end if;
        end if;
    end process;


    grp_normalization_layer_fu_28_ap_start_reg_assign_proc : process(ap_clk)
    begin
        if (ap_clk'event and ap_clk =  '1') then
            if (ap_rst = '1') then
                grp_normalization_layer_fu_28_ap_start_reg <= ap_const_logic_0;
            else
                if (((ap_const_logic_1 = ap_NS_fsm_state6) and (ap_const_logic_1 = ap_CS_fsm_state5))) then 
                    grp_normalization_layer_fu_28_ap_start_reg <= ap_const_logic_1;
                elsif ((grp_normalization_layer_fu_28_ap_ready = ap_const_logic_1)) then 
                    grp_normalization_layer_fu_28_ap_start_reg <= ap_const_logic_0;
                end if; 
            end if;
        end if;
    end process;

    process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if ((ap_const_logic_1 = ap_CS_fsm_state3)) then
                layer1_relu_out_0_V_reg_116 <= call_ret1_relu_fu_48_ap_return_0;
                layer1_relu_out_1_V_reg_121 <= call_ret1_relu_fu_48_ap_return_1;
                layer1_relu_out_2_V_reg_126 <= call_ret1_relu_fu_48_ap_return_2;
                layer1_relu_out_3_V_reg_131 <= call_ret1_relu_fu_48_ap_return_3;
            end if;
        end if;
    end process;
    process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if ((ap_const_logic_1 = ap_CS_fsm_state2)) then
                logits1_0_V_reg_96 <= grp_compute_layer_0_0_0_s_fu_34_ap_return_0;
                logits1_1_V_reg_101 <= grp_compute_layer_0_0_0_s_fu_34_ap_return_1;
                logits1_2_V_reg_106 <= grp_compute_layer_0_0_0_s_fu_34_ap_return_2;
                logits1_3_V_reg_111 <= grp_compute_layer_0_0_0_s_fu_34_ap_return_3;
            end if;
        end if;
    end process;
    process (ap_clk)
    begin
        if (ap_clk'event and ap_clk = '1') then
            if ((ap_const_logic_1 = ap_CS_fsm_state5)) then
                logits2_0_V_reg_136 <= grp_compute_layer_0_0_0_fu_40_ap_return_0;
                logits2_1_V_reg_141 <= grp_compute_layer_0_0_0_fu_40_ap_return_1;
            end if;
        end if;
    end process;

    ap_NS_fsm_assign_proc : process (ap_start, ap_CS_fsm, ap_CS_fsm_state1)
    begin
        case ap_CS_fsm is
            when ap_ST_fsm_state1 => 
                if (((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1))) then
                    ap_NS_fsm <= ap_ST_fsm_state2;
                else
                    ap_NS_fsm <= ap_ST_fsm_state1;
                end if;
            when ap_ST_fsm_state2 => 
                ap_NS_fsm <= ap_ST_fsm_state3;
            when ap_ST_fsm_state3 => 
                ap_NS_fsm <= ap_ST_fsm_state4;
            when ap_ST_fsm_state4 => 
                ap_NS_fsm <= ap_ST_fsm_state5;
            when ap_ST_fsm_state5 => 
                ap_NS_fsm <= ap_ST_fsm_state6;
            when ap_ST_fsm_state6 => 
                ap_NS_fsm <= ap_ST_fsm_state7;
            when ap_ST_fsm_state7 => 
                ap_NS_fsm <= ap_ST_fsm_state8;
            when ap_ST_fsm_state8 => 
                ap_NS_fsm <= ap_ST_fsm_state9;
            when ap_ST_fsm_state9 => 
                ap_NS_fsm <= ap_ST_fsm_state10;
            when ap_ST_fsm_state10 => 
                ap_NS_fsm <= ap_ST_fsm_state11;
            when ap_ST_fsm_state11 => 
                ap_NS_fsm <= ap_ST_fsm_state12;
            when ap_ST_fsm_state12 => 
                ap_NS_fsm <= ap_ST_fsm_state13;
            when ap_ST_fsm_state13 => 
                ap_NS_fsm <= ap_ST_fsm_state14;
            when ap_ST_fsm_state14 => 
                ap_NS_fsm <= ap_ST_fsm_state15;
            when ap_ST_fsm_state15 => 
                ap_NS_fsm <= ap_ST_fsm_state16;
            when ap_ST_fsm_state16 => 
                ap_NS_fsm <= ap_ST_fsm_state17;
            when ap_ST_fsm_state17 => 
                ap_NS_fsm <= ap_ST_fsm_state18;
            when ap_ST_fsm_state18 => 
                ap_NS_fsm <= ap_ST_fsm_state19;
            when ap_ST_fsm_state19 => 
                ap_NS_fsm <= ap_ST_fsm_state20;
            when ap_ST_fsm_state20 => 
                ap_NS_fsm <= ap_ST_fsm_state21;
            when ap_ST_fsm_state21 => 
                ap_NS_fsm <= ap_ST_fsm_state22;
            when ap_ST_fsm_state22 => 
                ap_NS_fsm <= ap_ST_fsm_state23;
            when ap_ST_fsm_state23 => 
                ap_NS_fsm <= ap_ST_fsm_state24;
            when ap_ST_fsm_state24 => 
                ap_NS_fsm <= ap_ST_fsm_state25;
            when ap_ST_fsm_state25 => 
                ap_NS_fsm <= ap_ST_fsm_state26;
            when ap_ST_fsm_state26 => 
                ap_NS_fsm <= ap_ST_fsm_state27;
            when ap_ST_fsm_state27 => 
                ap_NS_fsm <= ap_ST_fsm_state28;
            when ap_ST_fsm_state28 => 
                ap_NS_fsm <= ap_ST_fsm_state29;
            when ap_ST_fsm_state29 => 
                ap_NS_fsm <= ap_ST_fsm_state30;
            when ap_ST_fsm_state30 => 
                ap_NS_fsm <= ap_ST_fsm_state31;
            when ap_ST_fsm_state31 => 
                ap_NS_fsm <= ap_ST_fsm_state32;
            when ap_ST_fsm_state32 => 
                ap_NS_fsm <= ap_ST_fsm_state33;
            when ap_ST_fsm_state33 => 
                ap_NS_fsm <= ap_ST_fsm_state34;
            when ap_ST_fsm_state34 => 
                ap_NS_fsm <= ap_ST_fsm_state35;
            when ap_ST_fsm_state35 => 
                ap_NS_fsm <= ap_ST_fsm_state36;
            when ap_ST_fsm_state36 => 
                ap_NS_fsm <= ap_ST_fsm_state37;
            when ap_ST_fsm_state37 => 
                ap_NS_fsm <= ap_ST_fsm_state38;
            when ap_ST_fsm_state38 => 
                ap_NS_fsm <= ap_ST_fsm_state39;
            when ap_ST_fsm_state39 => 
                ap_NS_fsm <= ap_ST_fsm_state40;
            when ap_ST_fsm_state40 => 
                ap_NS_fsm <= ap_ST_fsm_state41;
            when ap_ST_fsm_state41 => 
                ap_NS_fsm <= ap_ST_fsm_state42;
            when ap_ST_fsm_state42 => 
                ap_NS_fsm <= ap_ST_fsm_state43;
            when ap_ST_fsm_state43 => 
                ap_NS_fsm <= ap_ST_fsm_state44;
            when ap_ST_fsm_state44 => 
                ap_NS_fsm <= ap_ST_fsm_state45;
            when ap_ST_fsm_state45 => 
                ap_NS_fsm <= ap_ST_fsm_state46;
            when ap_ST_fsm_state46 => 
                ap_NS_fsm <= ap_ST_fsm_state47;
            when ap_ST_fsm_state47 => 
                ap_NS_fsm <= ap_ST_fsm_state48;
            when ap_ST_fsm_state48 => 
                ap_NS_fsm <= ap_ST_fsm_state49;
            when ap_ST_fsm_state49 => 
                ap_NS_fsm <= ap_ST_fsm_state50;
            when ap_ST_fsm_state50 => 
                ap_NS_fsm <= ap_ST_fsm_state51;
            when ap_ST_fsm_state51 => 
                ap_NS_fsm <= ap_ST_fsm_state52;
            when ap_ST_fsm_state52 => 
                ap_NS_fsm <= ap_ST_fsm_state53;
            when ap_ST_fsm_state53 => 
                ap_NS_fsm <= ap_ST_fsm_state54;
            when ap_ST_fsm_state54 => 
                ap_NS_fsm <= ap_ST_fsm_state55;
            when ap_ST_fsm_state55 => 
                ap_NS_fsm <= ap_ST_fsm_state56;
            when ap_ST_fsm_state56 => 
                ap_NS_fsm <= ap_ST_fsm_state57;
            when ap_ST_fsm_state57 => 
                ap_NS_fsm <= ap_ST_fsm_state58;
            when ap_ST_fsm_state58 => 
                ap_NS_fsm <= ap_ST_fsm_state59;
            when ap_ST_fsm_state59 => 
                ap_NS_fsm <= ap_ST_fsm_state60;
            when ap_ST_fsm_state60 => 
                ap_NS_fsm <= ap_ST_fsm_state61;
            when ap_ST_fsm_state61 => 
                ap_NS_fsm <= ap_ST_fsm_state62;
            when ap_ST_fsm_state62 => 
                ap_NS_fsm <= ap_ST_fsm_state63;
            when ap_ST_fsm_state63 => 
                ap_NS_fsm <= ap_ST_fsm_state64;
            when ap_ST_fsm_state64 => 
                ap_NS_fsm <= ap_ST_fsm_state65;
            when ap_ST_fsm_state65 => 
                ap_NS_fsm <= ap_ST_fsm_state66;
            when ap_ST_fsm_state66 => 
                ap_NS_fsm <= ap_ST_fsm_state67;
            when ap_ST_fsm_state67 => 
                ap_NS_fsm <= ap_ST_fsm_state68;
            when ap_ST_fsm_state68 => 
                ap_NS_fsm <= ap_ST_fsm_state69;
            when ap_ST_fsm_state69 => 
                ap_NS_fsm <= ap_ST_fsm_state70;
            when ap_ST_fsm_state70 => 
                ap_NS_fsm <= ap_ST_fsm_state71;
            when ap_ST_fsm_state71 => 
                ap_NS_fsm <= ap_ST_fsm_state72;
            when ap_ST_fsm_state72 => 
                ap_NS_fsm <= ap_ST_fsm_state73;
            when ap_ST_fsm_state73 => 
                ap_NS_fsm <= ap_ST_fsm_state74;
            when ap_ST_fsm_state74 => 
                ap_NS_fsm <= ap_ST_fsm_state75;
            when ap_ST_fsm_state75 => 
                ap_NS_fsm <= ap_ST_fsm_state76;
            when ap_ST_fsm_state76 => 
                ap_NS_fsm <= ap_ST_fsm_state77;
            when ap_ST_fsm_state77 => 
                ap_NS_fsm <= ap_ST_fsm_state78;
            when ap_ST_fsm_state78 => 
                ap_NS_fsm <= ap_ST_fsm_state79;
            when ap_ST_fsm_state79 => 
                ap_NS_fsm <= ap_ST_fsm_state80;
            when ap_ST_fsm_state80 => 
                ap_NS_fsm <= ap_ST_fsm_state81;
            when ap_ST_fsm_state81 => 
                ap_NS_fsm <= ap_ST_fsm_state82;
            when ap_ST_fsm_state82 => 
                ap_NS_fsm <= ap_ST_fsm_state83;
            when ap_ST_fsm_state83 => 
                ap_NS_fsm <= ap_ST_fsm_state84;
            when ap_ST_fsm_state84 => 
                ap_NS_fsm <= ap_ST_fsm_state85;
            when ap_ST_fsm_state85 => 
                ap_NS_fsm <= ap_ST_fsm_state86;
            when ap_ST_fsm_state86 => 
                ap_NS_fsm <= ap_ST_fsm_state87;
            when ap_ST_fsm_state87 => 
                ap_NS_fsm <= ap_ST_fsm_state88;
            when ap_ST_fsm_state88 => 
                ap_NS_fsm <= ap_ST_fsm_state89;
            when ap_ST_fsm_state89 => 
                ap_NS_fsm <= ap_ST_fsm_state90;
            when ap_ST_fsm_state90 => 
                ap_NS_fsm <= ap_ST_fsm_state91;
            when ap_ST_fsm_state91 => 
                ap_NS_fsm <= ap_ST_fsm_state1;
            when others =>  
                ap_NS_fsm <= "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";
        end case;
    end process;
    ap_CS_fsm_state1 <= ap_CS_fsm(0);
    ap_CS_fsm_state2 <= ap_CS_fsm(1);
    ap_CS_fsm_state3 <= ap_CS_fsm(2);
    ap_CS_fsm_state4 <= ap_CS_fsm(3);
    ap_CS_fsm_state5 <= ap_CS_fsm(4);
    ap_CS_fsm_state6 <= ap_CS_fsm(5);
    ap_CS_fsm_state91 <= ap_CS_fsm(90);
    ap_NS_fsm_state6 <= ap_NS_fsm(5);

    ap_done_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_CS_fsm_state91)
    begin
        if (((ap_const_logic_1 = ap_CS_fsm_state91) or ((ap_start = ap_const_logic_0) and (ap_const_logic_1 = ap_CS_fsm_state1)))) then 
            ap_done <= ap_const_logic_1;
        else 
            ap_done <= ap_const_logic_0;
        end if; 
    end process;


    ap_idle_assign_proc : process(ap_start, ap_CS_fsm_state1)
    begin
        if (((ap_start = ap_const_logic_0) and (ap_const_logic_1 = ap_CS_fsm_state1))) then 
            ap_idle <= ap_const_logic_1;
        else 
            ap_idle <= ap_const_logic_0;
        end if; 
    end process;


    ap_ready_assign_proc : process(ap_CS_fsm_state91)
    begin
        if ((ap_const_logic_1 = ap_CS_fsm_state91)) then 
            ap_ready <= ap_const_logic_1;
        else 
            ap_ready <= ap_const_logic_0;
        end if; 
    end process;

    ap_return <= grp_normalization_layer_fu_28_ap_return;

    grp_compute_layer_0_0_0_fu_40_ap_ce_assign_proc : process(ap_CS_fsm_state5, ap_CS_fsm_state4)
    begin
        if (((ap_const_logic_1 = ap_CS_fsm_state4) or (ap_const_logic_1 = ap_CS_fsm_state5))) then 
            grp_compute_layer_0_0_0_fu_40_ap_ce <= ap_const_logic_1;
        else 
            grp_compute_layer_0_0_0_fu_40_ap_ce <= ap_const_logic_0;
        end if; 
    end process;


    grp_compute_layer_0_0_0_s_fu_34_ap_ce_assign_proc : process(ap_start, ap_CS_fsm_state1, ap_CS_fsm_state2)
    begin
        if (((ap_const_logic_1 = ap_CS_fsm_state2) or ((ap_start = ap_const_logic_1) and (ap_const_logic_1 = ap_CS_fsm_state1)))) then 
            grp_compute_layer_0_0_0_s_fu_34_ap_ce <= ap_const_logic_1;
        else 
            grp_compute_layer_0_0_0_s_fu_34_ap_ce <= ap_const_logic_0;
        end if; 
    end process;

    grp_normalization_layer_fu_28_ap_start <= grp_normalization_layer_fu_28_ap_start_reg;
end behav;
