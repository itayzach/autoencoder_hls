// ==============================================================
// File generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.2.1
// Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
// 
// ==============================================================

// ctrl
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - Channel 0 (ap_done)
//        bit 1  - Channel 1 (ap_ready)
//        others - reserved
// 0x10 : Data signal of SNR_REG_V
//        bit 7~0 - SNR_REG_V[7:0] (Read/Write)
//        others  - reserved
// 0x14 : reserved
// 0x18 : Data signal of AWGN_EN_REG
//        bit 31~0 - AWGN_EN_REG[31:0] (Read/Write)
// 0x1c : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XENCODER_DECODER_CTRL_ADDR_AP_CTRL          0x00
#define XENCODER_DECODER_CTRL_ADDR_GIE              0x04
#define XENCODER_DECODER_CTRL_ADDR_IER              0x08
#define XENCODER_DECODER_CTRL_ADDR_ISR              0x0c
#define XENCODER_DECODER_CTRL_ADDR_SNR_REG_V_DATA   0x10
#define XENCODER_DECODER_CTRL_BITS_SNR_REG_V_DATA   8
#define XENCODER_DECODER_CTRL_ADDR_AWGN_EN_REG_DATA 0x18
#define XENCODER_DECODER_CTRL_BITS_AWGN_EN_REG_DATA 32
