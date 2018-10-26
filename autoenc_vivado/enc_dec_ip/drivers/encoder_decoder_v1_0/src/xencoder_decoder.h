// ==============================================================
// File generated by Vivado(TM) HLS - High-Level Synthesis from C, C++ and SystemC
// Version: 2018.2.1
// Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
// 
// ==============================================================

#ifndef XENCODER_DECODER_H
#define XENCODER_DECODER_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xencoder_decoder_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
#else
typedef struct {
    u16 DeviceId;
    u32 Ctrl_BaseAddress;
} XEncoder_decoder_Config;
#endif

typedef struct {
    u32 Ctrl_BaseAddress;
    u32 IsReady;
} XEncoder_decoder;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XEncoder_decoder_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XEncoder_decoder_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XEncoder_decoder_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XEncoder_decoder_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XEncoder_decoder_Initialize(XEncoder_decoder *InstancePtr, u16 DeviceId);
XEncoder_decoder_Config* XEncoder_decoder_LookupConfig(u16 DeviceId);
int XEncoder_decoder_CfgInitialize(XEncoder_decoder *InstancePtr, XEncoder_decoder_Config *ConfigPtr);
#else
int XEncoder_decoder_Initialize(XEncoder_decoder *InstancePtr, const char* InstanceName);
int XEncoder_decoder_Release(XEncoder_decoder *InstancePtr);
#endif

void XEncoder_decoder_Start(XEncoder_decoder *InstancePtr);
u32 XEncoder_decoder_IsDone(XEncoder_decoder *InstancePtr);
u32 XEncoder_decoder_IsIdle(XEncoder_decoder *InstancePtr);
u32 XEncoder_decoder_IsReady(XEncoder_decoder *InstancePtr);
void XEncoder_decoder_EnableAutoRestart(XEncoder_decoder *InstancePtr);
void XEncoder_decoder_DisableAutoRestart(XEncoder_decoder *InstancePtr);

void XEncoder_decoder_Set_SNR_REG_V(XEncoder_decoder *InstancePtr, u32 Data);
u32 XEncoder_decoder_Get_SNR_REG_V(XEncoder_decoder *InstancePtr);
void XEncoder_decoder_Set_AWGN_EN_REG(XEncoder_decoder *InstancePtr, u32 Data);
u32 XEncoder_decoder_Get_AWGN_EN_REG(XEncoder_decoder *InstancePtr);

void XEncoder_decoder_InterruptGlobalEnable(XEncoder_decoder *InstancePtr);
void XEncoder_decoder_InterruptGlobalDisable(XEncoder_decoder *InstancePtr);
void XEncoder_decoder_InterruptEnable(XEncoder_decoder *InstancePtr, u32 Mask);
void XEncoder_decoder_InterruptDisable(XEncoder_decoder *InstancePtr, u32 Mask);
void XEncoder_decoder_InterruptClear(XEncoder_decoder *InstancePtr, u32 Mask);
u32 XEncoder_decoder_InterruptGetEnabled(XEncoder_decoder *InstancePtr);
u32 XEncoder_decoder_InterruptGetStatus(XEncoder_decoder *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
