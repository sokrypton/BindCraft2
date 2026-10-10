// The FlashAttention backward, at the two head dims (see flash.cuh).
//
//   fb    FD=32  triangle attention and MSA row attention
//   fb16  FD=16  the template pair stack
//
// -DFB_LDH / -DFB_LDF still sweep the 32-wide build's shared leading
// dimensions; at FD=16 the default LDH follows FD rather than staying 40,
// because LDH's floor IS FD and shared memory is this kernel's binding
// occupancy constraint.
#pragma once
#include "flash.cuh"

#define FB_NS fb
#define FB_FD 32
#include "flash_bwd_body.cuh"
#undef FB_NS
#undef FB_FD

#if NE_FD16
#define FB_NS fb16
#define FB_FD 16
// the sweep macros name the WIDE build's tuning; the narrow build derives.
#undef FB_LDH
#include "flash_bwd_body.cuh"
#undef FB_NS
#undef FB_FD
#endif
