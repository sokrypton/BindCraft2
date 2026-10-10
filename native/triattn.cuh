// AF2's triangle attention, at the two widths it is used at.
//
//   tattn    C=128, H=4, D=32  the Evoformer's 48 blocks and the extra-MSA
//                              stack, on the fa/fb flash build
//   tattn64  C= 64, H=4, D=16  the template pair stack, on fa16/fb16
//
// The head dim follows C/H, so the width choice here also picks which
// FlashAttention instantiation the module calls -- that pairing is what TA_FA
// and TA_FB carry into the body.
#pragma once
#include "flash.cuh"
#include "flash_bwd.cuh"

#define TA_NS tattn
#define TA_C 128
#define TA_FA fa
#define TA_FB fb
#include "triattn_body.cuh"
#undef TA_NS
#undef TA_C
#undef TA_FA
#undef TA_FB

#if NE_FD16
#define TA_NS tattn64
#define TA_C 64
#define TA_FA fa16
#define TA_FB fb16
#include "triattn_body.cuh"
#undef TA_NS
#undef TA_C
#undef TA_FA
#undef TA_FB
#endif
