// Triangle multiplication, instantiated at the TWO widths AF2 uses it at.
//
// The module is written against namespace-scope C and I because every tile
// size, warp count and shared-memory extent in it is derived from them at
// compile time.  Two widths therefore means two instantiations rather than a
// runtime parameter:
//
//   tmul    C=I=128  the Evoformer's 48 blocks and the extra-MSA stack
//   tmul64  C=I= 64  the template pair stack (model config: pair_channel 64,
//                    triangle_multiplication num_intermediate_channel 64)
//
// Nothing in the body is width-specific beyond those constants -- see the WF
// and WG notes there, which were the only two numbers written as 8 instead of
// derived.
#pragma once

#define TM_NS tmul
#define TM_C 128
#define TM_I 128
#include "trimul_body.cuh"
#undef TM_NS
#undef TM_C
#undef TM_I

// NE_TMUL64=0 drops the narrow build.  It is not free: it is a second copy of
// every kernel in the body, and on a translation unit that does not touch the
// template stack it is pure compile time.
#ifndef NE_TMUL64
#define NE_TMUL64 1
#endif
#if NE_TMUL64
#define TM_NS tmul64
#define TM_C 64
#define TM_I 64
#include "trimul_body.cuh"
#undef TM_NS
#undef TM_C
#undef TM_I
#endif
