// FlashAttention for AF2's triangle attention, at the TWO head dims it is
// asked for.
//
//   fa    FD=32  the Evoformer's triangle attention (pair 128 / 4 heads) and
//                the MSA row attention (msa 256 / 8 heads)
//   fa16  FD=16  the template pair stack (pair 64 / 4 heads)
//
// The head dim is a compile-time constant because every shared extent, warp
// sweep and wmma tile count derives from it -- see the body.
#pragma once

#define FA_NS fa
#define FA_FD 32
#include "flash_body.cuh"
#undef FA_NS
#undef FA_FD

#ifndef NE_FD16
#define NE_FD16 1
#endif
#if NE_FD16
#define FA_NS fa16
#define FA_FD 16
#include "flash_body.cuh"
#undef FA_NS
#undef FA_FD
#endif
