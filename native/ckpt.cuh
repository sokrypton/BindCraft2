// THE CHECKPOINT, READ IN C++: the last Python dependency.
//
// Until now `bindcraft` needed BC2_WEIGHT_DIRS pointing at the five
// directories bc2's native.py writes to /tmp when it installs a model.  So
// the RUN had no Python in it but getting to the run did, and a port whose
// first step is "now go and start the thing you are replacing" is not one.
//
// AF2's params are a .npz: a ZIP whose members are .npy arrays, and -- this
// is what makes a C++ reader short -- np.savez writes them STORED, not
// deflated.  So there is no decompressor here, only the ZIP central
// directory, the .npy header, and a memcpy.  (savez_compressed would need
// zlib; the check below says so rather than reading rubbish.)
//
// The five directories it writes are byte-for-byte what native.py writes,
// down to the sha1 in the directory name, which is what lets
// ckpt_gate.py diff them and what lets this reuse a dump bc2 has already
// made.  The mappings are transcribed from native.py: _MAIN, _EXTRA, _TPL,
// _TPE_FLAT, _SM/_HEADS/_IPA_ZERO_BIAS and _EMB_LEAVES.
#pragma once
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <algorithm>

#include "res_tables.cuh"

namespace ckpt {

struct Array {
  std::vector<int> shape;
  std::vector<float> v;
  size_t count() const { return v.size(); }
};

// ---- sha1, so the directory names match native.py's -----------------------
// hashlib.sha1 of the same bytes in the same order.  Not security; identity.
struct Sha1 {
  uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
  uint64_t len = 0;
  uint8_t buf[64];
  size_t have = 0;
  static uint32_t rol(uint32_t x, int c) { return (x << c) | (x >> (32 - c)); }
  void block(const uint8_t* p) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t)p[4*i] << 24 | (uint32_t)p[4*i+1] << 16 |
             (uint32_t)p[4*i+2] << 8 | (uint32_t)p[4*i+3];
    for (int i = 16; i < 80; ++i)
      w[i] = rol(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
    for (int i = 0; i < 80; ++i) {
      uint32_t f, k;
      if (i < 20)      { f = (b & c) | (~b & d);            k = 0x5A827999u; }
      else if (i < 40) { f = b ^ c ^ d;                      k = 0x6ED9EBA1u; }
      else if (i < 60) { f = (b & c) | (b & d) | (c & d);    k = 0x8F1BBCDCu; }
      else             { f = b ^ c ^ d;                      k = 0xCA62C1D6u; }
      const uint32_t t = rol(a, 5) + f + e + k + w[i];
      e = d; d = c; c = rol(b, 30); b = a; a = t;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
  }
  void update(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    len += n;
    while (n) {
      const size_t take = std::min(n, 64 - have);
      memcpy(buf + have, p, take);
      have += take; p += take; n -= take;
      if (have == 64) { block(buf); have = 0; }
    }
  }
  void update(const std::string& s) { update(s.data(), s.size()); }
  std::string hex() {
    uint64_t bits = len * 8;
    uint8_t pad = 0x80;
    update(&pad, 1);
    const uint8_t z = 0;
    while (have != 56) update(&z, 1);
    uint8_t be[8];
    for (int i = 0; i < 8; ++i) be[i] = (uint8_t)(bits >> (56 - 8 * i));
    update(be, 8);
    char out[41];
    for (int i = 0; i < 5; ++i) snprintf(out + i * 8, 9, "%08x", h[i]);
    return std::string(out, 40);
  }
};

// ---- the .npz --------------------------------------------------------------
// A haiku checkpoint names its leaves "<module>//<param>", which is the one
// place the double slash matters: `split` on it gives the module path the
// mappings below are written against.
// `sep` is what separates a module path from a parameter name.  AlphaFold's
// haiku checkpoint uses "//" and ProteinMPNN's uses "|"; everything else
// about the two files is the same, which is why this takes an argument
// rather than growing a second reader.
inline bool readNpz(const std::string& path,
                    std::map<std::string, std::map<std::string, Array>>* out,
                    const std::string& sep = "//") {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) { fprintf(stderr, "ckpt: cannot open %s\n", path.c_str()); return false; }
  fseek(f, 0, SEEK_END);
  const long size = ftell(f);
  std::vector<uint8_t> all((size_t)size);
  fseek(f, 0, SEEK_SET);
  if (fread(all.data(), 1, (size_t)size, f) != (size_t)size) {
    fprintf(stderr, "ckpt: short read on %s\n", path.c_str()); fclose(f); return false;
  }
  fclose(f);
  auto u16 = [&](size_t o) { return (uint32_t)all[o] | (uint32_t)all[o+1] << 8; };
  auto u32 = [&](size_t o) {
    return (uint32_t)all[o] | (uint32_t)all[o+1] << 8 |
           (uint32_t)all[o+2] << 16 | (uint32_t)all[o+3] << 24; };
  // the end-of-central-directory record, searched from the back (the comment
  // is almost always empty, but the format allows 64 KB of it)
  size_t eocd = 0;
  for (size_t i = all.size() >= 22 ? all.size() - 22 : 0; ; --i) {
    if (u32(i) == 0x06054b50u) { eocd = i; break; }
    if (i == 0) { fprintf(stderr, "ckpt: %s is not a zip\n", path.c_str()); return false; }
  }
  const uint32_t nent = u16(eocd + 10);
  size_t p = u32(eocd + 16);
  for (uint32_t e = 0; e < nent; ++e) {
    if (u32(p) != 0x02014b50u) { fprintf(stderr, "ckpt: bad central header\n"); return false; }
    const uint32_t method = u16(p + 10);
    const uint32_t csize = u32(p + 20);
    const uint32_t nlen = u16(p + 28), elen = u16(p + 30), clen = u16(p + 32);
    const uint32_t lho = u32(p + 42);
    std::string name((const char*)&all[p + 46], nlen);
    p += 46 + nlen + elen + clen;
    if (method != 0) {
      fprintf(stderr, "ckpt: %s is DEFLATED (savez_compressed); this reader "
                      "handles stored entries only\n", name.c_str());
      return false;
    }
    // the local header repeats the name and carries its own extra field
    const size_t lh = lho;
    const size_t data = lh + 30 + u16(lh + 26) + u16(lh + 28);
    // the .npy header: magic, version, then a 2- or 4-byte header length
    if (memcmp(&all[data], "\x93NUMPY", 6) != 0) {
      fprintf(stderr, "ckpt: %s is not a .npy member\n", name.c_str()); return false;
    }
    const int major = all[data + 6];
    size_t hlen, hstart;
    if (major == 1) { hlen = u16(data + 8); hstart = data + 10; }
    else            { hlen = u32(data + 8); hstart = data + 12; }
    const std::string hdr((const char*)&all[hstart], hlen);
    // A NON-FLOAT MEMBER IS SKIPPED, NOT A FAILURE.  ProteinMPNN's
    // checkpoint carries `num_edges` as an int32 scalar beside the
    // weights; refusing the whole file over it would be refusing the
    // model.  Only members that carry a parameter are wanted, and those
    // are all float32 -- so a non-float member that DOES look like a
    // parameter still has to be loud.
    if (hdr.find("'<f4'") == std::string::npos) {
      if (name.find(sep) != std::string::npos) {
        fprintf(stderr, "ckpt: %s is a parameter and not float32 (%s)\n",
                name.c_str(), hdr.c_str());
        return false;
      }
      continue;
    }
    // FORTRAN ORDER IS NOT AN ERROR, it is a transpose.  AlphaFold's
    // checkpoint has none; ProteinMPNN's has several (a weight saved from a
    // column-major view), and refusing them would have meant a second
    // reader.  The data is the same elements in the other traversal order.
    const bool fortran = hdr.find("'fortran_order': True") != std::string::npos;
    std::vector<int> shape;
    { const size_t a = hdr.find('('), b = hdr.find(')', a);
      std::string s = hdr.substr(a + 1, b - a - 1), cur;
      for (char c : s + ",")
        if (c == ',') { if (!cur.empty()) shape.push_back(atoi(cur.c_str())); cur.clear(); }
        else if (c != ' ') cur += c; }
    size_t n = 1; for (int d : shape) n *= (size_t)d;
    if (n * 4 + (hstart - data) + 0 > csize + 16) { /* tolerated: csize is the member's */ }
    // "<module>//<param>.npy" -> module, param
    std::string key = name;
    if (key.size() > 4 && key.compare(key.size() - 4, 4, ".npy") == 0)
      key.resize(key.size() - 4);
    const size_t dbl = key.rfind(sep);
    if (dbl == std::string::npos) continue;        // not a parameter leaf
    Array a;
    a.shape = shape;
    a.v.resize(n);
    const float* src = (const float*)&all[hstart + hlen];
    if (!fortran || shape.size() < 2) {
      memcpy(a.v.data(), src, n * 4);
    } else {
      // walk the C-order destination and compute the F-order source offset
      const size_t nd = shape.size();
      std::vector<size_t> fstride(nd, 1);
      for (size_t d = 1; d < nd; ++d) fstride[d] = fstride[d-1] * (size_t)shape[d-1];
      std::vector<size_t> idx(nd, 0);
      for (size_t o = 0; o < n; ++o) {
        size_t off = 0;
        for (size_t d = 0; d < nd; ++d) off += idx[d] * fstride[d];
        a.v[o] = src[off];
        for (size_t d = nd; d-- > 0; ) {
          if (++idx[d] < (size_t)shape[d]) break;
          idx[d] = 0;
        }
      }
    }
    (*out)[key.substr(0, dbl)][key.substr(dbl + sep.size())] = std::move(a);
  }
  return true;
}

// ---- writing a directory the Oracle can read -------------------------------
inline bool exists(const std::string& p) { struct stat s; return stat(p.c_str(), &s) == 0; }

// `flat` is name -> array, exactly the dict native.py builds before it
// hashes.  The hash is over sorted names and the FIRST 4096 BYTES of each
// array, which is what native.py does -- not the whole array, so a dump is
// cheap to identify and two checkpoints that agree on every array's first
// kilobyte would collide.  Transcribed rather than improved: the point is
// to land on the same directory bc2 would.
// THE MANIFEST'S ORDER IS PART OF THE BYTES.  native.py writes the two
// stack dumps with the BLOCK varying fastest -- `for mod: for pname: for b
// in range(depth)` -- while the other three write `for name in sorted(flat)`.
// A std::map gives the second order for free and the first not at all, so
// the caller hands in the order it wants.  It changes nothing the Oracle
// reads; it is what lets the gate say "identical" instead of "identical
// modulo line order", and an order-insensitive comparison is exactly where
// a later mismatch would hide.
inline std::string writeDir(const std::string& prefix,
                            const std::vector<std::pair<std::string, Array>>& flat,
                            const std::string& out_dir = "") {
  Sha1 h;
  { std::map<std::string, const Array*> sorted;
    for (const auto& kv : flat) sorted[kv.first] = &kv.second;
    for (const auto& kv : sorted) {
      h.update(kv.first);
      h.update(kv.second->v.data(),
               std::min<size_t>(kv.second->v.size() * 4, 4096));
    } }
  const std::string out = out_dir.empty()
      ? "/tmp/bc2_" + prefix + "_" + h.hex().substr(0, 16) : out_dir;
  if (exists(out + "/manifest.txt")) return out;
  std::string cmd = "mkdir -p '" + out + "'";
  if (system(cmd.c_str()) != 0) {
    fprintf(stderr, "ckpt: cannot create %s\n", out.c_str()); return "";
  }
  FILE* mf = fopen((out + "/manifest.txt").c_str(), "w");
  if (!mf) { fprintf(stderr, "ckpt: cannot write %s/manifest.txt\n", out.c_str()); return ""; }
  for (const auto& kv : flat) {
    std::string fn = kv.first;
    for (char& c : fn) if (c == '/') c = '\x01';
    std::string fn2;
    for (char c : fn) { if (c == '\x01') fn2 += "__"; else fn2 += c; }
    fn2 += ".f32";
    FILE* af = fopen((out + "/" + fn2).c_str(), "wb");
    if (!af) { fprintf(stderr, "ckpt: cannot write %s\n", fn2.c_str()); fclose(mf); return ""; }
    fwrite(kv.second.v.data(), 4, kv.second.v.size(), af);
    fclose(af);
    fprintf(mf, "%s %s %d", kv.first.c_str(), fn2.c_str(), (int)kv.second.shape.size());
    for (int d : kv.second.shape) fprintf(mf, " %d", d);
    fprintf(mf, "\n");
  }
  fclose(mf);
  return out;
}

// ---- the five mappings, from native.py -------------------------------------
static const char* MAIN  = "alphafold/alphafold_iteration/evoformer/evoformer_iteration";
static const char* EXTRA = "alphafold/alphafold_iteration/evoformer/extra_msa_stack";
static const char* TPL   = "alphafold/alphafold_iteration/evoformer/template_embedding/"
                           "single_template_embedding/template_embedding_iteration";
static const char* TPE   = "alphafold/alphafold_iteration/evoformer/template_embedding";
static const char* SM    = "alphafold/alphafold_iteration/structure_module";
static const char* IT    = "alphafold/alphafold_iteration";
static const char* EMB   = "alphafold/alphafold_iteration/evoformer";

using Tree = std::map<std::string, std::map<std::string, Array>>;

// A STACK'S WEIGHTS CARRY A LEADING BLOCK AXIS (haiku's layer_stack), and the
// dump splits it: one file per block.  `depth` is the minimum over every
// array's leading extent, as native.py takes it -- not a constant, because
// the extra stack is 4 deep and the trunk 48.
inline bool splitBlocks(const Tree& t, const std::string& prefix, int depth,
                        std::vector<std::pair<std::string, Array>>* flat) {
  bool any = false;
  for (const auto& mod : t) {
    if (mod.first.compare(0, prefix.size(), prefix) != 0) continue;
    any = true;
    const std::string path = mod.first.size() > prefix.size()
        ? mod.first.substr(prefix.size() + 1) : std::string();
    for (const auto& pr : mod.second) {
      const Array& w = pr.second;
      if (w.shape.empty()) return false;
      size_t per = 1;
      for (size_t i = 1; i < w.shape.size(); ++i) per *= (size_t)w.shape[i];
      for (int b = 0; b < depth; ++b) {
        Array a;
        a.shape.assign(w.shape.begin() + 1, w.shape.end());
        a.v.assign(w.v.begin() + (size_t)b * per, w.v.begin() + (size_t)(b + 1) * per);
        const std::string key = path.empty()
            ? "param/block" + std::to_string(b) + "/" + pr.first
            : "param/block" + std::to_string(b) + "/" + path + "/" + pr.first;
        flat->emplace_back(key, std::move(a));
      }
    }
  }
  return any;
}

inline int stackDepth(const Tree& t, const std::string& prefix) {
  int d = -1;
  for (const auto& mod : t) {
    if (mod.first.compare(0, prefix.size(), prefix) != 0) continue;
    for (const auto& pr : mod.second)
      if (!pr.second.shape.empty())
        d = (d < 0 || pr.second.shape[0] < d) ? pr.second.shape[0] : d;
  }
  return d;
}

// ---- THE MONOMER CHECKPOINT, ONTO THE SAME GRAPH --------------------------
//
// alphafold2_ptm is NOT a second network.  Diff the two parameter sets by
// name and shape and the whole difference is this:
//
//   evoformer/preprocess_1d, left_single, right_single   (22, .) vs (21, .)
//   evoformer/pair_activiations (65,128)   vs ~_relative_encoding/
//                                             position_activations (73,128)
//   triangle_multiplication_*  left/right projection+gate, two LayerNorms
//                                          vs one fused projection and gate
//   structure_module IPA       q_scalar / kv_scalar / q_point_local /
//                              kv_point_local, with biases
//                                          vs six separate projections
//   structure_module           affine_update (384,6)
//                                          vs quat_rigid/rigid (384,6)
//   template_embedding         a DIFFERENT module (see below)
//   masked_msa_head            23 classes vs 22      -- design never reads it
//
// Everything else -- every head, every Evoformer width, the whole structure
// module below the IPA's projections, the template PAIR stack -- is identical
// in name and shape.  So the monomer runs on these kernels with three runtime
// decisions (Plan::monomer: the 22-wide embedder, outer_product_mean last,
// position_scale 10) and the layout transforms below.  Two of them are
// genuinely the same operation under another name:
//
//   * affine_update IS quat_rigid.  QuatAffine::pre_compose normalises
//     q + q(0,v) = q(1,v), which is q composed with normalise(1,v) -- exactly
//     what QuatRigid builds from (1, v) -- and both add R_old @ t_update to
//     the old translation.  A rename, not a branch.
//   * the IPA's monomer projections are the multimer ones concatenated.  The
//     scalars reshape to (head, per) and split PER HEAD; the points are
//     COORD-major (x all heads, then y, then z) where multimer is HEAD-major,
//     so they transpose.
//
// WHAT IS NOT COVERED: the monomer template embedder.  model_1_ptm and
// model_2_ptm enable templates and theirs is a different module -- an
// embedding2d of 88 features into 64 channels and an attention over the
// template axis, against multimer's output_linear -- so a monomer step has to
// run with templates OFF until that is ported.  The template PAIR stack is
// the same module at the same width in both, with the tri-mul unfused, so it
// is converted here too and only the wrapper is missing.
//
// (The same transforms, in Python, are alphafold3/af2/convert.py, which is
// how the standalone library runs a monomer checkpoint on the multimer graph.
// This is that map in C++ so `bindcraft params=...ptm.npz` needs no Python.)

// concat along the LAST axis, keeping every leading axis (a stack's weights
// carry a leading block axis, so this cannot be a flat append)
inline Array concatLast(const Array& a, const Array& b) {
  Array o;
  o.shape = a.shape;
  const int la = a.shape.back(), lb = b.shape.back();
  o.shape.back() = la + lb;
  size_t outer = 1;
  for (size_t i = 0; i + 1 < a.shape.size(); ++i) outer *= (size_t)a.shape[i];
  o.v.resize(outer * (size_t)(la + lb));
  for (size_t r = 0; r < outer; ++r) {
    memcpy(&o.v[r * (la + lb)], &a.v[r * la], (size_t)la * 4);
    memcpy(&o.v[r * (la + lb) + la], &b.v[r * lb], (size_t)lb * 4);
  }
  return o;
}

inline bool looksMonomer(const Tree& t) {
  auto it = t.find(std::string(EMB) + "/preprocess_1d");
  if (it == t.end()) return false;
  auto w = it->second.find("weights");
  // 22 is monomer-and-unconverted, 21 is already the multimer layout.  The
  // test is the SHAPE for the same reason convert.py's is: it stays true after
  // a conversion, so running it twice cannot damage anything.
  return w != it->second.end() && !w->second.shape.empty() &&
         w->second.shape[0] == 22;
}

inline void monomerise(Tree* t) {
  Tree& T = *t;
  // ---- the template PAIR stack's scope.  haiku names a layer_stack after
  // what built it: the monomer builds it with layer_stack directly
  // (`template_pair_stack/__layer_stack_no_state`), the multimer through a
  // named iteration (`template_embedding_iteration`).  Same module, same
  // width, same two blocks -- only the scope -- so renaming it is what lets
  // the tri-mul pass below, and then tplstack.cuh, find it.
  {
    const std::string from = "/template_pair_stack/__layer_stack_no_state/";
    const std::string to = "/template_embedding_iteration/";
    std::vector<std::pair<std::string, std::string>> ren;
    for (const auto& mod : T) {
      const size_t at = mod.first.find(from);
      if (at != std::string::npos)
        ren.emplace_back(mod.first, mod.first.substr(0, at) + to +
                                        mod.first.substr(at + from.size()));
    }
    for (const auto& r : ren) { T[r.second] = T[r.first]; T.erase(r.first); }
  }
  // ---- the triangle multiplications, in BOTH stacks and the template stack
  std::vector<std::string> drop;
  std::vector<std::pair<std::string, std::map<std::string, Array>>> add;
  for (auto& mod : T) {
    const std::string& k = mod.first;
    const size_t slash = k.rfind('/');
    if (slash == std::string::npos) continue;
    const std::string leaf = k.substr(slash + 1);
    const std::string parent = k.substr(0, slash);
    const size_t pslash = parent.rfind('/');
    const std::string pleaf = pslash == std::string::npos
        ? parent : parent.substr(pslash + 1);
    // THE OUTER PRODUCT MEAN HAS layer_norm_input AND left/right_projection
    // IN BOTH CONFIGS.  Keying on the leaf name alone renames that module too,
    // which is the mistake weights.py's note records paying for once.
    if (pleaf.compare(0, 24, "triangle_multiplication_") != 0) continue;
    if (leaf == "layer_norm_input") {
      add.emplace_back(parent + "/left_norm_input", mod.second); drop.push_back(k);
    } else if (leaf == "center_layer_norm") {
      add.emplace_back(parent + "/center_norm", mod.second); drop.push_back(k);
    } else if (leaf == "left_projection" || leaf == "left_gate") {
      const std::string other = leaf == "left_projection" ? "right_projection"
                                                          : "right_gate";
      auto rit = T.find(parent + "/" + other);
      if (rit == T.end()) continue;
      std::map<std::string, Array> fused;
      // AF2's own split order: `a, b = jnp.split(projection, 2, axis=-1)`,
      // so left is the first half.
      for (const auto& pr : mod.second) {
        auto o = rit->second.find(pr.first);
        if (o == rit->second.end()) continue;
        fused[pr.first] = concatLast(pr.second, o->second);
      }
      add.emplace_back(parent + (leaf == "left_projection" ? "/projection"
                                                           : "/gate"),
                       std::move(fused));
      drop.push_back(k);
      drop.push_back(parent + "/" + other);
    }
  }
  for (const auto& k : drop) T.erase(k);
  for (auto& a : add) T[a.first] = std::move(a.second);

  // ---- the IPA's four fused projections
  const std::string IPA =
      std::string(SM) + "/fold_iteration/invariant_point_attention/";
  const int HH = 12;
  // scalars: (D, head*per) -> (D, head, per), split the LAST axis per head
  auto scalar = [&](const char* from,
                    std::vector<std::pair<const char*, int>> parts) {
    auto it = T.find(IPA + from);
    if (it == T.end()) return;
    const Array& w = it->second.at("weights");
    const Array& b = it->second.at("bias");
    const int D = w.shape[0], per = w.shape[1] / HH;
    int off = 0;
    for (const auto& pr : parts) {
      Array ow, ob;
      ow.shape = {D, HH, pr.second};
      ow.v.resize((size_t)D * HH * pr.second);
      ob.shape = {HH, pr.second};
      ob.v.resize((size_t)HH * pr.second);
      for (int d = 0; d < D; ++d)
        for (int h = 0; h < HH; ++h)
          for (int c = 0; c < pr.second; ++c)
            ow.v[((size_t)d * HH + h) * pr.second + c] =
                w.v[(size_t)d * HH * per + (size_t)h * per + off + c];
      for (int h = 0; h < HH; ++h)
        for (int c = 0; c < pr.second; ++c)
          ob.v[(size_t)h * pr.second + c] = b.v[(size_t)h * per + off + c];
      T[IPA + pr.first] = {{"weights", std::move(ow)}, {"bias", std::move(ob)}};
      off += pr.second;
    }
    T.erase(IPA + from);
  };
  scalar("q_scalar", {{"q_scalar_projection", 16}});
  scalar("kv_scalar", {{"k_scalar_projection", 16}, {"v_scalar_projection", 16}});

  // points: monomer is (coord, head, point), multimer is (head, coord, point)
  auto point = [&](const char* from,
                   std::vector<std::pair<const char*, int>> parts) {
    auto it = T.find(IPA + from);
    if (it == T.end()) return;
    const Array& w = it->second.at("weights");
    const Array& b = it->second.at("bias");
    const int D = w.shape[0], npts = w.shape[1] / (3 * HH);
    int off = 0;
    for (const auto& pr : parts) {
      const int P = pr.second;
      Array ow, ob;
      ow.shape = {D, HH, 3 * P};
      ow.v.resize((size_t)D * HH * 3 * P);
      ob.shape = {HH, 3 * P};
      ob.v.resize((size_t)HH * 3 * P);
      for (int d = 0; d < D; ++d)
        for (int h = 0; h < HH; ++h)
          for (int a = 0; a < 3; ++a)
            for (int p = 0; p < P; ++p) {
              const size_t src = (size_t)d * (3 * HH * npts) +
                                 (size_t)a * (HH * npts) + (size_t)h * npts +
                                 off + p;
              ow.v[(((size_t)d * HH + h) * 3 + a) * P + p] = w.v[src];
            }
      for (int h = 0; h < HH; ++h)
        for (int a = 0; a < 3; ++a)
          for (int p = 0; p < P; ++p)
            ob.v[((size_t)h * 3 + a) * P + p] =
                b.v[(size_t)a * (HH * npts) + (size_t)h * npts + off + p];
      T[IPA + pr.first] = {{"weights", std::move(ow)}, {"bias", std::move(ob)}};
      off += P;
    }
    T.erase(IPA + from);
  };
  point("q_point_local", {{"q_point_projection/point_projection", 4}});
  point("kv_point_local", {{"k_point_projection/point_projection", 4},
                           {"v_point_projection/point_projection", 8}});

  // ---- affine_update -> quat_rigid/rigid, a rename
  const std::string au = std::string(SM) + "/fold_iteration/affine_update";
  auto ait = T.find(au);
  if (ait != T.end()) {
    T[std::string(SM) + "/fold_iteration/quat_rigid/rigid"] = ait->second;
    T.erase(au);
  }
}

struct Dirs { std::string trunk, extra, tpl, post, emb; bool ok = false;
              // alphafold2_ptm: the caller must set Plan::monomer and run
              // with templates off
              bool monomer = false; };

inline Dirs convert(const std::string& npz, const std::string& out_root = "") {
  Dirs d;
  Tree t;
  if (!readNpz(npz, &t)) return d;
  // a monomer checkpoint becomes the multimer layout HERE, so every mapping
  // below, and every kernel, sees one layout
  d.monomer = looksMonomer(t);
  if (d.monomer) monomerise(&t);

  auto dirFor = [&](const char* tag) {
    return out_root.empty() ? std::string() : out_root + "/" + tag;
  };

  // 1 + 2: the two Evoformer stacks.  native.py's weights_from_params hashes
  // the WHOLE multi-block array, not the per-block slices, so the hash runs
  // over the tree and the split happens after.
  for (int which = 0; which < 2; ++which) {
    const std::string prefix = which == 0 ? MAIN : EXTRA;
    const int depth = which == 0 ? 48 : 4;
    Sha1 h;
    for (const auto& mod : t) {
      if (mod.first.compare(0, prefix.size(), prefix) != 0) continue;
      for (const auto& pr : mod.second) {
        h.update(mod.first); h.update(pr.first);
        h.update(pr.second.v.data(), std::min<size_t>(pr.second.v.size() * 4, 4096));
      }
    }
    std::vector<std::pair<std::string, Array>> flat;
    if (!splitBlocks(t, prefix, depth, &flat)) {
      fprintf(stderr, "ckpt: no %s* in %s\n", prefix.c_str(), npz.c_str());
      return d;
    }
    const std::string want = dirFor(which == 0 ? "trunk" : "extra");
    const std::string out = want.empty()
        ? "/tmp/bc2_native_" + h.hex().substr(0, 16) : want;
    const std::string got = writeDir("native", flat, out);
    if (got.empty()) return d;
    (which == 0 ? d.trunk : d.extra) = got;
  }

  // 3: the template embedding -- the two-block stack plus the flat modules.
  //
  // NOT THE MONOMER'S.  alphafold2_ptm's template embedder is a different
  // module: one `embedding2d` of 88 features into 64 channels where the
  // multimer has nine `template_pair_embedding_*`, an attention over the
  // template axis where the multimer has `output_linear`, and no
  // `query_embedding_norm` at all.  The pair STACK inside it is converted
  // (monomerise renames its scope and fuses its triangle multiplications), so
  // what is missing is the wrapper -- and a monomer step therefore has to run
  // with templates off.  Said out loud, because a missing dump that is
  // DELIBERATE and a missing dump that is a bug look the same on disk.
  {
    std::vector<std::pair<std::string, Array>> flat;
    const int depth = stackDepth(t, TPL);
    if (depth < 0 || !splitBlocks(t, TPL, depth, &flat)) {
      fprintf(stderr, "ckpt: no %s* in %s\n", TPL, npz.c_str()); return d;
    }
    const std::string ste = std::string(TPE) + "/single_template_embedding/";
    std::vector<std::pair<std::string, std::string>> leaves = {
        {ste + "query_embedding_norm", "query_embedding_norm"},
        {ste + "output_layer_norm", "output_layer_norm"},
        {std::string(TPE) + "/output_linear", "output_linear"},
    };
    // 0..7 ARE THE CONSTANT EIGHT and the step never uses them -- it gets
    // their SUM as act_const.  They travel anyway, because this driver
    // computes that sum itself.
    for (int i = 0; i <= 8; ++i)
      leaves.push_back({ste + "template_pair_embedding_" + std::to_string(i),
                        "template_pair_embedding_" + std::to_string(i)});
    // THE MONOMER'S WRAPPER IS A DIFFERENT WRAPPER, so it dumps different
    // leaves: one `embedding2d` of 88 features where the multimer has nine
    // `template_pair_embedding_*`, no `query_embedding_norm` at all, and an
    // ATTENTION over the template axis where the multimer has an
    // `output_linear`.  tplembed_ptm.cuh reads exactly these, and folds
    // value_w with output_w itself rather than being handed a product.
    if (d.monomer) {
      leaves = {
          {ste + "embedding2d", "single_template_embedding/embedding2d"},
          {ste + "output_layer_norm",
           "single_template_embedding/output_layer_norm"},
          {std::string(TPE) + "/attention", "attention"},
      };
    }
    for (const auto& lv : leaves) {
      auto it = t.find(lv.first);
      if (it == t.end()) {
        fprintf(stderr, "ckpt: template embedding: no %s\n", lv.first.c_str());
        return d;
      }
      for (const auto& pr : it->second)
        flat.emplace_back("param/" + lv.second + "/" + pr.first, pr.second);
    }
    // ...and this one IS sorted, because tplembed_weights_from_params writes
    // `for name in sorted(flat)` over a dict that holds both halves.
    std::sort(flat.begin(), flat.end(),
              [](const std::pair<std::string, Array>& a,
                 const std::pair<std::string, Array>& b) { return a.first < b.first; });
    const std::string got = writeDir("tpe", flat, dirFor("tpl"));
    if (got.empty()) return d;
    d.tpl = got;
  }

  // 4: the structure module and the four heads, with no block axis -- AF2's
  // structure module is RECURRENT and its eight iterations share one set.
  {
    std::map<std::string, Array> flat;
    static const char* HEADS[4] = {"distogram_head", "predicted_aligned_error_head",
                                   "predicted_lddt_head", "experimentally_resolved_head"};
    const std::string smp = SM, itp = std::string(IT) + "/";
    for (const auto& mod : t) {
      std::string path;
      if (mod.first == smp) path = "";
      else if (mod.first.compare(0, smp.size() + 1, smp + "/") == 0)
        path = mod.first.substr(smp.size() + 1);
      else if (mod.first.compare(0, itp.size(), itp) == 0) {
        const std::string rest = mod.first.substr(itp.size());
        const std::string head = rest.substr(0, rest.find('/'));
        bool isHead = false;
        for (const char* h : HEADS) isHead |= (head == h);
        if (!isHead) continue;
        path = rest;
      } else continue;
      for (const auto& pr : mod.second)
        flat[path.empty() ? "param/" + pr.first
                          : "param/" + path + "/" + pr.first] = pr.second;
    }
    if (flat.empty()) {
      fprintf(stderr, "ckpt: no structure_module or head parameters\n"); return d;
    }
    // AF2's IPA builds three scalar projections with use_bias=True so the
    // graph matches monomer and multimer, but NO checkpoint carries those
    // biases -- they were trained at zero.  Supplied, not special-cased.
    static const char* ZB[3] = {
        "fold_iteration/invariant_point_attention/q_scalar_projection",
        "fold_iteration/invariant_point_attention/k_scalar_projection",
        "fold_iteration/invariant_point_attention/v_scalar_projection"};
    for (const char* m : ZB) {
      const std::string wk = std::string("param/") + m + "/weights";
      const std::string bk = std::string("param/") + m + "/bias";
      auto it = flat.find(wk);
      if (it == flat.end() || flat.count(bk)) continue;
      size_t n = 1;
      for (size_t i = 1; i < it->second.shape.size(); ++i) n *= (size_t)it->second.shape[i];
      Array z; z.shape = {(int)n}; z.v.assign(n, 0.f);
      flat[bk] = std::move(z);
    }
    // the geometry travels WITH the dump, from the same residue_constants
    // the model uses, so there is no second copy of AF2's geometry here
    // FROM THE BIT PATTERNS, because AF2's geometry holds -0.0 in 190
    // places and -0.0 == 0.0 -- so a value check sees nothing while the
    // dump's sha1, and the gate's byte comparison, see everything.
    auto konst = [&](const char* name, const uint32_t* bits, std::vector<int> shape) {
      Array a; a.shape = shape;
      size_t n = 1; for (int s2 : shape) n *= (size_t)s2;
      a.v.resize(n);
      memcpy(a.v.data(), bits, n * 4);
      flat[std::string("const/") + name] = std::move(a);
    };
    konst("default_frame", RES_DEFAULT_FRAME_BITS, {21, 8, 4, 4});
    konst("atom14_to_group", RES_ATOM14_TO_GROUP_BITS, {21, 14});
    konst("atom14_lit_pos", RES_ATOM14_LIT_POS_BITS, {21, 14, 3});
    konst("atom14_mask", RES_ATOM14_MASK_21_BITS, {21, 14});
    std::vector<std::pair<std::string, Array>> ord(flat.begin(), flat.end());
    const std::string got = writeDir("post", ord, dirFor("post"));
    if (got.empty()) return d;
    d.post = got;
  }

  // 5: the embedder's eleven leaves.  THE SCOPE IS THE TRAP -- haiku scopes a
  // module created inside a METHOD under that method's name, so
  // position_activations lives under evoformer/~_relative_encoding/ while its
  // siblings sit directly under evoformer/.  Matching the LEAF is what makes
  // both land.
  {
    // ...and under alphafold2_ptm the relative encoding is spelled
    // `pair_activiations` (AF2's own typo) and is 65 bins instead of 73.
    // Dumped under its own name, not renamed to the multimer's: embed.cuh
    // runs the monomer embedder as itself, so a rename here would only make
    // the two sides disagree about which embedder is running.
    // NOT `static`: a function-local static with a runtime initialiser is
    // initialised ONCE, so a second convert() of a multimer checkpoint in the
    // same process would keep the monomer's spelling.
    const char* LEAVES[11] = {
        "preprocess_1d", "preprocess_msa", "left_single", "right_single",
        d.monomer ? "pair_activiations" : "position_activations",
        "single_activations",
        "template_single_embedding", "template_projection",
        "prev_pos_linear", "prev_msa_first_row_norm", "prev_pair_norm"};
    std::map<std::string, Array> flat;
    const std::string embp = EMB;
    for (const auto& mod : t) {
      if (mod.first.compare(0, embp.size(), embp) != 0) continue;
      const std::string leaf = mod.first.substr(mod.first.rfind('/') + 1);
      bool want = false;
      for (const char* L : LEAVES) want |= (leaf == L);
      if (!want) continue;
      for (const auto& pr : mod.second)
        flat["param/" + leaf + "/" + pr.first] = pr.second;
    }
    if (flat.size() != 2 * 11) {
      fprintf(stderr, "ckpt: embedder: found %zu arrays, expected 22\n", flat.size());
      return d;
    }
    std::vector<std::pair<std::string, Array>> ord(flat.begin(), flat.end());
    const std::string got = writeDir("emb", ord, dirFor("emb"));
    if (got.empty()) return d;
    d.emb = got;
  }

  d.ok = true;
  return d;
}

}  // namespace ckpt
