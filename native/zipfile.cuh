// ---------------------------------------------------------------------
// zipfile.cuh -- the one thing `archive_trajectories` needs that nothing
// else in this port does: a ZIP WRITER, and the reader that pairs with it
// ---------------------------------------------------------------------
//
// bc2's `archive_trajectory_folder` packs a finished trajectory folder
// into `<folder>.zip` with `zipfile.ZIP_DEFLATED` and DELETES the folder.
// That is not a cosmetic step: three readers go looking for a
// trajectory's losses.csv afterwards -- `summarize_campaign` (the
// campaign scope of summary.csv), `recorded_trajectory_rows` (which is
// what autotune and the parameter sweep review a block of trajectories
// with) and `trajectory_metric_rows` -- and each of them falls back to
// the ARCHIVE when the folder is gone.  A port that wrote the zip and
// could not read it back would archive a campaign into a summary with no
// trajectory scope at all, and nothing would say so.
//
// So both halves are here, and both are the real format rather than a
// convenient subset:
//
//   * deflate, level -1 (zlib's default 6, which is what
//     `zlib.compressobj` gets from `zipfile`), raw -- windowBits -15 --
//     because the member stream in a zip carries no zlib wrapper
//   * local header, central directory and end-of-central-directory, with
//     the DOS timestamp and the unix mode in `external_attr` that CPython
//     writes, so `zipfile.ZipFile` opens what this writes and reads the
//     mtimes back
//   * a reader that goes through the CENTRAL DIRECTORY, not by scanning
//     for local headers: a zip's directory is authoritative, and it is
//     the only place an entry that was overwritten can be told from the
//     dead copy still sitting in the file
//
// No zip64 and no encryption.  A trajectory folder is a handful of CSVs
// and structures; if one ever crosses 4 GB this writer must grow a zip64
// record rather than silently truncate, so `write` REFUSES past the
// 32-bit fields instead of writing a file CPython would reject.
#pragma once
#include <zlib.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <sys/stat.h>

namespace zipf {

struct Entry {
  std::string name;  // the path INSIDE the archive
  std::string path;  // the file on disk
};

inline void put16(std::string& out, uint32_t v) {
  out += (char)(v & 0xFF);
  out += (char)((v >> 8) & 0xFF);
}
inline void put32(std::string& out, uint32_t v) {
  put16(out, v & 0xFFFF);
  put16(out, (v >> 16) & 0xFFFF);
}
inline uint32_t get16(const std::string& b, size_t at) {
  return (uint8_t)b[at] | ((uint32_t)(uint8_t)b[at + 1] << 8);
}
inline uint32_t get32(const std::string& b, size_t at) {
  return get16(b, at) | (get16(b, at + 2) << 16);
}

inline bool readFile(const std::string& path, std::string* out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  char buf[65536];
  size_t n;
  out->clear();
  while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out->append(buf, n);
  fclose(f);
  return true;
}

// raw deflate, exactly as `zipfile` asks zlib for it
inline std::string deflateRaw(const std::string& in) {
  z_stream s;
  memset(&s, 0, sizeof(s));
  if (deflateInit2(&s, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8,
                   Z_DEFAULT_STRATEGY) != Z_OK)
    return std::string();
  std::string out;
  out.resize(deflateBound(&s, (uLong)in.size()) + 64);
  s.next_in = (Bytef*)in.data();
  s.avail_in = (uInt)in.size();
  s.next_out = (Bytef*)&out[0];
  s.avail_out = (uInt)out.size();
  const int rc = deflate(&s, Z_FINISH);
  const size_t produced = out.size() - s.avail_out;
  deflateEnd(&s);
  if (rc != Z_STREAM_END) return std::string();
  out.resize(produced);
  return out;
}

inline bool inflateRaw(const std::string& in, size_t want, std::string* out) {
  out->assign(want, '\0');
  if (want == 0) return true;
  z_stream s;
  memset(&s, 0, sizeof(s));
  if (inflateInit2(&s, -MAX_WBITS) != Z_OK) return false;
  s.next_in = (Bytef*)in.data();
  s.avail_in = (uInt)in.size();
  s.next_out = (Bytef*)&(*out)[0];
  s.avail_out = (uInt)want;
  const int rc = inflate(&s, Z_FINISH);
  inflateEnd(&s);
  return rc == Z_STREAM_END && s.avail_out == 0;
}

// CPython's `ZipInfo.from_file`: localtime, two-second resolution, and
// years counted from 1980.  A file older than 1980 clamps there, which
// is what zipfile's own struct packing would do with a negative year.
inline void dosTime(time_t when, uint32_t* date, uint32_t* tm) {
  struct tm parts;
  localtime_r(&when, &parts);
  int year = parts.tm_year + 1900;
  if (year < 1980) { year = 1980; parts.tm_mon = 0; parts.tm_mday = 1;
                     parts.tm_hour = parts.tm_min = parts.tm_sec = 0; }
  *date = (uint32_t)(((year - 1980) << 9) | ((parts.tm_mon + 1) << 5) |
                     parts.tm_mday);
  *tm = (uint32_t)((parts.tm_hour << 11) | (parts.tm_min << 5) |
                   (parts.tm_sec / 2));
}

// Write the entries, in the order given.  Returns false and writes
// nothing on any unreadable member or any field that does not fit.
inline bool write(const std::string& archive_path,
                  const std::vector<Entry>& entries) {
  std::string body, directory;
  uint32_t count = 0;
  for (const Entry& e : entries) {
    std::string raw;
    if (!readFile(e.path, &raw)) return false;
    if (raw.size() > 0xFFFFFFFEull) return false;  // zip64 territory
    struct stat st;
    const bool have = stat(e.path.c_str(), &st) == 0;
    uint32_t date = 0, tm = 0;
    dosTime(have ? st.st_mtime : time(nullptr), &date, &tm);
    const uint32_t mode = (uint32_t)(have ? (st.st_mode & 0xFFFF) : 0100644u);
    std::string packed = deflateRaw(raw);
    uint32_t method = 8;
    if (packed.empty() && !raw.empty()) return false;
    // zipfile stores rather than inflates a member compression made
    // BIGGER?  No -- it does not, and neither does this: a stored member
    // here would be a difference in the bytes with no difference in what
    // anything reads, and matching the vendor is the point.
    const uint32_t crc = (uint32_t)crc32(crc32(0L, Z_NULL, 0),
                                         (const Bytef*)raw.data(),
                                         (uInt)raw.size());
    const uint32_t offset = (uint32_t)body.size();
    if (body.size() > 0xFFFFFFFEull) return false;
    put32(body, 0x04034b50);
    put16(body, 20);            // version needed
    put16(body, 0);             // flags
    put16(body, method);
    put16(body, tm);
    put16(body, date);
    put32(body, crc);
    put32(body, (uint32_t)packed.size());
    put32(body, (uint32_t)raw.size());
    put16(body, (uint32_t)e.name.size());
    put16(body, 0);             // extra
    body += e.name;
    body += packed;

    put32(directory, 0x02014b50);
    put16(directory, (3 << 8) | 20);  // made by: unix, zip 2.0
    put16(directory, 20);
    put16(directory, 0);
    put16(directory, method);
    put16(directory, tm);
    put16(directory, date);
    put32(directory, crc);
    put32(directory, (uint32_t)packed.size());
    put32(directory, (uint32_t)raw.size());
    put16(directory, (uint32_t)e.name.size());
    put16(directory, 0);        // extra
    put16(directory, 0);        // comment
    put16(directory, 0);        // disk
    put16(directory, 0);        // internal attributes
    put32(directory, mode << 16);
    put32(directory, offset);
    directory += e.name;
    ++count;
  }
  std::string end;
  put32(end, 0x06054b50);
  put16(end, 0);
  put16(end, 0);
  put16(end, count);
  put16(end, count);
  put32(end, (uint32_t)directory.size());
  put32(end, (uint32_t)body.size());
  put16(end, 0);                // comment
  FILE* f = fopen(archive_path.c_str(), "wb");
  if (!f) return false;
  const bool ok = fwrite(body.data(), 1, body.size(), f) == body.size() &&
                  fwrite(directory.data(), 1, directory.size(), f) ==
                      directory.size() &&
                  fwrite(end.data(), 1, end.size(), f) == end.size();
  fclose(f);
  if (!ok) remove(archive_path.c_str());
  return ok;
}

struct Member {
  std::string name;
  uint32_t method = 0, csize = 0, usize = 0, offset = 0;
};

// the central directory, which is the authoritative list
inline std::vector<Member> members(const std::string& archive_path) {
  std::vector<Member> out;
  std::string blob;
  if (!readFile(archive_path, &blob) || blob.size() < 22) return out;
  size_t eocd = std::string::npos;
  const size_t lowest = blob.size() > 66000 ? blob.size() - 66000 : 0;
  for (size_t i = blob.size() - 22 + 1; i-- > lowest;)
    if (get32(blob, i) == 0x06054b50) { eocd = i; break; }
  if (eocd == std::string::npos) return out;
  const uint32_t entries = get16(blob, eocd + 10);
  size_t at = get32(blob, eocd + 16);
  for (uint32_t i = 0; i < entries; ++i) {
    if (at + 46 > blob.size() || get32(blob, at) != 0x02014b50) break;
    Member m;
    m.method = get16(blob, at + 10);
    m.csize = get32(blob, at + 20);
    m.usize = get32(blob, at + 24);
    const uint32_t namelen = get16(blob, at + 28);
    const uint32_t extralen = get16(blob, at + 30);
    const uint32_t commentlen = get16(blob, at + 32);
    m.offset = get32(blob, at + 42);
    m.name = blob.substr(at + 46, namelen);
    out.push_back(m);
    at += 46 + namelen + extralen + commentlen;
  }
  return out;
}

inline std::vector<std::string> namelist(const std::string& archive_path) {
  std::vector<std::string> out;
  for (const Member& m : members(archive_path)) out.push_back(m.name);
  return out;
}

// the bytes of one member, by its exact archive name
inline bool readMember(const std::string& archive_path, const std::string& name,
                       std::string* out) {
  std::string blob;
  if (!readFile(archive_path, &blob)) return false;
  for (const Member& m : members(archive_path)) {
    if (m.name != name) continue;
    if (m.offset + 30 > blob.size() || get32(blob, m.offset) != 0x04034b50)
      return false;
    // THE LOCAL HEADER'S OWN LENGTHS, not the directory's: an archiver
    // may pad the local extra field differently, and the data starts
    // after what the LOCAL record declares.
    const size_t at = m.offset + 30 + get16(blob, m.offset + 26) +
                      get16(blob, m.offset + 28);
    if (at + m.csize > blob.size()) return false;
    const std::string packed = blob.substr(at, m.csize);
    if (m.method == 0) { *out = packed; return packed.size() == m.usize; }
    if (m.method != 8) return false;
    return inflateRaw(packed, m.usize, out);
  }
  return false;
}

}  // namespace zipf
