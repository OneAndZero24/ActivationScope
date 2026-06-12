#include "utils.hpp"
#include <sys/stat.h>
#include <random>
#include <sstream>

namespace activationscope {

std::string sanitize_layer_name(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (char c : raw) {
        if (c == '/' || c == '\\' || c == ':' || c == '?' || c == '*') {
            out += '_';
        } else if (c == '.') {
            out += '_';
        } else {
            out += c;
        }
    }
    return out;
}

bool ensure_dir(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
        return true;
    return (mkdir(path.c_str(), 0700) == 0);
}


std::string make_session_temp_dir(uint64_t id) {
  // Use a random suffix to avoid collisions with concurrent sessions.
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(100000, 999999);
  int suffix = dis(gen);

  std::ostringstream oss;
  oss << "/tmp/activationscope_" << id << "_" << suffix;
  std::string dir = oss.str();

  if (mkdir(dir.c_str(), 0700) != 0) {
    // Fallback: try with a different suffix if directory already exists.
    suffix = dis(gen) + 1;
    oss.str("");
    oss << "/tmp/activationscope_" << id << "_" << suffix;
    dir = oss.str();
    mkdir(dir.c_str(), 0700); // best-effort
  }
  return dir;
}

CapturePolicy infer_capture_policy(int64_t sample_every, int64_t max_batches) {
  if (max_batches > 0)
    return CapturePolicy::MAX_K;
  if (sample_every > 1)
    return CapturePolicy::SAMPLE_N;
  return CapturePolicy::EVERY;
}

} // namespace activationscope
