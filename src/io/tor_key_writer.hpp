#pragma once

#include "io/verifier.hpp"

#include <expected>
#include <filesystem>

namespace onion::io {

enum class WriteError { create_dir_failed, open_failed, write_failed };

// Writes <outdir>/<56-char-address>/{hostname, hs_ed25519_secret_key,
// hs_ed25519_public_key} in the exact format Tor's HiddenServiceDir expects.
// Directory 0700, files 0600, O_EXCL (never overwrites). Returns the
// created directory.
[[nodiscard]] std::expected<std::filesystem::path, WriteError>
write_tor_keys(const VerifiedResult& result, const std::filesystem::path& outdir);

}  // namespace onion::io
