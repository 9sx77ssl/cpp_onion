#include "io/tor_key_writer.hpp"

#include <fcntl.h>
#include <sodium.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <cerrno>
#include <string>
#include <string_view>
#include <span>

namespace fs = std::filesystem;

namespace onion::io {
namespace {

constexpr std::string_view kSecretTag = "== ed25519v1-secret: type0 ==";
constexpr std::string_view kPublicTag = "== ed25519v1-public: type0 ==";

std::array<std::byte, 32> make_header(std::string_view tag) {
    std::array<std::byte, 32> h{};  // zero-padded to 32 bytes
    std::memcpy(h.data(), tag.data(), tag.size());
    return h;
}

std::expected<void, WriteError> write_file_0600(const fs::path& path,
                                                std::span<const std::byte> data) {
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return std::unexpected(WriteError::open_failed);
    std::size_t off = 0;
    while (off < data.size()) {
        const ssize_t n = ::write(fd, data.data() + off, data.size() - off);
        if (n <= 0) {
            ::close(fd);
            return std::unexpected(WriteError::write_failed);
        }
        off += static_cast<std::size_t>(n);
    }
    ::fsync(fd);
    ::close(fd);
    return {};
}

}  // namespace

std::expected<fs::path, WriteError>
write_tor_keys(const VerifiedResult& result, const fs::path& outdir) {
    const std::string addr56{result.address.view()};
    const fs::path dir = outdir / addr56;

    std::error_code ec;
    // Create parent components first (they hold no secrets), then create the
    // leaf result directory atomically with mode 0700. Doing the leaf via
    // create_directories + a later chmod would leave a brief window where it
    // exists with umask-default (group/other-readable) permissions.
    if (!outdir.empty()) {
        fs::create_directories(outdir, ec);
        if (ec) return std::unexpected(WriteError::create_dir_failed);
    }
    if (::mkdir(dir.c_str(), 0700) != 0 && errno != EEXIST)
        return std::unexpected(WriteError::create_dir_failed);

    // hostname
    const std::string hostname = result.address.to_string() + "\n";
    if (auto r = write_file_0600(
            dir / "hostname",
            {reinterpret_cast<const std::byte*>(hostname.data()), hostname.size()});
        !r)
        return std::unexpected(r.error());

    // hs_ed25519_secret_key: header || scalar || prf_prefix
    std::array<std::byte, 96> secret_file;
    const auto sec_header = make_header(kSecretTag);
    std::ranges::copy(sec_header, secret_file.begin());
    std::ranges::copy(result.secret.scalar, secret_file.begin() + 32);
    std::ranges::copy(result.secret.prf_prefix, secret_file.begin() + 64);
    const auto sec_result = write_file_0600(dir / "hs_ed25519_secret_key", secret_file);
    sodium_memzero(secret_file.data(), secret_file.size());
    if (!sec_result) return std::unexpected(sec_result.error());

    // hs_ed25519_public_key: header || pubkey
    std::array<std::byte, 64> public_file;
    const auto pub_header = make_header(kPublicTag);
    std::ranges::copy(pub_header, public_file.begin());
    std::ranges::copy(result.pubkey, public_file.begin() + 32);
    if (auto r = write_file_0600(dir / "hs_ed25519_public_key", public_file); !r)
        return std::unexpected(r.error());

    return dir;
}

}  // namespace onion::io
