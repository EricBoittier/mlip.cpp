#pragma once
// Print a symbolised backtrace on SIGSEGV/SIGABRT (core dumps go to
// systemd-coredump on the compute nodes, and ptrace is not permitted).
#include <csignal>
#include <execinfo.h>
#include <unistd.h>

namespace mlipcpp::train {

inline void install_crash_backtrace() {
  auto handler = [](int sig) {
    void *frames[64];
    const int n = backtrace(frames, 64);
    const char msg[] = "\n*** fatal signal, backtrace:\n";
    (void)!write(STDERR_FILENO, msg, sizeof msg - 1);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);
    std::signal(sig, SIG_DFL);
    std::raise(sig);
  };
  std::signal(SIGSEGV, handler);
  std::signal(SIGABRT, handler);
  std::signal(SIGBUS, handler);
}

} // namespace mlipcpp::train
