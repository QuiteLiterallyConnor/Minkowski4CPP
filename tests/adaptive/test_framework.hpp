// Minimal dependency-free test harness for the adaptive voxelization tests.
#ifndef MINK_ADAPTIVE_TEST_FRAMEWORK_HPP
#define MINK_ADAPTIVE_TEST_FRAMEWORK_HPP

#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mink_test {

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> r;
    return r;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> fn) {
        registry().push_back({name, std::move(fn)});
    }
};

inline int run_all(int argc, char** argv) {
    std::string filter = argc > 1 ? argv[1] : "";
    int passed = 0, failed = 0, skipped = 0;
    for (auto& t : registry()) {
        if (!filter.empty() && t.name.find(filter) == std::string::npos) {
            ++skipped;
            continue;
        }
        try {
            t.fn();
            std::cout << "[ PASS ] " << t.name << "\n";
            ++passed;
        } catch (const std::exception& e) {
            std::cout << "[ FAIL ] " << t.name << "\n         " << e.what() << "\n";
            ++failed;
        }
    }
    std::cout << "\n" << passed << " passed, " << failed << " failed, " << skipped
              << " skipped\n";
    return failed == 0 ? 0 : 1;
}

}  // namespace mink_test

#define MINK_CONCAT_(a, b) a##b
#define MINK_CONCAT(a, b) MINK_CONCAT_(a, b)

#define TEST(name)                                                              \
    static void name();                                                         \
    static ::mink_test::Registrar MINK_CONCAT(reg_, name)(#name, name);         \
    static void name()

#define FAIL_AT(msg)                                                            \
    do {                                                                        \
        std::ostringstream os_;                                                 \
        os_ << __FILE__ << ":" << __LINE__ << ": " << msg;                      \
        throw ::mink_test::Failure(os_.str());                                  \
    } while (0)

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) FAIL_AT("CHECK failed: " #cond);                           \
    } while (0)

#define CHECK_EQ(a, b)                                                          \
    do {                                                                        \
        auto va_ = (a);                                                         \
        auto vb_ = (b);                                                         \
        if (!(va_ == vb_)) FAIL_AT("CHECK_EQ failed: " #a " == " #b " (" << va_ \
                                   << " vs " << vb_ << ")");                    \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                   \
    do {                                                                        \
        double va_ = static_cast<double>(a);                                    \
        double vb_ = static_cast<double>(b);                                    \
        if (!(std::fabs(va_ - vb_) <= (tol)))                                   \
            FAIL_AT("CHECK_NEAR failed: " #a " ~ " #b " (" << va_ << " vs "     \
                    << vb_ << ", tol " << (tol) << ")");                        \
    } while (0)

#define CHECK_THROWS(expr)                                                      \
    do {                                                                        \
        bool thrown_ = false;                                                   \
        try {                                                                   \
            (void)(expr);                                                       \
        } catch (const std::exception&) {                                       \
            thrown_ = true;                                                     \
        }                                                                       \
        if (!thrown_) FAIL_AT("CHECK_THROWS failed: " #expr " did not throw");  \
    } while (0)

#define CHECK_ALLCLOSE(a, b, rtol, atol)                                        \
    do {                                                                        \
        if (!torch::allclose((a), (b), (rtol), (atol)))                         \
            FAIL_AT("CHECK_ALLCLOSE failed: " #a " vs " #b " (max abs diff "    \
                    << ((a) - (b)).abs().max().item<double>() << ")");          \
    } while (0)

#endif
