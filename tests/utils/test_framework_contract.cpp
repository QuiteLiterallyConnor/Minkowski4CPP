#include "test_framework.hpp"

namespace {

struct Capture {
    std::ostringstream out, err;
    std::streambuf* old_out = std::cout.rdbuf(out.rdbuf());
    std::streambuf* old_err = std::cerr.rdbuf(err.rdbuf());
    ~Capture() {
        std::cout.rdbuf(old_out);
        std::cerr.rdbuf(old_err);
    }
};

std::string executed;

void check_run(const char* filter, int expected_code, const char* expected_summary,
               const char* expected_executed, bool unmatched = false) {
    char program[] = "fixture";
    std::string argument = filter ? filter : "";
    char* argv[] = {program, argument.data()};
    executed.clear();
    int code;
    std::string out, err;
    {
        Capture capture;
        code = mink_test::run_all(filter ? 2 : 1, argv);
        out = capture.out.str();
        err = capture.err.str();
    }
    CHECK_EQ(code, expected_code);
    CHECK(out.find(expected_summary) != std::string::npos);
    CHECK_EQ(executed, expected_executed);
    if (unmatched) {
        CHECK(err.find("No tests matched filter: " + argument) != std::string::npos);
    } else {
        CHECK(err.empty());
    }
}

}  // namespace

int main() {
    auto& tests = mink_test::registry();
    tests.push_back({"fixture_pass", [] { executed += 'p'; }});
    tests.push_back({"fixture_fail", [] {
        executed += 'f';
        throw mink_test::Failure("intentional failure");
    }});
    tests.push_back({"fixture_skip", [] {
        executed += 's';
        throw mink_test::Skip("intentional skip");
    }});
    try {
        check_run("does_not_exist", 1, "0 passed, 0 failed, 3 skipped", "", true);
        check_run("pass", 0, "1 passed, 0 failed, 2 skipped", "p");
        check_run("fail", 1, "0 passed, 1 failed, 2 skipped", "f");
        check_run("skip", 0, "0 passed, 0 failed, 3 skipped", "s");
        check_run("fixture", 1, "1 passed, 1 failed, 1 skipped", "pfs");
        check_run(nullptr, 1, "1 passed, 1 failed, 1 skipped", "pfs");
        check_run("", 1, "1 passed, 1 failed, 1 skipped", "pfs");
        tests.clear();
        check_run("missing", 1, "0 passed, 0 failed, 0 skipped", "", true);
        check_run(nullptr, 0, "0 passed, 0 failed, 0 skipped", "");
        std::cout << "9 test-framework contract cases passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
