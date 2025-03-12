#include <iostream>
#include <cstdlib>

void checkPlagiarism(std::string code1, std::string code2) {
    // Escape `<<` for Windows CMD
    size_t pos;
    while ((pos = code1.find("<<")) != std::string::npos) {
        code1.replace(pos, 2, "^<^<");
    }
    while ((pos = code2.find("<<")) != std::string::npos) {
        code2.replace(pos, 2, "^<^<");
    }

    std::string command = "curl -X POST http://127.0.0.1:5000/compare "
                          "-H \"Content-Type: application/json\" "
                          "-d \"{\\\"code1\\\": \\\"" + code1 + "\\\", \\\"code2\\\": \\\"" + code2 + "\\\"}\"";

    std::cout << "Executing command: " << command << std::endl;
    int result = system(command.c_str());

    if (result != 0) {
        std::cerr << "Error executing curl command!" << std::endl;
    }
}

int main() {
    std::string code1 = "int main() { int a = 5; std::cout << a; return 0; }";
    std::string code2 = "int main() { int x = 5; std::cout << x; return 0; }";

    checkPlagiarism(code1, code2);
    return 0;
}
