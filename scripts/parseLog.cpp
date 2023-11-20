// parse the cudaHisto.log from the dir ../results/

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>

int main() {
    // read the file
    freopen("../results/cudaHisto.log", "r", stdin);
    std::ofstream bout{"../results/block.csv"};
    std::ofstream gout{"../results/global.csv"};
    std::ofstream sout{"../results/serial.csv"};

    std::string line;
    uint64_t n;
    int32_t h, nr;
    std::string identifier;
    uint64_t time_ns;
    double time_ms;
    double throughput;
    char sep = ';';
    std::string header = "n;time(ms);throughput(mflop/s)";
    std::string content;

    bout << header << std::endl;
    gout << header << std::endl;
    sout << header << std::endl;

    while (std::getline(std::cin, line)) {
        // parse the first line
        // n=100 h=1024 nr=20
        sscanf(line.c_str(), "n=%ld h=%d nr=%d", &n, &h, &nr);
        std::getline(std::cin, line);

        // start to parse the data
        for (int i = 0; i < 3; ++i) {
            std::getline(std::cin, line);
            // first line: <identifier> deltaT(ns): <time_ns> ns for <nr> ops
            identifier = line.substr(0, line.find(" deltaT(ns):"));

            std::getline(std::cin, line);
            // second line:         ==> each op takes <time_ns> ns
            sscanf(line.c_str(), "        ==> each op takes %ld ns", &time_ns);
            time_ms = time_ns / 1e6;

            std::getline(std::cin, line);
            // third line:         ==> throughput: <throughput> MFLOP/s
            sscanf(line.c_str(), "        ==> throughput: %lf MFLOP/s", &throughput);

            // define content to write
            content = std::to_string(n) + sep + std::to_string(time_ms) + sep + std::to_string(throughput);

            // std::cout << n << "," << time_ns << "," << throughput << "," << identifier << std::endl;
            std::cout << identifier << std::endl;
            if (identifier == "BlockHisto") {
                bout << content << std::endl;
            } else if (identifier == "GlobalHisto") {
                gout << content << std::endl;
            } else if (identifier == "SerialHisto") {
                sout << content << std::endl;
            } else {
                std::cout << "error: identifier " << identifier << " not found" << std::endl;
            }
            std::cout << content << std::endl << std::endl;
            std::getline(std::cin, line);
        }
    }
    bout.close();
    gout.close();
    sout.close();

    return 0;
}
