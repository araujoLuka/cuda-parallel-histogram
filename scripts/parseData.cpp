// parse the cudaHisto.log from the dir ../results/

#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>

constexpr static char sep = ';';

void parseFile(std::array<std::ofstream, 3>& out, int size, std::string input) {
    uint64_t n;
    int32_t h, nr;
    std::string identifier;
    uint64_t time_ns;
    double time_ms;
    double throughput;
    std::string content;
    std::string line;

    // open the file to read
    std::ifstream cin{input};
    if (!cin.is_open()) {
        std::cout << "error: file not found" << std::endl;
        return;
    }

    // start to parse the file
    while (std::getline(cin, line)) {
        // parse the first line
        // n=100 h=1024 nr=20
        sscanf(line.c_str(), "n=%ld h=%d nr=%d", &n, &h, &nr);

        while (std::getline(cin, line) && line[0] == '>') {
            // skip empty line and lines that start with '>'
        }

        // start to parse the data
        for (int i = 0; i < size; ++i) {
            std::getline(cin, line);
            // first line: <identifier> deltaT(ns): <time_ns> ns for <nr> ops
            identifier = line.substr(0, line.find(" deltaT(ns):"));

            std::getline(cin, line);
            // second line:         ==> each op takes <time_ns> ns
            sscanf(line.c_str(), "        ==> each op takes %ld ns", &time_ns);
            time_ms = time_ns / 1e6;

            std::getline(cin, line);
            // third line:         ==> throughput: <throughput> MFLOP/s
            sscanf(line.c_str(), "        ==> throughput: %lf MFLOP/s", &throughput);

            // define content to write
            if (input == "../results/hVariation.data")
                content = std::to_string(h) + sep + std::to_string(time_ms) + sep + std::to_string(throughput);
            else
                content = std::to_string(n) + sep + std::to_string(time_ms) + sep + std::to_string(throughput);

            // std::cout << n << "," << time_ns << "," << throughput << "," << identifier << std::endl;
            std::cout << identifier << std::endl;
            if (identifier == "BlockHisto") {
                out[0] << content << std::endl;
            } else if (identifier == "GlobalHisto") {
                out[1] << content << std::endl;
            } else if (identifier == "SerialHisto") {
                out[2] << content << std::endl;
            } else {
                std::cout << "error: identifier " << identifier << " not found" << std::endl;
            }
            std::cout << content << std::endl << std::endl;
            std::getline(cin, line);
        }
    }
    for (int i{0}; i < size; ++i) {
        out[i].close();
    }
    cin.close();
}

int main() {
    // define the output files
    std::array<std::ofstream, 3> nout{
        std::ofstream{"../results/nBlock.csv"},
        std::ofstream{"../results/nGlobal.csv"},
        std::ofstream{"../results/nSerial.csv"}
    };
    std::array<std::ofstream, 3> hout{
        std::ofstream{"../results/hBlock.csv"},
        std::ofstream{"../results/hGlobal.csv"}
    };

    std::string header = "n;time(ms);throughput(mflop/s)";

    for (int i{0}; i < 3; ++i) {
        nout[i] << header << std::endl;
        hout[i] << header << std::endl;
    }

    std::cout << "Start parsing the files" << std::endl << std::endl;

    std::cout << "nVariation.data" << std::endl;
    parseFile(nout, 3, "../results/nVariation.data");

    std::cout << "hVariation.data" << std::endl;
    parseFile(hout, 2, "../results/hVariation.data");

    return 0;
}
