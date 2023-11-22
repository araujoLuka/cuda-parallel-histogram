#!/usr/bin/gnuplot -c

# Script para plotar os gráficos de desempenho do algoritmo CudaHisto
# Os dados de entrada devem estar no formato csv, com os valores separados por ponto e vírgula (;)

# Uso: gnuplot plotData.gp

# Eixo x: Tamanho da entrada (n) (int)
# Eixo y1: Tempo de execução (ms) (float)
# Eixo y2: Vazão (MFLOPS) (int)

# Exemplo de arquivo de entrada:
# 2;0.000000;0.000000

set encoding utf8

set grid

set style line 1 lc rgb '#0060ad' lt 1 lw 2 pt 7 pi -1 ps 1.5
set style line 2 lc rgb '#dd181f' lt 1 lw 2 pt 5 pi -1 ps 1.5
set style line 3 lc rgb '#00a000' lt 1 lw 2 pt 9 pi -1 ps 1.5

set xlabel 'Tamanho da Entrada (n)'
set ylabel 'Tempo (ms)'  # Eixo y primário
set y2label 'Vazão (MFLOPS)'  # Eixo y secundário

set ytics nomirror
set y2tics nomirror

set key left top

set datafile separator ';'

# Dados em csv no diretório ../results/

# Plotagem de dados com n variando de 10 a 100000000

set xrange [10:100000000]
set logscale x 10

# Plotagem de BlockHisto (block.csv)
set title 'BlockHisto (h=2048)'
set terminal qt 1 title 'BlockHisto'

plot '../results/nBlock.csv' using 1:2 title 'Tempo de execução' with linespoints axes x1y1 ls 1,\
     '../results/nBlock.csv' using 1:3 title 'Vazão' with linespoints axes x1y2 ls 2

pause -1

# Plotagem de GlobalHisto (global.csv)
set title 'GlobalHisto (h=2048)'
set terminal qt 2 title 'GlobalHisto'

plot '../results/nGlobal.csv' using 1:2 title 'Tempo de execução' with linespoints axes x1y1 ls 1,\
     '../results/nGlobal.csv' using 1:3 title 'Vazão' with linespoints axes x1y2 ls 2

pause -1

# Plotagem de SerialHisto (serial.csv)
set title 'SerialHisto (h=2048)'
set terminal qt 3 title 'SerialHisto'

plot '../results/nSerial.csv' using 1:2 title 'Tempo de execução' with linespoints axes x1y1 ls 1,\
     '../results/nSerial.csv' using 1:3 title 'Vazão' with linespoints axes x1y2 ls 2

pause -1

unset y2label
unset y2tics

# Plotagem de comparacao de tempo entre os algoritmos
set title 'Comparação de tempo (h=2048)'
set terminal qt 4 title 'Comparação de tempo'

plot '../results/nBlock.csv' using 1:2 title 'BlockHisto' with linespoints axes x1y1 ls 1,\
     '../results/nGlobal.csv' using 1:2 title 'GlobalHisto' with linespoints axes x1y1 ls 2

pause -1

# Plotagem de comparacao de vazao entre os algoritmos
set title 'Comparação de vazão (h=2048)'
set terminal qt 5 title 'Comparação de vazão'

plot '../results/nBlock.csv' using 1:3 title 'BlockHisto' with linespoints axes x1y1 ls 1,\
     '../results/nGlobal.csv' using 1:3 title 'GlobalHisto' with linespoints axes x1y1 ls 2

pause -1

# Plotagem de dados com h variando de 2 a 12288

set y2label 'Vazão (MFLOPS)'  # Eixo y secundário
set y2tics nomirror

set xrange [2:12288]
set logscale x 2
set xlabel 'Tamanho do histograma (h)'

# Plotagem de BlockHisto (block.csv)
set title 'BlockHisto (n=100000000)'
set terminal qt 6 title 'BlockHisto'

plot '../results/hBlock.csv' using 1:2 title 'Tempo de execução' with linespoints axes x1y1 ls 1,\
     '../results/hBlock.csv' using 1:3 title 'Vazão' with linespoints axes x1y2 ls 2

pause -1

# Plotagem de GlobalHisto (global.csv)
set title 'GlobalHisto (n=100000000)'
set terminal qt 7 title 'GlobalHisto'

plot '../results/hGlobal.csv' using 1:2 title 'Tempo de execução' with linespoints axes x1y1 ls 1,\
     '../results/hGlobal.csv' using 1:3 title 'Vazão' with linespoints axes x1y2 ls 2

pause -1

# Plotagem de Comparacao de tempo entre os algoritmos
set title 'Comparação de tempo (n=100000000)'
set terminal qt 8 title 'Comparação de tempo'

plot '../results/hBlock.csv' using 1:2 title 'BlockHisto' with linespoints axes x1y1 ls 1,\
     '../results/hGlobal.csv' using 1:2 title 'GlobalHisto' with linespoints axes x1y1 ls 2

pause -1

# Plotagem de Comparacao de vazao entre os algoritmos
set title 'Comparação de vazão (n=100000000)'
set terminal qt 9 title 'Comparação de vazão'

plot '../results/hBlock.csv' using 1:3 title 'BlockHisto' with linespoints axes x1y1 ls 1,\
     '../results/hGlobal.csv' using 1:3 title 'GlobalHisto' with linespoints axes x1y1 ls 2

pause -1
