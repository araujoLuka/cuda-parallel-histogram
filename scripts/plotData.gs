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

set datafile separator ';'

# Dados em csv no diretório ../results/

# Plotagem de BlockHisto (block.csv)
set title 'BlockHisto'
set terminal qt 1 title 'BlockHisto'

plot '../results/block.csv' using 1:2 title 'Tempo de execução' with linespoints axes x1y1 ls 1,\
     '../results/block.csv' using 1:3 title 'Vazão' with linespoints axes x1y2 ls 2

pause -1

# Plotagem de GlobalHisto (global.csv)
set title 'GlobalHisto'
set terminal qt 2 title 'GlobalHisto'

plot '../results/global.csv' using 1:2 title 'Tempo de execução' with linespoints axes x1y1 ls 1,\
     '../results/global.csv' using 1:3 title 'Vazão' with linespoints axes x1y2 ls 2

pause -1

# Plotagem de SerialHisto (serial.csv)
set title 'SerialHisto'
set terminal qt 3 title 'SerialHisto'

plot '../results/serial.csv' using 1:2 title 'Tempo de execução' with linespoints axes x1y1 ls 1,\
     '../results/serial.csv' using 1:3 title 'Vazão' with linespoints axes x1y2 ls 2

pause -1

unset y2label
unset y2tics

# Plotagem de comparacao de tempo entre os algoritmos
set title 'Comparação de tempo'
set terminal qt 4 title 'Comparação de tempo'

plot '../results/block.csv' using 1:2 title 'BlockHisto' with linespoints axes x1y1 ls 1,\
     '../results/global.csv' using 1:2 title 'GlobalHisto' with linespoints axes x1y1 ls 2,\

pause -1

# Plotagem de comparacao de vazao entre os algoritmos
set title 'Comparação de vazão'
set terminal qt 5 title 'Comparação de vazão'

plot '../results/block.csv' using 1:3 title 'BlockHisto' with linespoints axes x1y1 ls 1,\
     '../results/global.csv' using 1:3 title 'GlobalHisto' with linespoints axes x1y1 ls 2,\

pause -1
