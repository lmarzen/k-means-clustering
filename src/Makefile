CC = gcc

kmeans: kmeans.c
	$(CC) kmeans.c -o kmeans -Wall -lm -fopenmp -O2
kmeans-gpu-offload: kmeans-gpu-offload.c
	$(CC) kmeans-gpu-offload.c -o kmeans-gpu-offload -Wall -lm -fopenmp -O2
linux: kmeans.c
	$(CC) kmeans.c -o kmeans -Wall -lm -fopenmp -O2
win32: kmeans.c
	i686-w64-mingw32-gcc kmeans.c -o kmeans_x86.exe -Wall -fopenmp -O2
win64: kmeans.c
	x86_64-w64-mingw32-gcc kmeans.c -o kmeans_x64.exe -Wall -fopenmp -O2
unoptimized: kmeans.c
	$(CC) kmeans.c -o kmeans -Wall -lm -fopenmp
debug: kmeans.c
	$(CC) kmeans.c -o kmeans -Wall -lm -fopenmp -g
clean:
	rm -f *- kmeans kmeans-gpu-offload kmeans_x86.exe kmeans_x64.exe core
