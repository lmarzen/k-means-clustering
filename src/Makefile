kmeans: kmeans.c
	gcc kmeans.c -o kmeans -Wall -lm -fopenmp -O2
	i686-w64-mingw32-gcc kmeans.c -o kmeans_x86.exe -Wall -fopenmp -O2
	x86_64-w64-mingw32-gcc kmeans.c -o kmeans_x64.exe -Wall -fopenmp -O2
linux: kmeans.c
	gcc kmeans.c -o kmeans -Wall -lm -fopenmp -O2
win32: kmeans.c
	i686-w64-mingw32-gcc kmeans.c -o kmeans_x86.exe -Wall -fopenmp -O2
win64: kmeans.c
	x86_64-w64-mingw32-gcc kmeans.c -o kmeans_x64.exe -Wall -fopenmp -O2
unoptimized: kmeans.c
	gcc kmeans.c -o kmeans -Wall -lm -fopenmp
debug: kmeans.c
	gcc kmeans.c -o kmeans -Wall -lm -fopenmp -g
clean:
	rm -f *- kmeans kmeans_x86.exe kmeans_x64.exe core
