omp:
	nvcc DisplayImage_omp_ouyang.cu -Xcompiler -fopenmp `pkg-config opencv --cflags --libs ~/.local/lib/pkgconfig/opencv.pc`  -o DisplayImage_omp
cuda:
	nvcc DisplayImage_CU_ouyang.cu -Xcompiler -fopenmp `pkg-config opencv --cflags --libs ~/.local/lib/pkgconfig/opencv.pc`  -o DisplayImage_cuda
upup:
	nvcc DisplayImage_upupsheep.cu -Xcompiler -fopenmp `pkg-config opencv --cflags --libs ~/.local/lib/pkgconfig/opencv.pc`  -o DisplayImage_upup
noomp:
	nvcc DisplayImage_omp_ouyang.cu -Xcompiler `pkg-config opencv --cflags --libs ~/.local/lib/pkgconfig/opencv.pc`  -o DisplayImage_noomp