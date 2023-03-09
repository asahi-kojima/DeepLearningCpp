//#include <cuda_runtime.h>
//#include <stdio.h>
//#include <thread>
//#include <iostream>
//
//#include "commonGPU.cuh"
//
//
//void initialData(float* ip, int size)
//{
//    // generate different seed for random number
//    time_t t;
//    srand((unsigned)time(&t));
//
//    for (int i = 0; i < size; i++)
//    {
//        ip[i] = (float)(rand() & 0xFF) / 10.0f;
//    }
//
//    return;
//}
//
//int checker(float* a, float* b, float* c, int size)
//{
//    int noMatch = 0;
//    float ep = 0.000001;
//    for (int i = 0; i < size; i++)
//    {
//        float result = a[i] + b[i];
//        if (abs(result - c[i]) > ep)
//        {
//            noMatch++;
//        }
//    }
//
//    return noMatch;
//}
//__global__ void testGPUgunc(float* A, float* C)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    C[idx] = A[idx] + A[idx];
//}
//
//
////class A
////{
////public:
////    int byteSize;
////    int dataSize;
////    float* p;
////    float* d_A;
////    float* d_C;
////    A(int dataSize)
////        :dataSize(dataSize)
////        , byteSize(dataSize * sizeof(float))
////        , p(new float[dataSize])
////    {
////        for (int i = 0; i < dataSize; i++)
////        {
////            p[i] = 12.0f;
////        }
////
////    }
////    ~A()
////    {
////        delete[] p;
////    }
////
////    void f(dim3 grid, dim3 block)
////    {
////        cudaMalloc((float**)&d_A, byteSize);
////        cudaMalloc((float**)&d_C, byteSize);
////        cudaMemcpy(d_A, p, byteSize, cudaMemcpyHostToDevice);
////        testGPUgunc << <grid, block >> > (d_A, d_C);
////        cudaMemcpy(p, d_C, byteSize, cudaMemcpyDeviceToHost);
////        std::cout << p[12] << std::endl;
////    }
////
////   void g(float* A, float* C)
////    {
////        int idx = blockIdx.x * blockDim.x + threadIdx.x;
////        C[idx] = A[idx] + A[idx];
////        printf("t");
////    }
////};
//
//
//
//
////__global__ void testGPUfun(float* A, float* B, float* C)
////{
////    int idx = blockIdx.x * blockDim.x + threadIdx.x;
////    C[idx] = A[idx] + B[idx];
////}
//
//int main(int argc, char** argv)
//{
//    printGpuDriverInfo();
//
//
//    const int dataSize = 1 << 10;
//    const int byteSize = dataSize * sizeof(float);
//    float* h_A = new float[dataSize];
//    float* h_B = new float[dataSize];
//    float* h_C = new float[dataSize];
//
//    float* d_A, * d_B, * d_C;
//    cudaMalloc((float**)&d_A, byteSize);
//    cudaMalloc((float**)&d_B, byteSize);
//    cudaMalloc((float**)&d_C, byteSize);
//
//    initialData(h_A, dataSize);
//    initialData(h_B, dataSize);
//
//    cudaMemcpy(d_A, h_A, byteSize, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_B, h_B, byteSize, cudaMemcpyHostToDevice);
//
//
//    dim3 block(1 << 9);
//    dim3 grid((dataSize + block.x - 1) / block.x);
//
//    //A a{ dataSize };
//    //a.f(grid, block);
//    //a.g(d_A, d_C);
//    //testGPUfun << <grid, block >> > (d_A, d_B, d_C);
//    //cudaMemcpy(h_C, d_C, byteSize, cudaMemcpyDeviceToHost);
//
//
//    //std::cout << h_A[0] << std::endl;
//    //std::cout << h_B[0] << std::endl;
//    //std::cout << h_C[0] << std::endl;
//
//    std::cout << checker(h_A, h_B, h_C, dataSize);
//
//    return(0);
//}
