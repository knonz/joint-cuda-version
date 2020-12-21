#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <omp.h>
#include "CycleTimer.h"
using namespace cv;
using namespace std;

#define atc at<char>
#define atf at<float>
#define atv at<Vec3f>

float greenValue(Mat A,int type,float w1,float w2,int n,int i, int j);
vector<float> WghtDir(Mat A,float k,float T);
vector<float> CalcWeights(Mat A,float k);

// __global__ void cuda_bayer_reverse(unsigned char* img_input_cuda, unsigned char* img_output_cuda,int img_col, int img_row,  int img_border){
//     int cuda_col = blockIdx.x * blockDim.x + threadIdx.x;
//     int cuda_row = blockIdx.y * blockDim.y + threadIdx.y;

//     unsigned int tmp = 0;
//     int target = 0;
//     int a, b;

//     // for(int j = 0; j < filter_row; j++){
//     //     for(int i = 0; i < filter_row; i++){
//     //         a = cuda_col + i - (filter_row / 2);
//     //         b = cuda_row + j - (filter_row / 2);

//     //         target = 3 * (b * img_col + a) + shift;
//     //         if (target >= img_border || target < 0){
//     //             continue;
//     //         }
// 		// 	tmp += filter_cuda[j * filter_row + i] * img_input_cuda[target];  
//     //     }
//     // }
//     // tmp /= filter_scale;

//     // if(tmp > 255){
//     //     tmp = 255;
//     // }
//     // img_output_cuda[3 * (cuda_row * img_col + cuda_col) + shift] = tmp;
// }
__global__ void kernel_bayer_reverse(const cv::cuda::PtrStepSzf input,cv::cuda::PtrStepSzf output)
{
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x <= input.cols - 1 && y <= input.rows - 1 && y >= 0 && x >= 0)
        {
           output(y, x) = input(y, x);
        }
}

void bayer_reverse_callKernel(cv::InputArray _input,cv::OutputArray _output,cv::cuda::Stream _stream)
{
    const cv::cuda::GpuMat input = _input.getGpuMat();

    _output.create(input.size(), input.type()); 
    cv::cuda::GpuMat output = _output.getGpuMat();

    dim3 cthreads(16, 16);
    dim3 cblocks(
        static_cast<int>(std::ceil(input1.size().width /
            static_cast<double>(cthreads.x))),
        static_cast<int>(std::ceil(input1.size().height / 
            static_cast<double>(cthreads.y))));

    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);
    myKernel<<<cblocks, cthreads, 0, stream>>>(input, output);

    cudaSafeCall(cudaGetLastError());
}

Mat bayer_reverse_CU(Mat img){
  MAT convert to _input

  bayer_reverse_callKernel(_input,_output,_stream);

  output convert to MAT
  // int img_col = img.size().width;
  // int img_row = img.size().height;
  // int resolution = 3 * (img_col * img_row);
  // cv::cuda::GpuMat myGpuMat,myGpuMat2;
  // myGpuMat.upload(img);
  // float* img_input_cuda;
  // // Grid and block, divide the image into 1024 per block
  // const dim3 block_size(block_row, block_row);
  // const dim3 grid_size((img_col + block_row - 1) / block_row, (img_row + block_row - 1) / block_row);
  // cudaMalloc((void**) &img_input_cuda, resolution * sizeof(float));
  // cudaMemcpy(img_input_cuda, img_input, resolution * sizeof(float), cudaMemcpyHostToDevice)
  // https://docs.opencv.org/master/df/dfc/group__cudev.html
  // https://docs.opencv.org/master/d5/dc3/group__cudalegacy.html
  // https://docs.opencv.org/3.4/d4/d03/cudaarithm_8hpp.html
  // kernel_bayer_reverse<<<grid_size, block_size>>>(img_input_cuda, img_output_cuda);
  // cv::Mat result(myGpuMat2);
  return result;
}

Mat bayer_reverse(Mat img){
  int width = img.size().width;
  int height = img.size().height;
  Mat tmp(cv::Size(width, height), CV_32F ); 
  vector<Mat> channels(3);
  split(img, channels);
  #pragma omp parallel for
  for(int i = 0;i<height;i++){
    for(int j = 0;j<width;j++){
      //cout<<img.at<Vec3f>(i,j)<<endl;
      
      if(i % 2==0){
        if(j % 2==0)
          tmp.atf(i,j) = channels[1].at<float>(i,j);
        else
          tmp.atf(i,j) = channels[2].at<float>(i,j);
      }else{
        if(j % 2==0)
          tmp.atf(i,j) = channels[0].at<float>(i,j);
        else
          tmp.atf(i,j) = channels[1].at<float>(i,j);
      }
      //tmp.atf(i,j)=(channels[0]).atf(i,j);
    }
  }
  return tmp; 
  //return channels[0]; 
}

Mat ACPIgreenH(Mat CFA){

  int nRow = CFA.size().height;
  int nCol = CFA.size().width;
  Mat green;
  CFA.copyTo(green);
  #pragma omp parallel for
  for(int i = 3; i < nRow - 1; i += 2){
    for(int j = 2; j < nCol - 2; j += 2){
      float a = CFA.atf(i,j-1) + CFA.atf(i,j+1);
      float b = 2*CFA.atf(i,j) - CFA.atf(i,j-2) - CFA.atf(i,j+2);
      green.atf(i,j) = (a+b)/2;
    }
  }
  #pragma omp parallel for
  for(int i = 2; i < nRow - 2; i += 2){
    for(int j = 3; j < nCol - 1; j += 2){
      float a = CFA.atf(i,j-1) + CFA.atf(i,j+1);
      float b = CFA.atf(i,j) - CFA.atf(i,j-2) - CFA.atf(i,j+2);
      green.atf(i,j) = (a+b)/2;
    }
  }

  return green;

}

Mat ACPIgreenV(Mat CFA){

  int nRow = CFA.size().height;
  int nCol = CFA.size().width;
  Mat green;
  CFA.copyTo(green);
  #pragma omp parallel for
  for(int i = 3; i < nRow - 2; i += 2){
    for(int j = 2; j < nCol - 3; j += 2){
      float a = CFA.atf(i-1,j) + CFA.atf(i+1,j);
      float b = 2*CFA.atf(i,j) - CFA.atf(i-2,j) - CFA.atf(i+2,j);
      green.atf(i,j) = (a+b)/2;
    }
  }
  #pragma omp parallel for
  for(int i = 2; i < nRow - 2; i += 2){
    for(int j = 3; j < nCol - 1; j += 2){
      float a = CFA.atf(i-1,j) + CFA.atf(i+1,j);
      float b = CFA.atf(i,j) - CFA.atf(i-2,j) - CFA.atf(i+2,j);
      green.atf(i,j) = (a+b)/2;
    }
  }

  return green;

}

typedef struct DDFW{
  Mat G;
  Mat DM;
}DDFW;

DDFW jointDDFWgreen(Mat CFA,Mat gh,Mat gv){
  int nRow = CFA.size().height;
  int nCol = CFA.size().width;
  
  Mat G; 
  CFA.copyTo(G);
  Mat DM(cv::Size(nCol, nRow), CV_8U ); 

  Mat CH(cv::Size(nCol, nRow), CV_32F ); 
  Mat CV(cv::Size(nCol, nRow), CV_32F ); 
  
  
  CH = CFA - gh;
  CV = CFA - gv;
  
  Mat DH(cv::Size(nCol, nRow), CV_32F ); 
  #pragma omp parallel for
  for(int i = 0;i<nRow-2;i+=2){
    for(int j = 1;j<nCol-3;j+=2){
      DH.atf(i,j) = abs( CH.atf(i,j) - CH.atf(i+2,j) );
    }
  }
  #pragma omp parallel for
  for(int i = 1;i<nRow-1;i+=2){
    for(int j = 0;j<nCol-4;j+=2){
      DH.atf(i,j) = abs( CH.atf(i,j) - CH.atf(i+2,j) );
    }
  }


  Mat DV(cv::Size(nCol, nRow), CV_32F ); 
  #pragma omp parallel for
  for(int i = 0;i<nRow-4;i+=2){
    for(int j = 1;j<nCol-1;j+=2){
      DV.atf(i,j) = abs( CV.atf(i,j) - CV.atf(i+2,j) );
    }
  }
  #pragma omp parallel for
  for(int i = 1;i<nRow-1;i+=2){
    for(int j = 0;j<nCol-4;j+=2){
      DV.atf(i,j) = abs( CV.atf(i,j) - CV.atf(i+2,j) );
    }
  }

  Mat DeltaH(cv::Size(nCol, nRow), CV_32F );
  Mat DeltaV(cv::Size(nCol, nRow), CV_32F );
  
  // positions of red pixels
  #pragma omp parallel for
  for(int i = 2;i<nRow-2;i+=2){
    for(int j = 3;j<nCol-1;j+=2){
      DeltaH.atf(i,j) = DH.atf(i-2,j-2) + DH.atf(i-2,j) + DH.atf(i,j-2) + DH.atf(i,j) + DH.atf(i+2,j-2) + DH.atf(i+2,j) + DH.atf(i-1,j-1) + DH.atf(i+1,j-1);
      DeltaV.atf(i,j) = DV.atf(i-2,j-2) + DV.atf(i-2,j) + DV.atf(i-2,j+2) + DH.atf(i,j-2) + DH.atf(i,j) + DH.atf(i,j+2) + DH.atf(i-1,j-1) + DH.atf(i-1,j+1);      
    }
  }


  // positions of blue pixels
  #pragma omp parallel for
  for(int i = 3;i<nRow-2;i+=2){
    for(int j = 2;j<nCol-3;j+=2){
      DeltaH.atf(i,j) = DH.atf(i-2,j-2)+DH.atf(i-2,j)+DH.atf(i,j-2)+DH.atf(i,j)+DH.atf(i+2,j-2)+DH.atf(i+2,j)+DH.atf(i-1,j-1)+DH.atf(i+1,j-1);
      DeltaV.atf(i,j) = DV.atf(i-2,j-2)+DV.atf(i-2,j)+DV.atf(i-2,j+2)+DV.atf(i,j-2)+DV.atf(i,j)+DV.atf(i,j+2)+DV.atf(i-1,j-1)+DV.atf(i-1,j+1);
      
    }
  }
  
  // Decide between the horizontal and vertical interpolations
  float T = 1.5;
  #pragma omp parallel for
  for(int i = 2;i<nRow-2;i+=2){
    for(int j = 3;j<nCol-1;j+=2){
      if( (1+DeltaH.atf(i,j))/(1+DeltaV.atf(i,j))>T ){
        G.atf(i,j) = gv.atf(i,j);
        DM.atc(i,j) = 1;
      }else if( (1+DeltaV.atf(i,j))/(1+DeltaH.atf(i,j))>T ){
        G.atf(i,j) = gh.atf(i,j);
        DM.atc(i,j) = 2;
      }else{
        float h1 = CFA.atf(i, j+1)-CFA.atf(i, j-1);
        float h2 = 2*CFA.atf(i, j)-CFA.atf(i, j-2)-CFA.atf(i, j+2);
        float v1 = CFA.atf(i+1, j)-CFA.atf(i-1, j);
        float v2 = 2*CFA.atf(i, j)-CFA.atf(i-2, j)-CFA.atf(i+2, j);
        float HG = abs(h1)+abs(h2);
        float VG = abs(v1)+abs(v2);
        float w1 = 1/(1+HG);
        float w2 = 1/(1+VG);
        G.atf(i,j) = (w1*gh.atf(i,j)+w2*gv.atf(i,j))/(w1+w2);
        DM.atc(i,j) = 3;

      }
    }
  }
  #pragma omp parallel for
  for(int i = 3;i<nRow-2;i+=2){
    for(int j = 2;j<nCol-3;j+=2){
      if( (1+DeltaH.atf(i,j))/(1+DeltaV.atf(i,j))>T ){
        G.atf(i,j) = gv.atf(i,j);
        DM.atc(i,j) = 1;
      }else if ((1+DeltaV.atf(i,j))/(1+DeltaH.atf(i,j))>T){
        G.atf(i,j) = gh.atf(i,j);
        DM.atc(i,j) = 2;
      }else{
        float h1 = CFA.atf(i, j-1)-CFA.atf(i, j+1);
        float h2 = 2*CFA.atf(i, j)-CFA.atf(i, j-2)-CFA.atf(i, j+2);
        float v1 = CFA.atf(i-1, j)-CFA.atf(i+1, j);
        float v2 = 2*CFA.atf(i, j)-CFA.atf(i-2, j)-CFA.atf(i+2, j);
        float HG = abs(h1)+abs(h2);
        float VG = abs(v1)+abs(v2);
        float w1 = 1/(1+HG);
        float w2 = 1/(1+VG);
        G.atf(i,j) = (w1*gh.atf(i,j)+w2*gv.atf(i,j))/(w1+w2);
        DM.atc(i,j) = 3;
      }
    }
  }

  DDFW tmp = {.G =G,.DM = DM};
  return tmp;
  
  
}

vector<float> DDFWweights(Mat CFA,int i,int j,int type){
  float cf = 2;
  vector<float> weights(4);
  if(type == 1){
    float h = abs( CFA.atf(i,j-1)-CFA.atf(i,j+1));
    float v = abs( CFA.atf(i-1,j)-CFA.atf(i+1,j));
    weights[0] = 1/(1+h+cf*abs(CFA.atf(i,j-3)-CFA.atf(i,j-1)));
    weights[1] = 1/(1+v+cf*abs(CFA.atf(i+3,j)-CFA.atf(i+1,j)));
    weights[2] = 1/(1+h+cf*abs(CFA.atf(i,j+3)-CFA.atf(i,j+1)));
    weights[3] = 1/(1+v+cf*abs(CFA.atf(i-3,j)-CFA.atf(i-1,j))); 
  }else if(type==2){
    float h = abs(CFA.atf(i-1,j-1)-CFA.atf(i+1,j+1));
    float v = abs(CFA.atf(i-1,j+1)-CFA.atf(i+1,j-1));
    weights[0] = 1/(1+h+cf*abs(CFA.atf(i-3,j-3)-CFA.atf(i-1,j-1)));
    weights[1] = 1/(1+v+cf*abs(CFA.atf(i+3,j-3)-CFA.atf(i+1,j-1)));
    weights[2] = 1/(1+h+cf*abs(CFA.atf(i+3,j+3)-CFA.atf(i+1,j+1)));
    weights[3] = 1/(1+v+cf*abs(CFA.atf(i-3,j+3)-CFA.atf(i-1,j+1)));
  }else{
    cout<<"AAAA"<<endl;
  }
  return weights;
}

Mat DDFW_RG_diff(Mat CFA, Mat green){
  int nRow = CFA.size().height;
  int nCol = CFA.size().width;

  Mat KR(cv::Size(nCol, nRow), CV_32F );
  KR = CFA-green;

  //cout<<"hi"<<endl;
  #pragma omp parallel for
  for(int i = 3;i<nRow-3;i+=2){
    for(int j = 4;j<nCol-2;j+=2){
      vector<float> w(4);
      w = DDFWweights(KR,i,j,2);
      float a = KR.atf(i-1,j-1)*w[0]+KR.atf(i+1,j-1)*w[1]+KR.atf(i+1,j+1)*w[2]+KR.atf(i-1,j+1)*w[3];
      float b = w[0]+w[1]+w[2]+w[3];
      KR.atf(i,j) = a/b;
    }
  }
  #pragma omp parallel for
  for(int i = 4;i<nRow-3;i+=2){
    for(int j = 4;j<nCol-3;j+=2){
      vector<float> w(4);
      w = DDFWweights(KR,i,j,1);

      float a = KR.atf(i,j-1)*w[0]+KR.atf(i+1,j)*w[1]+KR.atf(i,j+1)*w[2]+KR.atf(i-1,j)*w[3];
      float b = w[0]+w[1]+w[2]+w[3];
      KR.atf(i,j) = a/b;
    }
  }
  #pragma omp parallel for
  for(int i = 3;i<nRow-3;i+=2){
    for(int j = 3;j<nCol-3;j+=2){
      vector<float> w = DDFWweights(KR,i,j,1);

      float a = KR.atf(i,j-1)*w[0]+KR.atf(i+1,j)*w[1]+KR.atf(i,j+1)*w[2]+KR.atf(i-1,j)*w[3];
      float b = w[0]+w[1]+w[2]+w[3];
      KR.atf(i,j) = a/b;
    }
  }
  
  return KR;
}


Mat DDFW_BG_diff(Mat CFA, Mat green){
  int nRow = CFA.size().height;
  int nCol = CFA.size().width;

  Mat KB(cv::Size(nCol, nRow), CV_32F );
  KB = CFA-green;
  #pragma omp parallel for
  for(int i = 4;i<nRow-3;i+=2){
    for(int j = 3;j<nCol-4;j+=2){
      vector<float> w = DDFWweights(KB,i,j,2);
      float a = KB.atf(i-1,j-1)*w[0]+KB.atf(i+1,j-1)*w[1]+KB.atf(i+1,j+1)*w[2]+KB.atf(i-1,j+1)*w[3];
      float b = w[0]+w[1]+w[2]+w[3];
      KB.atf(i,j) = a/b;
    }
  }
  #pragma omp parallel for
  for(int i = 4;i<nRow-3;i+=2){
    for(int j = 4;j<nCol-3;j+=2){
      vector<float> w = DDFWweights(KB,i,j,1);

      float a = KB.atf(i,j-1)*w[0]+KB.atf(i+1,j)*w[1]+KB.atf(i,j+1)*w[2]+KB.atf(i-1,j)*w[3];
      float b = w[0]+w[1]+w[2]+w[3];
      KB.atf(i,j) = a/b;
    }
  }

#pragma omp parallel for
  for(int i = 3;i<nRow-3;i+=2){
    for(int j = 3;j<nCol-3;j+=2){
      vector<float> w = DDFWweights(KB,i,j,1);

      float a = KB.atf(i,j-1)*w[0]+KB.atf(i+1,j)*w[1]+KB.atf(i,j+1)*w[2]+KB.atf(i-1,j)*w[3];
      float b = w[0]+w[1]+w[2]+w[3];
      KB.atf(i,j) = a/b;
    }
  }
  
  return KB;
}

Mat DDFW_refine_green(Mat CFA, Mat KR, Mat KB){
  int nRow = CFA.size().height;
  int nCol = CFA.size().width;
  Mat G;
  CFA.copyTo(G);
#pragma omp parallel for
  for(int i = 4;i<nRow-3;i+=2){
    for(int j = 3;j<nCol-4;j+=2){
      vector<float> w = DDFWweights(KR,i,j,1);

      float a = KR.atf(i,j-1)*w[0]+KR.atf(i+1,j)*w[1]+KR.atf(i,j+1)*w[2]+KR.atf(i-1,j)*w[3];
      float b = w[0]+w[1]+w[2]+w[3];
      G.atf(i,j) = CFA.atf(i,j) - a/b;
    }
  }
#pragma omp parallel for
  for(int i = 3;i<nRow-3;i+=2){
    for(int j = 4;j<nCol-2;j+=2){
      vector<float> w = DDFWweights(KB,i,j,1);

      float a = KB.atf(i,j-1)*w[0]+KB.atf(i+1,j)*w[1]+KB.atf(i,j+1)*w[2]+KB.atf(i-1,j)*w[3];
      float b = w[0]+w[1]+w[2]+w[3];
      G.atf(i,j) = CFA.atf(i,j) - a/b;
    }
  }

  return G;
}


Mat DDFW_refine_RG_diff(Mat KR){

  int nRow = KR.size().height;
  int nCol = KR.size().width;
#pragma omp parallel for
  for(int i = 3;i<nRow-3;i+=2){
    for(int j = 4;j<nCol-2;j+=2){
      vector<float> w = DDFWweights(KR,i,j,1);

      float a = KR.atf(i,j-1)*w[0]+KR.atf(i+1,j)*w[1]+KR.atf(i,j+1)*w[2]+KR.atf(i-1,j)*w[3];
      float b = w[0]+w[1]+w[2]+w[3];
      KR.atf(i,j) = a/b;
    }
  }
#pragma omp parallel for
  for(int i = 4;i<nRow-3;i+=2){
    for(int j = 4;j<nCol-3;j+=2){
      vector<float> w = DDFWweights(KR,i,j,1);

      float a = KR.atf(i,j-1)*w[0]+KR.atf(i+1,j)*w[1]+KR.atf(i,j+1)*w[2]+KR.atf(i-1,j)*w[3];
      float b = w[0]+w[1]+w[2]+w[3];
      KR.atf(i,j) =  a/b;
    }
  }
#pragma omp parallel for
  for(int i = 3;i<nRow-3;i+=2){
    for(int j = 3;j<nCol-3;j+=2){
      vector<float> w = DDFWweights(KR,i,j,1);

      float a = KR.atf(i,j-1)*w[0]+KR.atf(i+1,j)*w[1]+KR.atf(i,j+1)*w[2]+KR.atf(i-1,j)*w[3];
      float b = w[0]+w[1]+w[2]+w[3];
      KR.atf(i,j) =  a/b;
    }
  }
  
  return KR;
}


Mat DDFW_refine_BG_diff(Mat KB){

  int nRow = KB.size().height;
  int nCol = KB.size().width;
#pragma omp parallel for
  for(int i = 4;i<nRow-3;i+=2){
    for(int j = 3;j<nCol-4;j+=2){
      vector<float> w = DDFWweights(KB,i,j,1);

      float a = KB.atf(i,j-1)*w[0]+KB.atf(i+1,j)*w[1]+KB.atf(i,j+1)*w[2]+KB.atf(i-1,j)*w[3];
      float b = w[0]+w[1]+w[2]+w[3];
      KB.atf(i,j) = a/b;
    }
  }
#pragma omp parallel for
  for(int i = 4;i<nRow-3;i+=2){
    for(int j = 4;j<nCol-3;j+=2){
      vector<float> w = DDFWweights(KB,i,j,1);

      float a = KB.atf(i,j-1)*w[0]+KB.atf(i+1,j)*w[1]+KB.atf(i,j+1)*w[2]+KB.atf(i-1,j)*w[3];
      float b = w[0]+w[1]+w[2]+w[3];
      KB.atf(i,j) =  a/b;
    }
  }
#pragma omp parallel for  
  for(int i = 3;i<nRow-3;i+=2){
    for(int j = 3;j<nCol-3;j+=2){
      vector<float> w = DDFWweights(KB,i,j,1);

      float a = KB.atf(i,j-1)*w[0]+KB.atf(i+1,j)*w[1]+KB.atf(i,j+1)*w[2]+KB.atf(i-1,j)*w[3];
      float b = w[0]+w[1]+w[2]+w[3];
      KB.atf(i,j) =  a/b;
    }
  }
  
  return KB;
}

Mat RoundImage(Mat img){
  int nRow = img.size().height;
  int nCol = img.size().width;
#pragma omp parallel for  
  for(int i = 0;i<nRow;i++){
    for(int j = 0;j<nCol;j++){
      img.atv(i,j)[0] = round( img.atv(i,j)[0] );

      if(img.atv(i,j)[0] > 255)
        img.atv(i,j)[0] = 255;
      if(img.atv(i,j)[0] < 0)
        img.atv(i,j)[0] = 0;

      if(img.atv(i,j)[1] > 255)
        img.atv(i,j)[1] = 255;
      if(img.atv(i,j)[1] < 0)
        img.atv(i,j)[1] = 0;

      if(img.atv(i,j)[2] > 255)
        img.atv(i,j)[2] = 255;
      if(img.atv(i,j)[2] < 0)
        img.atv(i,j)[2] = 0;
    }
  }
  
  return img;
}

DDFW jointDDFW(Mat CFA){
  //cout<<"ACPIGreenH"<<endl;
  Mat gh,gv;
//  //#pragma omp parallel sections
{ 
//  //#pragma omp section
  gh = ACPIgreenH(CFA);
//  //#pragma omp section
  gv = ACPIgreenV(CFA);
}
  //cout<<"jointDDFWgreen"<<endl;
  DDFW a = jointDDFWgreen(CFA,gh,gv);
  Mat G0 = a.G;
  Mat DM = a.DM;
  Mat KR,KB;
//  //#pragma omp parallel sections
{ 
  //cout<<"DDFW_RG_Diff"<<endl;
//  //#pragma omp section
  KR = DDFW_RG_diff(CFA,G0);
//  //#pragma omp section
  KB = DDFW_BG_diff(CFA,G0);
}
  //cout<<"DDFW_refine_green"<<endl;
  Mat G;
  G = DDFW_refine_green(CFA,KR,KB);
//  //#pragma omp parallel sections
{ 
//  //#pragma omp section
  KR = DDFW_refine_RG_diff(KR+G0-G);
//  //#pragma omp section
  KB = DDFW_refine_BG_diff(KB+G0-G);
}
  int width = G.size().width;
  int height = G.size().height;
  Mat d_out(cv::Size(width, height), CV_32FC3 );
  vector<Mat> t;
  Mat GKB = G+KB;
  Mat GKR = G+KR;
  t.push_back(GKB);
  t.push_back(G);
  t.push_back(GKR);

  
  merge(t, d_out);
  d_out = RoundImage(d_out);
  DDFW tmp = {.G = d_out,.DM=DM};
  
  return tmp;
}


DDFW zoomGreen(Mat green,Mat DM,float k,float T){
  int nRow = green.size().height;
  int nCol = green.size().width; 

  nRow = nRow * 2;
  nCol = nCol * 2;
  
  Mat A(cv::Size(nCol, nRow), CV_32F );
  Mat temp(cv::Size(nCol, nRow), CV_32F );
#pragma omp parallel for  
  for(int i = 0;i<nRow/2;i++){
    for(int j = 0;j<nCol/2;j++){
      A.atf(i*2,j*2) = green.atf(i,j);
      temp.atf(i*2,j*2) = DM.atf(i,j);
    }
  }
  //cout<<"Asub"<<endl;
#pragma omp parallel for
  for(int i = 3;i<nRow-3;i+=2){
    for(int j = 3;j<nCol-3;j+=2){
      Rect roi1( i, j, 7, 7 );
      Mat Asub = A(Range(i-3,i+4),Range(j-3,j+4) );
      
      vector<float> w = WghtDir(Asub,k,T);
      A.atf(i,j) = greenValue(Asub,1,w[0],w[1],w[2],i,j);
      temp.atf(i,j)=w[2];
    }
  }
#pragma omp parallel for 
  for(int i = 4;i<nRow-4;i+=2){
    for(int j = 3;j<nCol-3;j+=2){


      Mat Asub = A(Range(i-2,i+3),Range(j-2,j+3));
      Mat Asub7 = A(Range(i-3,i+4),Range(j-3,j+4));
      
      
      vector<float> w = CalcWeights(Asub,k);
      float n = temp.atf(i,j-1) + temp.atf(i,j+1);
      A.atf(i,j) = greenValue( Asub7,2,w[0],w[1],n,i,j);
    }
  }

#pragma omp parallel for
  for(int i = 3;i<nRow-3;i+=2){
    for(int j = 4;j<nCol-4;j+=2){
      Rect roi1( i, j, 5, 5 );
      Rect roi2( i, j, 7, 7 );
      
      Mat Asub = A(Range(i-2,i+3),Range(j-2,j+3));
      Mat Asub7 = A(Range(i-3,i+4),Range(j-3,j+4));
      
      vector<float> w = CalcWeights(Asub,k);
      float n = temp.atf(i-1,j) + temp.atf(i+1,j);
      A.atf(i,j) = greenValue( Asub7,3,w[0],w[1],n,i,j);
    }
  }
   
  
  DDFW tmp = {.G = A,.DM=temp};
  return tmp;
}

vector<float> WghtDir(Mat A,float k,float T){
  vector<float> w(3);
  float t1 = abs(A.atf(2,0)-A.atf(0,2));   
  float t2 = abs(A.atf(4,0)-A.atf(2,2))+abs(A.atf(2,2)-A.atf(0,4));     
  float t3 = abs(A.atf(6,0)-A.atf(4,2))+abs(A.atf(4,2)-A.atf(2,4))+abs(A.atf(2,4)-A.atf(0,6)); 
  float t4 = abs(A.atf(6,2)-A.atf(4,4))+abs(A.atf(4,4)-A.atf(2,6)); 
  float t5 = abs(A.atf(6,4)-A.atf(4,6)); 
  float d1 = t1+t2+t3+t4+t5;
  
  t1 = abs(A.atf(0,4)-A.atf(2,6));   
  t2 = abs(A.atf(0,2)-A.atf(2,4))+abs(A.atf(2,4)-A.atf(4,6));   
  t3 = abs(A.atf(0,0)-A.atf(2,2))+abs(A.atf(2,2)-A.atf(4,4))+abs(A.atf(4,4)-A.atf(6,6)); 
  t4 = abs(A.atf(2,0)-A.atf(4,2))+abs(A.atf(4,2)-A.atf(6,4));
  t5 = abs(A.atf(4,0)-A.atf(6,2));
  float d2 = t1+t2+t3+t4+t5;

  w[0] = 1/(1+pow(d1,k) ); 
  w[1] = 1/(1+pow(d2,k) );

  
  if( (1+d1)/(1+d2) > T )
     w[2] = 1; 
  else if( (1+d2)/(1+d1) > T )
     w[2] = 2; 
  else
     w[2] = 3;

  return w;
}

vector<float> CalcWeights(Mat A,float k){
  vector<float> w(2);
  float t1 = abs(A.atf(0,1)-A.atf(0,3))+abs(A.atf(2,1)-A.atf(2,3))+abs(A.atf(4,1)-A.atf(4,3));
  float t2 = abs(A.atf(1,0)-A.atf(1,2))+abs(A.atf(1,2)-A.atf(1,4));
  float t3 = abs(A.atf(3,0)-A.atf(3,2))+abs(A.atf(3,2)-A.atf(3,4));
  float d1 = t1+t2+t3;

  t1 = abs(A.atf(1,0)-A.atf(3,0))+abs(A.atf(1,2)-A.atf(3,2))+abs(A.atf(1,4)-A.atf(3,4));
  t2 = abs(A.atf(0,1)-A.atf(2,1))+abs(A.atf(2,1)-A.atf(4,1));
  t3 = abs(A.atf(0,3)-A.atf(2,3))+abs(A.atf(2,3)-A.atf(4,3));
  float d2 = t1+t2+t3;

  w[0] = 1/(1+pow(d1,k)); 
  w[1] = 1/(1+pow(d2,k));

  return w;
}

float greenValue(Mat A,int type,float w1,float w2,int n,int i, int j){
  vector<float> f(4);
  f[0] = -1/16;
  f[1] = 9/16;
  f[2] = 9/16;
  f[3] = -1/16;

  vector<float> v1(4);
  vector<float> v2(4);
  if(type == 1){
    v1[0] = A.atf(6,0);
    v1[1] = A.atf(4,2);
    v1[2] = A.atf(2,4);
    v1[3] = A.atf(0,6);
    
    v2[0] = A.atf(0,0);
    v2[1] = A.atf(2,2);
    v2[2] = A.atf(4,4);
    v2[3] = A.atf(6,6);
  }else{
    v1[0] = A.atf(3,1);
    v1[1] = A.atf(3,2);
    v1[2] = A.atf(3,4);
    v1[3] = A.atf(3,6);
    
    v2[0] = A.atf(1,3);
    v2[1] = A.atf(2,3);
    v2[2] = A.atf(4,3);
    v2[3] = A.atf(6,3);
    
  }
  float p; 
  if(n==1){
    p =v2[0]*f[0] + v2[1]*f[1] + v2[2]*f[2] + v2[3]*f[3];
  }else if(n==2){
    p =v1[0]*f[0] + v1[1]*f[1] + v1[2]*f[2] + v1[3]*f[3];
  }else{
    float p1 =v1[0]*f[0] + v1[1]*f[1] + v1[2]*f[2] + v1[3]*f[3];
    float p2 =v2[0]*f[0] + v2[1]*f[1] + v2[2]*f[2] + v2[3]*f[3];
    p = (w1*p1+w2*p2)/(w2+w2);
  }

  return p;
}

Mat zoomColorDiff(Mat color_diff, Mat DM){
  int nRow = color_diff.size().height;
  int nCol = color_diff.size().width;

  nRow = nRow * 2;
  nCol = nCol * 2;
  
  Mat A(cv::Size(nCol, nRow), CV_32F );
  Mat temp(cv::Size(nCol, nRow), CV_32F );
#pragma omp parallel for  
  for(int i = 0;i<nRow/2;i++){
    for(int j = 0;j<nCol/2;j++){
      A.atf(i*2,j*2) = color_diff.atf(i,j);
    }
  }
#pragma omp parallel for
  for(int i = 3;i<nRow-3;i+=2){
    for(int j = 3;j<nCol-3;j+=2){
      if( DM.atf(i,j) == 1)
        A.atf(i,j) = ( A.atf(i-1,j-1) + A.atf(i+1,j+1) ) / 2;
      else if( DM.atf(i,j) == 2)
        A.atf(i,j) = ( A.atf(i-1,j+1) + A.atf(i+1,j-1) ) / 2;
      else
        A.atf(i,j) = ( A.atf(i-1,j-1) + A.atf(i-1,j+1) + A.atf(i+1,j-1) + A.atf(i+1,j+1) ) / 4;
    }
  }
#pragma omp parallel for
  for(int i = 4;i<nRow-4;i+=2){
    for(int j = 3;j<nCol-3;j+=2){
      if( DM.atf(i,j-1) + DM.atf(i,j+1) == 1 )
        A.atf(i,j) = ( A.atf(i-1,j) + A.atf(i+1,j) ) / 2;
      else if ( DM.atf(i,j-1) + DM.atf(i,j+1) == 2)
        A.atf(i,j) = ( A.atf(i,j-1) + A.atf(i,j+1) ) / 2;
      else
        A.atf(i,j) = ( A.atf(i-1,j) + A.atf(i+1,j) + A.atf(i,j-1) + A.atf(i,j+1) ) / 4;
    }
  }
#pragma omp parallel for
  for(int i = 3;i<nRow-3;i+=2){
    for(int j = 4;j<nCol-4;j+=2){
      if( DM.atf(i-1,j) + DM.atf(i+1,j) == 1 )
        A.atf(i,j) = ( A.atf(i-1,j) + A.atf(i+1,j) ) / 2;
      else if( DM.atf(i-1,j) + DM.atf(i+1,j) == 2)
        A.atf(i,j) = ( A.atf(i,j-1) + A.atf(i,j+1) ) / 2;
      else
        A.atf(i,j) = ( A.atf(i-1,j) + A.atf(i+1,j) + A.atf(i,j-1) + A.atf(i,j+1) ) / 4;
    }
  }


  return A;
}

Mat jointZoom(Mat img, Mat DM, float k, float T){
  vector<Mat> channels(3);
  split(img, channels);
  Mat G = channels[1];
  Mat grDiff = channels[2] - G;
  Mat gbDiff = channels[0] - G;

  //cout<<"zoomGreen"<<endl;
  DDFW ZG = zoomGreen(G,DM,k,T);
  Mat zoomedG = ZG.G; 
  Mat Dm = ZG.DM;
  
  //cout<<"zoomColorDiff"<<endl;
  Mat zoomedR = zoomedG + zoomColorDiff(grDiff,Dm);
  Mat zoomedB = zoomedG + zoomColorDiff(gbDiff,Dm);
  //cout<<"EEE"<<endl;


  int width = zoomedR.size().width;
  int height = zoomedR.size().height;

  Mat out(cv::Size(width, height), CV_32FC3 );
  
  
  vector<Mat> OUT;
  OUT.push_back(zoomedB);
  OUT.push_back(zoomedG);
  OUT.push_back(zoomedR);
  merge(OUT, out);
  
  return out;
}

Mat DZ(Mat CFA){
  float k = 5;
  float T = 1.15;

  //cout<<"jointDDFW"<<endl;
  DDFW d_out = jointDDFW(CFA);
  
  Mat DOUT = d_out.G;
  Mat DM = d_out.DM;
  //cout<<"jointZoom"<<endl;
  Mat OUT = jointZoom(DOUT,DM,k,T);
  OUT = RoundImage(OUT);
  return OUT;
}


int main(int argc, char* argv[]) {

  // 檢查是否有指定輸入影像檔案
  if ( argc != 2 ) {
    printf("usage: DisplayImage.out <Image_Path>n");
    return -1;
  }
  double start,out_time,sum=0.0,avg=0.0;
  // 讀取影像檔案
  Mat image,imageFloat;
  image = imread( argv[1] ,CV_32F);
  image.convertTo(imageFloat, CV_32FC3, 1/255.0);
  // 檢查影像是否正確讀入
  if ( !image.data ) {
    printf("No image data n");
    return -1;
  }
  // Mat small = imageFloat;
  Mat small;
  resize(imageFloat,small, Size(),0.5,0.5 );
  //cout<< image.channels()<<endl;
  Mat tmp,out;
  for(int i=0;i<5;i++){
    start = CycleTimer::currentSeconds();
    tmp = bayer_reverse(small);
    out = DZ(tmp); 
    out_time = CycleTimer::currentSeconds() - start;
    sum+=out_time;
  }

  cout<<"avg out_time:"<<sum/5.0<<endl;
  // Mat img;
  // img.create(2,2,CV_8UC1);
  // Mat img2;
  // out.convertTo(img2, CV_8UC3);
  image.convertTo(imageFloat, CV_8UC3, 255);
  imwrite("PPOUTPUT.bmp",image );
  // 建立視窗
  // namedWindow("Display Image", WINDOW_AUTOSIZE);
  
  // vector<unsigned char> d(3840*2160);
  // vector<unsigned char> e(3840*2160,87);
  // start = CycleTimer::currentSeconds();
  // for(int i = 0;i<3840*2160;i++){
  //   d[i] = e[i] + 1;
  // }
  // out_time = CycleTimer::currentSeconds() - start;
  // cout<<"no pragma omp parallel for:"<<out_time<<endl;
  
  // start = CycleTimer::currentSeconds();
  // // #pragma omp parallel for
  // //#pragma omp for schedule(dynamic)
  // for(int i = 0;i<3840*2160;i++){
  //   d[i] = e[i] + 1;
  // }
  // out_time = CycleTimer::currentSeconds() - start;
  // cout<<"omp for schedule(dynamic):"<<out_time<<endl;
  
  // start = CycleTimer::currentSeconds();
  // #pragma omp parallel for
  // for(int i = 0;i<3840*2160;i++){
  //   d[i] = e[i] + 1;
  // }
  // out_time = CycleTimer::currentSeconds() - start;
  // cout<<"omp parallel for:"<<out_time<<endl;

  // 用視窗顯示影像
  // imshow("Display Image", image);

  // 顯示視窗，直到任何鍵盤輸入後才離開
  // waitKey(0);

  return 0;
}
