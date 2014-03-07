#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2
#define O_TILE_WIDTH 32
#define BLOCK_WIDTH O_TILE_WIDTH + (Mask_width-1)
//@@ INSERT CODE HERE
__global__ void convolution_2D_kernel(float* in, float* out,const float* __restrict__ mask, int width, int height,int channels) 
{
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	
	int row =by*O_TILE_WIDTH+ty;
	int col = bx*O_TILE_WIDTH+tx;
	
	int row_i = row-Mask_radius;
	int col_i = col-Mask_radius;
	/*
	__shared__ float sImageC0[O_TILE_WIDTH][O_TILE_WIDTH];
	__shared__ float sImageC1[O_TILE_WIDTH][O_TILE_WIDTH];
	__shared__ float sImageC2[O_TILE_WIDTH][O_TILE_WIDTH];
	*/
	__shared__ float sImageC0[BLOCK_WIDTH][BLOCK_WIDTH];
	//__shared__ float sImageC1[BLOCK_WIDTH][BLOCK_WIDTH];
	//__shared__ float sImageC2[BLOCK_WIDTH][BLOCK_WIDTH];
			
	for(int k=0; k < 3;k++)
	{
		
	
			if(row_i>=0 && row_i < height && col_i>=0 && col_i < width)
			{
				sImageC0[ty][tx] = in[(row_i*width+col_i)*3+k];
				//sImageC1[ty][tx] = in[(row_i*width+col_i)*3 + 1];
				//sImageC2[ty][tx] = in[(row_i*width+col_i)*3 + 2];
			}
			else 
			{
				sImageC0[ty][tx] = 0.0f;
				//sImageC1[ty][tx] = 0.0f;
				//sImageC2[ty][tx] = 0.0f;
			}
			
			__syncthreads();
			float sum0 = 0.0f;
			//float sum1 = 0.0f;
			//float sum2 = 0.0f;
			if(tx < O_TILE_WIDTH && ty < O_TILE_WIDTH)
			{
				//if(ty == 15)
				//{
			//		printf("calculating for image index(%d,%d)\n",row,col);
			//	}
				
				for(int y =0; y < Mask_width; y++)
				{
					for(int x = 0; x<Mask_width;x++)
					{
						sum0 += mask[y*Mask_width+x]*sImageC0[y+ty][x+tx];
						//sum1 += mask[y*Mask_width+x]*sImageC1[y+ty][x+tx];
						//sum2 += mask[y*Mask_width+x]*sImageC2[y+ty][x+tx];
					}
				}
			}
			__syncthreads();
			if(row < height && col < width)
			{
				out[(row*width+col)*3+k] = sum0;
				//out[(row*width+col)*3+1] = sum1;
				//out[(row*width+col)*3+2] = sum2;
			}
		__syncthreads();
	}
}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;
	//return 0;
    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");

	wbLog(TRACE,"width: ",imageWidth," height: ",imageHeight, " channels: ",imageChannels );
	wbLog(TRACE, "maskrows: ",maskRows," maskcolumns: ", maskColumns);
    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
	dim3 dimGrid(((imageWidth-1)/O_TILE_WIDTH)+1, ((imageHeight-1)/O_TILE_WIDTH)-1, 1);
	dim3 dimBlock(BLOCK_WIDTH,BLOCK_WIDTH,1);
	convolution_2D_kernel<<<dimGrid,dimBlock>>>(deviceInputImageData,
												  deviceOutputImageData,
												  deviceMaskData,
												  imageWidth,
												  imageHeight,
												  imageChannels);
    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
