#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define MASK_WIDTH 5
#define MASK_RADIUS MASK_WIDTH/2

#define TILE_WIDTH 16
#define O_TILE_WIDTH (TILE_WIDTH - MASK_WIDTH + 1)

#define CHANNELS 3		// RGB 3 channels.

//@@ INSERT CODE HERE

// a helper function called from device
__device__ float clamp(float x, float start, float end){
	return min(max(x, start), end);
}

// tiled 2D convolution kernel with adjustments for channels
__global__ void convolution_2D(float * inputImageData, float * outputImageData, const float * __restrict__ MASK,
			             int imageWidth, int imageHeight) {
	
	// use shared memory to reduce the number of global accesses
	__shared__ float ds_input[TILE_WIDTH][TILE_WIDTH][CHANNELS];
	
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	int rowIdx_o = by * O_TILE_WIDTH + ty;		// blockDim.y = TILE_WIDTH, is not equal to O_TILE_WIDTH
	int colIdx_o = bx * O_TILE_WIDTH + tx;		// blockDim.y = TILE_WIDTH, is not equal to O_TILE_WIDTH
	int rowIdx_i = rowIdx_o - MASK_RADIUS;
	int colIdx_i = colIdx_o - MASK_RADIUS;
	float output[CHANNELS] = {0.0f, 0.0f, 0.0f};
	
	// use all threads in the thread block to load input list elements into the shared memory
	if((rowIdx_i >= 0) && (rowIdx_i < imageHeight) && (colIdx_i >= 0) && (colIdx_i < imageWidth) ) {
		for (int k = 0; k < CHANNELS; k++)
			ds_input[ty][tx][k] = inputImageData[(rowIdx_i * imageWidth + colIdx_i) * CHANNELS + k];
	} else{
		for (int k = 0; k < CHANNELS; k++)
			ds_input[ty][tx][k] = 0.0f;
	}
	
	// sync before calculation
	__syncthreads();
	
	// calculate convoluted image pixels, only for the threads within the output tile 
	if((ty < O_TILE_WIDTH) && (tx < O_TILE_WIDTH)){
		for(int i = 0; i < MASK_WIDTH; i++) {
			for(int j = 0; j < MASK_WIDTH; j++) {
				for (int k = 0; k < CHANNELS; k++)
					output[k] += MASK[i * MASK_WIDTH + j] * ds_input[i+ty][j+tx][k];
			}
		}
		
		// sync before copying the result
		__syncthreads();
		
		// write the result back to the global memory 
		if(rowIdx_o < imageHeight && colIdx_o < imageWidth){
			for (int k = 0; k < CHANNELS; k++)
				outputImageData[(rowIdx_o * imageWidth + colIdx_o) * CHANNELS + k] = clamp(output[k], 0, 1);
		}
		
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
	
	wbLog(TRACE, "The image width is ", imageWidth, "; height is ", imageHeight,
		  "; number of channels is ", imageChannels, "; pitch is ", wbImage_getPitch(inputImage));
	wbLog(TRACE, "mask radius is ", MASK_RADIUS);
	wbLog(TRACE, "out put tile width is ", O_TILE_WIDTH);
	
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
	
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
	
    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");
	
	// allocate device memory
    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");
	
	// copy host memory to device
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
	// initialize thread block and kernel grid dimensions
	dim3 DimGrid((imageWidth-1)/O_TILE_WIDTH + 1, (imageHeight-1)/O_TILE_WIDTH + 1, 1);
	dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	// invoke CUDA kernel
	convolution_2D<<<DimGrid, DimBlock>>>(deviceInputImageData, deviceOutputImageData, deviceMaskData,
			             imageWidth, imageHeight);
	
    wbTime_stop(Compute, "Doing the computation on the GPU");

	// copy results from device to host
    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");
	
    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");
	
    wbSolution(args, outputImage);
	
	// deallocate device memory
    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);
	
    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
