#include	<wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
    //@@ Insert code to implement vector addition here
	int index = threadIdx.x+blockIdx.x*blockDim.x;
	if(index < len)
	{
		out[index] = in1[index]+in2[index];
	}
}

int main(int argc, char ** argv) {
	
	cudaStream_t stream1, stream2, stream3, stream4;
	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);
	cudaStreamCreate(&stream3);
	cudaStreamCreate(&stream4);
	
    wbArg_t args;
    int inputLength;
    float * hostInput1;
    float * hostInput2;
    float * hostOutput;
    
	float * dA1, * dB1, * dC1;
	float * dA2, * dB2, * dC2;
	float * dA3, * dB3, * dC3;
	float * dA4, * dB4, * dC4;
	


	
    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput1 = (float *) wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *) wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *) malloc(inputLength * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

	
	
	wbCheck(cudaMalloc((void **) &dA1, inputLength*sizeof(float)));
	wbCheck(cudaMalloc((void **) &dB1, inputLength*sizeof(float)));
	wbCheck(cudaMalloc((void **) &dC1, inputLength*sizeof(float)));
	
	wbCheck(cudaMalloc((void **) &dA2, inputLength*sizeof(float)));
	wbCheck(cudaMalloc((void **) &dB2, inputLength*sizeof(float)));
	wbCheck(cudaMalloc((void **) &dC2, inputLength*sizeof(float)));
	
	wbCheck(cudaMalloc((void **) &dA3, inputLength*sizeof(float)));
	wbCheck(cudaMalloc((void **) &dB3, inputLength*sizeof(float)));
	wbCheck(cudaMalloc((void **) &dC3, inputLength*sizeof(float)));
	
	wbCheck(cudaMalloc((void **) &dA4, inputLength*sizeof(float)));
	wbCheck(cudaMalloc((void **) &dB4, inputLength*sizeof(float)));
	wbCheck(cudaMalloc((void **) &dC4, inputLength*sizeof(float)));
	
	dim3 dimGrid((inputLength-1)/512 + 1, 1, 1);
	dim3 dimBlock(512, 1, 1);
	
	int SegmentSize = inputLength/4;
	
	//for improvements try using cudaDevceSynchronize();
	for(int i = 0; i < inputLength; i += SegmentSize*4)
	{
		
		//first copy input arrays to device
		//then call kernel function, passing device input and output arrays
		//copy back to host output array
		
		cudaMemcpyAsync(dA1, hostInput1+i, SegmentSize*sizeof(float),cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(dB1, hostInput2+i, SegmentSize*sizeof(float),cudaMemcpyHostToDevice, stream1);
		vecAdd<<<dimGrid, dimBlock, 0, stream1>>>(dA1, dB1, dC1, inputLength);
		cudaMemcpyAsync(hostOutput+i, dC1, SegmentSize*sizeof(float), cudaMemcpyDeviceToHost, stream1);
		
		
		cudaMemcpyAsync(dA2, hostInput1+i+SegmentSize, SegmentSize*sizeof(float),cudaMemcpyHostToDevice, stream2);
		cudaMemcpyAsync(dB2, hostInput2+i+SegmentSize, SegmentSize*sizeof(float),cudaMemcpyHostToDevice, stream2);
		vecAdd<<<dimGrid, dimBlock, 0, stream2>>>(dA2, dB2, dC2, inputLength);
		cudaMemcpyAsync(hostOutput+i+SegmentSize, dC2, SegmentSize*sizeof(float), cudaMemcpyDeviceToHost, stream2);
		
		
		cudaMemcpyAsync(dA3, hostInput1+i+SegmentSize*2, SegmentSize*sizeof(float),cudaMemcpyHostToDevice, stream3);
		cudaMemcpyAsync(dB3, hostInput2+i+SegmentSize*2, SegmentSize*sizeof(float),cudaMemcpyHostToDevice, stream3);
		vecAdd<<<dimGrid, dimBlock, 0, stream3>>>(dA3, dB3, dC3, inputLength);
		cudaMemcpyAsync(hostOutput+i+SegmentSize*2, dC3, SegmentSize*sizeof(float), cudaMemcpyDeviceToHost, stream3);
		
		
		cudaMemcpyAsync(dA4, hostInput1+i+SegmentSize*3, SegmentSize*sizeof(float),cudaMemcpyHostToDevice, stream4);
		cudaMemcpyAsync(dB4, hostInput2+i+SegmentSize*3, SegmentSize*sizeof(float),cudaMemcpyHostToDevice, stream4);
		vecAdd<<<dimGrid, dimBlock, 0, stream4>>>(dA4, dB4, dC4, inputLength);		
		cudaMemcpyAsync(hostOutput+i+SegmentSize*3, dC4, SegmentSize*sizeof(float), cudaMemcpyDeviceToHost, stream4);
	}
	
    wbSolution(args, hostOutput, inputLength);

	cudaFree(dA1);
	cudaFree(dB1);
	cudaFree(dC1);
	
	cudaFree(dA2);
	cudaFree(dB2);
	cudaFree(dC2);
	
	cudaFree(dA3);
	cudaFree(dB3);
	cudaFree(dC3);
	
	cudaFree(dA4);
	cudaFree(dB4);
	cudaFree(dC4);
	
    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}

