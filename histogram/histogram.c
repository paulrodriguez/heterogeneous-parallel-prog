// Histogram Equalization

#include    <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here
/**
*  each thread will compute a pixel for all three channels.
**/
__global__ void convertToChar(float * input, unsigned char * ucharInput, int width, int height)
{
	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	
	int row = by*HISTOGRAM_LENGTH+ty;
	int col = bx*HISTOGRAM_LENGTH+tx;
	int index = row*width + col;
	
	if(row < height && col < width)
	{
		ucharInput[index*3]   = (unsigned char) (255 * input[index*3]); //r
		ucharInput[index*3+1] = (unsigned char) (255 * input[index*3+1]); //g
		ucharInput[index*3+2] = (unsigned char) (255 * input[index*3+2]); //b
	}
	
	
}

__global__ void convertToGrayScale(unsigned char * ucharImg, unsigned char * grayImg, int width, int height)
{
	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	
	int row = by*blockDim.y+ty;
	int col = bx*blockDim.x+tx;
	int index = row*width + col;
	
	if(row < height && col < width)
	{
		grayImg[index] = (unsigned char) (0.21*ucharImg[index*3] + 0.71*ucharImg[index*3 + 1] + 0.07*ucharImg[index*3 + 2]);
	}
	
}

/*
__global__ void hist_eq(unsigned char * ucharImg, int width, int height)
{
	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;
	
	int row = by*blockDim.y+ty;
	int col = bx*blockDim.x+tx;
	int index = row*width + col;
	
	if(row < height && col < width)
	{
		grayImg[index] = (unsigned char) (0.21*ucharImg[index*3] + 0.71*ucharImg[index*3 + 1] + 0.07*ucharImg[index*3 + 2]);
	}
}
*/
/*
__global__ void histo_kernel(unsigned char * buffer, long size, unsigned int * histo)
{

}*/

float prob(int x, int width, int height)
{
	return 1.0*x/(width*height);
}

int main(int argc, char ** argv) {
    wbArg_t args;
    int imageWidth;
    int imageHeight;
    int imageChannels;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    const char * inputImageFile;

    //@@ Insert more code here
	//  device variables
	float * deviceInputImageData;
	float * deviceOutputImageData;
	unsigned char * ucharImage;
	unsigned char * deviceGrayImg;
	
	//  more host variables
	unsigned char * hostGrayImg;
	unsigned char * hostCharImg;
	
    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);

    wbTime_start(Generic, "Importing data and creating memory on host");
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
    wbTime_stop(Generic, "Importing data and creating memory on host");

    //@@ insert code here
	hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);
	
	hostGrayImg = (unsigned char *)malloc(imageWidth*imageHeight*sizeof(unsigned char));
	hostCharImg = (unsigned char *)malloc(imageWidth*imageHeight*imageChannels*sizeof(unsigned char));
	
	cudaMalloc((void **) &deviceInputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));
	cudaMalloc((void **) &deviceOutputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));
	cudaMalloc((void **) &ucharImage, imageWidth*imageHeight*imageChannels*sizeof(unsigned char));
	cudaMalloc((void **) &deviceGrayImg, imageWidth*imageHeight*sizeof(unsigned char));
	
	cudaMemcpy(deviceInputImageData, 
			   hostInputImageData, 
			   imageWidth*imageHeight*imageChannels*sizeof(float), 
			   cudaMemcpyHostToDevice);
	cudaMemcpy(deviceOutputImageData, 
		 	   hostOutputImageData, 
		   	   imageWidth*imageHeight*imageChannels*sizeof(float), 
		  	   cudaMemcpyHostToDevice);
	
    wbLog(TRACE, "image width: ",imageWidth,", image height: ",imageHeight);

    //@@ insert code here
	dim3 dimBlock(256,256,1);
	dim3 dimGrid((imageWidth - 1)/256 + 1, (imageHeight-1)/256 + 1, 1);
	
	convertToChar<<<dimGrid, dimBlock>>>(deviceInputImageData, ucharImage, imageWidth, imageHeight);
	//  copy char image to host
	cudaMemcpy(hostCharImg, ucharImage, imageWidth*imageHeight*imageChannels*sizeof(unsigned char), cudaMemcpyDeviceToHost);
		
	convertToGrayScale<<<dimGrid, dimBlock>>>(ucharImage, deviceGrayImg, imageWidth, imageHeight);
	
	cudaMemcpy(hostGrayImg, deviceGrayImg, imageWidth*imageHeight*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	//  compute the histogram
	int * histogram;
	histogram = (int *)malloc(HISTOGRAM_LENGTH*sizeof(int));	
	for(int i = 0; i < HISTOGRAM_LENGTH; i++)
	{
		histogram[i] = 0;
	}
	for(int i = 0; i < imageWidth*imageHeight; i++)
	{
		histogram[hostGrayImg[i]]++;
	}
	
	
	//  compute scan operation for histogram
	float * cdf;
	cdf = (float *)malloc(HISTOGRAM_LENGTH*sizeof(float));
	cdf[0] = prob(histogram[0], imageWidth, imageHeight);
	for(int i = 1; i < HISTOGRAM_LENGTH; i++)
	{
		cdf[i] = cdf[i-1]+prob(histogram[i],imageWidth,imageHeight);
	}
	
	float cdfmin = cdf[0];
	for(int i = 1; i < HISTOGRAM_LENGTH;i++)
	{
		cdfmin = min(cdfmin, cdf[i]);
	}
	
	//  histogram equalization function
	for(int i = 0; i < imageWidth*imageHeight*imageChannels; i++)
	{
		hostCharImg[i] = min(max(255*(cdf[hostCharImg[i]] - cdfmin)/(1 - cdfmin),0.0),255.0);
		hostOutputImageData[i] = (float)(hostCharImg[i]/255.0);
		printf("image  value: %f\n ",hostOutputImageData);
	}
	
	wbSolution(args, outputImage);
	
	cudaFree(ucharImage);
	cudaFree(deviceGrayImg);
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);
	
	free(hostInputImageData);
	free(hostOutputImageData);
	//free(inputImageFile);
	
	wbImage_delete(outputImage);
    wbImage_delete(inputImage);
    
	return 0;
}

