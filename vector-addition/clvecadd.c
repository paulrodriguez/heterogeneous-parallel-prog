#include <wb.h> //@@ wb include opencl.h for you

//@@ OpenCL Kernel
const char * vaddsrc = 
	"__kernel void vadd(__global float* in1, __global float* in2, __global float* out, int len) {\n"\
	"int id = get_global_id(0);\n"\
	"if (id < len) {out[id] = in1[id] + in2[id];}}\n";

int main(int argc, char **argv) {
  	wbArg_t args;
  	int inputLength;
  	float *hostInput1;
	float *hostInput2;
  	float *hostOutput;
  	float *deviceInput1;
  	float *deviceInput2;
  	float *deviceOutput;
	
	cl_context clctx;
	cl_context_properties properties[3];
	cl_program program;
	
	cl_device_id device_id;
	cl_uint num_of_platforms;
	cl_uint num_of_devices;
	cl_command_queue command_queue;
	cl_kernel kernel;
	cl_int clerr = CL_SUCCESS;
	
 	args = wbArg_read(argc, argv);

  	wbTime_start(Generic, "Importing data and creating memory on host");
  	hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  	hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  	hostOutput = (float *)malloc(inputLength * sizeof(float));
  	wbTime_stop(Generic, "Importing data and creating memory on host");

  	wbLog(TRACE, "The input length is ", inputLength);
	for (int i = 0; i < inputLength; i++)
	{
		printf("position:%d [%f, %f]\n", i, hostInput1[i], hostInput2[i]);
	}
	
	/*printf("CL_INVALID_PROGRAM: %d\n", CL_INVALID_PROGRAM);
	printf("CL_INVALID_VALUE: %d\n", CL_INVALID_VALUE);
	printf("CL_INVALID_CONTEXT: %d \n", CL_INVALID_CONTEXT);
	*/
  	wbTime_start(GPU, "Allocating GPU memory.");
  	//@@ Allocate GPU memory here
	//get number of platforms
	if(clGetPlatformIDs(0, NULL, &num_of_platforms) != CL_SUCCESS)
	{
		printf("unable to get number of platforms\n");
		return 1;
	}
	
	//now get all the platforms
	cl_platform_id platform[num_of_platforms];
  	if (clGetPlatformIDs(num_of_platforms, platform, NULL)!= CL_SUCCESS)
	{
		printf("Unable to get platform_id\n");
		return 1;
	}
 
	// try to get a supported GPU device
	/*if (clGetDeviceIDs(&platform, CL_DEVICE_TYPE_GPU, 1, &device_id, &num_of_devices) != CL_SUCCESS)
	{
		printf("Unable to get device_id\n");
		return 1;
	}*/
	
	properties[0]= (cl_context_properties) CL_CONTEXT_PLATFORM;
	properties[1]= (cl_context_properties) platform[0];
	properties[2]= (cl_context_properties) 0;
	
	clctx = clCreateContextFromType(properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &clerr);
	if(clerr != CL_SUCCESS)
	{
		printf("error creating context.\n");
		return 1;
	}
	
	size_t stuff;
	clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, 0, NULL, &stuff);
	if(clerr != CL_SUCCESS)
	{
		printf("error getting context info.\n");
		return 1;
	}
	
	cl_device_id* cldevs = (cl_device_id *) malloc(stuff);
	clerr = clGetContextInfo(clctx, CL_CONTEXT_DEVICES, stuff, cldevs, NULL);
	if(clerr != CL_SUCCESS)
	{
		printf("could not get context infor.\n");
		return 1;
	}
	
	command_queue = clCreateCommandQueue(clctx, cldevs[0], 0, &clerr);
	if (clerr != CL_SUCCESS)
	{
		printf("could not create command queue\n");
		return 1;
	}
	
	program = clCreateProgramWithSource(clctx, 1, &vaddsrc, NULL, &clerr);
	printf("create program function: %d\n", clerr);
	//char clcompileflags[4096];
	//sprintf(clcompileflags, "-cl-mad-enable");
	
	clerr = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if( clerr != CL_SUCCESS)
	{
		
		printf("unable to build program.%d \n", clerr);
		return 1;
	}
	
	kernel = clCreateKernel(program, "vadd", &clerr);
	
	
 	cl_mem d_A, d_B, d_C;
	int mem_size = inputLength*sizeof(float);
	d_A = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, hostInput1, &clerr);
	d_B = clCreateBuffer(clctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mem_size, hostInput2, &clerr);
	d_C = clCreateBuffer(clctx, CL_MEM_WRITE_ONLY, mem_size, NULL, &clerr);
	//printf("%f", d_A[0]);
	
  	wbTime_stop(GPU, "Allocating GPU memory.");
	
	wbTime_start(GPU, "Copying input memory to the GPU.");
	
	//@@ Copy memory to the GPU here

	//pritnf("%f", &d_A[0]);
	wbTime_stop(GPU, "Copying input memory to the GPU.");

	//@@ Initialize the grid and block dimensions here
	//size_t size = inputLength;
	const size_t grid = (inputLength - 1)/512+1;
	const size_t block = 512;
	
	
	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Launch the GPU Kernel here
	
	kernel = clCreateKernel(program, "vadd", NULL);
	clerr = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
	clerr = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
	clerr = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
	clerr = clSetKernelArg(kernel, 3, sizeof(size_t), &inputLength);
	
	cl_event event = NULL;
	clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &grid, &block, 0, NULL, NULL);
	//clFinish(command_queue);
	clerr = clWaitForEvents(1, &event);
	clerr = clReleaseEvent(event);
	cudaThreadSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

  	wbTime_start(Copy, "Copying output memory to the CPU");
  	//@@ Copy the GPU memory back to the CPU here
	clEnqueueReadBuffer(command_queue, d_C, CL_TRUE, 0, mem_size, hostOutput, 0, NULL, NULL);
	
  	for(int i = 0; i < inputLength; i++)
	{
		//hostOutput[i] = hostInput2[i] + hostInput1[i];
		printf("at pos:%d, [%f]\n", i, hostOutput[i]);
	}

	
	wbSolution(args, hostOutput, inputLength);
  	wbTime_start(GPU, "Freeing GPU Memory");
  	//@@ Free the GPU memory here
	clReleaseMemObject(d_A);
	clReleaseMemObject(d_B);
	clReleaseMemObject(d_C);
  	
	wbTime_stop(GPU, "Freeing CPU Memory");

  	

  	free(hostInput1);
  	free(hostInput2);
  	free(hostOutput);

 	return 0;
}
