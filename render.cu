#include <render.h>
#include <config.h>

__global__ void clearScreen(Uint32* d_pixels, Uint32 pixel) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    d_pixels[index] = pixel;
}

__global__ void drawCircle(Uint32* d_pixels,Circle sourcCircle, Circle *circlesObject) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int indexX = index % WIDTH;
    int indexY = index / WIDTH;
    __shared__ Circle circles[NUM_CIRCLE_OBJECTS+1];

    int blockIndex = threadIdx.x;
    if(blockIndex < NUM_CIRCLE_OBJECTS){
        circles[blockIndex] = circlesObject[blockIndex];
    }
    else if(blockIndex == NUM_CIRCLE_OBJECTS){
        circles[blockIndex] = sourcCircle;
    }
    __syncthreads();

    for(Circle circle: circles){
        int distance_squared = (indexX-circle.x)*(indexX-circle.x) + (indexY-circle.y)*(indexY-circle.y);

        if(distance_squared <= circle.radius_square){
            d_pixels[index] = circle.pixel;
        }
    }
}

__global__ void drawRays(Uint32* d_pixels, Ray *rays, Circle source) {
    Ray ray = rays[threadIdx.x];
    double dx = cos(ray.angle);
    double dy = sin(ray.angle);
    

    int x = ray.x + source.x;
    int y = ray.y + source.y;

    for (int j = 0; j < ray.length; j++) {
        int px = x + (int)(j * dx);
        int py = y + (int)(j * dy);

        // Controleer of de pixel binnen de grenzen van het scherm valt
        if (j > source.radius && px >= 0 && px < WIDTH && py >= 0 && py < HEIGHT) {
            d_pixels[py * WIDTH + px] = ray.pixel;
        }
    }
}

