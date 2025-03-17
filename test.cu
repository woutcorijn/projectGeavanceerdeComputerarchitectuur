#include <SDL3/SDL.h>
#include <stdio.h>

__global__ void setToGreen(Uint32* d_pixels, int size, Uint32 pixel) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
        i < size; 
        i += blockDim.x * gridDim.x) 
    {
        d_pixels[i] = pixel;
    }
}


// Function to modify pixel data in the surface
void ModifySurfaceMatrix(SDL_Surface *surface) {
    int width = surface->w;
    int height = surface->h;
    const SDL_PixelFormat format = surface->format;
    const SDL_PixelFormatDetails *formatDetail = SDL_GetPixelFormatDetails(format);

    // Create a pointer to the pixel data
    Uint32 *pixels = (Uint32*)surface->pixels;
    Uint32 pixel = SDL_MapRGBA(formatDetail,NULL, 0, 255, 0, 255);
    Uint32 *d_pixels;

    cudaMalloc((void**)&d_pixels, width * height * sizeof(Uint32));
    //cudaMemcpy(d_pixels, pixels,  width * height * sizeof(Uint32), cudaMemcpyHostToDevice);

    // Modify pixel data (for example, change the color to green)

    int threadsPerBlock = 1024;
    int blocks = (width*height + threadsPerBlock - 1) / threadsPerBlock;

    setToGreen<<<blocks, threadsPerBlock>>>(d_pixels, width*height, pixel);
    //sync cuda
    cudaMemcpy(pixels, d_pixels, width * height * sizeof(Uint32), cudaMemcpyDeviceToHost);
    cudaFree(d_pixels);
}

// Function to render the modified surface to the screen
void RenderSurface(SDL_Renderer *renderer, SDL_Surface *surface) {
    // Create a texture from the surface
    SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
    if (!texture) {
        printf("Failed to create texture: %s\n", SDL_GetError());
        return;
    }

    // Render the texture to the screen (position at (0, 0))
    SDL_RenderTexture(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);

    // Clean up the texture
    SDL_DestroyTexture(texture);
}

int main() {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL: %s", SDL_GetError());
        return 3;
    }
    
    SDL_Window *window = SDL_CreateWindow("Raytracing", 800, 600, 0);

    SDL_Renderer *renderer = SDL_CreateRenderer(window, NULL);
    if (!renderer) {
        printf("Failed to create renderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // Create an initial surface (e.g., 100x100)
    SDL_Surface *surface = SDL_CreateSurface(100, 100, SDL_PIXELFORMAT_RGBA32);
    if (!surface) {
        printf("Failed to create surface: %s\n", SDL_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    // Modify the surface (set all pixels to green)
    ModifySurfaceMatrix(surface);

    // Render the modified surface
    SDL_RenderClear(renderer);
    RenderSurface(renderer, surface);

    // Wait for a while before quitting
    SDL_Delay(3000);  // Wait for 3 seconds

    // Clean up
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
