#include <render.h>
#include <config.h>

int BLOCKS;
int BLOCKS_RAYS;
int THREADSPERBLOCK;
bool gpu = true;

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

void updateFPS(std::chrono::nanoseconds *totalTime, int *totalLoops, std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point stop){
    (*totalTime) += stop-start;
    (*totalLoops) ++;
    if(*totalLoops == 500){
        double totalSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(*totalTime).count();
        double fps = *totalLoops / totalSeconds;
        std::cout << "FPS: " << fps << std::endl;
        *totalLoops = 0;
        *totalTime = std::chrono::nanoseconds::zero();
    }
}

int main() {
    assert(NUM_RAYS < 1024 && "Rays worden nog berekend met threads.id ");

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL: %s", SDL_GetError());
        return 3;
    }
    
    SDL_Window *window = SDL_CreateWindow("Raytracing", WIDTH, HEIGHT, 0);

    SDL_Renderer *renderer = SDL_CreateRenderer(window, NULL);
    if (!renderer) {
        printf("Failed to create renderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }

    SDL_Surface *surface = SDL_CreateSurface(WIDTH, HEIGHT, SDL_PIXELFORMAT_RGBA32);
    if (!surface) {
        printf("Failed to create surface: %s\n", SDL_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return -1;
    }
    Uint32 *pixels = (Uint32*)surface->pixels;
    const SDL_PixelFormat format = surface->format;
    const SDL_PixelFormatDetails *formatDetail = SDL_GetPixelFormatDetails(format);

    Uint32 blackPixel = SDL_MapRGBA(formatDetail,NULL, 0, 0, 0, 255);
    
    Uint32 greenPixel = SDL_MapRGBA(formatDetail,NULL, 0, 255, 0, 255);

    printf("Byte 0: %u\n", (greenPixel & 0xFF000000) >> 24);
    printf("Byte 1: %u\n", (greenPixel & 0x00FF0000) >> 16);
    printf("Byte 2: %u\n", (greenPixel & 0x0000FF00) >> 8);
    printf("Byte 3: %u\n", (greenPixel & 0x000000FF));

    Uint32 whitePixel = SDL_MapRGBA(formatDetail,NULL, 255, 255, 255, 255);

    struct Circle sourceCircle ={200,200,80,80*80, greenPixel};
    struct Circle object1 ={900,300,50,50*50, whitePixel};
    struct Circle object2 ={800,100,40,40*40, whitePixel};
    Circle *circles = (Circle*)malloc(NUM_CIRCLE_OBJECTS*sizeof(Circle));
    Ray *rays = (Ray*)malloc(NUM_RAYS*sizeof(Ray));

    circles[0] = object1;
    circles[1] = object2;

    for(int i = 0; i < NUM_RAYS; i++){
        double angle = (double) i * 2* M_PI / NUM_RAYS;
        struct Ray ray = {0, 0, angle, 300, whitePixel};
        rays[i] = ray;
    }

    if(gpu) {
        // Cuda resource allocation
        Uint32 *d_pixels;
        Circle *d_circleObjects;
        Ray *d_rays;
        cudaMalloc((void**)&d_pixels, WIDTH * HEIGHT * sizeof(Uint32));
        cudaMalloc((void**)&d_circleObjects, NUM_CIRCLE_OBJECTS*sizeof(Circle));
        cudaMalloc((void**)&d_rays, NUM_RAYS*sizeof(Ray));
        cudaMemcpy(d_circleObjects, circles, NUM_CIRCLE_OBJECTS*sizeof(Circle), cudaMemcpyHostToDevice);
        cudaMemcpy(d_rays, rays, NUM_RAYS*sizeof(Ray), cudaMemcpyHostToDevice);

        THREADSPERBLOCK = 1024;
        BLOCKS = (WIDTH*HEIGHT + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
        BLOCKS_RAYS = (WIDTH*HEIGHT + THREADSPERBLOCK - 1) / THREADSPERBLOCK;

        SDL_Event event;
        bool running{true};

        std::chrono::nanoseconds totalTime = std::chrono::nanoseconds::zero();
        int totalLoops = 0;

        while(running) {
            auto start = std::chrono::high_resolution_clock::now();
            
            while(SDL_PollEvent(&event)){
                if(event.key.key == 120){
                    // 'x'
                    running = false;
                }
                if(event.type == SDL_EVENT_MOUSE_MOTION && 
                    event.motion.state != 0){
                    sourceCircle.x = event.motion.x;
                    sourceCircle.y = event.motion.y;
                }
            }
            clearScreen<<<BLOCKS, THREADSPERBLOCK>>>(d_pixels, blackPixel);

            drawCircle<<<BLOCKS, THREADSPERBLOCK>>>(d_pixels,sourceCircle, d_circleObjects);

            calculateLengthRays<<<1, NUM_RAYS>>>(d_rays, d_circleObjects, sourceCircle);

            drawRays<<<1, NUM_RAYS>>>(d_pixels, d_rays, sourceCircle);

            cudaMemcpy(pixels, d_pixels, WIDTH * HEIGHT * sizeof(Uint32), cudaMemcpyDeviceToHost);

            SDL_RenderClear(renderer);
            RenderSurface(renderer, surface);
            // max 200 fps
            SDL_Delay(5);

            auto stop = std::chrono::high_resolution_clock::now();
            updateFPS(&totalTime, &totalLoops, start, stop);

        }

        cudaFree(d_pixels);
        cudaFree(d_circleObjects);
        cudaFree(d_rays);

    } else {
        //CPU
        SDL_Event event;
        bool running{true};

        std::chrono::nanoseconds totalTime = std::chrono::nanoseconds::zero();
        int totalLoops = 0;

        while(running){
            auto start = std::chrono::high_resolution_clock::now();
            
            while(SDL_PollEvent(&event)){
                if(event.key.key == 120){
                    // 'x'
                    running = false;
                }
                if(event.type == SDL_EVENT_MOUSE_MOTION && 
                    event.motion.state != 0){
                    sourceCircle.x = event.motion.x;
                    sourceCircle.y = event.motion.y;
                }
            }
            clearScreenCpu(pixels, blackPixel);

            drawCircleCpu(pixels, sourceCircle);
            for (int i = 0; i < NUM_CIRCLE_OBJECTS; i++) {
                drawCircleCpu(pixels, circles[i]);
            }

            calculateLengthRaysCpu(rays, circles, sourceCircle);

            drawRaysCpu(pixels, rays, sourceCircle);

            SDL_RenderClear(renderer);
            RenderSurface(renderer, surface);
            // max 200 fps
            SDL_Delay(5);

            auto stop = std::chrono::high_resolution_clock::now();
            updateFPS(&totalTime, &totalLoops, start, stop);
        }
    }

    free(circles);
    free(pixels);
    free(rays);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}