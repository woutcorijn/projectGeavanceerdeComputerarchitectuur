#include <render.h>
#include <config.h>

int BLOCKS;
int BLOCKS_RAYS;
int THREADSPERBLOCK;
bool gpu = true;
double meanFps = 0;
int fpsCounter = 0;

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
        meanFps += fps;
        fpsCounter++;
        std::cout << "FPS: " << fps << std::endl;
        *totalLoops = 0;
        *totalTime = std::chrono::nanoseconds::zero();
    }
}

int main() {
    //assert(NUM_RAYS < 1024 && "Rays worden nog berekend met threads.id ");

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

    Uint32 blackPixel = SDL_MapRGBA(formatDetail,NULL, 0, 0, 0, 0);
    Uint32 greenPixel = SDL_MapRGBA(formatDetail,NULL, 0, 255, 0, 255);
    Uint32 whitePixel = SDL_MapRGBA(formatDetail,NULL, 255, 255, 255, 255);
    Uint32 yellowPixel = SDL_MapRGBA(formatDetail,NULL, 255, 222, 0, 255);
    Uint32 grayPixel = SDL_MapRGBA(formatDetail,NULL, 209, 209, 209, 255);
    Uint32 orangePixel = SDL_MapRGBA(formatDetail,NULL, 200, 70, 0, 255);

    struct Circle sourceCircle ={200,200,80,80*80, yellowPixel};
    struct Circle object1 ={900,600,40,40*40, grayPixel};
    struct Circle object2 ={800,400,70,70*70, orangePixel};
    Circle *circles = (Circle*)malloc(NUM_CIRCLE_OBJECTS*sizeof(Circle));
    Ray *rays = (Ray*)malloc(NUM_RAYS*sizeof(Ray));

    circles[0] = object1;
    circles[1] = object2;

    for(int i = 0; i < NUM_RAYS; i++){
        double angle = (double) i * 2* M_PI / NUM_RAYS;
        struct Ray ray = {{0}, {0}, {angle}, {INT32_MAX}, 0, {yellowPixel}};
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
        BLOCKS_RAYS = (NUM_RAYS + THREADSPERBLOCK - 1) / THREADSPERBLOCK;

        printf("Blocksize: %d\n", THREADSPERBLOCK);
        printf("Blocks-pixels: %d\n", BLOCKS);
        printf("Blocks-rays: %d\n", BLOCKS_RAYS);

        SDL_Event event;
        bool running{true};

        std::chrono::nanoseconds totalTime = std::chrono::nanoseconds::zero();
        int totalLoops = 0;

        while(running){
            auto start = std::chrono::high_resolution_clock::now();
            
            while(SDL_PollEvent(&event)){
                if(event.key.key == 120 || event.key.key == 27 || event.type == SDL_EVENT_QUIT){
                    // 'x', 'esc'
                    running = false;
                }
                if(event.type == SDL_EVENT_MOUSE_MOTION && 
                    event.motion.state != 0){
                    sourceCircle.x = event.motion.x;
                    sourceCircle.y = event.motion.y;
                }
            }
            clearScreen<<<BLOCKS, THREADSPERBLOCK>>>(d_pixels, blackPixel);

            for(int i = 0; i < NUM_REFLECTIONS - 1; i++){
                calculateLengthRays<<<BLOCKS_RAYS, THREADSPERBLOCK>>>(d_rays, d_circleObjects, sourceCircle, i);
                calculateReflection<<<BLOCKS_RAYS, THREADSPERBLOCK>>>(d_rays, d_circleObjects, sourceCircle, i);
            }

            drawRays<<<BLOCKS_RAYS, THREADSPERBLOCK>>>(d_pixels, d_rays, sourceCircle);

            drawCircle<<<BLOCKS, THREADSPERBLOCK>>>(d_pixels,sourceCircle, d_circleObjects);

            cudaMemcpy(pixels, d_pixels, WIDTH * HEIGHT * sizeof(Uint32), cudaMemcpyDeviceToHost);

            SDL_RenderClear(renderer);
            RenderSurface(renderer, surface);
            //SDL_Delay(1);

                auto stop = std::chrono::high_resolution_clock::now();
                updateFPS(&totalTime, &totalLoops, start, stop);
        }

        printf("Free cuda resources\n");
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

            for(int i = 0; i < NUM_REFLECTIONS-1; i++){
                calculateLengthRaysCpu(rays, circles, sourceCircle, i);
                calculateReflectionCpu(rays, circles, sourceCircle, i);
            }

            drawRaysCpu(pixels, rays, sourceCircle);

            drawCircleCpu(pixels, sourceCircle);
            for (int i = 0; i < NUM_CIRCLE_OBJECTS; i++) {
                drawCircleCpu(pixels, circles[i]);
            }

            SDL_RenderClear(renderer);
            RenderSurface(renderer, surface);
            // max 200 fps
            SDL_Delay(5);

            auto stop = std::chrono::high_resolution_clock::now();
            updateFPS(&totalTime, &totalLoops, start, stop);
        }
    }

    printf("mean fps: %f\n", meanFps/fpsCounter);

    printf("Free host resources\n");
    free(circles);
    free(rays);
    printf("Free SDL resources\n");
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    printf("CLOSE PROGRAM\n");
    return 0;
}