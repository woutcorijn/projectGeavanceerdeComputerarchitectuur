#include <stdio.h>
#include <cuda_runtime.h>
#include <SDL3/SDL.h>
#include <atomic>
#include <thread>

#define WIDTH 1100
#define HEIGHT 800
#define COLOR_WHITE 0xffffffff
#define COLOR_BLACK 0x00000000
#define COLOR_GRAY 0x808080
#define RAYS_NUMBER 100

# define M_PI 3.14159265358979323846

struct Circle
{
    int x;
    int y;
    int radius;
};

struct Ray
{
    int x;
    int y;
    double angle;
    int length;
};

void fillCircle(SDL_Surface *surface, struct Circle *circle, Uint32 color){
    int circleR = circle->radius;
    int circleX = circle->x;
    int circleY = circle->y;

    int radius_squared = circleR*circleR;

    for (int x=circleX-circleR; x <= circleX+circleR; x++) {
        for(int y=circleY-circleR; y <= circleY+circleR; y++) {
            int distance_squared = (x-circleX)*(x-circleX) + (y-circleY)*(y-circleY);
            
            if(distance_squared < radius_squared){
                SDL_Rect pixel = {x,y,1,1};
                SDL_FillSurfaceRect(surface, &pixel, color);
            }
        }
    }
}

void fillRays(SDL_Surface *surface, struct Ray rays[RAYS_NUMBER], Uint32 color){
    for(int i = 0; i < RAYS_NUMBER; i++){
        for(int j = 0; j < rays[i].length; j++){
            SDL_Rect pixel = {rays[i].x, rays[i].y, 1, 1};
            SDL_FillSurfaceRect(surface, &pixel, color);
        }
    }
}

/*void rays(int x_start, int y_start, int angle, int length){

}*/
void gererate_rays(struct Circle *circle, struct Ray rays[RAYS_NUMBER])
{
    for(int i = 0; i < RAYS_NUMBER; i++)
    {
        double angle = (double) i * 2* M_PI / RAYS_NUMBER;
        struct Ray ray = {circle->x, circle->y, angle, 300};
        rays[i] = ray;
    }
}

int main() {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL: %s", SDL_GetError());
        return 3;
    }
    
    SDL_Window *window = SDL_CreateWindow("Raytracing", WIDTH, HEIGHT, 0);
    SDL_Surface *surface = SDL_GetWindowSurface(window);

    struct Circle circle ={200,200,80};
    struct Circle objectCircle = {800, 400, 140};
    SDL_Rect erase_rect = {0,0,WIDTH,HEIGHT};

    struct Ray rays[RAYS_NUMBER];
    gererate_rays(&circle, rays);

    bool running{true};
    SDL_Event event;
    
    while(running){
        while(SDL_PollEvent(&event)){
            if(event.key.key == 120){
                // 'x'
                running = false;
            }
            if(event.type == SDL_EVENT_MOUSE_MOTION && 
                event.motion.state != 0){
                circle.x = event.motion.x;
                circle.y = event.motion.y;
            }
        }
        SDL_FillSurfaceRect(surface, &erase_rect, COLOR_BLACK);
        fillCircle(surface, &circle, COLOR_WHITE);

        fillCircle(surface, &objectCircle, COLOR_WHITE);

        fillRays(surface, rays, COLOR_WHITE);

        SDL_UpdateWindowSurface(window);

        // 100 fps
        SDL_Delay(10);
    }

    SDL_DestroyWindow(window);

    SDL_Quit();

    return 0;
}
