#ifndef RENDER_H
#define RENDER_H

#include <config.h>

struct Circle
{
    int x;
    int y;
    int radius;
    int radius_square;
    Uint32 pixel;
};

struct Ray
{
    int x;
    int y;
    double angle;
    int length;
    Uint32 pixel;
};

__global__ void clearScreen(Uint32* d_pixels, Uint32 pixel);
__global__ void drawCircle(Uint32* d_pixels,Circle sourcCircle, Circle *circlesObject);
__global__ void drawRays(Uint32* d_pixels, Ray *rays, Circle source);

#endif