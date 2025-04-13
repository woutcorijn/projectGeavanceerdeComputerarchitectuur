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
    int x[8];
    int y[8];
    double angle[8];
    int length[8];
    int reflections;
    Uint32 pixel[8];
};

__global__ void clearScreen(Uint32* d_pixels, Uint32 pixel);
__global__ void drawCircle(Uint32* d_pixels,Circle sourceCircle, Circle *circlesObject);
__global__ void drawRays(Uint32* d_pixels, Ray *rays, Circle source);
__global__ void calculateLengthRays(Ray *rays, Circle *circlesObject, Circle source);

void clearScreenCpu(Uint32* d_pixels, Uint32 pixel);
void drawCircleCpu(Uint32* d_pixels,Circle circle);
void drawRaysCpu(Uint32* d_pixels, Ray *rays, Circle source);
void calculateLengthRaysCpu(Ray *rays, Circle *circlesObject, Circle source, int rayIndex);
void calculateReflectionCpu(Ray *rays, Circle *circlesObject, Circle source, int rayIndex);

#endif