#ifndef CONFIG_H
#define CONFIG_H

#include <SDL3/SDL.h>
#include <stdio.h>
#include <cstdint>
#include <chrono>
#include <iostream>
#include <cmath>
#include <cassert>


#define WIDTH 1300
#define HEIGHT 1000
#define M_PI 3.14159265358979323846
#define COLOR_WHITE 0xffffffff
#define COLOR_BLACK 0x00000000

constexpr int NUM_CIRCLE_OBJECTS = 2;
constexpr int NUM_REFLECTIONS = 5;
constexpr int NUM_RAYS = 1021;

#endif