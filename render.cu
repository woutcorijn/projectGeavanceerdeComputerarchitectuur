#include <render.h>
#include <config.h>

__device__ uint8_t saturating_add(uint8_t a, uint8_t b) {
    uint16_t sum = (uint16_t)a + (uint16_t)b;
    return (sum > 255) ? 255 : (uint8_t)sum;
}

__global__ void clearScreen(Uint32* d_pixels, Uint32 pixel) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    d_pixels[index] = pixel;
}

__global__ void drawCircle(Uint32* d_pixels,Circle sourceCircle, Circle *circlesObject) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int indexX = index % WIDTH;
    int indexY = index / WIDTH;
    __shared__ Circle circles[NUM_CIRCLE_OBJECTS+1];

    int blockIndex = threadIdx.x;
    if(blockIndex < NUM_CIRCLE_OBJECTS){
        circles[blockIndex] = circlesObject[blockIndex];
    }
    else if(blockIndex == NUM_CIRCLE_OBJECTS){
        circles[blockIndex] = sourceCircle;
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

    double fadeLength = 16;
    double fadeFactor = 0.996;
    double fadeByte = fadeLength * (ray.pixel[0] >> 24);

    for (int rayIndex = 0; rayIndex < NUM_REFLECTIONS; rayIndex++) {

        if(ray.length[rayIndex] == 0){
            return;
        }

        double dx = cos(ray.angle[rayIndex]);
        double dy = sin(ray.angle[rayIndex]);
        

        int x = ray.x[rayIndex] + source.x;
        int y = ray.y[rayIndex] + source.y;

        for (int j = 0; j < ray.length[rayIndex]; j++) {

            int px = x + (int)(j * dx);
            int py = y + (int)(j * dy);

            // Controleer of de pixel binnen de grenzen van het scherm valt
            if (px >= 0 && px < WIDTH && py >= 0 && py < HEIGHT) {
                fadeByte = fadeByte*fadeFactor;
                uint32_t pixel = ray.pixel[rayIndex];
                pixel = (pixel & 0x00FFFFFF) | ((uint32_t)(fadeByte / fadeLength) << 24);
                
                uint8_t a = (pixel >> 0) & 0xFF;
                uint8_t r = (pixel >> 8) & 0xFF;
                uint8_t g = (pixel >> 16) & 0xFF;
                uint8_t b = (pixel >> 24) & 0xFF;

                uint32_t d_pixel = d_pixels[py * WIDTH + px];

                uint8_t a_old = (d_pixel >> 0) & 0xFF;
                uint8_t r_old = (d_pixel >> 8) & 0xFF;
                uint8_t g_old = (d_pixel >> 16) & 0xFF;
                uint8_t b_old = (d_pixel >> 24) & 0xFF;

                uint8_t a_new = saturating_add(a, a_old);
                uint8_t r_new = saturating_add(r, r_old);
                uint8_t g_new = saturating_add(g, g_old);
                uint8_t b_new = saturating_add(b, b_old);

                uint32_t new_pixel = (b_new << 24) | (g_new << 16) | (r_new << 8) | (a_new);
                atomicExch(&d_pixels[py * WIDTH + px], new_pixel);
            }
            else if(px < 0 || px > WIDTH || py < 0 || py > HEIGHT || fadeByte < 1){
                return;
            }
        }
    }
}

__global__ void calculateLengthRays(Ray *rays, Circle *circlesObject, Circle source, int rayIndex) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ Circle circles[NUM_CIRCLE_OBJECTS];

    int blockIndex = threadIdx.x;
    if(threadIdx.x < NUM_CIRCLE_OBJECTS){
        circles[blockIndex] = circlesObject[blockIndex];
    }
    __syncthreads();

    Ray &ray = rays[index];

        double rayDirX = cos(ray.angle[rayIndex]);
        double rayDirY = sin(ray.angle[rayIndex]);
        double originX = ray.x[rayIndex] + source.x;
        double originY = ray.y[rayIndex] + source.y;
    
        double minLength = WIDTH + HEIGHT; // very large number
    
        for (Circle circle : circles) {
            double dx = rayDirX;
            double dy = rayDirY;
    
            double cx = circle.x;
            double cy = circle.y;
            double r2 = circle.radius_square;
    
            // Compute quadratic coefficients
            double a = dx * dx + dy * dy;
            double b = 2.0 * (dx * (originX - cx) + dy * (originY - cy));
            double c = (originX - cx) * (originX - cx) + (originY - cy) * (originY - cy) - r2;
    
            double discriminant = b * b - 4 * a * c;
    
            if (discriminant >= 0.0) {
                double sqrtDisc = sqrt(discriminant);
                double t1 = (-b + sqrtDisc) / (2.0 * a);
                double t2 = (-b - sqrtDisc) / (2.0 * a);
    
                // Only consider points in front of ray origin (t > 0)
                if (t1 > 0) {
                    double length1 = t1 * sqrt(a);
                    if (length1 < minLength) minLength = length1;
                }
    
                if (t2 > 0) {
                    double length2 = t2 * sqrt(a);
                    if (length2 < minLength) minLength = length2;
                }
            }
        }
        ray.length[rayIndex] = (int)minLength;

}


__global__ void calculateReflection(Ray *rays, Circle *circlesObject, Circle source, int rayIndex) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    /*__shared__ Circle circles[NUM_CIRCLE_OBJECTS];

    int blockIndex = threadIdx.x;
    if(threadIdx.x < NUM_CIRCLE_OBJECTS){
        circles[blockIndex] = circlesObject[blockIndex];
    }
    __syncthreads();*/

    Ray &ray = rays[index];

    // Raakpunt berekenen
    ray.x[rayIndex + 1] = ray.x[rayIndex] + ray.length[rayIndex] * cos(ray.angle[rayIndex]);
    ray.y[rayIndex + 1] = ray.y[rayIndex] + ray.length[rayIndex] * sin(ray.angle[rayIndex]);

    float dx = cos(ray.angle[rayIndex]);
    float dy = sin(ray.angle[rayIndex]);

    //float x0 = ray.x[rayIndex];
    //float y0 = ray.y[rayIndex];
    float x0 = source.x + ray.x[rayIndex];
    float y0 = source.y + ray.y[rayIndex];

    Circle *closestCircle = nullptr;
    float closestT = INT_MAX;
    float hitX = 0, hitY = 0;

    // Zoek de dichtste cirkel
    for (int j = 0; j < NUM_CIRCLE_OBJECTS; j++) {
        Circle &circle = circlesObject[j];

        float cx = circle.x;
        float cy = circle.y;
        float r = circle.radius;

        // vector van de cirkel naar de ray
        float fx = x0 - cx;
        float fy = y0 - cy;

        // Discriminant van de kwadratische vergelijking
        float a = dx * dx + dy * dy;
        float b = 2 * (fx * dx + fy * dy);
        float c = fx * fx + fy * fy - r * r;
        float discriminant = b * b - 4 * a * c;

        if (discriminant >= 0) {
            // Compute both intersection points
            float sqrtD = sqrt(discriminant);
            float t1 = (-b - sqrtD) / (2 * a);
            float t2 = (-b + sqrtD) / (2 * a);

            // Choose the smallest positive intersection distance (if any)
            float t = (t1 > 0) ? t1 : ((t2 > 0) ? t2 : -1);
            if (t > 0 && t < closestT) {
                closestT = t;
                closestCircle = &circle;
                hitX = x0 + dx * t;
                hitY = y0 + dy * t;
            }
        }
    }

    // Reflectie
    if (closestCircle) {
        float nx = hitX - closestCircle->x;
        float ny = hitY - closestCircle->y;
        float nLen = sqrt(nx * nx + ny * ny);
        nx /= nLen;
        ny /= nLen;

        float dot = dx * nx + dy * ny;
        float rx = dx - 2 * dot * nx;
        float ry = dy - 2 * dot * ny;

        ray.angle[rayIndex + 1] = atan2(ry, rx);
        ray.length[rayIndex + 1] = 600;
        ray.pixel[rayIndex + 1] = ray.pixel[rayIndex];
    }

}