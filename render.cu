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
    int rayIndex = 0;

    Ray ray = rays[threadIdx.x];
    if(ray.length[rayIndex] == 0){
        return;
    }
    double dx = cos(ray.angle[rayIndex]);
    double dy = sin(ray.angle[rayIndex]);
    

    int x = ray.x[rayIndex] + source.x;
    int y = ray.y[rayIndex] + source.y;

    double fadeLength = 16;
    double fadeFactor = 0.997;
    double fadeByte = fadeLength * (ray.pixel[rayIndex] >> 24);

    for (int j = 0; j < ray.length[rayIndex]; j++) {
        

        int px = x + (int)(j * dx);
        int py = y + (int)(j * dy);

        // Controleer of de pixel binnen de grenzen van het scherm valt
        if (j > source.radius && px >= 0 && px < WIDTH && py >= 0 && py < HEIGHT) {
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

__global__ void calculateLengthRays(Ray *rays, Circle *circlesObject, Circle source) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ Circle circles[NUM_CIRCLE_OBJECTS];

    int blockIndex = threadIdx.x;
    if(threadIdx.x < NUM_CIRCLE_OBJECTS){
        circles[blockIndex] = circlesObject[blockIndex];
    }
    __syncthreads();

    Ray &ray = rays[index];

    int rayIndex = 0;

    // y = helling * x + a
    
    //printf("x: %d, y: %d", ray.x[0], ray.y[0]);

    // (x - c.x)^2 + (y - c.y)^2 = r^2 -> circle

    // (x - c.x)^2 + ((helling * x + a) - c.y)^2 = r^2 -> ray in circle vergelijking steken

    // (x - c.x)^2 + (helling * x + (a - c.y))^2 = r^2

    // (x^2 - 2 * c.x * x + c.x^2) + (helling^2 * x^2 + 2 * helling * x * (a - c.y) + (a - c.y)^2) - r^2 = 0

    // (1 + helling^2) *x^2 + (2 * helling * (a - c.y) - 2 *c.x) * x + c.x^2 + (a - c.y)^2 - r^2 = 0

    // A = 1 + helling^2
    // B = 2 * helling * (a - c.y) - 2 *c.x
    // C = c.x^2 + (a - c.y)^2 - r^2
    // A*x^2 + B*x + C = 0

    // Diskriminant = B^2 - 4*A*C
    // x1 = (-B + sqrt(B^2 - 4*A*C)) / (2*A)
    // x2 = (-B - sqrt(B^2 - 4*A*C)) / (2*A)
    double rayDirX = cos(ray.angle[rayIndex]);
    double rayDirY = sin(ray.angle[rayIndex]);

    double helling = rayDirY/rayDirX;
    double a = (ray.y[rayIndex]+source.y) - helling * (ray.x[rayIndex]+source.x);
    
    double A = 1.0 + helling*helling;

    bool lengthSet = false;
    ray.length[rayIndex] = INT32_MAX;

    for(Circle circle: circles){
        double B = 2.0 * helling * (a - circle.y) - 2.0 * circle.x;
        double C = circle.x*circle.x + (a - circle.y)*(a - circle.y) - circle.radius_square;
        double discriminant = B*B - 4.0*A*C;

        if (discriminant >= 0.0) {
            double x1 = (-B + sqrt(discriminant)) / (2.0*A);
            double x2 = (-B - sqrt(discriminant)) / (2.0*A);
            // bereken y op circle
            double y1 = helling * x1 + a;
            double y2 = helling * x2 + a;

            // scalair product om te kijken of de circle in de richting van de ray valt
            double vec1X = x1 - (ray.x[rayIndex] + source.x);
            double vec1Y = y1 - (ray.y[rayIndex] + source.y);
            double dot1 = rayDirX * vec1X + rayDirY * vec1Y;
            
            double length1 = sqrt((x1 - ray.x[rayIndex]-source.x) * (x1 - ray.x[rayIndex]-source.x) + (y1 - ray.y[rayIndex]-source.y) * (y1 - ray.y[rayIndex]-source.y));
            double length2 = sqrt((x2 - ray.x[rayIndex]-source.x) * (x2 - ray.x[rayIndex]-source.x) + (y2 - ray.y[rayIndex]-source.y) * (y2 - ray.y[rayIndex]-source.y));

            // Scalair product berekenen om richting te bepalen
            if(dot1 > 0){
                if (length1 < length2) {
                    if(lengthSet == true && length1 < ray.length[rayIndex] || lengthSet == false){
                        lengthSet = true;
                        ray.length[rayIndex] = (int)length1;
                    }
                } else {
                    if(lengthSet == true && length2 < ray.length[rayIndex] || lengthSet == false){
                        lengthSet = true;
                        ray.length[rayIndex] = (int)length2;
                    }
                }
                if(ray.length[rayIndex] > WIDTH+HEIGHT){
                    ray.length[rayIndex] = WIDTH+HEIGHT;
                } 
            }
            else{
                ray.length[rayIndex] = WIDTH+HEIGHT;
            }
        }
        else{
            if(lengthSet == false){
                ray.length[rayIndex] = WIDTH+HEIGHT;
            }
        }

    }

}