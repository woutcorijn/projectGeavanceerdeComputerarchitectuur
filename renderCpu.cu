#include <render.h>
#include <config.h>

void clearScreenCpu(Uint32* d_pixels, Uint32 pixel) {
    for(int i = 0; i < WIDTH * HEIGHT; i++){
        d_pixels[i] = pixel;
    }
}

void drawCircleCpu(Uint32* d_pixels, Circle circle) {
    for(int index = 0; index < WIDTH * HEIGHT; index++){
    
        int indexX = index % WIDTH;
        int indexY = index / WIDTH;

        int distance_squared = (indexX-circle.x)*(indexX-circle.x) + (indexY-circle.y)*(indexY-circle.y);

        if(distance_squared <= circle.radius_square){
            d_pixels[index] = circle.pixel;
        }
    }
}

void drawRaysCpu(Uint32* d_pixels, Ray *rays, Circle source) {
    for(int i = 0; i < NUM_RAYS; i++) {
        Ray ray = rays[i];

        double fadeLength = 16;
        double fadeFactor = 0.997;
        double fadeByte = fadeLength * (ray.pixel[0] >> 24);
        for(int rayIndex = 0; rayIndex < 4; rayIndex++) {
            
            double dx = cos(ray.angle[rayIndex]);
            double dy = sin(ray.angle[rayIndex]);

            int x = ray.x[rayIndex] + source.x;
            int y = ray.y[rayIndex] + source.y;

            for (int j = 0; j < ray.length[rayIndex]; j++) {
                fadeByte = fadeByte*fadeFactor;

                int px = x + (int)(j * dx);
                int py = y + (int)(j * dy);

                // Controleer of de pixel binnen het scherm valt
                if (px >= 0 && px < WIDTH && py >= 0 && py < HEIGHT) {
                    uint32_t pixel = ray.pixel[rayIndex];
                    pixel = (pixel & 0x00FFFFFF) | ((uint32_t)(fadeByte / fadeLength) << 24);
                    d_pixels[py * WIDTH + px] = pixel+d_pixels[py * WIDTH + px];
                }
            }
        }
    }
}

void calculateLengthRaysCpu(Ray *rays, Circle *circlesObject, Circle source, int rayIndex) {
    Circle circles[NUM_CIRCLE_OBJECTS];
    for(int i = 0; i < NUM_CIRCLE_OBJECTS; i++) {
        circles[i] = circlesObject[i];
    }

    for (int i = 0; i < NUM_RAYS; i++) {
        Ray &ray = rays[i];
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
}

void calculateReflectionCpu(Ray *rays, Circle *circlesObject, Circle source, int rayIndex) {
    for(int i = 0; i < NUM_RAYS; i++){
        Ray &ray = rays[i];

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
        float closestT = std::numeric_limits<float>::max(); // kortste afstand tot raakpunt
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
}