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

        Uint32 fadeFactor = 6;
        Uint32 fadeByte = fadeFactor * 0xFF;
        Uint32 oldFadeByte = fadeFactor * 0xFF;

        for(int rayIndex = 0; rayIndex < 2; rayIndex++) {
            double dx = cos(ray.angle[rayIndex]);
            double dy = sin(ray.angle[rayIndex]);

            int x = ray.x[rayIndex] + source.x;
            int y = ray.y[rayIndex] + source.y;

            for (int j = 0; j < ray.length[rayIndex]; j++) {
                fadeByte -= 0x01;
                if(fadeByte > oldFadeByte)
                    fadeByte = oldFadeByte;
                else
                    oldFadeByte = fadeByte;

                int px = x + (int)(j * dx);
                int py = y + (int)(j * dy);

                // Controleer of de pixel binnen het scherm valt
                if (px >= 0 && px < WIDTH && py >= 0 && py < HEIGHT) {
                    uint32_t pixel = ray.pixel[rayIndex];
                    pixel = (pixel & 0x00FFFFFF) | ((fadeByte / fadeFactor) << 24);
                    d_pixels[py * WIDTH + px] = pixel;
                }
            }
        }
    }
}

void calculateLengthRaysCpu(Ray *rays, Circle *circlesObject, Circle source) {
    Circle circles[NUM_CIRCLE_OBJECTS];
    for(int i = 0; i < NUM_CIRCLE_OBJECTS; i++) {
        circles[i] = circlesObject[i];
    }

    int rayIndex = 0;

    for(int i = 0; i < NUM_RAYS; i++){
        Ray &ray = rays[i];

        // y = helling * x + a
        
        //printf("x: %d, y: %d", ray.x, ray.y);

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

}

void calculateReflectionCpu(Ray *rays, Circle *circlesObject, Circle source) {
    for(int i = 0; i < NUM_RAYS; i++){
        Ray &ray = rays[i];

        // Raakpunt berekenen
        ray.x[1] = ray.length[0] * cos(ray.angle[0]);
        ray.y[1] = ray.length[0] * sin(ray.angle[0]);

        float dx = cos(ray.angle[0]);
        float dy = sin(ray.angle[0]);

        float x0 = source.x;
        float y0 = source.y;

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

            ray.angle[1] = atan2(ry, rx);
            ray.length[1] = ray.length[0];
            ray.pixel[1] = ray.pixel[0];
        }
    }
}