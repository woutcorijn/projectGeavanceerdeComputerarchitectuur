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
    for(int i = 0; i < NUM_RAYS; i++){
        Ray ray = rays[i];
        if(ray.length == 0){
            return;
        }
        double dx = cos(ray.angle);
        double dy = sin(ray.angle);
        

        int x = ray.x + source.x;
        int y = ray.y + source.y;

        Uint32 fadeFactor = 10;
        Uint32 fadeByte = fadeFactor * 0xFF;
        Uint32 oldFadeByte = fadeFactor * 0xFF;

        for (int j = 0; j < ray.length; j++) {
            fadeByte = fadeByte - 0x01;
            if(fadeByte > oldFadeByte)
                fadeByte = oldFadeByte;
            else
                oldFadeByte = fadeByte;

            int px = x + (int)(j * dx);
            int py = y + (int)(j * dy);

            // Controleer of de pixel binnen de grenzen van het scherm valt
            if (j > source.radius && px >= 0 && px < WIDTH && py >= 0 && py < HEIGHT) {
                uint32_t pixel = ray.pixel;
                pixel = (pixel & 0x00FFFFFF) | ((fadeByte / fadeFactor) << 24);
                
                d_pixels[py * WIDTH + px] = pixel;
            }
        }
    }
}

void calculateLengthRaysCpu(Ray *rays, Circle *circlesObject, Circle source) {
    Circle circles[NUM_CIRCLE_OBJECTS];
    for(int i = 0; i < NUM_CIRCLE_OBJECTS; i++) {
        circles[i] = circlesObject[i];
    }

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
        double rayDirX = cos(ray.angle);
        double rayDirY = sin(ray.angle);

        double helling = rayDirY/rayDirX;
        double a = (ray.y+source.y) - helling * (ray.x+source.x);
        
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
                double vec1X = x1 - (ray.x + source.x);
                double vec1Y = y1 - (ray.y + source.y);
                double dot1 = rayDirX * vec1X + rayDirY * vec1Y;
                
                double length1 = sqrt((x1 - ray.x-source.x) * (x1 - ray.x-source.x) + (y1 - ray.y-source.y) * (y1 - ray.y-source.y));
                double length2 = sqrt((x2 - ray.x-source.x) * (x2 - ray.x-source.x) + (y2 - ray.y-source.y) * (y2 - ray.y-source.y));
                if(dot1 > 0){
                    if (length1 < length2) {
                        if(lengthSet == true && length1 < ray.length || lengthSet == false){
                            lengthSet = true;
                            ray.length = (int)length1;
                        }
                    } else {
                        if(lengthSet == true && length2 < ray.length || lengthSet == false){
                            lengthSet = true;
                            ray.length = (int)length2;
                        }
                    }
                    if(ray.length > WIDTH+HEIGHT){
                        ray.length = WIDTH+HEIGHT;
                    } 
                }
                else{
                    ray.length = WIDTH+HEIGHT;
                }
            }
            else{
                if(lengthSet == false){
                    ray.length = WIDTH+HEIGHT;
                }
            }

        }
    }

}