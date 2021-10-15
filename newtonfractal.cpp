#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <ctime>

#include <SDL2/SDL.h>

#include "Perlin_Noise/PerlinNoise.h"

// Stuff to define noise function to move points
#define EDGEREPEL 0.1
#define SPEED 0.01
#define EVOLVE 0.001
#define PERIOD 0.3

using namespace std;

void randomComplex(int n, complex<double> *points) {
	for(int x=0; x<n; x++)
		points[x] = complex<double>(((double)rand()/RAND_MAX)*2-1, ((double)rand()/RAND_MAX)*2-1);
}

void randomDouble(int n, double *numbers) {
	for(int x=0; x<n; x++)
		numbers[x] = ((double)rand()/RAND_MAX)*2-1;
}

class Polynomial {
public:
	Polynomial(int order, complex<double> *roots) {
		this->order = order;
		coefficients = (complex<double> *)calloc(order+1, sizeof(complex<double>));
		genCoefficients(roots);
	}
	
	~Polynomial() {
		free(coefficients);
	}
	
	void genCoefficients(complex<double> *roots) {
		memset(coefficients, 0, sizeof(complex<double>)*(order+1));
		// First order polynomial of x-roots[0]
		coefficients[0] = -roots[0];
		coefficients[1] = 1;
		for(int x=1; x<order; x++) { // For all the other points
			coefficients[order] = coefficients[order-1]; // Move highest coefficient up
			for(int y=order-1; y>=1; y--) // For the rest of the coefficients
				coefficients[y] = coefficients[y-1] + coefficients[y] * -roots[x];
			coefficients[0] *= -roots[x];
		}
	}
	
	complex<double> eval(complex<double> x) {
		complex<double> p = 1, ret = 0;
		for(int i=0; i<=order; i++) {
			ret += p*coefficients[i];
			p *= x;
		}
		return ret;
	}
	
	complex<double> d(complex<double> x) {
		complex<double> p = 1, ret = 0;
		for(int i=1; i<=order; i++) {
			ret += complex<double>(i, 0) * p * coefficients[i];
			p *= x;
		}
		return ret;
	}
	
private:
	int order;
	complex<double> *coefficients;
};

typedef struct {
	double r;
	double g;
	double b;
} COLOR;

void createColors(int n, double *hues, COLOR *colors) {
	for(int x=0; x<n; x++) {
		colors[x].r = (sin(2*M_PI* hues[x])     +1)/2;
		colors[x].g = (sin(2*M_PI*(hues[x]+1.0/3))+1)/2;
		colors[x].b = (sin(2*M_PI*(hues[x]+2.0/3))+1)/2;
	}
}

int main(int argc, char *argv[]) {
	srand(time(NULL));
	PerlinNoise perlin(time(NULL));
	
	int order = 5;
	int maxIters = 15;
	double thresh = 0.0001;
	complex<double> *points = (complex<double> *)calloc(order, sizeof(complex<double>));
	double *colors = (double *)calloc(order, sizeof(double));
	COLOR *rgbColors = (COLOR *)calloc(order, sizeof(COLOR));
	
	
	randomComplex(order, points);
	randomDouble(order, colors);
	createColors(order, colors, rgbColors);
	
	Polynomial *p = new Polynomial(order, points);
	
	SDL_Renderer *renderer;
	SDL_Window *window;
	SDL_Init(SDL_INIT_VIDEO);
	SDL_CreateWindowAndRenderer(512, 512, SDL_WINDOW_RESIZABLE|SDL_WINDOW_ALLOW_HIGHDPI, &window, &renderer);
	
	SDL_Event event;
	int frame = 0;
	int run = 1;
	while(run) {
		Uint64 start = SDL_GetPerformanceCounter();
		
		SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
		SDL_RenderClear(renderer);
		
		int w, h;
		SDL_GetRendererOutputSize(renderer, &w, &h);
		for(int x=0; x<w; x++) {
			for(int y=0; y<h; y++) {
				complex<double> point(((double)x/w)*2-1, ((double)y/h)*2-1);
				int n, closestPoint = 0;
				double minDist = DBL_MAX;
				for(n=0; n<maxIters; n++) {
					point -= p->eval(point) / p->d(point);
					for(int i=0; i<order; i++) {
						double dist = norm(point - points[i]);
						if(dist < minDist) {
							minDist = dist;
							closestPoint = i;
						}
					}
					if(minDist < thresh) break;
				}
				//double brightness = fmax(0, 1-(n+fmin(minDist/thresh, 1.0))/maxIters);
				double brightness = 1-(double)n/maxIters;
				SDL_SetRenderDrawColor(renderer, rgbColors[closestPoint].r*brightness*254.999, rgbColors[closestPoint].g*brightness*254.999, rgbColors[closestPoint].b*brightness*254.999, 255);
				SDL_RenderDrawPoint(renderer, x, y);
			}
		}
		
		SDL_RenderPresent(renderer);
		
		for(int i=0; i<order; i++) {
			double x = points[i].real(), y = points[i].imag();
			x += (perlin.noise(x*PERIOD, y*PERIOD, frame*EVOLVE       +i*132.32) + EDGEREPEL/(x+1) + EDGEREPEL/(x-1))*SPEED;
			y += (perlin.noise(x*PERIOD, y*PERIOD, frame*EVOLVE+142.12+i*132.32) + EDGEREPEL/(y+1) + EDGEREPEL/(y-1))*SPEED;
			if(x > 1) x = 1;
			if(x < -1) x = -1;
			if(y > 1) y = 1;
			if(y < -1) y = -1;
			points[i] = complex<double>(x, y);
			
			colors[i] += perlin.noise(frame*EVOLVE + i*132.32, 0, 0) * SPEED;
		}
		p->genCoefficients(points);
		createColors(order, colors, rgbColors);
		frame++;
		
		Uint64 end = SDL_GetPerformanceCounter();
		float elapsed = (end - start) / (float)SDL_GetPerformanceFrequency();
		SDL_Delay(fmaxf(1/60.0 - elapsed, 0));
		
		while(SDL_PollEvent(&event))
			if(event.type == SDL_QUIT) {
				run = 0;
				break;
			}
	}
	
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
