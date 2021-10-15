#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <ctime>
#include <string>

#include <SDL2/SDL.h>

#include "Perlin_Noise/PerlinNoise.h"

// Stuff to define noise function to move points
#define EDGEREPEL 1
#define SPEED 0.003
#define EVOLVE 0.001
#define PERIOD 0.3
#define MOUSEPOWER 0.01
#define REPELPOWER 0.001

// Window dimensions
#define WINDOW_W 2048 // Needs to be divisible by 256
#define WINDOW_H 2048

// Newton fractal settings
#define ORDER 8
#define MAX_ITERS 15
#define THRESH 0.0001

// How much to make the brightnes fall off with more iterations (looks like space if this is really high)
#define BRIGHTNESS_POW 2

using namespace std;

// Custom complex number class since the standard library one doesn't work with CUDA
class Complex {
public:
	__host__ __device__ Complex(double r, double i) {
		this->r = r;
		this->i = i;
	}
	
	__host__ __device__ Complex(int r) {
		this->r = r;
		this->i = 0;
	}
	
	__host__ __device__ double real() {return r;}
	__host__ __device__ double imag() {return i;}
	
	__host__ __device__ Complex operator+(const Complex &other) {return Complex(this->r + other.r, this->i + other.i);}
	__host__ __device__ Complex &operator+=(const Complex &other) {
		this->r += other.r;
		this->i += other.i;
		return *this;
	}
	
	__host__ __device__ Complex operator-(const Complex &other) {return Complex(this->r - other.r, this->i - other.i);}
	__host__ __device__ Complex &operator-=(const Complex &other) {
		this->r -= other.r;
		this->i -= other.i;
		return *this;
	}
	
	__host__ __device__ Complex operator-() {return Complex(-this->r, -this->i);}
	
	__host__ __device__ Complex operator*(const Complex &other) {return Complex(this->r*other.r - this->i*other.i, this->r*other.i + this->i*other.r);}
	__host__ __device__ Complex &operator*=(const Complex &other) {
		double tr = this->r*other.r - this->i*other.i;
		this->i = this->r*other.i + this->i*other.r;
		this->r = tr;
		return *this;
	}
	
	__host__ __device__ Complex operator/(const Complex &other) {return Complex((this->r*other.r + this->i*other.i)/(other.r*other.r + other.i*other.i), (this->i*other.r - this->r*other.i)/(other.r*other.r + other.i*other.i));}
	__host__ __device__ Complex &operator/=(const Complex &other) {
		double tr = (this->r*other.r + this->i*other.i)/(other.r*other.r + other.i*other.i);
		this->i = (this->i*other.r - this->r*other.i)/(other.r*other.r + other.i*other.i);
		this->r = tr;
		return *this;
	}
	
	__host__ __device__ static double norm(const Complex &arg) {
		return arg.r * arg.r + arg.i * arg.i;
	}
	
private:
	double r, i;
};

// Represent a polynomial
class Polynomial {
public:
	// Create a polynomial with some complex roots
	Polynomial(int order, Complex *roots) {
		this->order = order;
		cudaMallocManaged(&coefficients, (order+1)*sizeof(Complex));
		genCoefficients(roots);
	}
	
	~Polynomial() {
		cudaFree(coefficients);
	}
	
	void genCoefficients(Complex *roots) {
		memset(coefficients, 0, sizeof(Complex)*(order+1));
		// First order polynomial of x-roots[0]
		coefficients[0] = -roots[0];
		coefficients[1] = Complex(1, 0);
		for(int x=1; x<order; x++) { // For all the other points
			coefficients[order] = coefficients[order-1]; // Move highest coefficient up
			for(int y=order-1; y>=1; y--) // For the rest of the coefficients
				coefficients[y] = coefficients[y-1] + coefficients[y] * -roots[x];
			coefficients[0] *= -roots[x];
		}
	}
	
	// Evaluate at point
	__host__ __device__ Complex eval(Complex x) {
		Complex p = 1, ret = 0;
		for(int i=0; i<=order; i++) {
			ret += p*coefficients[i];
			p *= x;
		}
		return ret;
	}
	
	// Derivative at point
	__host__ __device__ Complex d(Complex x) {
		Complex p = 1, ret = 0;
		for(int i=1; i<=order; i++) {
			ret += Complex(i, 0) * p * coefficients[i];
			p *= x;
		}
		return ret;
	}
	
private:
	int order;
	Complex *coefficients;
};

typedef struct {
	double r;
	double g;
	double b;
} COLOR;

// Hue to RGB
void createColors(int n, double *hues, COLOR *colors) {
	for(int x=0; x<n; x++) {
		colors[x].r = (sin(2*M_PI* hues[x])     +1)/2;
		colors[x].g = (sin(2*M_PI*(hues[x]+1.0/3))+1)/2;
		colors[x].b = (sin(2*M_PI*(hues[x]+2.0/3))+1)/2;
	}
}

void randomComplex(int n, Complex *points) {
	for(int x=0; x<n; x++)
		points[x] = Complex(((double)rand()/RAND_MAX)*2-1, ((double)rand()/RAND_MAX)*2-1);
}

void randomDouble(int n, double *numbers) {
	for(int x=0; x<n; x++)
		numbers[x] = ((double)rand()/RAND_MAX)*2-1;
}

// Apply a force (into dx, dy) that pushes away from the other thing
void computeRepel(double &dx, double &dy, double x, double y, double otherx, double othery, double power) {
	double otherdx = otherx - x;
	double otherdy = othery - y;
	double otherDist = otherdx*otherdx + otherdy*otherdy;
	dx -= otherdx / otherDist * power;
	dy -= otherdy / otherDist * power;
}

// CUDA kernel
__global__ void compute(uint32_t *buf, Complex *points, COLOR *rgbColors, Polynomial p) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y;
	
	Complex point(((double)x/WINDOW_W)*2-1, ((double)y/WINDOW_H)*2-1); // Point on complex plane (normalized)
	int n, closestPoint = 0;
	double minDist = DBL_MAX;
	for(n=0; n<MAX_ITERS; n++) { // Do Newton's method of finding root
		point -= p.eval(point) / p.d(point);
		for(int i=0; i<ORDER; i++) { // Keep track of closest actual root, exit if close enough
			double dist = Complex::norm(point - points[i]);
			if(dist < minDist) {
				minDist = dist;
				closestPoint = i;
			}
		}
		if(minDist < THRESH) break;
	}
	
	// Coolor based on closest point
	double smoothAdj = minDist/THRESH; // Smooth a bit between numbers of iterations
	if(smoothAdj > 1) smoothAdj = 1;
	double brightness = 1-(n+smoothAdj)/MAX_ITERS;
	if(brightness < 0) brightness = 0;
	brightness = powf(brightness, BRIGHTNESS_POW);
	//double brightness = 1-(double)n/MAX_ITERS;
	//const double brightness = 1;
	unsigned char r = rgbColors[closestPoint].r*brightness*254.999;
	unsigned char g = rgbColors[closestPoint].g*brightness*254.999;
	unsigned char b = rgbColors[closestPoint].b*brightness*254.999;
	
	buf[y*WINDOW_W+x] = 0xFF000000 | (r << 16) | (g << 8) | b;
}

int main(int argc, char *argv[]) {
	// Seed randoms
	srand(time(NULL));
	PerlinNoise perlin(time(NULL));
	
	// Allocate memory (using unified/managed memory for this stuff)
	Complex *points; // Actual roots (which will move around)
	cudaMallocManaged(&points, ORDER*sizeof(Complex));
	double *colors = (double *)calloc(ORDER, sizeof(double)); // Hues assigned to each root
	COLOR *rgbColors; // RGB version of the above hues
	cudaMallocManaged(&rgbColors, ORDER*sizeof(COLOR));
	
	// Start with random data
	randomComplex(ORDER, points);
	randomDouble(ORDER, colors);
	createColors(ORDER, colors, rgbColors);
	
	Polynomial p = Polynomial(ORDER, points);
	
	SDL_Init(SDL_INIT_VIDEO);
	SDL_Window *window = SDL_CreateWindow("Newton's Fractal", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_W, WINDOW_H, SDL_WINDOW_SHOWN);
	SDL_Surface *screen = SDL_CreateRGBSurface( 0, WINDOW_W, WINDOW_H, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000);
	if(!screen) {
		SDL_Log("SDL_CreateRGBSurface() failed: %s", SDL_GetError());
		exit(1);
	}
	// Stuff to make it so we can write to a texture (not zero-copy but whatever; this might be limiting performance)
	SDL_Renderer *sdlRenderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
	SDL_Texture *sdlTexture = SDL_CreateTexture(sdlRenderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING | SDL_TEXTUREACCESS_TARGET, WINDOW_W, WINDOW_H);
	
	uint32_t *gpuMem;
	cudaMalloc(&gpuMem, WINDOW_W*WINDOW_H*4);
	
	// CUDA kernel size
	dim3 threadDim(256); // 1024 threads per block
	dim3 blockDim(WINDOW_W/256, WINDOW_H);
	
	
	SDL_Event event;
	int frame = 0;
	int run = 1;
	while(run) {
		Uint64 start = SDL_GetPerformanceCounter();
		
		SDL_LockSurface(screen);
		compute<<<blockDim, threadDim>>>(gpuMem, points, rgbColors, p); // Compute the frame
		cudaMemcpy(screen->pixels, gpuMem, WINDOW_W*WINDOW_H*4, cudaMemcpyDeviceToHost); // Copy frame back to host (so we can then copy it back ugh)
		SDL_UnlockSurface(screen);
		
		// Display frame
		SDL_UpdateTexture(sdlTexture, NULL, screen->pixels, screen->pitch);
		SDL_RenderClear(sdlRenderer);
		SDL_RenderCopy(sdlRenderer, sdlTexture, NULL, NULL);
		SDL_RenderPresent(sdlRenderer);
		
		// Check if mosue is over the window and get its position if it is
		int mousex = -1, mousey = -1;
		if(window == SDL_GetMouseFocus())
			SDL_GetMouseState(&mousex, &mousey);
		
		// Update all the roots/points
		for(int i=0; i<ORDER; i++) {
			double x = points[i].real(), y = points[i].imag();
			
			// Use Perlin noise to move them around; make them repelled from the edges of the frame (adding random numbers to get some different noise for each point)
			double dx = (perlin.noise(x*PERIOD, y*PERIOD, frame*EVOLVE       +i*132.32)*2-1 + EDGEREPEL/(x+1.1) + EDGEREPEL/(x-1.1))*SPEED;
			double dy = (perlin.noise(x*PERIOD, y*PERIOD, frame*EVOLVE+142.12+i*132.32)*2-1 + EDGEREPEL/(y+1.1) + EDGEREPEL/(y-1.1))*SPEED;
			
			// If mouse is over the window, make it repel the points
			if(mousex > 0 && mousey > 0)
				computeRepel(dx, dy, x, y, (double)mousex/WINDOW_W*2-1, (double)mousey/WINDOW_H*2-1, MOUSEPOWER);
			
			// Points repel each other
			for(int j=0; j<ORDER; j++) {
				if(j == i) continue;
				computeRepel(dx, dy, x, y, points[j].real(), points[j].imag(), REPELPOWER);
			}
			
			// Apply movement
			x += dx;
			y += dy;
			
			// Check bounds
			if(x > 1) x = 1;
			if(x < -1) x = -1;
			if(y > 1) y = 1;
			if(y < -1) y = -1;
			points[i] = Complex(x, y);
			
			// Change color hues with Perlin noise as well
			colors[i] += (perlin.noise(frame*EVOLVE + i*132.32, 0, 0)*2-1) * SPEED;
		}
		p.genCoefficients(points); // Regenerate coefficients for the polynomial with the new roots
		createColors(ORDER, colors, rgbColors); // Regenerate RGB versions of the hues
		frame++;
		
		// Limit framerate
		Uint64 end = SDL_GetPerformanceCounter();
		float elapsed = (end - start) / (float)SDL_GetPerformanceFrequency();
		SDL_Delay(fmaxf(1/60.0 - elapsed, 0));
		
		// Allow program to exit
		while(SDL_PollEvent(&event)) {
			if(event.type == SDL_QUIT) {
				run = 0;
				break;
			}
		}
	}
	
	cudaFree(gpuMem);
	
	cudaFree(points);
	free(colors);
	cudaFree(rgbColors);
	
	SDL_DestroyTexture(sdlTexture);
	SDL_DestroyRenderer(sdlRenderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
	return 0;
}
