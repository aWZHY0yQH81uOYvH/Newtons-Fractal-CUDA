SDL2 = `pkg-config --cflags --libs sdl2`
CFLAGS = --std=c++11 -O2

newtonfractal: newtonfractal.cpp Perlin_Noise/PerlinNoise.cpp
	g++ $(CFLAGS) $^ $(SDL2) -I. -o $@

newtonfractal_cuda: newtonfractal.cu Perlin_Noise/PerlinNoise.cpp
	nvcc $(CFLAGS) $^ $(SDL2) -I. -o $@

.PHONY: run run_cuda
run: newtonfractal
	@./newtonfractal

run_cuda: newtonfractal_cuda
	@./newtonfractal_cuda
