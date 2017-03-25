#ifndef __CONTAINER_PROJECT__
#define __CONTAINER_PROJECT__

// NArmedBandit functions

extern "C" __declspec(dllexport) float* StationaryAnalysis();

extern "C" __declspec(dllexport) float* NonStationaryAnalysis();

// MountainCar functions

extern "C" __declspec(dllexport) void SolveMountainCar();

// Qlearning functions

extern "C" __declspec(dllexport) int* SolveWindyGridWorld();

#endif // __CONTAINER_PROJECT__
