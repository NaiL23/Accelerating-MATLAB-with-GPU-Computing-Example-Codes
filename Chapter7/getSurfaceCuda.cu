#include "mex.h"
#include <vector>

// To compile: mexcuda getSurfaceCuda.cu

__constant__ unsigned int edgeTable[256] =
{
       0,  265,  515,  778, 1030, 1295, 1541, 1804,
    2060, 2309, 2575, 2822, 3082, 3331, 3593, 3840,
     400,  153,  915,  666, 1430, 1183, 1941, 1692,
    2460, 2197, 2975, 2710, 3482, 3219, 3993, 3728,
     560,  825,   51,  314, 1590, 1855, 1077, 1340,
    2620, 2869, 2111, 2358, 3642, 3891, 3129, 3376,
     928,  681,  419,  170, 1958, 1711, 1445, 1196,
    2988, 2725, 2479, 2214, 4010, 3747, 3497, 3232,
    1120, 1385, 1635, 1898,  102,  367,  613,  876,
    3180, 3429, 3695, 3942, 2154, 2403, 2665, 2912,
    1520, 1273, 2035, 1786,  502,  255, 1013,  764,
    3580, 3317, 4095, 3830, 2554, 2291, 3065, 2800,
    1616, 1881, 1107, 1370,  598,  863,   85,  348,
    3676, 3925, 3167, 3414, 2650, 2899, 2137, 2384,
    1984, 1737, 1475, 1226,  966,  719,  453,  204,
    4044, 3781, 3535, 3270, 3018, 2755, 2505, 2240,
    2240, 2505, 2755, 3018, 3270, 3535, 3781, 4044,
     204,  453,  719,  966, 1226, 1475, 1737, 1984,
    2384, 2137, 2899, 2650, 3414, 3167, 3925, 3676,
     348,   85,  863,  598, 1370, 1107, 1881, 1616,
    2800, 3065, 2291, 2554, 3830, 4095, 3317, 3580,
     764, 1013,  255,  502, 1786, 2035, 1273, 1520,
    2912, 2665, 2403, 2154, 3942, 3695, 3429, 3180,
     876,  613,  367,  102, 1898, 1635, 1385, 1120,
    3232, 3497, 3747, 4010, 2214, 2479, 2725, 2988,
    1196, 1445, 1711, 1958,  170,  419,  681,  928,
    3376, 3129, 3891, 3642, 2358, 2111, 2869, 2620,
    1340, 1077, 1855, 1590,  314,   51,  825,  560,
    3728, 3993, 3219, 3482, 2710, 2975, 2197, 2460,
    1692, 1941, 1183, 1430,  666,  915,  153,  400,
    3840, 3593, 3331, 3082, 2822, 2575, 2309, 2060,
    1804, 1541, 1295, 1030,  778,  515,  265,    0
};

__constant__ int triTable[256][16] =
{
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  1,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  8,  3,  9,  8,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  8,  3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  2, 10,  0,  2,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  2,  8,  3,  2, 10,  8, 10,  9,  8, -1, -1, -1, -1, -1, -1, -1 },
    {  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0, 11,  2,  8, 11,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  9,  0,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  1, 11,  2,  1,  9, 11,  9,  8, 11, -1, -1, -1, -1, -1, -1, -1 },
    {  3, 10,  1, 11, 10,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0, 10,  1,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  9,  0,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  3,  0,  7,  3,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  1,  9,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  1,  9,  4,  7,  1,  7,  3,  1, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  2, 10,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  4,  7,  3,  0,  4,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  2, 10,  9,  0,  2,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1 },
    {  2, 10,  9,  2,  9,  7,  2,  7,  3,  7,  9,  4, -1, -1, -1, -1 },
    {  8,  4,  7,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 11,  4,  7, 11,  2,  4,  2,  0,  4, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  0,  1,  8,  4,  7,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  7, 11,  9,  4, 11,  9, 11,  2,  9,  2,  1, -1, -1, -1, -1 },
    {  3, 10,  1,  3, 11, 10,  7,  8,  4, -1, -1, -1, -1, -1, -1, -1 },
    {  1, 11, 10,  1,  4, 11,  1,  0,  4,  7, 11,  4, -1, -1, -1, -1 },
    {  4,  7,  8,  9,  0, 11,  9, 11, 10, 11,  0,  3, -1, -1, -1, -1 },
    {  4,  7, 11,  4, 11,  9,  9, 11, 10, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  5,  4,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  5,  4,  1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  8,  5,  4,  8,  3,  5,  3,  1,  5, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  0,  8,  1,  2, 10,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1 },
    {  5,  2, 10,  5,  4,  2,  4,  0,  2, -1, -1, -1, -1, -1, -1, -1 },
    {  2, 10,  5,  3,  2,  5,  3,  5,  4,  3,  4,  8, -1, -1, -1, -1 },
    {  9,  5,  4,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0, 11,  2,  0,  8, 11,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  5,  4,  0,  1,  5,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1 },
    {  2,  1,  5,  2,  5,  8,  2,  8, 11,  4,  8,  5, -1, -1, -1, -1 },
    { 10,  3, 11, 10,  1,  3,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  9,  5,  0,  8,  1,  8, 10,  1,  8, 11, 10, -1, -1, -1, -1 },
    {  5,  4,  0,  5,  0, 11,  5, 11, 10, 11,  0,  3, -1, -1, -1, -1 },
    {  5,  4,  8,  5,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  7,  8,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  3,  0,  9,  5,  3,  5,  7,  3, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  7,  8,  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  7,  8,  9,  5,  7, 10,  1,  2, -1, -1, -1, -1, -1, -1, -1 },
    { 10,  1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3, -1, -1, -1, -1 },
    {  8,  0,  2,  8,  2,  5,  8,  5,  7, 10,  5,  2, -1, -1, -1, -1 },
    {  2, 10,  5,  2,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1 },
    {  7,  9,  5,  7,  8,  9,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7, 11, -1, -1, -1, -1 },
    {  2,  3, 11,  0,  1,  8,  1,  7,  8,  1,  5,  7, -1, -1, -1, -1 },
    { 11,  2,  1, 11,  1,  7,  7,  1,  5, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  5,  8,  8,  5,  7, 10,  1,  3, 10,  3, 11, -1, -1, -1, -1 },
    {  5,  7,  0,  5,  0,  9,  7, 11,  0,  1,  0, 10, 11, 10,  0, -1 },
    { 11, 10,  0, 11,  0,  3, 10,  5,  0,  8,  0,  7,  5,  7,  0, -1 },
    { 11, 10,  5,  7, 11,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  8,  3,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  0,  1,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  8,  3,  1,  9,  8,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  6,  5,  2,  6,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  6,  5,  1,  2,  6,  3,  0,  8, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1, -1, -1, -1 },
    {  5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8, -1, -1, -1, -1 },
    {  2,  3, 11, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 11,  0,  8, 11,  2,  0, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  1,  9,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1 },
    {  5, 10,  6,  1,  9,  2,  9, 11,  2,  9,  8, 11, -1, -1, -1, -1 },
    {  6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, -1, -1, -1, -1 },
    {  3, 11,  6,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1, -1, -1, -1 },
    {  6,  5,  9,  6,  9, 11, 11,  9,  8, -1, -1, -1, -1, -1, -1, -1 },
    {  5, 10,  6,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  3,  0,  4,  7,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  9,  0,  5, 10,  6,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1 },
    { 10,  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4, -1, -1, -1, -1 },
    {  6,  1,  2,  6,  5,  1,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7, -1, -1, -1, -1 },
    {  8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6, -1, -1, -1, -1 },
    {  7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9, -1 },
    {  3, 11,  2,  7,  8,  4, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1 },
    {  5, 10,  6,  4,  7,  2,  4,  2,  0,  2,  7, 11, -1, -1, -1, -1 },
    {  0,  1,  9,  4,  7,  8,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1 },
    {  9,  2,  1,  9, 11,  2,  9,  4, 11,  7, 11,  4,  5, 10,  6, -1 },
    {  8,  4,  7,  3, 11,  5,  3,  5,  1,  5, 11,  6, -1, -1, -1, -1 },
    {  5,  1, 11,  5, 11,  6,  1,  0, 11,  7, 11,  4,  0,  4, 11, -1 },
    {  0,  5,  9,  0,  6,  5,  0,  3,  6, 11,  6,  3,  8,  4,  7, -1 },
    {  6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, -1, -1, -1, -1 },
    { 10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  4, 10,  6,  4,  9, 10,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1 },
    { 10,  0,  1, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1, -1, -1, -1 },
    {  8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1, 10, -1, -1, -1, -1 },
    {  1,  4,  9,  1,  2,  4,  2,  6,  4, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4, -1, -1, -1, -1 },
    {  0,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  8,  3,  2,  8,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1 },
    { 10,  4,  9, 10,  6,  4, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  8,  2,  2,  8, 11,  4,  9, 10,  4, 10,  6, -1, -1, -1, -1 },
    {  3, 11,  2,  0,  1,  6,  0,  6,  4,  6,  1, 10, -1, -1, -1, -1 },
    {  6,  4,  1,  6,  1, 10,  4,  8,  1,  2,  1, 11,  8, 11,  1, -1 },
    {  9,  6,  4,  9,  3,  6,  9,  1,  3, 11,  6,  3, -1, -1, -1, -1 },
    {  8, 11,  1,  8,  1,  0, 11,  6,  1,  9,  1,  4,  6,  4,  1, -1 },
    {  3, 11,  6,  3,  6,  0,  0,  6,  4, -1, -1, -1, -1, -1, -1, -1 },
    {  6,  4,  8, 11,  6,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  7, 10,  6,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  7,  3,  0, 10,  7,  0,  9, 10,  6,  7, 10, -1, -1, -1, -1 },
    { 10,  6,  7,  1, 10,  7,  1,  7,  8,  1,  8,  0, -1, -1, -1, -1 },
    { 10,  6,  7, 10,  7,  1,  1,  7,  3, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7, -1, -1, -1, -1 },
    {  2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9, -1 },
    {  7,  8,  0,  7,  0,  6,  6,  0,  2, -1, -1, -1, -1, -1, -1, -1 },
    {  7,  3,  2,  6,  7,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  2,  3, 11, 10,  6,  8, 10,  8,  9,  8,  6,  7, -1, -1, -1, -1 },
    {  2,  0,  7,  2,  7, 11,  0,  9,  7,  6,  7, 10,  9, 10,  7, -1 },
    {  1,  8,  0,  1,  7,  8,  1, 10,  7,  6,  7, 10,  2,  3, 11, -1 },
    { 11,  2,  1, 11,  1,  7, 10,  6,  1,  6,  7,  1, -1, -1, -1, -1 },
    {  8,  9,  6,  8,  6,  7,  9,  1,  6, 11,  6,  3,  1,  3,  6, -1 },
    {  0,  9,  1, 11,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  7,  8,  0,  7,  0,  6,  3, 11,  0, 11,  6,  0, -1, -1, -1, -1 },
    {  7, 11,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  0,  8, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  1,  9, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  8,  1,  9,  8,  3,  1, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1 },
    { 10,  1,  2,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  2, 10,  3,  0,  8,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1 },
    {  2,  9,  0,  2, 10,  9,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1 },
    {  6, 11,  7,  2, 10,  3, 10,  8,  3, 10,  9,  8, -1, -1, -1, -1 },
    {  7,  2,  3,  6,  2,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  7,  0,  8,  7,  6,  0,  6,  2,  0, -1, -1, -1, -1, -1, -1, -1 },
    {  2,  7,  6,  2,  3,  7,  0,  1,  9, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6, -1, -1, -1, -1 },
    { 10,  7,  6, 10,  1,  7,  1,  3,  7, -1, -1, -1, -1, -1, -1, -1 },
    { 10,  7,  6,  1,  7, 10,  1,  8,  7,  1,  0,  8, -1, -1, -1, -1 },
    {  0,  3,  7,  0,  7, 10,  0, 10,  9,  6, 10,  7, -1, -1, -1, -1 },
    {  7,  6, 10,  7, 10,  8,  8, 10,  9, -1, -1, -1, -1, -1, -1, -1 },
    {  6,  8,  4, 11,  8,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  6, 11,  3,  0,  6,  0,  4,  6, -1, -1, -1, -1, -1, -1, -1 },
    {  8,  6, 11,  8,  4,  6,  9,  0,  1, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  4,  6,  9,  6,  3,  9,  3,  1, 11,  3,  6, -1, -1, -1, -1 },
    {  6,  8,  4,  6, 11,  8,  2, 10,  1, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  2, 10,  3,  0, 11,  0,  6, 11,  0,  4,  6, -1, -1, -1, -1 },
    {  4, 11,  8,  4,  6, 11,  0,  2,  9,  2, 10,  9, -1, -1, -1, -1 },
    { 10,  9,  3, 10,  3,  2,  9,  4,  3, 11,  3,  6,  4,  6,  3, -1 },
    {  8,  2,  3,  8,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8, -1, -1, -1, -1 },
    {  1,  9,  4,  1,  4,  2,  2,  4,  6, -1, -1, -1, -1, -1, -1, -1 },
    {  8,  1,  3,  8,  6,  1,  8,  4,  6,  6, 10,  1, -1, -1, -1, -1 },
    { 10,  1,  0, 10,  0,  6,  6,  0,  4, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  6,  3,  4,  3,  8,  6, 10,  3,  0,  3,  9, 10,  9,  3, -1 },
    { 10,  9,  4,  6, 10,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  9,  5,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  8,  3,  4,  9,  5, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1 },
    {  5,  0,  1,  5,  4,  0,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1 },
    { 11,  7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5, -1, -1, -1, -1 },
    {  9,  5,  4, 10,  1,  2,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1 },
    {  6, 11,  7,  1,  2, 10,  0,  8,  3,  4,  9,  5, -1, -1, -1, -1 },
    {  7,  6, 11,  5,  4, 10,  4,  2, 10,  4,  0,  2, -1, -1, -1, -1 },
    {  3,  4,  8,  3,  5,  4,  3,  2,  5, 10,  5,  2, 11,  7,  6, -1 },
    {  7,  2,  3,  7,  6,  2,  5,  4,  9, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7, -1, -1, -1, -1 },
    {  3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0, -1, -1, -1, -1 },
    {  6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8, -1 },
    {  9,  5,  4, 10,  1,  6,  1,  7,  6,  1,  3,  7, -1, -1, -1, -1 },
    {  1,  6, 10,  1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4, -1 },
    {  4,  0, 10,  4, 10,  5,  0,  3, 10,  6, 10,  7,  3,  7, 10, -1 },
    {  7,  6, 10,  7, 10,  8,  5,  4, 10,  4,  8, 10, -1, -1, -1, -1 },
    {  6,  9,  5,  6, 11,  9, 11,  8,  9, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  6, 11,  0,  6,  3,  0,  5,  6,  0,  9,  5, -1, -1, -1, -1 },
    {  0, 11,  8,  0,  5, 11,  0,  1,  5,  5,  6, 11, -1, -1, -1, -1 },
    {  6, 11,  3,  6,  3,  5,  5,  3,  1, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  2, 10,  9,  5, 11,  9, 11,  8, 11,  5,  6, -1, -1, -1, -1 },
    {  0, 11,  3,  0,  6, 11,  0,  9,  6,  5,  6,  9,  1,  2, 10, -1 },
    { 11,  8,  5, 11,  5,  6,  8,  0,  5, 10,  5,  2,  0,  2,  5, -1 },
    {  6, 11,  3,  6,  3,  5,  2, 10,  3, 10,  5,  3, -1, -1, -1, -1 },
    {  5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2, -1, -1, -1, -1 },
    {  9,  5,  6,  9,  6,  0,  0,  6,  2, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8, -1 },
    {  1,  5,  6,  2,  1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  3,  6,  1,  6, 10,  3,  8,  6,  5,  6,  9,  8,  9,  6, -1 },
    { 10,  1,  0, 10,  0,  6,  9,  5,  0,  5,  6,  0, -1, -1, -1, -1 },
    {  0,  3,  8,  5,  6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 10,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 11,  5, 10,  7,  5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { 11,  5, 10, 11,  7,  5,  8,  3,  0, -1, -1, -1, -1, -1, -1, -1 },
    {  5, 11,  7,  5, 10, 11,  1,  9,  0, -1, -1, -1, -1, -1, -1, -1 },
    { 10,  7,  5, 10, 11,  7,  9,  8,  1,  8,  3,  1, -1, -1, -1, -1 },
    { 11,  1,  2, 11,  7,  1,  7,  5,  1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2, 11, -1, -1, -1, -1 },
    {  9,  7,  5,  9,  2,  7,  9,  0,  2,  2, 11,  7, -1, -1, -1, -1 },
    {  7,  5,  2,  7,  2, 11,  5,  9,  2,  3,  2,  8,  9,  8,  2, -1 },
    {  2,  5, 10,  2,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1 },
    {  8,  2,  0,  8,  5,  2,  8,  7,  5, 10,  2,  5, -1, -1, -1, -1 },
    {  9,  0,  1,  5, 10,  3,  5,  3,  7,  3, 10,  2, -1, -1, -1, -1 },
    {  9,  8,  2,  9,  2,  1,  8,  7,  2, 10,  2,  5,  7,  5,  2, -1 },
    {  1,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  8,  7,  0,  7,  1,  1,  7,  5, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  0,  3,  9,  3,  5,  5,  3,  7, -1, -1, -1, -1, -1, -1, -1 },
    {  9,  8,  7,  5,  9,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  5,  8,  4,  5, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1 },
    {  5,  0,  4,  5, 11,  0,  5, 10, 11, 11,  3,  0, -1, -1, -1, -1 },
    {  0,  1,  9,  8,  4, 10,  8, 10, 11, 10,  4,  5, -1, -1, -1, -1 },
    { 10, 11,  4, 10,  4,  5, 11,  3,  4,  9,  4,  1,  3,  1,  4, -1 },
    {  2,  5,  1,  2,  8,  5,  2, 11,  8,  4,  5,  8, -1, -1, -1, -1 },
    {  0,  4, 11,  0, 11,  3,  4,  5, 11,  2, 11,  1,  5 , 1, 11, -1 },
    {  0,  2,  5,  0,  5,  9,  2, 11,  5,  4,  5,  8, 11,  8,  5, -1 },
    {  9,  4,  5,  2, 11,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  2,  5, 10,  3,  5,  2,  3,  4,  5,  3,  8,  4, -1, -1, -1, -1 },
    {  5, 10,  2,  5,  2,  4,  4,  2,  0, -1, -1, -1, -1, -1, -1, -1 },
    {  3, 10,  2,  3,  5, 10,  3,  8,  5,  4,  5,  8,  0,  1,  9, -1 },
    {  5, 10,  2,  5,  2,  4,  1,  9,  2,  9,  4,  2, -1, -1, -1, -1 },
    {  8,  4,  5,  8,  5,  3,  3,  5,  1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  4,  5,  1,  0,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5, -1, -1, -1, -1 },
    {  9,  4,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  4, 11,  7,  4,  9, 11,  9, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  8,  3,  4,  9,  7,  9, 11,  7,  9, 10, 11, -1, -1, -1, -1 },
    {  1, 10, 11,  1, 11,  4,  1,  4,  0,  7,  4, 11, -1, -1, -1, -1 },
    {  3,  1,  4,  3,  4,  8,  1, 10,  4,  7,  4, 11, 10, 11,  4, -1 },
    {  4, 11,  7,  9, 11,  4,  9,  2, 11,  9,  1,  2, -1, -1, -1, -1 },
    {  9,  7,  4,  9, 11,  7,  9,  1, 11,  2, 11,  1,  0,  8,  3, -1 },
    { 11,  7,  4, 11,  4,  2,  2,  4,  0, -1, -1, -1, -1, -1, -1, -1 },
    { 11,  7,  4, 11,  4,  2,  8,  3,  4,  3,  2,  4, -1, -1, -1, -1 },
    {  2,  9, 10,  2,  7,  9,  2,  3,  7,  7,  4,  9, -1, -1, -1, -1 },
    {  9, 10,  7,  9,  7,  4, 10,  2,  7,  8,  7,  0,  2,  0,  7, -1 },
    {  3,  7, 10,  3, 10,  2,  7,  4, 10,  1, 10,  0,  4,  0, 10, -1 },
    {  1, 10,  2,  8,  7,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  9,  1,  4,  1,  7,  7,  1,  3, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1, -1, -1, -1, -1 },
    {  4,  0,  3,  7,  4,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  4,  8,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  9, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  0,  9,  3,  9, 11, 11,  9, 10, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  1, 10,  0, 10,  8,  8, 10, 11, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  1, 10, 11,  3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  2, 11,  1, 11,  9,  9, 11,  8, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  0,  9,  3,  9, 11,  1,  2,  9,  2, 11,  9, -1, -1, -1, -1 },
    {  0,  2, 11,  8,  0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  3,  2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  2,  3,  8,  2,  8, 10, 10,  8,  9, -1, -1, -1, -1, -1, -1, -1 },
    {  9, 10,  2,  0,  9,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  2,  3,  8,  2,  8, 10,  0,  1,  8,  1, 10,  8, -1, -1, -1, -1 },
    {  1, 10,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  1,  3,  8,  9,  1,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  9,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    {  0,  3,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
    { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }
};

__device__ float3 interpolatePos(float isovalue, float4 voxel1, float4 voxel2)
{
    float scale = (isovalue - voxel1.w) / (voxel2.w - voxel1.w);
    float3 pos;
    pos.x = voxel1.x + scale * (voxel2.x - voxel1.x);
    pos.y = voxel1.y + scale * (voxel2.y - voxel1.y);
    pos.z = voxel1.z + scale * (voxel2.z - voxel1.z);
    return pos;
}

#define MAX_VERTICES 15

__global__ void getTriangles(float isovalue,
                             float* X, float* Y, float* Z, float* V,
                             int sizeX, int sizeY, int sizeZ,
                             float3* vertexBin, int* triCounter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // compute capability >= 2.x
    //int j = blockIdx.y * blockDim.y + threadIdx.y;
    //int k = blockIdx.z * blockDim.z + threadIdx.z;

    // compute capability < 2.x
    int gy = (sizeY + blockDim.y - 1) / blockDim.y;
    int j = (blockIdx.y % gy) * blockDim.y + threadIdx.y;
    int k = (blockIdx.y / gy) * blockDim.z + threadIdx.z;

    if (i >= sizeX - 1 || j >= sizeY - 1 || k >= sizeZ - 1)
        return;
 
    float4 voxels[8];
    float3 isoPos[12];

    int idx[8];
    idx[0] = sizeX * (sizeY * k + j) + i;
    idx[1] = sizeX * (sizeY * k + j + 1) + i;
    idx[2] = sizeX * (sizeY * k + j + 1) + i + 1;
    idx[3] = sizeX * (sizeY * k + j) + i + 1;
    idx[4] = sizeX * (sizeY * (k + 1) + j) + i;
    idx[5] = sizeX * (sizeY * (k + 1) + j + 1) + i;
    idx[6] = sizeX * (sizeY * (k + 1) + j + 1) + i + 1;
    idx[7] = sizeX * (sizeY * (k + 1) + j) + i + 1;

    // cube
    for (int n = 0; n < 8; ++n)
    {
        voxels[n].w = V[idx[n]];
        voxels[n].x = X[idx[n]];
        voxels[n].y = Y[idx[n]];
        voxels[n].z = Z[idx[n]];
    }

    // find the cube index
    unsigned int cubeIndex = 0;
    for (int n = 0; n < 8; ++n)
    {
       if (voxels[n].w >= isovalue)
           cubeIndex |= (1 << n);
    }

    // get edges from edgeTable
    unsigned int edges = edgeTable[cubeIndex];
    if (edges == 0)
        return;

    // check 12 edges
    if (edges & 1)
        isoPos[0] = interpolatePos(isovalue, voxels[0], voxels[1]);
    if (edges & 2)
        isoPos[1] = interpolatePos(isovalue, voxels[1], voxels[2]);
    if (edges & 4)
        isoPos[2] = interpolatePos(isovalue, voxels[2], voxels[3]);
    if (edges & 8)
        isoPos[3] = interpolatePos(isovalue, voxels[3], voxels[0]);
    if (edges & 16)
        isoPos[4] = interpolatePos(isovalue, voxels[4], voxels[5]);
    if (edges & 32)
        isoPos[5] = interpolatePos(isovalue, voxels[5], voxels[6]);
    if (edges & 64)
        isoPos[6] = interpolatePos(isovalue, voxels[6], voxels[7]);
    if (edges & 128)
        isoPos[7] = interpolatePos(isovalue, voxels[7], voxels[4]);
    if (edges & 256)
        isoPos[8] = interpolatePos(isovalue, voxels[0], voxels[4]);
    if (edges & 512)
        isoPos[9] = interpolatePos(isovalue, voxels[1], voxels[5]);
    if (edges & 1024)
        isoPos[10] = interpolatePos(isovalue, voxels[2], voxels[6]);
    if (edges & 2048)
        isoPos[11] = interpolatePos(isovalue, voxels[3], voxels[7]);

    // walk through the triTable and get the triangle(s) vertices
    float3 vertices[15];
    int numTriangles = 0;
    int numVertices = 0;

    for (int n = 0; n < 15; n += 3)
    {
        int edgeNumger = triTable[cubeIndex][n];
        if (edgeNumger < 0)
            break;

        vertices[numVertices++] = isoPos[edgeNumger];
        vertices[numVertices++] = isoPos[triTable[cubeIndex][n+1]];
        vertices[numVertices++] = isoPos[triTable[cubeIndex][n+2]];
        ++numTriangles;
    }

    triCounter[idx[0]] = numTriangles;
    for (int n = 0; n < numVertices; ++n)
        vertexBin[MAX_VERTICES * idx[0] + n] = vertices[n];
}

void marchingCubes(float isovalue,
                   float* X, float* Y, float* Z, float* V,
                   int sizeX, int sizeY, int sizeZ,
                   float* vertices[],
                   unsigned int* indices[],
                   int& numVertices, int& numTriangles)
{
    float* devX = 0;
    float* devY = 0;
    float* devZ = 0;
    float* devV = 0;
    float3* devVertexBin = 0;
    int* devTriCounter = 0;

    int totalSize = sizeX * sizeY * sizeZ;
    cudaMalloc(&devX, sizeof(float) * totalSize);
    cudaMalloc(&devY, sizeof(float) * totalSize);
    cudaMalloc(&devZ, sizeof(float) * totalSize);
    cudaMalloc(&devV, sizeof(float) * totalSize);
    cudaMemcpy(devX, X, sizeof(float) * totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devY, Y, sizeof(float) * totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devZ, Z, sizeof(float) * totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devV, V, sizeof(float) * totalSize, cudaMemcpyHostToDevice);

    cudaMalloc(&devVertexBin, sizeof(float3) * totalSize * MAX_VERTICES);
    cudaMemset(devVertexBin, 0, sizeof(float3) * totalSize * MAX_VERTICES);
    cudaMalloc(&devTriCounter, sizeof(int) * totalSize);
    cudaMemset(devTriCounter, 0, sizeof(int) * totalSize);

    dim3 blockSize(4, 4, 4);

    // compute capability >= 2.x
    //dim3 gridSize((sizeX + blockSize.x - 1) / blockSize.x,
    //              (sizeY + blockSize.y - 1) / blockSize.y,
    //              (sizeZ + blockSize.z - 1) / blockSize.z);

    // compute capabiltiy < 2.x
    int gy = (sizeY + blockSize.y - 1) / blockSize.y;
    int gz = (sizeZ + blockSize.z - 1) / blockSize.z;
    dim3 gridSize((sizeX + blockSize.x - 1) / blockSize.x,
                  gy * gz,
                  1);

    getTriangles<<<gridSize, blockSize>>>(isovalue,
                                          devX, devY, devZ, devV,
                                          sizeX, sizeY, sizeZ,
                                          devVertexBin, devTriCounter);

    float3* vertexBin = (float3*)malloc(sizeof(float3) * totalSize * MAX_VERTICES);
    cudaMemcpy(vertexBin, devVertexBin, sizeof(float3) * totalSize * MAX_VERTICES, cudaMemcpyDeviceToHost);

    int* triCounter = (int*)malloc(sizeof(int) * totalSize);
    cudaMemcpy(triCounter, devTriCounter, sizeof(int) * totalSize, cudaMemcpyDeviceToHost);

    numTriangles = 0;
    for (int i = 0; i < totalSize; ++i)
        numTriangles += triCounter[i];
    numVertices = 3 * numTriangles;

    for (int i = 0; i < 3; ++i)
    {
        vertices[i] = (float*)malloc(sizeof(float) * numVertices);
        indices[i] = (unsigned int*)malloc(sizeof(unsigned int) * numTriangles);
    }

    int tIdx = 0, vIdx = 0;
    for (int i = 0; i < totalSize; ++i)
    {
        int triCount = triCounter[i];
        if (triCount < 1)
            continue;

        int binIdx = i * MAX_VERTICES;
        for (int c = 0; c < triCount; ++c)
        {
            for (int v = 0; v < 3; ++v)
            {
                vertices[0][vIdx] = vertexBin[binIdx].x;
                vertices[1][vIdx] = vertexBin[binIdx].y;
                vertices[2][vIdx] = vertexBin[binIdx].z;
                indices[v][tIdx] = 3 * tIdx + v + 1;
                ++vIdx;
                ++binIdx;
            }
            ++tIdx;
        }
    }
    
    cudaFree(devX);
    cudaFree(devY);
    cudaFree(devZ);
    cudaFree(devV);
    cudaFree(devVertexBin);
    cudaFree(devTriCounter);
}


// [Vertices, Indices] = getSurface(X, Y, Z, V, isovalue)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 5)
        mexErrMsgTxt("Invaid number of input arguments");
    
    if (nlhs != 2)
        mexErrMsgTxt("Invalid number of outputs");
    
    if (!mxIsSingle(prhs[0]) && !mxIsSingle(prhs[1]) &&
        !mxIsSingle(prhs[2]) && !mxIsSingle(prhs[3]) &&
        !mxIsSingle(prhs[4]))
        mexErrMsgTxt("input vector data type must be single");
    
    const mwSize* size = mxGetDimensions(prhs[3]);
    int sizeX = size[0];
    int sizeY = size[1];
    int sizeZ = size[2];

    float* X = (float*)mxGetData(prhs[0]);
    float* Y = (float*)mxGetData(prhs[1]);
    float* Z = (float*)mxGetData(prhs[2]);
    float* V = (float*)mxGetData(prhs[3]);
    float isovalue = *((float*)mxGetData(prhs[4]));
    
    float* vertices[3];
    unsigned int* indices[3];
    int numVertices = 0;
    int numTriangles = 0;

    marchingCubes(isovalue,
                  X, Y, Z, V,
                  sizeX, sizeY, sizeZ,
                  vertices, indices,
                  numVertices, numTriangles);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        mexPrintf("%s\n", cudaGetErrorString(error));
        mexErrMsgTxt("CUDA failed\n");
    }

    mexPrintf("numVertices = %d\n", numVertices);
    mexPrintf("numTriangles = %d\n", numTriangles);
    
    plhs[0] = mxCreateNumericMatrix(numVertices, 3, mxSINGLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericMatrix(numTriangles, 3, mxUINT32_CLASS, mxREAL);
    float* Vertices = (float*)mxGetData(plhs[0]);
    unsigned int* Indices = (unsigned int*)mxGetData(plhs[1]);

    for (int i = 0; i < 3; ++i)
    {
        memcpy(Vertices + i * numVertices, vertices[i], numVertices * sizeof(float));
        free(vertices[i]);
    }
    for (int i = 0; i < 3; ++i)
    {
        memcpy(Indices + i * numTriangles, indices[i], numTriangles * sizeof(unsigned int));
        free(indices[i]);
    }
}