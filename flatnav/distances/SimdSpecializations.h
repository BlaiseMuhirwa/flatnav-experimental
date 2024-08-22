
// #pragma once
// #include <flatnav/distances/SquaredL2Distance.h>



// namespace flatnav {

// class SquaredL216AVX512Distance : public SquaredL2Distance {
// public:
//     SquaredL216AVX512Distance() = default;

//     explicit SquaredL216AVX512Distance(size_t dim)
//         : SquaredL2Distance(dim) {}

//     float distanceImpl(const void *x, const void *y, bool asymmetric = false) {
//         (void) asymmetric;
//         float *pointer_x = (float*)(x);
//         float *pointer_y = (float*)(y);

//         float PORTABLE_ALIGN64 temp_result[16];

//         // Pointer to the 
//         const float* pointer_end_x = pointer_x + this->dimension >> 4 << 4;



//     }

// };



// } // namespace flatnav