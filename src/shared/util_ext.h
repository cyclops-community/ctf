#ifndef __UTIL_EXT_H__
#define __UTIL_EXT_H__

int CTF_alloc_ptr(int64_t len, void ** const ptr);
int CTF_mst_alloc_ptr(int64_t len, void ** const ptr);
void * CTF_alloc(int64_t len);
void * CTF_mst_alloc(int64_t len);
int CTF_free(void * ptr, int const tid);
int CTF_free(void * ptr);


void csgemm(char transa,      char transb,
            int m,            int n,
            int k,            float a,
            float const * A,  int lda,
            float const * B,  int ldb,
            float b,          float * C,
                              int ldc);

void cdgemm(char transa,      char transb,
            int m,            int n,
            int k,            double a,
            double const * A, int lda,
            double const * B, int ldb,
            double b,         double * C,
                              int ldc);

void ccgemm(char transa,                    char transb,
            int m,                          int n,
            int k,                          std::complex<float> a,
            const std::complex<float> * A,  int lda,
            const std::complex<float> * B,  int ldb,
            std::complex<float> b,          std::complex<float> * C,
                                            int ldc);


void czgemm(char transa,                    char transb,
            int m,                          int n,
            int k,                          std::complex<double> a,
            const std::complex<double> * A, int lda,
            const std::complex<double> * B, int ldb,
            std::complex<double> b,         std::complex<double> * C,
                                            int ldc);

void csaxpy(int n,              float  dA,
            const float  * dX,  int incX,
            float  * dY,        int incY);

void cdaxpy(int n,              double dA,
            const double * dX,  int incX,
            double * dY,        int incY);

void ccaxpy(int n,                            std::complex<float> dA,
            const std::complex<float> * dX,   int incX,
            std::complex<float> * dY,         int incY);

void czaxpy(int n,                            std::complex<double> dA,
            const std::complex<double> * dX,  int incX,
            std::complex<double> * dY,        int incY);

void csscal(int n, float dA, float * dX, int incX);

void cdscal(int n, double dA, double * dX, int incX);

void ccscal(int n, std::complex<float> dA, std::complex<float> * dX, int incX);

void czscal(int n, std::complex<double> dA, std::complex<double> * dX, int incX);


int conv_idx(int          ndim,
             char const * cidx,
             int **       iidx);

int conv_idx(int          ndim_A,
             char const * cidx_A,
             int **       iidx_A,
             int          ndim_B,
             char const * cidx_B,
             int **       iidx_B);

int conv_idx(int          ndim_A,
             char const * cidx_A,
             int **       iidx_A,
             int          ndim_B,
             char const * cidx_B,
             int **       iidx_B,
             int          ndim_C,
             char const * cidx_C,
             int **       iidx_C);



#endif
