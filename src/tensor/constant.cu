// 그리드 차원
#define GDim_X gridDim.x
#define GDim_Y gridDim.y
#define GDim_Z gridDim.z

// 블록 차원
#define BDim_X blockDim.x
#define BDim_Y blockDim.y
#define BDim_Z blockDim.z

#define NUM_THREAD_IN_BLOCK (BDim_X * BDim_Y * BDim_Z)

__device__ __forceinline__ int TIdInBlock(int TId_X, int TId_Y, int TId_Z)
{
    return TId_Z * (BDim_Y * BDim_X) + TId_Y * BDim_X + TId_X;
}

__device__ __forceinline__ int Grid1DTID(int BId_X, int TId_X, int TId_Y, int TId_Z)
{
    return BId_X * NUM_THREAD_IN_BLOCK + TIdInBlock(TId_X, TId_Y, TId_Z);
}

__device__ __forceinline__ int Grid2DTID(int BId_X, int BId_Y, int TId_X, int TId_Y, int TId_Z)
{
    return GDim_X * BId_Y + Grid1DTID(BId_X, TId_X, TId_Y, TId_Z);
}

__device__ __forceinline__ int Grid3DTID(int BId_X, int BId_Y, int BId_Z, int TId_X, int TId_Y, int TId_Z)
{
    return GDim_X * GDim_Y * BId_Z + Grid2DTID(BId_X, BId_Y, TId_X, TId_Y, TId_Z);
}