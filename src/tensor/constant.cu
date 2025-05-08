// 그리드 차원
const int GDim_X = gridDim.x;
const int GDim_Y = gridDim.y;
const int GDim_Z = gridDim.z;

// 블록 차원
const int BDim_X = blockDim.x;
const int BDim_Y = blockDim.y;
const int BDim_Z = blockDim.z;

const int NUM_THREAD_IN_BLOCK = BDim_X * BDim_Y * BDim_Z;

__device__ __forceinline__ int TIdInBlock(int TId_X, int TId_Y, int TId_Z)
{
    return TId_Z * (BDim_Y * BDim_X) + TId_Y * BDim_X + TId_X;
}

__device__ __forceinline__ int Grid1DTID(int BId_X, int TId_X, int TId_Y, int TId_Z)
{
    return BId_X * NUM_THREAD_IN_BLOCK + TIdInBlock(TId_X, TId_Y, TId_Z);
}

__device__ __forceinline__ int Grid2DTID()