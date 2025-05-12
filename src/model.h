#pragma once

#ifdef USE_CUDNN

#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

namespace ushionn
{
class Model
{
};
}  // namespace ushionn

#else
#endif