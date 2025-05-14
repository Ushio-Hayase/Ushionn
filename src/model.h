#pragma once

#include <list>

#include "layers.h"

#ifdef USE_CUDNN

#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

namespace ushionn
{
class Model
{
   public:
    virtual Tensor Forward(Tensor input);

   private:
    fe::graph::Graph graph;
    std::list<Layer> layers;
};
}  // namespace ushionn

#else
#endif