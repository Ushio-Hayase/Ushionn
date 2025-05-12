#pragma once

#include <memory>

#ifdef USE_CUDNN

#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

#endif
namespace ushionn
{
enum class ActivationType
{
    kSigmoid,
    kReLU,
    kSoftmax,
    kTanh,
    kLeakyRelu,
    kELU,
    kGELU
};

#ifdef USE_CUDNN

class ActivationFunc
{
   public:
    virtual std::shared_ptr<fe::graph::Tensor_attributes> Forward(std::shared_ptr<fe::graph::Tensor_attributes> input);
    virtual std::shared_ptr<fe::graph::Tensor_attributes> Backward(
        std::shared_ptr<fe::graph::Tensor_attributes> grad_output);
};

#else
#endif

}  // namespace ushionn