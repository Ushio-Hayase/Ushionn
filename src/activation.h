#pragma once

#include <memory>

#include "tensor.h"

#ifdef USE_CUDNN

#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

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

class Activation
{
   public:
    virtual std::shared_ptr<fe::graph::Tensor_attributes> Forward(std::shared_ptr<fe::graph::Tensor_attributes> input,
                                                                  fe::graph::Graph& graph);
    virtual std::shared_ptr<fe::graph::Tensor_attributes> Backward(
        std::shared_ptr<fe::graph::Tensor_attributes> grad_output, fe::graph::Graph& graph);

    static ActivationType value;

   protected:
    std::shared_ptr<fe::graph::Tensor_attributes> input_;
    std::shared_ptr<fe::graph::Tensor_attributes> output_;
    Device device_;
    DataType dtype_;
};

class Sigmoid : public Activation
{
   public:
    std::shared_ptr<fe::graph::Tensor_attributes> Forward(std::shared_ptr<fe::graph::Tensor_attributes> input,
                                                          fe::graph::Graph& graph);
    std::shared_ptr<fe::graph::Tensor_attributes> Forward(size_t axis,
                                                          std::shared_ptr<fe::graph::Tensor_attributes> input,
                                                          fe::graph::Graph& graph);

    std::shared_ptr<fe::graph::Tensor_attributes> Backward(std::shared_ptr<fe::graph::Tensor_attributes> grad_output,
                                                           fe::graph::Graph& graph);
};

class ReLU : public Activation
{
   public:
    std::shared_ptr<fe::graph::Tensor_attributes> Forward(std::shared_ptr<fe::graph::Tensor_attributes> input,
                                                          fe::graph::Graph& graph);

    std::shared_ptr<fe::graph::Tensor_attributes> Forward(size_t axis,
                                                          std::shared_ptr<fe::graph::Tensor_attributes> input,
                                                          fe::graph::Graph& graph);
    std::shared_ptr<fe::graph::Tensor_attributes> Backward(std::shared_ptr<fe::graph::Tensor_attributes> grad_output,
                                                           fe::graph::Graph& graph);
};

}  // namespace ushionn

#else

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

class Activation
{
   public:
    virtual std::shared_ptr<Tensor> Forward(std::shared_ptr<Tensor> input);
    virtual std::shared_ptr<Tensor> Backward(std::shared_ptr<Tensor> grad_output);
};

}  // namespace ushionn
#endif
