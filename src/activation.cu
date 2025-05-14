#include <cmath>

#include "activation.h"

#ifdef USE_CUDNN

#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

std::shared_ptr<fe::graph::Tensor_attributes> ushionn::Sigmoid::Forward(
    std::shared_ptr<fe::graph::Tensor_attributes> input, fe::graph::Graph& graph)
{
    auto sigmoid_operation = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::SIGMOID_FWD);

    return graph.tensor(graph.pointwise(input, sigmoid_operation)->set_is_virtual(true));
}

std::shared_ptr<fe::graph::Tensor_attributes> ushionn::Sigmoid::Forward(
    size_t axis, std::shared_ptr<fe::graph::Tensor_attributes> input, fe::graph::Graph& graph)
{
    auto sigmoid_operation =
        fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::SIGMOID_FWD).set_axis(axis);

    return graph.tensor(graph.pointwise(input, sigmoid_operation)->set_is_virtual(true));
}

std::shared_ptr<fe::graph::Tensor_attributes> ushionn::Sigmoid::Backward(
    std::shared_ptr<fe::graph::Tensor_attributes> grad_output, fe::graph::Graph& graph)
{
    auto bwd_sigmoid_operation = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::SIGMOID_BWD);

    return graph.tensor(graph.pointwise(grad_output, bwd_sigmoid_operation)->set_is_virtual(true));
}

std::shared_ptr<fe::graph::Tensor_attributes> ushionn::ReLU::Forward(
    std::shared_ptr<fe::graph::Tensor_attributes> input, fe::graph::Graph& graph)
{
    auto relu_attribute = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);

    return graph.tensor(graph.pointwise(input, relu_attribute)->set_is_virtual(true));
}

std::shared_ptr<fe::graph::Tensor_attributes> ushionn::ReLU::Forward(
    size_t size, std::shared_ptr<fe::graph::Tensor_attributes> grad_output, fe::graph::Graph& graph)
{
    auto bwd_relu_operation = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_BWD);

    return graph.tensor(graph.pointwise(grad_output, bwd_relu_operation)->set_is_virtual(true));
}

#else
#endif
