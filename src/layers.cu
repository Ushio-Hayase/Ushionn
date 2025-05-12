#include "layers.cuh"

namespace ushionn
{

#ifdef USE_CUDNN
std::pair<const fe::graph::Tensor_attributes&, const fe::graph::Tensor_attributes&> Layer::Parameters() const
{
    return {*weights_, *bias_};
}

std::pair<const fe::graph::Tensor_attributes&, const fe::graph::Tensor_attributes&> Layer::Gradients() const
{
    return {*weights_grads_, *bias_grads_};
}

template <typename T>
void Layer::AttributeSetComputeDataType(fe::graph::Attributes<T>& attr)
{
    if (dtype_ == DataType::FLOAT32)
        attr.set_compute_data_type(fe::DataType_t::FLOAT);
    else if (dtype_ == DataType::FLOAT64)
        attr.set_compute_data_type(fe::DataType_t::DOUBLE);
    else if (dtype_ == DataType::INT32)
        attr.set_compute_data_type(fe::DataType_t::INT32);
}

std::shared_ptr<fe::graph::Tensor_attributes> DenseLayer::Forward(
    const std::shared_ptr<fe::graph::Tensor_attributes> input, fe::graph::Graph& graph)
{
    auto matmul_attribute = fe::graph::Matmul_attributes();
    AttributeSetComputeDataType<fe::graph::Matmul_attributes>(matmul_attribute);
    auto WX = graph.matmul(input, weights_, matmul_attribute);

    auto pointwise_attribute = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    AttributeSetComputeDataType<fe::graph::Pointwise_attributes>(pointwise_attribute);

    auto WXplusB = graph.pointwise(WX, bias_, pointwise_attribute);
    return WXplusB;
}

std::shared_ptr<fe::graph::Tensor_attributes> DenseLayer::Backward(
    const std::shared_ptr<fe::graph::Tensor_attributes> grad_output, fe::graph::Graph& graph)
{
}

#else
std::pair<const Tensor&, const Tensor&> Layer::Parameters() const
{
    return {*weights_, *bias_};
}

std::pair<const Tensor&, const Tensor&> Layer::Gradients() const
{
    return {*weights_grads_, *bias_grads_};
}
#endif

}  // namespace ushionn