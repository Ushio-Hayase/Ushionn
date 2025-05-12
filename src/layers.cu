#include "layers.h"

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
    input_ = input;

    auto matmul_attribute = fe::graph::Matmul_attributes();
    AttributeSetComputeDataType<fe::graph::Matmul_attributes>(matmul_attribute);
    auto WX = graph.matmul(input, weights_, matmul_attribute);

    auto pointwise_attribute = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    AttributeSetComputeDataType<fe::graph::Pointwise_attributes>(pointwise_attribute);

    auto WXplusB = graph.pointwise(WX, bias_, pointwise_attribute);

    output_ = WXplusB;

    return WXplusB;
}

std::shared_ptr<fe::graph::Tensor_attributes> DenseLayer::Backward(
    const std::shared_ptr<fe::graph::Tensor_attributes> grad_output, fe::graph::Graph& graph)
{
    auto weight_matmul_attribute = fe::graph::Matmul_attributes();
    AttributeSetComputeDataType<fe::graph::Matmul_attributes>(weight_matmul_attribute);

    auto input_shape = input_->get_dim();
    size_t input_shape_size = input_shape.size();

    std::swap(input_shape[input_shape_size - 1], input_shape[input_shape_size - 2]);
    input_->set_dim(input_shape);

    weights_grads_ = graph.matmul(input_, grad_output, weight_matmul_attribute);
    bias_grads_ = grad_output;

    std::swap(input_shape[input_shape_size - 1], input_shape[input_shape_size - 2]);
    input_->set_dim(input_shape);

    auto weight_shape = weights_->get_dim();
    size_t weight_shape_size = weight_shape.size();

    std::swap(weight_shape[weight_shape_size - 1], weight_shape[weight_shape_size - 2]);
    weights_->set_dim(weight_shape);

    auto grad_input = graph.matmul(grad_output, weights_, weight_matmul_attribute);

    std::swap(weight_shape[weight_shape_size - 1], weight_shape[weight_shape_size - 2]);
    weights_->set_dim(weight_shape);

    return grad_input;
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