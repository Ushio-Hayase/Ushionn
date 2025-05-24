#include "layers/layers.h"

namespace ushionn
{
namespace nn
{
std::shared_ptr<fe::graph::Tensor_attributes> DenseLayer::add_forward_to_graph(
    std::shared_ptr<fe::graph::Graph>& graph,
    const std::shared_ptr<fe::graph::Tensor_attributes>& input_tensor_graph_ref, int64_t output_uid)
{
    auto matmul_operation = fe::graph::Matmul_attributes().set_compute_data_type(weights_.get_data_type());
    auto add_operation = fe::graph::Pointwise_attributes()
                             .set_compute_data_type(bias_.get_data_type())
                             .set_mode(fe::PointwiseMode_t::ADD);

    auto weights_tensor_attributes = weights_.create_graph_tensor_attributes(graph);
    auto bias_tensor_attributes = bias_.create_graph_tensor_attributes(graph);

    // 가중치 행렬곱
    auto weights_output_tensor_attributes =
        graph->matmul(input_tensor_graph_ref, weights_tensor_attributes, matmul_operation);

    weights_output_tensor_attributes->set_is_virtual(true)
        .set_name(name_ + "_weights_matmul_out")
        .set_data_type(data_type_);

    // 편향 덧셈
    auto output_tensor_attributes =
        graph->pointwise(weights_output_tensor_attributes, bias_tensor_attributes, add_operation);
    output_tensor_attributes->set_is_virtual(false)
        .set_name(name_ + "_out")
        .set_data_type(data_type_)
        .set_uid(output_uid);

    return output_tensor_attributes;
}

std::shared_ptr<fe::graph::Tensor_attributes> DenseLayer::add_backward_to_graph(
    std::shared_ptr<fe::graph::Graph>& graph,
    const std::shared_ptr<fe::graph::Tensor_attributes>& output_grad_graph_ref,
    const std::shared_ptr<fe::graph::Tensor_attributes>& fwd_input_tensor_ref,  // 순전파 시의 입력
    const std::shared_ptr<fe::graph::Tensor_attributes>& fwd_output_tensor_ref)
// 순전파 시의 출력 (필요시)

{
    auto matmul_operation = fe::graph::Matmul_attributes().set_compute_data_type(weights_grad_.get_data_type());

    std::vector<int64_t> fwd_input_tensor_ref_shape = fwd_input_tensor_ref->get_dim();
    size_t fwd_input_tensor_ref_shape_size = fwd_input_tensor_ref_shape.size();
    std::swap(fwd_input_tensor_ref_shape[fwd_input_tensor_ref_shape_size - 1],
              fwd_input_tensor_ref_shape[fwd_input_tensor_ref_shape_size - 2]);

    auto transpose1 =
        fe::graph::Pointwise_attributes().set_compute_data_type(data_type_).set_mode(fe::PointwiseMode_t::IDENTITY);

    auto fwd_input_tensor = graph->pointwise(fwd_input_tensor_ref, transpose1);
    fwd_input_tensor->set_is_virtual(true).set_data_type(data_type_).set_dim(fwd_input_tensor_ref_shape);

    // 손실함수에 대한 가중치의 기울기 계산
    auto weights_grad_tensor_attributes = graph->matmul(fwd_input_tensor, output_grad_graph_ref, matmul_operation);
    weights_grad_tensor_attributes->set_data_type(data_type_).set_is_virtual(false);

    // reduction 작업 attributes 정의
    auto reduction_operation = fe::graph::Reduction_attributes()
                                   .set_compute_data_type(bias_grad_.get_data_type())
                                   .set_mode(fe::ReductionMode_t::ADD);

    // 손실함수에 대한 편향의 기울기 계산
    auto bias_grad_tensor_attributes = graph->reduction(output_grad_graph_ref, reduction_operation);
    bias_grad_tensor_attributes->set_data_type(data_type_).set_is_virtual(false).set_dim(bias_grad_.get_shape());

    auto weights_tensor_attributes = weights_.create_graph_tensor_attributes(graph);

    std::vector<int64_t> weights_tensor_attributes_shape = weights_tensor_attributes->get_dim();
    size_t weights_tensor_attributes_shape_size = weights_tensor_attributes_shape.size();
    std::swap(weights_tensor_attributes_shape[weights_tensor_attributes_shape_size - 1],
              weights_tensor_attributes_shape[weights_tensor_attributes_shape_size - 2]);

    auto transpose2 =
        fe::graph::Pointwise_attributes().set_compute_data_type(data_type_).set_mode(fe::PointwiseMode_t::IDENTITY);

    weights_tensor_attributes = graph->pointwise(weights_tensor_attributes, transpose2);
    weights_tensor_attributes->set_is_virtual(true).set_data_type(data_type_).set_dim(weights_tensor_attributes_shape);

    // 손실함수에 대한 입력의 기울기 계산
    auto output_tensor_attributes = graph->matmul(output_grad_graph_ref, weights_tensor_attributes, matmul_operation);
    output_tensor_attributes->set_is_virtual(true).set_name(name_ + "_output_bwd").set_data_type(data_type_);

    return output_tensor_attributes;
}

std::vector<Tensor*> DenseLayer::get_parameters()
{
    return {&weights_, &bias_};
}

std::vector<Tensor*> DenseLayer::get_gradients()
{
    return {&weights_grad_, &bias_grad_};
}

void DenseLayer::initialize_parameters_norm(unsigned long long seed)
{
    USHIONN_ASSERT(weights_.is_on_host(), "The weight to initialize must be on the host");

    size_t num_elem = weights_.get_num_elements();

    std::mt19937 gen(seed);

    if (data_type_ == fe::DataType_t::FLOAT)
    {
        std::normal_distribution<float> dist;
        for (int i = 0; i < num_elem; ++i) static_cast<float*>(weights_.get_mutable_host_ptr())[i] = dist(gen);
    }
    else if (data_type_ == fe::DataType_t::DOUBLE)
    {
        std::normal_distribution<double> dist;
        for (int i = 0; i < num_elem; ++i) static_cast<double*>(weights_.get_mutable_host_ptr())[i] = dist(gen);
    }
    else if (data_type_ == fe::DataType_t::INT32)
    {
        USHIONN_LOG_FATAL("This library does not yet support integer weight initialization");
    }
}

std::vector<int64_t> DenseLayer::get_output_shape(std::vector<int64_t> input_dims) const
{
    USHIONN_ASSERT(!input_dims.empty(), "Input dimensions cannot be empty for DenseLayer.");

    // 입력 차원 마지막이 가중치 차원 중간, 가중치 행렬의 첫 차원과 일치해야함
    USHIONN_ASSERT(input_dims.back() == weights_.get_shape()[1],
                   "DenseLayer input feature size mismatch with weight input size.");
    input_dims.back() = weights_.get_shape()[2];
    return input_dims;
}

}  // namespace nn
}  // namespace ushionn