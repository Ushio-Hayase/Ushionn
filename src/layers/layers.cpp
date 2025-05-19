#include "layers/layers.h"

namespace ushionn
{

std::shared_ptr<fe::graph::Tensor_attributes> DenseLayer::add_forward_to_graph(
    std::shared_ptr<fe::graph::Graph>& graph,
    const std::shared_ptr<fe::graph::Tensor_attributes>& input_tensor_graph_ref)
{
    auto matmul_operation = fe::graph::Matmul_attributes().set_compute_data_type(weights_.get_data_type());
    auto add_operation = fe::graph::Pointwise_attributes()
                             .set_compute_data_type(bias_.get_data_type())
                             .set_mode(fe::PointwiseMode_t::ADD);

    auto weights_tensor_attributes = weights_.create_graph_tensor_attributes(graph);
    auto bias_tensor_attributes = bias_.create_graph_tensor_attributes(graph);

    // 가중치 행렬곱
    auto wegihts_output_tensor_attrubutes =
        graph->tensor(graph->matmul(input_tensor_graph_ref, weights_tensor_attributes, matmul_operation)
                          ->set_is_virtual(true)
                          .set_name(name_ + "_weights_matmul_out"));

    // 편향 덧셈
    auto output_tensor_attributes =
        graph->tensor(graph->pointwise(wegihts_output_tensor_attrubutes, bias_tensor_attributes, add_operation)
                          ->set_is_virtual(true)
                          .set_name("_bias_add_out"));

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
    auto weights_grad_tensor_attributes = weights_grad_.create_graph_tensor_attributes(graph);

    // 순전파 입력 행렬 전치
    auto fwd_input_tensor_ref_shape = fwd_input_tensor_ref->get_dim();
    auto fwd_input_tensor_ref_shape_size = fwd_input_tensor_ref_shape.size();
    std::swap(fwd_input_tensor_ref_shape[fwd_input_tensor_ref_shape_size - 1],
              fwd_input_tensor_ref_shape[fwd_input_tensor_ref_shape_size - 2]);

    auto fwd_input_graph_ref_T =
        graph->tensor(fwd_input_tensor_ref->set_dim(fwd_input_tensor_ref_shape).set_is_virtual(true));

    // 손실함수에 대한 가중치의 기울기 계산
    weights_grad_tensor_attributes =
        graph->tensor(graph->matmul(fwd_input_graph_ref_T, output_grad_graph_ref, matmul_operation)
                          ->set_is_virtual(true)
                          .set_name(name_ + "_weights_matmul_out_bwd"));

    // reduction 작업 attributes 정의
    auto reduction_operation = fe::graph::Reduction_attributes()
                                   .set_compute_data_type(bias_grad_.get_data_type())
                                   .set_mode(fe::ReductionMode_t::ADD);
    auto bias_grad_tensor_attributes = bias_grad_.create_graph_tensor_attributes(graph);

    // 손실함수에 대한 편향의 기울기 계산
    bias_grad_tensor_attributes = graph->tensor(graph->reduction(output_grad_graph_ref, reduction_operation)
                                                    ->set_is_virtual(true)
                                                    .set_name(name_ + "_bias_add_out_bwd"));

    // 가중치 행렬 전치
    auto weights_tensor_attribute = weights_.create_graph_tensor_attributes(graph);
    auto weights_tensor_shape = weights_grad_tensor_attributes->get_dim();
    auto weights_tensor_shape_size = weights_tensor_shape.size();
    std::swap(weights_tensor_shape[weights_tensor_shape_size - 1], weights_tensor_shape[weights_tensor_shape_size - 2]);

    auto weights_tensor_T = graph->tensor(weights_tensor_attribute->set_dim(weights_tensor_shape).set_is_virtual(true));

    // 손실함수에 대한 입력의 기울기 계산
    return graph->tensor(graph->matmul(output_grad_graph_ref, weights_tensor_T, matmul_operation)
                             ->set_is_virtual(true)
                             .set_name(name_ + "_output_bwd"));
}

std::vector<Tensor*> DenseLayer::get_parameters()
{
    return {&weights_, &bias_};
}

std::vector<Tensor*> DenseLayer::get_gradients()
{
    return {&weights_grad_, &bias_grad_};
}

}  // namespace ushionn
