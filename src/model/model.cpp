#include "model/model.h"

#include <algorithm>
namespace ushionn
{
namespace model
{
void Sequential::add_layer(std::unique_ptr<nn::Layer> layer)
{
    layers_.push_back(layer);
}

void Sequential::build_model_graph(const Tensor& input_shape_template, bool training)
{
    model_input_template_ = input_shape_template;

    auto fwd_input_tensor_attributes =
        model_input_template_.create_graph_tensor_attributes(forward_graph_, true, false);

    // 순전파시 입출력 텐서 속성을 기록할 변수
    std::vector<std::pair<std::shared_ptr<fe::graph::Tensor_attributes>, std::shared_ptr<fe::graph::Tensor_attributes>>>
        fwd_input_output_tensor_attribute_pair_;

    // gpu에 할당할 포인터 변수 선언
    void* input_fwd_gpu_ptr = nullptr;
    void* output_fwd_gpu_ptr = nullptr;
    void* input_bwd_gpu_ptr = nullptr;
    void* output_bwd_gpu_ptr = nullptr;

    // 순전파, 역전파 출력 텐서 속성
    std::shared_ptr<fe::graph::Tensor_attributes> output_tensor_attributes;
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_output_tensor_attributes;

    cudaMalloc(&input_fwd_gpu_ptr, model_input_template_.get_size_in_bytes());

    variant_pack_fwd_.insert({fwd_input_tensor_attributes, input_fwd_gpu_ptr});

    auto fwd_tensor_attributes = fwd_input_tensor_attributes;

    for (auto layer = layers_.begin(); layer != layers_.end(); layer++)
    {
        // 입출력 텐서 속성 저장 변수
        std::pair<std::shared_ptr<fe::graph::Tensor_attributes>, std::shared_ptr<fe::graph::Tensor_attributes>>
            input_output_pair = {fwd_input_tensor_attributes, nullptr};

        // 순전파 처리
        fwd_tensor_attributes = (*layer)->add_forward_to_graph(forward_graph_, fwd_tensor_attributes);

        // 입출력 텐서 속성 저장 변수 중 출력을 저장 후 멤버 변수에 추가
        input_output_pair.second = fwd_tensor_attributes;
        fwd_input_output_tensor_attribute_pair_.push_back(input_output_pair);

        // 마지막 레이어일때 처리
        if (layer == layers_.end() - 1)
            output_tensor_attributes =
                forward_graph_->tensor(fwd_tensor_attributes->set_output(true).set_is_virtual(false));
    }

    Tensor model_output_template_(output_tensor_attributes->get_dim(), fe::DataType_t::FLOAT, false, "model_output");

    cudaMalloc(&output_fwd_gpu_ptr, model_output_template_.get_size_in_bytes());
    variant_pack_fwd_.insert({output_tensor_attributes, output_fwd_gpu_ptr});

    auto bwd_input_tensor_attribute = backward_graph_->tensor(output_tensor_attributes->set_is_virtual(false));

    cudaMalloc(&input_bwd_gpu_ptr, model_output_template_.get_size_in_bytes());
    variant_pack_bwd_.insert({bwd_input_tensor_attribute, input_bwd_gpu_ptr});

    auto bwd_tensor_attributes = bwd_input_tensor_attribute;

    for (auto layer = layers_.begin(); layer != layers_.end(); layer--)
    {
        const int dist = std::distance(layers_.begin(), layer);

        // 순전파시 입출력 텐서
        const auto& fwd_pair = fwd_input_output_tensor_attribute_pair_[dist];

        bwd_tensor_attributes =
            (*layer)->add_backward_to_graph(backward_graph_, bwd_tensor_attributes, fwd_pair.first, fwd_pair.second);

        if (layer == layers_.begin())
            bwd_output_tensor_attributes =
                backward_graph_->tensor(bwd_tensor_attributes->set_output(true).set_is_virtual(false));
    }

    cudaMalloc(&output_bwd_gpu_ptr, model_input_template_.get_size_in_bytes());
    variant_pack_bwd_.insert({bwd_output_tensor_attributes, output_bwd_gpu_ptr});
}

Tensor Sequential::predict(const Tensor& x_input)
{
}

}  // namespace model
}  // namespace ushionn