#include "model/model.h"

#include <algorithm>
namespace ushionn
{
namespace model
{
void Sequential::add_layer(std::unique_ptr<nn::Layer> layer)
{
    layers_.push_back(std::move(layer));
}

bool Sequential::build_model_graph(const Tensor& input_shape_template, bool training)
{
    USHIONN_ASSERT(!layers_.empty(), "No layers in the model");

    auto fwd_input_tensor_attributes = input_shape_template.create_graph_tensor_attributes(forward_graph_, true, false);

    // 순전파시 입출력 텐서 속성을 기록할 변수
    std::vector<std::pair<std::shared_ptr<fe::graph::Tensor_attributes>, std::shared_ptr<fe::graph::Tensor_attributes>>>
        fwd_input_output_tensor_attribute_pair_;

    // 순전파, 역전파 출력 텐서 속성
    std::shared_ptr<fe::graph::Tensor_attributes> output_tensor_attributes;
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_output_tensor_attributes;

    if (!fwd_graph_built_)
    {
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
            fwd_input_output_tensor_attribute_pair_.push_back(std::move(input_output_pair));

            // 마지막 레이어일때 처리
            if (layer == layers_.end() - 1)
            {
                output_tensor_attributes = fwd_tensor_attributes;
                output_tensor_attributes->set_output(true).set_is_virtual(false).set_data_type(data_type_);
            }
        }

        Tensor model_output_ = Tensor(output_tensor_attributes->get_dim(), data_type_, false, "model_output");

        /* --- 순전파 그래프 빌드 --- */

        bool fwd_build_graph_err_value = build_graph(forward_graph_);

        if (!fwd_build_graph_err_value) return fwd_build_graph_err_value;

        /* ------ */
    }

    if (!bwd_graph_built_)
    {
        auto bwd_input_tensor_attributes = backward_graph_->tensor(output_tensor_attributes->set_is_virtual(false));

        auto bwd_tensor_attributes = bwd_input_tensor_attributes;

        for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++)
        {
            const int dist = std::distance(layer, layers_.rend());

            // 순전파시 입출력 텐서
            const auto& fwd_pair = fwd_input_output_tensor_attribute_pair_.at(dist - 1);

            bwd_tensor_attributes = (*layer)->add_backward_to_graph(backward_graph_, bwd_tensor_attributes,
                                                                    fwd_pair.first, fwd_pair.second);

            if (layer == layers_.rbegin())
            {
                bwd_output_tensor_attributes = bwd_tensor_attributes;
                bwd_output_tensor_attributes->set_output(true).set_is_virtual(false);
            }
        }

        /* --- 역전파 그래프 빌드 --- */

        bool bwd_build_graph_err_value = build_graph(backward_graph_);

        if (!bwd_build_graph_err_value) return bwd_build_graph_err_value;

        /* ------ */
    }
    return true;
}

Tensor Sequential::predict(const Tensor& x_input)
{
    USHIONN_ASSERT(!layers_.empty(), "No layers in the model");
    USHIONN_ASSERT(x_input.get_data_location() == ushionn::Tensor::DataLocation::DEVICE,
                   "Input tensor must already be on the DEVICE");

    USHIONN_ASSERT(fwd_graph_built_,
                   "Forward Graph is not built. Call build_model_graph() with appropriate input template first.");

    USHIONN_ASSERT(bwd_graph_built_,
                   "Backward Graph is not built. Call build_model_graph() with appropriate input template first.");

    variant_pack_fwd_.clear();
    variant_pack_bwd_.clear();

    auto input_tensor_attribute = x_input.create_graph_tensor_attributes(forward_graph_, true, false);

    variant_pack_fwd_[input_tensor_attribute] = const_cast<void*>(x_input.get_device_ptr());

    for (const auto& layer : layers_)
    {
        for (auto param : layer->get_parameters())
        {
            USHIONN_ASSERT(param->is_on_device(), "Parameter tensor is not on device");
            variant_pack_fwd_[input_tensor_attribute] = const_cast<void*>(param->get_device_ptr());
        }
    }

    model_output_.allocate_device_memory(model_output_.get_size_in_bytes());

    auto output_tensor_attribute = model_output_.create_graph_tensor_attributes(forward_graph_);

    variant_pack_fwd_[output_tensor_attribute] = const_cast<void*>(model_output_.get_device_ptr());

    workspace_size_fwd_ = forward_graph_->get_workspace_size();

    cudaMalloc(&workspace_fwd_, workspace_size_fwd_);

    auto execute_err = forward_graph_->execute(*cudnn_handle_, variant_pack_fwd_, workspace_fwd_);

    if (execute_err.is_bad()) USHIONN_LOG_FATAL("Forward graph Executing Error : " + execute_err.get_message());

    return model_output_;
}

bool Sequential::build_graph(std::shared_ptr<fe::graph::Graph> graph)
{
    auto validata_err = graph->validate();

    if (validata_err.is_bad())
    {
        USHIONN_WARN("Graph validate Error : " + validata_err.get_message());
        return false;
    }

    auto build_operation_err = graph->build_operation_graph(*cudnn_handle_);

    if (build_operation_err.is_bad())
    {
        USHIONN_WARN("Graph building operation Error : " + build_operation_err.get_message());
        return false;
    }

    auto execution_plans_err = graph->create_execution_plans({fe::HeurMode_t::A});

    if (execution_plans_err.is_bad())
    {
        USHIONN_WARN("Graph Creating execution plans Error : " + execution_plans_err.get_message());
        return false;
    }

    auto check_support_err = graph->check_support(*cudnn_handle_);

    if (check_support_err.is_bad())
    {
        USHIONN_WARN("Graph checking support Error : " + check_support_err.get_message());
        return false;
    }

    auto build_plans_err = graph->build_plans(*cudnn_handle_, fe::BuildPlanPolicy_t::ALL);

    if (build_plans_err.is_bad())
    {
        USHIONN_WARN("Graph building plans Error : " + build_plans_err.get_message());
        return false;
    }

    return true;
}
}  // namespace model
}  // namespace ushionn