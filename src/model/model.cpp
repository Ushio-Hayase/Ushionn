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

    std::vector<int64_t> input_shape(input_shape_template.get_shape());

    intermediate_tensors_fwd_.resize(layers_.size());

    // 순전파, 역전파 출력 텐서 속성
    std::shared_ptr<fe::graph::Tensor_attributes> output_tensor_attributes;
    std::shared_ptr<fe::graph::Tensor_attributes> bwd_output_tensor_attributes;

    if (!fwd_graph_built_)
    {
        auto fwd_tensor_attributes = input_tensor_template_attr_;

        for (auto layer = layers_.begin(); layer != layers_.end(); layer++)
        {
            size_t dist = std::distance(layers_.begin(), layer);

            input_shape = (*layer)->get_output_shape(input_shape);

            intermediate_tensors_fwd_[dist] =
                Tensor(input_shape, data_type_, false, (*layer)->get_name() + "_output_fwd");

            // 순전파 처리
            fwd_tensor_attributes = (*layer)->add_forward_to_graph(forward_graph_, fwd_tensor_attributes,
                                                                   intermediate_tensors_fwd_[dist].get_uid());

            intermediate_tensors_fwd_[dist].allocate_device_memory(intermediate_tensors_fwd_[dist].get_size_in_bytes());
            variant_pack_fwd_[intermediate_tensors_fwd_[dist].get_uid()] =
                intermediate_tensors_fwd_[dist].get_mutable_device_ptr();
        }

        // 마지막 레이어일 때 처리
        output_tensor_attributes = fwd_tensor_attributes;
        output_tensor_attributes->set_output(true)
            .set_is_virtual(false)
            .set_data_type(data_type_)
            .set_name("model_output");

        /* --- 순전파 그래프 빌드 --- */

        bool fwd_build_graph_err_value = build_graph(forward_graph_);

        if (!fwd_build_graph_err_value) return fwd_build_graph_err_value;

        fwd_graph_built_ = true;

        /* ------ */
    }

    if (!bwd_graph_built_)
    {
        std::shared_ptr<fe::graph::Tensor_attributes> bwd_tensor_attributes;

        std::vector<std::shared_ptr<fe::graph::Tensor_attributes>> tmp_bwd_tensor_attributes;

        for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++)
        {
            const int dist = std::distance(layer, layers_.rend());

            // 순전파시 입출력 텐서
            const auto fwd_input =
                intermediate_tensors_fwd_.at(dist - 1).create_graph_tensor_attributes(backward_graph_);
            const auto fwd_output = intermediate_tensors_fwd_.at(dist).create_graph_tensor_attributes(backward_graph_);

            if (layer == layers_.rbegin())
            {
                bwd_tensor_attributes = fwd_output;
                bwd_tensor_attributes->set_is_virtual(false);
            }

            tmp_bwd_tensor_attributes.push_back(bwd_tensor_attributes);

            bwd_tensor_attributes =
                (*layer)->add_backward_to_graph(backward_graph_, bwd_tensor_attributes, fwd_input, fwd_output);
        }

        bwd_tensor_attributes->set_is_virtual(false).set_output(true).set_data_type(data_type_);

        /* --- 역전파 그래프 빌드 --- */

        bool bwd_build_graph_err_value = build_graph(backward_graph_);

        if (!bwd_build_graph_err_value) return bwd_build_graph_err_value;

        /* ------ */

        grad_output_ = std::make_unique<Tensor>(bwd_tensor_attributes->get_dim(), data_type_, false, "grad_output");
    }

    for (const auto& layer : layers_)
    {
        for (auto param : layer->get_parameters())
        {
            USHIONN_ASSERT(param->is_on_device(), "Parameter tensor is not on device");
            variant_pack_fwd_[param->get_uid()] = const_cast<void*>(param->get_device_ptr());
        }
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

    variant_pack_fwd_[input_tensor_template_attr_->get_uid()] = const_cast<void*>(x_input.get_device_ptr());

    cudaMalloc(&workspace_fwd_, workspace_size_fwd_);

    auto execute_err = forward_graph_->execute(*cudnn_handle_, variant_pack_fwd_, workspace_fwd_);

    if (execute_err.is_bad()) USHIONN_LOG_FATAL("Forward graph Executing Error : " + execute_err.get_message());

    std::cout << forward_graph_->print();

    float* ptr = new float[3];
    CUDA_CHECK(cudaMemcpy(ptr, intermediate_tensors_fwd_.back().get_mutable_device_ptr(), 12, cudaMemcpyDeviceToHost));
    std::cout << ptr[0] << ' ' << ptr[1] << ' ' << ptr[2] << '\n';
    delete[] ptr;

    return intermediate_tensors_fwd_.back();
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

    auto build_plans_err = graph->build_plans(*cudnn_handle_, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE);

    if (build_plans_err.is_bad())
    {
        USHIONN_WARN("Graph building plans Error : " + build_plans_err.get_message());
        return false;
    }

    auto get_workspace_size_err = forward_graph_->get_workspace_size(workspace_size_fwd_);

    if (get_workspace_size_err.is_bad())
    {
        USHIONN_WARN("Graph get workspace size Error : " + get_workspace_size_err.get_message());
        return false;
    }

    return true;
}
}  // namespace model
}  // namespace ushionn