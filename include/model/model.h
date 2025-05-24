#pragma once

#include <cudnn_frontend.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/tensor.h"
#include "layers/layers.h"

namespace fe = cudnn_frontend;

namespace ushionn
{
namespace model
{

class Sequential
{
   public:
    Sequential()
        : cudnn_handle_(std::make_unique<cudnnHandle_t>()),
          forward_graph_(std::make_shared<fe::graph::Graph>()),
          backward_graph_(std::make_shared<fe::graph::Graph>())
    {
        forward_graph_->set_dynamic_shape_enabled(true).set_compute_data_type(data_type_);
        backward_graph_->set_dynamic_shape_enabled(true).set_compute_data_type(data_type_);
        cudnnCreate(cudnn_handle_.get());

    };  // cuDNN 핸들 주입
    virtual ~Sequential() = default;

    void add_layer(std::unique_ptr<nn::Layer> layer);

    // 전체 모델 그래프 빌드 (순전파 및 역전파)
    // input_shape_template: 모델 입력의 shape, data_type 등을 가진 템플릿 Tensor
    bool build_model_graph(const Tensor& input_shape_template, bool training = true);

    // 학습 (단일 배치)
    // train_step 내에서 variant_pack 구성 및 execute 호출
    /*float train_step(const Tensor& x_batch, const Tensor& y_batch, Optimizer& optimizer,
                     Loss& loss_fn);  // 옵티마이저, 손실함수는 현재 CPU 기반으로 가정
*/
    // 추론
    Tensor predict(const Tensor& x_input);

   private:
    // 내부 헬퍼
    void allocate_workspace(std::shared_ptr<fe::graph::Graph>& graph, void*& workspace_ptr, size_t& workspace_size);

    bool build_graph(std::shared_ptr<fe::graph::Graph> graph);

   private:
    std::unique_ptr<cudnnHandle_t> cudnn_handle_;
    std::vector<std::unique_ptr<nn::Layer>> layers_;

    /* 순전파 변수 */
    std::shared_ptr<fe::graph::Graph> forward_graph_;

    std::vector<Tensor> intermediate_tensors_fwd_;  // 각 레이어의 출력을 담을 실제 Tensor 객체들

    std::unordered_map<int64_t, void*> variant_pack_fwd_;

    /* 역전파 변수 */

    std::shared_ptr<fe::graph::Graph> backward_graph_;

    std::unordered_map<int64_t, void*> variant_pack_bwd_;

    void* workspace_fwd_ = nullptr;
    int64_t workspace_size_fwd_ = 0;

    void* workspace_bwd_ = nullptr;
    int64_t workspace_size_bwd_ = 0;

    std::shared_ptr<fe::graph::Tensor_attributes> input_tensor_template_attr_;
    std::unique_ptr<Tensor> grad_output_;

    bool fwd_graph_built_ = false;
    bool bwd_graph_built_ = true;

    fe::DataType_t data_type_ = fe::DataType_t::FLOAT;
};
}  // namespace model
}  // namespace ushionn