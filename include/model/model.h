#pragma once

#include <cudnn_frontend.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "core/tensor.h"
#include "layers/layers.h"

namespace ushionn
{
namespace model
{

class Sequential
{
   public:
    Sequential(cudnnHandle_t cudnn_handle) : cudnn_handle_(cudnn_handle) {};  // cuDNN 핸들 주입
    ~Sequential();

    void add_layer(std::unique_ptr<Layer> layer);

    // 전체 모델 그래프 빌드 (순전파 및 역전파)
    // input_shape_template: 모델 입력의 shape, data_type 등을 가진 템플릿 Tensor
    void build_model_graph(const Tensor& input_shape_template, bool build_for_training = true);

    // 학습 (단일 배치)
    // train_step 내에서 variant_pack 구성 및 execute 호출
    /*float train_step(const Tensor& x_batch, const Tensor& y_batch, Optimizer& optimizer,
                     Loss& loss_fn);  // 옵티마이저, 손실함수는 현재 CPU 기반으로 가정
*/
    // 추론
    Tensor predict(const Tensor& x_input);

   private:
    cudnnHandle_t cudnn_handle_;
    std::vector<std::unique_ptr<Layer>> layers_;

    // --- Forward Pass Graph ---
    std::shared_ptr<cudnn_frontend::graph::Graph> forward_graph_;
    // std::shared_ptr<cudnn_frontend::graph::Execution_plan> forward_plan_; // 캐싱된 실행 계획
    Tensor model_input_template_;                          // 그래프 빌드 시 사용된 입력 템플릿
    std::vector<Tensor> intermediate_tensors_fwd_;         // 각 레이어의 출력을 담을 실제 Tensor 객체들
    std::unordered_map<int64_t, void*> variant_pack_fwd_;  // 실행 시 사용할 variant pack
    void* workspace_fwd_ = nullptr;
    size_t workspace_size_fwd_ = 0;

    // --- Backward Pass Graph (선택적, 유사하게 구성) ---
    // std::shared_ptr<cudnn_frontend::graph::Graph> backward_graph_;
    // ...

    bool graph_built_ = false;

    // 내부 헬퍼
    void allocate_workspace(std::shared_ptr<cudnn_frontend::graph::Graph>& graph, void*& workspace_ptr,
                            size_t& workspace_size);
    void prepare_variant_pack_forward(const Tensor& actual_input);
};
}  // namespace model
}  // namespace ushionn