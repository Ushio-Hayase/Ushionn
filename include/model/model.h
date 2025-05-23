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
    Sequential() = delete;
    Sequential(cudnnHandle_t cudnn_handle)
        : cudnn_handle_(cudnn_handle), forward_graph_(new fe::graph::Graph()) {};  // cuDNN 핸들 주입
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
    void cuda_malloc_and_variant_pack_insert(
        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*>& variant_pack,
        std::shared_ptr<fe::graph::Tensor_attributes>& key, void* value, size_t size);

    bool build_graph(std::shared_ptr<fe::graph::Graph> graph);

   private:
    cudnnHandle_t cudnn_handle_;
    std::vector<std::unique_ptr<nn::Layer>> layers_;

    /* 순전파 변수 */
    std::shared_ptr<fe::graph::Graph> forward_graph_;

    std::vector<Tensor> intermediate_tensors_fwd_;  // 각 레이어의 출력을 담을 실제 Tensor 객체들

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack_fwd_;

    /* 역전파 변수 */

    std::shared_ptr<fe::graph::Graph> backward_graph_;

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack_bwd_;

    void* workspace_fwd_ = nullptr;
    size_t workspace_size_fwd_ = 0;

    void* workspace_bwd_ = nullptr;
    size_t workspace_size_bwd_ = 0;

    Tensor model_input_template_;
    Tensor model_output_;

    bool fwd_graph_built_ = false;
    bool bwd_graph_built_ = false;
};
}  // namespace model
}  // namespace ushionn