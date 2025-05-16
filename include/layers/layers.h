#pragma once

#include <cudnn_frontend.h>

#include <memory>  // for std::shared_ptr
#include <string>
#include <vector>

#include "core/tensor.h"

namespace ushionn
{
class Layer
{
   public:
    Layer(std::string name) : name_(std::move(name)) {}
    virtual ~Layer() = default;

    const std::string& get_name() const { return name_; }

    // --- 핵심 메소드: 그래프 구성 ---
    // 이 레이어의 forward 연산을 주어진 Graph에 추가하고, 출력 Tensor의 "정의"를 반환.
    // input_tensor_graph_ref는 이전 레이어의 출력이거나 모델의 입력에 해당하는
    // 이미 Graph에 정의된 cudnn_frontend::graph::Tensor 객체에 대한 참조.
    virtual std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> add_forward_to_graph(
        cudnnHandle_t cudnn_handle,  // cuDNN 핸들
        std::shared_ptr<cudnn_frontend::graph::Graph>& graph,
        const std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& input_tensor_graph_ref,
        // 필요시 추가 인자 (예: 학습 모드 여부)
        Tensor& actual_output_tensor  // 이 레이어의 실제 출력을 담을 (메모리 할당된) Tensor 객체 (ID 설정용)
        ) = 0;

    // 역전파 그래프 구성 (유사한 방식)
    virtual std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> add_backward_to_graph(
        cudnnHandle_t cudnn_handle, std::shared_ptr<cudnn_frontend::graph::Graph>& graph,
        const std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& output_grad_graph_ref,
        const std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>& fwd_input_graph_ref,  // 순전파 시의 입력
        const std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>&
            fwd_output_graph_ref,         // 순전파 시의 출력 (필요시)
        Tensor& actual_input_grad_tensor  // 이 레이어의 실제 입력 그래디언트를 담을 Tensor 객체
        ) = 0;

    // --- 파라미터 관리 (기존과 유사) ---
    // 학습 가능한 파라미터 (가중치, 편향) Tensor 객체들을 반환.
    // 이 Tensor들은 실제 메모리를 가지고 있어야 하며, 그래프 구성 시 사용됨.
    virtual std::vector<Tensor*> get_parameters() { return {}; }
    virtual std::vector<Tensor*> get_gradients() { return {}; }         // 파라미터에 대한 그래디언트
    virtual void initialize_parameters(unsigned long long seed = 0) {}  // 파라미터 초기화

   protected:
    std::string name_;

    // 파라미터로 사용될 Tensor 객체들은 각 구체 레이어가 멤버로 소유.
    // 예: Tensor weights_; Tensor biases_;
    // 예: Tensor grad_weights_; Tensor grad_biases_;
};
}  // namespace ushionn