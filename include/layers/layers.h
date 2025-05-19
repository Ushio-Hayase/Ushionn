#pragma once

#include <cudnn_frontend.h>

#include <memory>  // for std::shared_ptr
#include <string>
#include <vector>

#include "core/tensor.h"

namespace fe = cudnn_frontend;

namespace ushionn
{

class Layer
{
   public:
    Layer(std::string name) : name_(std::move(name)) {}
    virtual ~Layer() = default;

    const inline std::string& get_name() const { return name_; }

    // --- 핵심 메소드: 그래프 구성 ---
    // 이 레이어의 forward 연산을 주어진 Graph에 추가하고, 출력 Tensor의 "정의"를 반환.
    // input_tensor_graph_ref는 이전 레이어의 출력이거나 모델의 입력에 해당하는
    // 이미 Graph에 정의된 fe::graph::Tensor 객체에 대한 참조.
    virtual std::shared_ptr<fe::graph::Tensor_attributes> add_forward_to_graph(
        std::shared_ptr<fe::graph::Graph>& graph,
        const std::shared_ptr<fe::graph::Tensor_attributes>& input_tensor_graph_ref) = 0;

    // 역전파 그래프 구성 (유사한 방식)
    virtual std::shared_ptr<fe::graph::Tensor_attributes> add_backward_to_graph(
        std::shared_ptr<fe::graph::Graph>& graph,
        const std::shared_ptr<fe::graph::Tensor_attributes>& output_grad_graph_ref,
        const std::shared_ptr<fe::graph::Tensor_attributes>& fwd_input_graph_ref,    // 순전파 시의 입력
        const std::shared_ptr<fe::graph::Tensor_attributes>& fwd_output_graph_ref);  // 순전파 시의 출력 (필요시)) = 0;

    // --- 파라미터 관리 (기존과 유사) ---
    // 학습 가능한 파라미터 (가중치, 편향) Tensor 객체들을 반환.
    // 이 Tensor들은 실제 메모리를 가지고 있어야 하며, 그래프 구성 시 사용됨.
    virtual std::vector<Tensor*> get_parameters() = 0;
    virtual std::vector<Tensor*> get_gradients() = 0;                     // 파라미터에 대한 그래디언트
    virtual void initialize_parameters(unsigned long long seed = 0) = 0;  // 파라미터 초기화

   protected:
    std::string name_;
};

class DenseLayer : public Layer
{
    DenseLayer(int64_t batch_size, int64_t input_size, int64_t output_size, std::string name)
        : Layer(name),
          weights_({batch_size, input_size, output_size}),
          bias_({batch_size, input_size, output_size}),
          weights_grad_({batch_size, input_size, output_size}),
          bias_grad_({batch_size, input_size, output_size})
    {
    }

    /// @brief 그래프에 순전파 작업을 추가합니다
    /// @param[in] graph 그래프 shared 포인터
    /// @param[in] input_tensor_graph_ref 입력 텐서 속성 shared 포인터
    /// @return 출력 텐서 속성 shared 포인터
    std::shared_ptr<fe::graph::Tensor_attributes> add_forward_to_graph(
        std::shared_ptr<fe::graph::Graph>& graph,
        const std::shared_ptr<fe::graph::Tensor_attributes>& input_tensor_graph_ref) override;

    /// @brief 그래프에 역전파 작업을 추가합니다
    /// @param[in] graph 그래프 shared 포인터
    /// @param[in] output_grad_graph_ref 손실함수에 대한 순전파시 출력의 가중치 기울기 텐서 속성 shared 포인터
    /// @param[in] fwd_input_graph_ref 순전파시 입력의 텐서 속성 shared 포인터
    /// @param[in] fwd_output_graph_ref 순전파시 출력의 텐서 속성 shard 포인터
    /// @return 손실함수에 대한 입력의 가중치 기울기 텐서 속성 shared 포인터
    std::shared_ptr<fe::graph::Tensor_attributes> add_backward_to_graph(
        std::shared_ptr<fe::graph::Graph>& graph,
        const std::shared_ptr<fe::graph::Tensor_attributes>& output_grad_graph_ref,
        const std::shared_ptr<fe::graph::Tensor_attributes>& fwd_input_graph_ref,  // 순전파 시의 입력
        const std::shared_ptr<fe::graph::Tensor_attributes>& fwd_output_graph_ref) override;

    /// @brief 가중치와 편향을 가져옵니다
    /// @return 가중치와 편향 배열
    std::vector<Tensor*> get_parameters() override;

    /// @brief 가중치 기울기와 편향 기울기를 가져옵니다
    /// @return 가중치 기울기와 편향 기울기 배열
    std::vector<Tensor*> get_gradients() override;

    void initialize_parameters(unsigned long long seed = 0);

   private:
    Tensor weights_;
    Tensor bias_;
    Tensor weights_grad_;
    Tensor bias_grad_;
};

}  // namespace ushionn