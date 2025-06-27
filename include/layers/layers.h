#pragma once

#include <memory>  // for std::shared_ptr
#include <random>
#include <string>
#include <vector>

#include "core/tensor.h"

namespace ushionn
{
namespace nn
{

class Layer
{
   public:
    Layer(std::string name) : name_(std::move(name)), data_type_(fe::DataType_t::FLOAT) {}
    Layer(std::string name, fe::DataType_t type) : name_(std::move(name)), data_type_(type) {}
    virtual ~Layer() = default;

    /// @brief 레이어의 이름을 반환
    /// @return 레이어의 이름
    const inline std::string& get_name() const { return name_; }

    // 순전파 그래프 구성 순수 가상 함수
    virtual std::shared_ptr<fe::graph::Tensor_attributes> add_forward_to_graph(
        std::shared_ptr<fe::graph::Graph>& graph,
        const std::shared_ptr<fe::graph::Tensor_attributes>& input_tensor_graph_ref, int64_t output_uid) = 0;

    // 역전파 그래프 구성 순수 가상 함수
    virtual std::shared_ptr<fe::graph::Tensor_attributes> add_backward_to_graph(
        std::shared_ptr<fe::graph::Graph>& graph,
        const std::shared_ptr<fe::graph::Tensor_attributes>& output_grad_graph_ref,
        const std::shared_ptr<fe::graph::Tensor_attributes>& fwd_input_graph_ref,  // 순전파 시의 입력
        const std::shared_ptr<fe::graph::Tensor_attributes>&
            fwd_output_graph_ref) = 0;  // 순전파 시의 출력 (필요시)) = 0;

    /* 파라미터 관리 */
    virtual std::vector<Tensor*> get_parameters() = 0;
    virtual std::vector<Tensor*> get_gradients() = 0;                          // 파라미터에 대한 그래디언트
    virtual void initialize_parameters_norm(unsigned long long seed = 0) = 0;  // 파라미터 초기화

    virtual std::vector<int64_t> get_output_shape(std::vector<int64_t> input_dims) const = 0;

   protected:
    std::string name_;
    fe::DataType_t data_type_ = fe::DataType_t::FLOAT;
};

class DenseLayer : public Layer
{
   public:
    DenseLayer(int64_t batch_size, int64_t input_size, int64_t output_size, std::string name)
        : Layer(name),
          weights_({batch_size, input_size, output_size}, fe::DataType_t::FLOAT, false, name + "_weight"),
          bias_({batch_size, 1, output_size}, fe::DataType_t::FLOAT, false, name + "_bias"),
          weights_grad_({batch_size, input_size, output_size}, fe::DataType_t::FLOAT, false, name + "_weight_grad"),
          bias_grad_({batch_size, 1, output_size}, fe::DataType_t::FLOAT, false, name + "_bias_grad")
    {
    }

    /// @brief 그래프에 순전파 작업을 추가합니다
    /// @param[in] graph 그래프 shared 포인터
    /// @param[in] input_tensor_graph_ref 입력 텐서 속성 shared 포인터
    /// @return 출력 텐서 속성 shared 포인터
    std::shared_ptr<fe::graph::Tensor_attributes> add_forward_to_graph(
        std::shared_ptr<fe::graph::Graph>& graph,
        const std::shared_ptr<fe::graph::Tensor_attributes>& input_tensor_graph_ref, int64_t output_uid) override;

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

    void initialize_parameters_norm(unsigned long long seed = 0);

    std::vector<int64_t> get_output_shape(std::vector<int64_t> input_dims) const override;

   private:
    Tensor weights_;
    Tensor bias_;
    Tensor weights_grad_;
    Tensor bias_grad_;
};
}  // namespace nn

}  // namespace ushionn