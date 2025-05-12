#pragma once

#include <memory>
#include <string>

#include "activation.cuh"
#include "tensor.cuh"

#ifdef USE_CUDNN

#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

namespace ushionn
{
class Layer
{
   public:
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    virtual ~Layer();

    virtual void Initialize();

    virtual std::shared_ptr<fe::graph::Tensor_attributes> Forward(
        const std::shared_ptr<fe::graph::Tensor_attributes> input, fe::graph::Graph& graph);
    virtual std::shared_ptr<fe::graph::Tensor_attributes> Backward(
        const std::shared_ptr<fe::graph::Tensor_attributes> grad_output, fe::graph::Graph& graph);

    std::pair<const fe::graph::Tensor_attributes&, const fe::graph::Tensor_attributes&> Parameters() const;
    std::pair<const fe::graph::Tensor_attributes&, const fe::graph::Tensor_attributes&> Gradients() const;

    virtual void To(Device);

    virtual bool Save(const std::string&);
    virtual bool Load(const std::string&);

   protected:
    template <typename T>
    void AttributeSetComputeDataType(fe::graph::Attributes<T>& attr);

   protected:
    std::shared_ptr<fe::graph::Tensor_attributes> weights_;
    std::shared_ptr<fe::graph::Tensor_attributes> bias_;
    std::shared_ptr<fe::graph::Tensor_attributes> weights_grads_;
    std::shared_ptr<fe::graph::Tensor_attributes> bias_grads_;
    ActivationType activation_;
    Device device_;
    DataType dtype_;
};

class DenseLayer : public Layer
{
   public:
    void Initialize() override;

    std::shared_ptr<fe::graph::Tensor_attributes> Forward(const std::shared_ptr<fe::graph::Tensor_attributes>,
                                                          fe::graph::Graph& graph) override;
    std::shared_ptr<fe::graph::Tensor_attributes> Backward(
        const std::shared_ptr<fe::graph::Tensor_attributes> grad_output, fe::graph::Graph& graph) override;

    void To(Device device) override;

    bool Save(const std::string& path) override;
    bool Load(const std::string& path) override;
};

}  // namespace ushionn

#else
namespace ushionn
{
class Layer
{
   public:
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    virtual ~Layer();

    virtual graph::Tensor_attributes Forward(const graph::Tensor_attributes& input);
    virtual graph::Tensor_attributes Backward(const graph::Tensor_attributes& grad_output);

    std::pair<const graph::Tensor_attributes&, const graph::Tensor_attributes&> Parameters() const;
    std::pair<const graph::Tensor_attributes&, const graph::Tensor_attributes&> Gradients() const;

    virtual void To(Device);

    virtual bool Save(const std::string&);
    virtual bool Load(const std::string&);

   private:
    std::unique_ptr<graph::Tensor_attributes> weights_;
    std::unique_ptr<graph::Tensor_attributes> bias_;
    std::unique_ptr<graph::Tensor_attributes> weights_grads_;
    std::unique_ptr<graph::Tensor_attributes> bias_grads_;
    ActivationType activation_;
    Device device_;
};

class DenseLayer : public Layer
{
    graph::Tensor_attributes Forward(const graph::Tensor_attributes& input) override;
    graph::Tensor_attributes Backward(const graph::Tensor_attributes& grad_output) override;

    void To(Device device) override;

    bool Save(const std::string& path) override;
    bool Load(const std::string& path) override;
};

}  // namespace ushionn

#endif