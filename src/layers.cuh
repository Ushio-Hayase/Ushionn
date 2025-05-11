#pragma once

#include <memory>
#include <string>

#include "activation.cuh"
#include "tensor.cuh"

#ifdef USE_CUDNN

#include <cudnn_frontend.h>

namespace ushionn
{
class Layer
{
   public:
    Layer(const Layer&) = delete;
    Layer& operator=(const Layer&) = delete;
    virtual ~Layer();

    virtual Tensor Forward(const Tensor& input);
    virtual Tensor Backward(const Tensor& grad_output);

    virtual Tensor Forward(const cudnn_frontend::Tensor& input);
    virtual Tensor Backward(const cudnn_frontend::Tensor& grad_output);

    std::pair<const cudnn_frontend::Tensor&, const cudnn_frontend::Tensor&> Parameters() const;
    std::pair<const cudnn_frontend::Tensor&, const cudnn_frontend::Tensor&> Gradients() const;

    virtual void To(Device);

    virtual bool Save(const std::string&);
    virtual bool Load(const std::string&);

   private:
    std::unique_ptr<cudnn_frontend::Tensor> weights_;
    std::unique_ptr<cudnn_frontend::Tensor> bias_;
    std::unique_ptr<cudnn_frontend::Tensor> weights_grads_;
    std::unique_ptr<cudnn_frontend::Tensor> bias_grads_;
    ActivationType activation_;
    Device device_;
};

class DenseLayer : public Layer
{
    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;

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

    virtual Tensor Forward(const Tensor& input);
    virtual Tensor Backward(const Tensor& grad_output);

    std::pair<const Tensor&, const Tensor&> Parameters() const;
    std::pair<const Tensor&, const Tensor&> Gradients() const;

    virtual void To(Device);

    virtual bool Save(const std::string&);
    virtual bool Load(const std::string&);

   private:
    std::unique_ptr<Tensor> weights_;
    std::unique_ptr<Tensor> bias_;
    std::unique_ptr<Tensor> weights_grads_;
    std::unique_ptr<Tensor> bias_grads_;
    ActivationType activation_;
    Device device_;
};

class DenseLayer : public Layer
{
    Tensor Forward(const Tensor& input) override;
    Tensor Backward(const Tensor& grad_output) override;

    void To(Device device) override;

    bool Save(const std::string& path) override;
    bool Load(const std::string& path) override;
};

}  // namespace ushionn

#endif