#include "layers.cuh"

namespace ushionn
{

#ifdef USE_CUDNN
std::pair<const cudnn_frontend::Tensor&, const cudnn_frontend::Tensor&> Layer::Parameters() const
{
    return {*weights_, *bias_};
}

std::pair<const cudnn_frontend::Tensor&, const cudnn_frontend::Tensor&> Layer::Gradients() const
{
    return {*weights_grads_, *bias_grads_};
}
#else
std::pair<const Tensor&, const Tensor&> Layer::Parameters() const
{
    return {*weights_, *bias_};
}

std::pair<const Tensor&, const Tensor&> Layer::Gradients() const
{
    return {*weights_grads_, *bias_grads_};
}
#endif

}  // namespace ushionn