#include "layers.cuh"

namespace ushionn
{
std::pair<const Tensor&, const Tensor&> Layer::Parameters() const
{
    return {*weights_, *bias_};
}

std::pair<const Tensor&, const Tensor&> Layer::Gradients() const
{
    return {*weights_grads_, *bias_grads_};
}

}  // namespace ushionn