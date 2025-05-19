#include "model/model.h"

namespace ushionn
{
namespace model
{
void Sequential::add_layer(std::unique_ptr<Layer> layer)
{
    layers_.push_back(layer);
}

}  // namespace model
}  // namespace ushionn