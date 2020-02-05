#ifndef TVM_RELAY_ATTRS_STRING_H_
#define TVM_RELAY_ATTRS_STRING_H_

#include <tvm/ir/attrs.h>
#include <tvm/relay/base.h>
#include <string>

namespace tvm{
namespace relay{

/*! \brief Attributes used in hash-table operator */
struct ReduceJoinAttrs : public tvm::AttrsNode<ReduceJoinAttrs> {
  bool keep_dims;
  std::string separator;

  TVM_DECLARE_ATTRS(ReduceJoinAttrs, "relay.attrs.ReduceJoinAttrs") {
    TVM_ATTR_FIELD(keep_dims).set_default(false)
      .describe("Whether keep reduced dims.");
    TVM_ATTR_FIELD(separator).set_default("")
      .describe("Separator in joined strings.");
  }
};




} // namespace relay
} // namespace tvm
#endif //TVM_RELAY_ATTRS_HASHTABLE_H_
