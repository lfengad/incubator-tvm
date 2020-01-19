/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file nms.cc
 * \brief Non-maximum suppression operators
 */
#include <tvm/relay/op.h>
#include <tvm/relay/attrs/string.h>
#include <tvm/runtime/packed_func.h>

namespace tvm {
namespace relay {


using namespace runtime;


TVM_REGISTER_NODE_TYPE(ReduceJoinAttrs);

bool ReduceJoinRel(const Array<Type>& types,
            int num_inputs,
            const Attrs& attrs,
            const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  //const HashTableAttrs* param = attrs.as<HashTableAttrs>();

  // assign output type
  const auto* inputs = types[0].as<TensorTypeNode>();
  CHECK(inputs);
  const auto* reduction_indices = types[1].as<TensorTypeNode>();
  CHECK(reduction_indices);
  const ReduceJoinAttrs* param =
    attrs.as<ReduceJoinAttrs>();
  const auto& string_shape = inputs->shape;
  const auto& string_type = inputs->dtype;
  reporter->Assign(types[2], TensorTypeNode::make(string_shape, string_type));
  return true;
}

Expr MakeReduceJoin(Expr inputs,
             Expr reduction_indices,
             bool keep_dims,
             std::string separator) {
  auto attrs = make_object<ReduceJoinAttrs>();
  attrs->keep_dims = keep_dims;
  attrs->separator = separator;
  static const Op& op = Op::Get("contrib.reduce_join");
  return CallNode::make(op, {inputs, reduction_indices}, Attrs(attrs), {});
}

TVM_REGISTER_GLOBAL("relay.op.contrib._make.reduce_join")
.set_body_typed(MakeReduceJoin);


RELAY_REGISTER_OP("contrib.reduce_join")
.describe(R"doc(ReduceJoin for strings. Inputs is the string type tensor needed to be reduced. Reduction_indices is the indices along the dimension needed to be reduced. Returns a new string type tensor been reduced.)doc" TVM_ADD_FILELINE)
.set_attrs_type<ReduceJoinAttrs>()
.set_num_inputs(2)
.add_argument("inputs", "Tensor", "Input strings.")
.add_argument("reduction_indices", "Tensor", "Indices for reduction.")
.set_support_level(5)
.add_type_rel("ReduceJoin", ReduceJoinRel);



}  // namespace relay
}  // namespace tvm
