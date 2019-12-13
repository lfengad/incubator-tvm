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
#include <tvm/relay/attrs/hashtable.h>
#include <tvm/runtime/packed_func.h>

namespace tvm {
namespace relay {


using namespace runtime;


TVM_REGISTER_NODE_TYPE(HashTableAttrs);

bool HashTableRel(const Array<Type>& types,
            int num_inputs,
            const Attrs& attrs,
            const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 1);
  //const HashTableAttrs* param = attrs.as<HashTableAttrs>();

  // assign output type
  std::vector<IndexExpr> oshape({1});
  DataType table_type = TVMType2Type(String2TVMType("custom[hashtable]64"));
  reporter->Assign(types[0], TensorTypeNode::make(oshape, table_type));
  return true;
}

Expr MakeHashTable(
             DataType key_dtype,
             DataType value_dtype,
             DataType dtype) {
  auto attrs = make_node<HashTableAttrs>();
  attrs->key_dtype = key_dtype;
  attrs->value_dtype = value_dtype;
  attrs->dtype = dtype; 
  static const Op& op = Op::Get("contrib.hash_table");
  return CallNode::make(op, {}, Attrs(attrs), {});
}

TVM_REGISTER_API("relay.op.contrib._make.hash_table")
.set_body_typed(MakeHashTable);


RELAY_REGISTER_OP("contrib.hash_table")
.describe(R"doc(Hash Table reference. No input needed. Returns a Hash Table object, which can
be passed to the lookup_table_* style table operations. The two attributes define the key and 
value data types in the hash table.)doc" TVM_ADD_FILELINE)
.set_attrs_type<HashTableAttrs>()
.set_num_inputs(0)
.set_support_level(5)
.add_type_rel("HashTable", HashTableRel);



TVM_REGISTER_NODE_TYPE(LookupTableFindAttrs);

bool LookupTableFindRel(const Array<Type>& types,
            int num_inputs,
            const Attrs& attrs,
            const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4);
  const auto* table_reference = types[0].as<TensorTypeNode>();
  CHECK(table_reference);
  const auto* key_to_check = types[1].as<TensorTypeNode>();
  CHECK(key_to_check);
  const auto* default_value = types[2].as<TensorTypeNode>();
  CHECK(default_value);
  const LookupTableFindAttrs* param =
    attrs.as<LookupTableFindAttrs>();
  const auto& dshape = key_to_check->shape;
  CHECK_EQ(param->key_dtype, key_to_check->dtype);
  //CHECK_EQ(param->value_dtype, default_value->dtype);

  reporter->Assign(types[3], TensorTypeNode::make(dshape, param->value_dtype));
  return true;

}

Expr MakeLookupTableFind(Expr table_reference,
                        Expr key_to_check,
                        Expr default_value,
                        DataType key_dtype,
                        DataType value_dtype,
                        DataType dtype) 
{
  auto attrs = make_node<LookupTableFindAttrs>();
  attrs->key_dtype = key_dtype;
  attrs->value_dtype = value_dtype;
  attrs->dtype = dtype;
  static const Op& op = Op::Get("contrib.lookup_table_find");
  return CallNode::make(op, {table_reference, key_to_check, default_value}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.contrib._make.lookup_table_find")
.set_body_typed(MakeLookupTableFind);


RELAY_REGISTER_OP("contrib.lookup_table_find")
.describe(R"doc(Find the value for the key in an given Hash Table. Inputs are [Hash Table, 
Keys to be checked, Default Value if not found]. Returns a set of corresponding values to 
the checked keys, The two attributes define the key and value data types in the hash 
table.)doc" TVM_ADD_FILELINE)
.set_attrs_type<LookupTableFindAttrs>()
.set_num_inputs(3)
.add_argument("table_reference", "Tensor", "Referred hashtable.")
.add_argument("key_to_check", "Tensor", "Input key to check.")
.add_argument("default_value", "Tensor", "Default value for missing entry.")
.set_support_level(5)
.add_type_rel("LookupTableFind", LookupTableFindRel);



TVM_REGISTER_NODE_TYPE(LookupTableImportAttrs);

bool LookupTableImportRel(const Array<Type>& types,
            int num_inputs,
            const Attrs& attrs,
            const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 4);
  const auto* table_reference = types[0].as<TensorTypeNode>();
  CHECK(table_reference);
  const auto* keys = types[1].as<TensorTypeNode>();
  CHECK(keys);
  const auto* values = types[2].as<TensorTypeNode>();
  CHECK(values);
  const LookupTableImportAttrs* param =
    attrs.as<LookupTableImportAttrs>();
  //CHECK_EQ(keys->shape, values->shape);
  CHECK_EQ(param->key_dtype, keys->dtype);
  CHECK_EQ(param->value_dtype, values->dtype);

  std::vector<IndexExpr> oshape({1});
  DataType fake_type = TVMType2Type(String2TVMType("int32"));
  reporter->Assign(types[3], TensorTypeNode::make(oshape, fake_type));
  return true;

}

Expr MakeLookupTableImport(Expr table_reference,
                        Expr keys,
                        Expr values,
                        DataType key_dtype,
                        DataType value_dtype) {
  auto attrs = make_node<LookupTableImportAttrs>();
  attrs->key_dtype = key_dtype;
  attrs->value_dtype = value_dtype;
  static const Op& op = Op::Get("contrib.lookup_table_import");
  return CallNode::make(op, {table_reference, keys, values}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.contrib._make.lookup_table_import")
.set_body_typed(MakeLookupTableImport);


RELAY_REGISTER_OP("contrib.lookup_table_import")
.describe(R"doc(Import the given value and key pairs to initialize the given Hash Table. 
Inputs are [Hash Table, Keys, Values]. No Returns. The two attributes define the key and 
value data types in the hash table.)doc" TVM_ADD_FILELINE
)
.set_attrs_type<LookupTableImportAttrs>()
.set_num_inputs(3)
.add_argument("table_reference", "Tensor", "Referred hashtable.")
.add_argument("keys", "Tensor", "Keys for initilize.")
.add_argument("values", "Tensor", "Values for initialize.")
.set_support_level(5)
.add_type_rel("LookupTableImport", LookupTableImportRel);


TVM_REGISTER_NODE_TYPE(InitializeTableFromTextFileAttrs);

bool InitializeTableFromTextFileRel(const Array<Type>& types,
            int num_inputs,
            const Attrs& attrs,
            const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 3);
  const auto* table_reference = types[0].as<TensorTypeNode>();
  CHECK(table_reference);
  const auto* files = types[1].as<TensorTypeNode>();
  CHECK(files);
  const InitializeTableFromTextFileAttrs* param =
    attrs.as<InitializeTableFromTextFileAttrs>();
  //CHECK_EQ(keys->shape, values->shape);
  std::vector<IndexExpr> oshape({1});
  DataType fake_type = TVMType2Type(String2TVMType("int32"));
  reporter->Assign(types[2], TensorTypeNode::make(oshape, fake_type));
  return true;

}

Expr MakeInitializeTableFromTextFile(Expr table_reference,
                        Expr files,
                        int64_t vocab_size,
                        int64_t key_index,
                        int64_t value_index,
                        std::string delim) {
  auto attrs = make_node<InitializeTableFromTextFileAttrs>();
  attrs->vocab_size = vocab_size;
  attrs->key_index = key_index;
  attrs->value_index = value_index;
  attrs->delim = delim;
  static const Op& op = Op::Get("contrib.initialize_table_from_text_file");
  return CallNode::make(op, {table_reference, files}, Attrs(attrs), {});
}


TVM_REGISTER_API("relay.op.contrib._make.initialize_table_from_text_file")
.set_body_typed(MakeInitializeTableFromTextFile);


RELAY_REGISTER_OP("contrib.initialize_table_from_text_file")
.describe(R"doc(Import the given value and key pairs to initialize the given Hash Table. 
Inputs are [Hash Table, Keys, Values]. No Returns. The two attributes define the key and 
value data types in the hash table.)doc" TVM_ADD_FILELINE
)
.set_attrs_type<InitializeTableFromTextFileAttrs>()
.set_num_inputs(2)
.add_argument("table_reference", "Tensor", "Referred hashtable.")
.add_argument("files", "Tensor", "file for initilize.")
.set_support_level(5)
.add_type_rel("InitializeTableFromTextFile", InitializeTableFromTextFileRel);



}  // namespace relay
}  // namespace tvm
