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
 * \file tvm/relay/attrs/hashtable.h
 * \brief The set of Relay analysis passes written in C++.
 */
#ifndef TVM_RELAY_ATTRS_HASHTABLE_H_
#define TVM_RELAY_ATTRS_HASHTABLE_H_

#include <tvm/ir/attrs.h>
#include <tvm/relay/base.h>
#include <string>

namespace tvm {
namespace relay {

/*! \brief Attributes used in hash-table operator */
struct HashTableAttrs : public tvm::AttrsNode<HashTableAttrs> {
  DataType key_dtype;
  DataType value_dtype;

  TVM_DECLARE_ATTRS(HashTableAttrs, "relay.attrs.HashTableAttrs") {
    TVM_ATTR_FIELD(key_dtype).set_default(NullValue<DataType>())
      .describe("Key data type.");
    TVM_ATTR_FIELD(value_dtype).set_default(NullValue<DataType>())
      .describe("Value data type.");
  }
};


/*! \brief Attributes used in lookup-table-find operator */
struct LookupTableFindAttrs : public tvm::AttrsNode<LookupTableFindAttrs> {
  DataType key_dtype;
  DataType value_dtype;
  DataType dtype;

  TVM_DECLARE_ATTRS(LookupTableFindAttrs, "relay.attrs.LookupTableFindAttrs") {
    TVM_ATTR_FIELD(key_dtype).set_default(NullValue<DataType>())
      .describe("Key data type.");
    TVM_ATTR_FIELD(value_dtype).set_default(NullValue<DataType>())
      .describe("Value data type.");
    TVM_ATTR_FIELD(dtype).set_default(NullValue<DataType>())
      .describe("Output data type.");
  }
};


/*! \brief Attributes used in lookup-table-import operator */
struct LookupTableImportAttrs : public tvm::AttrsNode<LookupTableImportAttrs> {
  DataType key_dtype;
  DataType value_dtype;

  TVM_DECLARE_ATTRS(LookupTableImportAttrs, "relay.attrs.LookupTableImportAttrs") {
    TVM_ATTR_FIELD(key_dtype).set_default(NullValue<DataType>())
      .describe("Key data type.");
    TVM_ATTR_FIELD(value_dtype).set_default(NullValue<DataType>())
      .describe("Value data type.");
  }
};


/*! \brief Attributes used in initialize-table-from-text-file operator */
struct InitializeTableFromTextFileAttrs : public tvm::AttrsNode<InitializeTableFromTextFileAttrs> {
  int64_t vocab_size;
  int64_t key_index;
  int64_t value_index;
  std::string delim;

  TVM_DECLARE_ATTRS(InitializeTableFromTextFileAttrs,
     "relay.attrs.InitializeTableFromTextFileAttrs") {
    TVM_ATTR_FIELD(vocab_size).set_default(-1)
      .describe("Vocabulary size.");
    TVM_ATTR_FIELD(key_index).set_default(-2)
      .describe("Key index for parsing");
    TVM_ATTR_FIELD(value_index).set_default(-1)
      .describe("Value index for parsing");
    TVM_ATTR_FIELD(delim).set_default(" ")
      .describe("Delim for parsing");
  }
};

}  // namespace relay
}  // namespace tvm
#endif  // TVM_RELAY_ATTRS_HASHTABLE_H_
