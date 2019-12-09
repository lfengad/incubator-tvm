
#ifndef TVM_RELAY_ATTRS_HASHTABLE_H_
#define TVM_RELAY_ATTRS_HASHTABLE_H_

#include <tvm/attrs.h>
#include <tvm/relay/base.h>
#include <string>

namespace tvm{
namespace relay{

/*! \brief Attributes used in hash-table operator */
struct HashTableAttrs : public tvm::AttrsNode<HashTableAttrs> {
  DataType key_dtype;
  DataType value_dtype;
  DataType dtype; 

  TVM_DECLARE_ATTRS(HashTableAttrs, "relay.attrs.HashTableAttrs") {
    TVM_ATTR_FIELD(key_dtype).set_default(NullValue<DataType>())
      .describe("Key data type.");
    TVM_ATTR_FIELD(value_dtype).set_default(NullValue<DataType>())
      .describe("Value data type.");
    TVM_ATTR_FIELD(dtype).set_default(NullValue<DataType>())
      .describe("Hashtable data type.");
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



} // namespace relay
} // namespace tvm
#endif //TVM_RELAY_ATTRS_HASHTABLE_H_
