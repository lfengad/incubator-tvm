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
 * \file Use standard C library call.
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dlpack/dlpack.h>
#include <algorithm>
#include <vector>
#include "hashtable.h"


namespace tvm {
namespace contrib {

using namespace runtime;

// Argsort implemented C library sort for nms.
// Return indices of sorted tensor.
// By default, the last axis will be used to sort.
// sort_num specify the number of elements to be sorted.
// If input tensor has dimension (d0, d1, ..., d(k-1), dk, d(k+1), ..., d(n-1))
// and sort axis is dk. sort_num should have dimension of
// (d1, d2, ..., d(k-1), d(k+1), ..., dn).
TVM_REGISTER_GLOBAL("tvm.contrib.hashtable_handler.create")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor *output = args[0];
  //DLTensor *key_example = args[1];
  //DLTensor *value_example = args[2];
  auto table_ptr = static_cast<void**>(output->data); 
  if(table_ptr[0] == nullptr) {
   //auto key_dtype = key_example->dtype;
   //auto value_dtype = value_example->dtype;
   //auto key_dtype_str = TVMType2String(key_dtype);
   //auto value_dtype_str = TVMType2String(value_dtype);
   std::string key_dtype_str = args[1];  
   std::string value_dtype_str = args[2]; 
   auto key_dtype = String2TVMType(key_dtype_str);
   auto value_dtype = String2TVMType(value_dtype_str);
   
   if (key_dtype_str == "float32") {
    if (value_dtype_str == "int32") {
      table_ptr[0] = (void*)(new HashTable<float, int32_t>(key_dtype, value_dtype));
    } else if (value_dtype_str == "int64") {
      table_ptr[0] = (void*)new HashTable<float, int64_t>(key_dtype, value_dtype);
    } else if (value_dtype_str == "float32") {
      table_ptr[0] = (void*)new HashTable<float, float>(key_dtype, value_dtype);
    } else if (value_dtype_str == "float64") {
      table_ptr[0] = (void*)new HashTable<float, double>(key_dtype, value_dtype);
    } else {
      LOG(FATAL) << "Unsupported value dtype: " << value_dtype_str;
    }
  } else if (key_dtype_str == "float64") {
    if (value_dtype_str == "int32") {
      table_ptr[0] = (void*)new HashTable<double, int32_t>(key_dtype, value_dtype);
    } else if (value_dtype_str == "int64") {
      table_ptr[0] = (void*)new HashTable<double, int64_t>(key_dtype, value_dtype);
    } else if (value_dtype_str == "float32") {
      table_ptr[0] = (void*)new HashTable<double, float>(key_dtype, value_dtype);
    } else if (value_dtype_str == "float64") {
      table_ptr[0] = (void*)new HashTable<double, double>(key_dtype, value_dtype);
    } else {
      LOG(FATAL) << "Unsupported value dtype: " << value_dtype_str;
    }
  } else if (key_dtype_str == "int32") {
    if (value_dtype_str == "int32") {
      table_ptr[0] = (void*)new HashTable<int32_t, int32_t>(key_dtype, value_dtype);
    } else if (value_dtype_str == "int64") {
      table_ptr[0] = (void*)new HashTable<int32_t, int64_t>(key_dtype, value_dtype);
    } else if (value_dtype_str == "float32") {
      table_ptr[0] = (void*)new HashTable<int32_t, float>(key_dtype, value_dtype);
    } else if (value_dtype_str == "float64") {
      table_ptr[0] = (void*)new HashTable<int32_t, double>(key_dtype, value_dtype);
    } else {
      LOG(FATAL) << "Unsupported value dtype: " << value_dtype_str;
    }
  } else if (key_dtype_str == "int64") {
    if (value_dtype_str == "int32") {
      table_ptr[0] = (void*)new HashTable<int64_t, int32_t>(key_dtype, value_dtype);
    } else if (value_dtype_str == "int64") {
      table_ptr[0] = (void*)new HashTable<int64_t, int64_t>(key_dtype, value_dtype);
    } else if (value_dtype_str == "float32") {
      table_ptr[0] = (void*)new HashTable<int64_t, float>(key_dtype, value_dtype);
    } else if (value_dtype_str == "float64") {
      table_ptr[0] = (void*)new HashTable<int64_t, double>(key_dtype, value_dtype);
    } else {
      LOG(FATAL) << "Unsupported value dtype: " << value_dtype_str;
    }
  } else {
    LOG(FATAL) << "Unsupported key dtype: " << key_dtype_str;
  }
  }
});




// Argsort implemented C library sort for nms.
// Return indices of sorted tensor.
// By default, the last axis will be used to sort.
// sort_num specify the number of elements to be sorted.
// If input tensor has dimension (d0, d1, ..., d(k-1), dk, d(k+1), ..., d(n-1))
// and sort axis is dk. sort_num should have dimension of
// (d1, d2, ..., d(k-1), d(k+1), ..., dn).
TVM_REGISTER_GLOBAL("tvm.contrib.hashtable_handler.find")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor *table_tensor = args[0];
  DLTensor *keys = args[1];
  DLTensor *default_value = args[2];
  DLTensor *values = args[3];
   
  printf("here execute find\n");
  auto table_ptr = static_cast<BaseTable **>(table_tensor->data);

  CHECK(table_ptr[0] != NULL) << "Hashtable pointer is NULL and  not prepared.";
  
  auto table_interface = static_cast<BaseTable *>(table_ptr[0]); 
  CHECK(table_interface->KeyDtype().code == keys->dtype.code) << "Key data types not match.";
  CHECK(table_interface->ValueDtype().code == values->dtype.code) << "Value data types not match.";
  CHECK(table_interface->ValueDtype().code == default_value->dtype.code) << "Default value data types not match.";

  table_interface->DoFind(keys, values, default_value);
});



// Argsort implemented C library sort for nms.
// Return indices of sorted tensor.
// By default, the last axis will be used to sort.
// sort_num specify the number of elements to be sorted.
// If input tensor has dimension (d0, d1, ..., d(k-1), dk, d(k+1), ..., d(n-1))
// and sort axis is dk. sort_num should have dimension of
// (d1, d2, ..., d(k-1), d(k+1), ..., dn).
TVM_REGISTER_GLOBAL("tvm.contrib.hashtable_handler.init")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor *table_tensor = args[0];
  DLTensor *keys = args[1];
  DLTensor *values = args[2];
  DLTensor *fakesymbol = args[3];
   
  printf("here execute init\n");
  auto table_ptr = static_cast<BaseTable **>(table_tensor->data); 

  CHECK(table_ptr[0] != NULL) << "Hashtable pointer is NULL and  not prepared.";

  auto table_interface = table_ptr[0]; 

  if (!table_interface->is_initialized()) {
    CHECK(table_interface->KeyDtype().code == keys->dtype.code) << "Key data types not match.";
    CHECK(table_interface->ValueDtype().code == values->dtype.code) << "Value data types not match.";
    table_interface->DoInsert(keys, values);
  }
  auto symbol = static_cast<int32_t *>(fakesymbol->data); 
  symbol[0] = 1;
});





}// contrib namespace
}//tvm namespace











