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


namespace tvm {
namespace contrib {

using namespace runtime;


inline size_t GetSize(int dim, int64_t* shape)
{
    size_t size = 1;
    for(uint32_t i=0; i<dim; i++) {
       size *= shape[i];  
    }
    return size;
}


inline std::vector<size_t> GetStride(int dim, int64_t* shape)
{
    sts::vector<size_t> strides(dim) 
    size_t product = 1;
    for(uint32_t i=dim-1; i >= 0; --i) {
       strides[i] = product;
       product *= shape[i];  
    }
    return strides;   
}

inline size_t LinearSubIndexToFullIndex(
    size_t output_index, std::vector<int32_t>& dim_list,
    int64_t* input_shape,
    std::vector<size_t>& strides) {
  
    size_t result = 0;
    size_t quotient = output_index;
    for (int32_t i = dim_list.size() - 1; i >= 0; --i) {
      int32_t dim = dim_list[i];
      size_t dim_value = quotient % input_shape[dim];
      quotient = quotient / input_shape[dim];
      result += strides[dim] * dim_value;
    }
    return result;
}

std::string* StringJoin(std::vector<std::string*>&sub_strings, std::string* delim) {
   std::string * joined_string = new std::string(*sub_strings[0]);
   size_t len = joined_string->size();
   size_t len_b = delim->size();
   for (uint32_t i = 1; i<sub_strings.size(); ++i) {
   std::string* sub_string = sub_strings[i];
   size_t len_a = sub_string->size();
   joined_string->resize(len + len_a + len_b); 
   char* data_ = joined_string->data();
   memcpy(data_+len, delim->data(), len_b);
   memcpy(data_+len+lenb, sub_string->data(), len_b);
   len += lenb+lena;
   }
   return joined_string;  
}

// Argsort implemented C library sort for nms.
// Return indices of sorted tensor.
// By default, the last axis will be used to sort.
// sort_num specify the number of elements to be sorted.
// If input tensor has dimension (d0, d1, ..., d(k-1), dk, d(k+1), ..., d(n-1))
// and sort axis is dk. sort_num should have dimension of
// (d1, d2, ..., d(k-1), d(k+1), ..., dn).
TVM_REGISTER_GLOBAL("tvm.contrib.strings.reducejoin")
.set_body([](TVMArgs args, TVMRetValue *ret) {
  DLTensor *string_tensor = args[0];
  DLTensor *reduction_indice_tensor = args[1];
  bool keep_dims = args[2];
  std::string separator = args[3];
  DLTensor *output_tensor = args[4];
   
  auto strings = static_cast<std::string**>(string_tensor->data); 
  auto out_strings = static_cast<std::string**>(output_tensor->data); 
  auto reduction_indices = static_cast<int32_t *>(reduction_indice_tensor->data);

  int reduction_dim = reduction_indice_tensor->ndim;
  int string_dim = string_tensor->ndim;

  int64_t* reduction_shape = reduction_indice_tensor->shape;   
  int64_t* string_shape = string_tensor->shape; 

  size_t reduction_size = GetSize(reduction_dim, reduction_shape); 
  size_t output_size = GetSize(output_tensor->ndim, output_tensor->shape); 
  std::vector<size_t> strides = GetStride(string_dim, string_shape);

  std::vector<bool> is_reduced(string_dim, false);
  std::vector<int32_t> reduced_indices(reduction_size);


  size_t reduction_iter_size = 1;

  for(uint32_t i=0; i<reduction_size; i++) {
    int32_t indice_tmp = reduction_indices[i];
    if(indice_tmp<0)
      indice_tmp = string_dim - indice_tmp;
    reduced_indices[reduction_szie-1-i] = indice_tmp;
    is_reduced[indice_tmp] = true;
    reduction_iter_size *= string_shape[indice_tmp];
  }

  std::vector<int32_t> unreduced_indices;
  for(uint32_t i=0; i<string_dim; i++) {
    if(!is_reduced[i])
      unreduced_indices.push_back(i);        
  }
  
  std::vector<std::string*> sub_strings(reduction_iter_size);

  for (size_t idx = 0; idx < output_size; ++idx) {
      size_t output_full_index = LinearSubIndexToFullIndex(
          idx, unreduced_indices, string_shape, strides);
      for (size_t reduction_index = 0; reduction_index < reduction_iter_size;
           ++reduction_index) {
        size_t reduction_full_index = LinearSubIndexToFullIndex(
            reduction_index, reduced_indices, string_shape, strides);
        sub_strings[reduction_index] =
            strings[output_full_index + reduction_full_index];
      }
      out_strings[output_index] =
          StringJoin(sub_strings, &separator);
    }
 
});




}// contrib namespace
}//tvm namespace











