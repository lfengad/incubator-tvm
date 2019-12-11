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
 * \file ndarray.cc
 * \brief NDArray container infratructure.
 */
#include <dmlc/logging.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/device_api.h>
#include "runtime_base.h"

extern "C" {
// C-mangled dlpack deleter.
static void TVMNDArrayDLPackDeleter(DLManagedTensor* tensor);
// helper function to get NDArray's type index, only used by ctypes.
TVM_DLL int TVMArrayGetTypeIndex(TVMArrayHandle handle, unsigned* out_tindex);
}

namespace tvm {
namespace runtime {

inline void VerifyDataType(DLDataType dtype) {
  CHECK_GE(dtype.lanes, 1);
  if (dtype.code == kDLFloat) {
    CHECK_EQ(dtype.bits % 8, 0);
  } else {
    // allow uint1 as a special flag for bool.
    if (dtype.bits == 1 && dtype.code == kDLUInt) return;
    CHECK_EQ(dtype.bits % 8, 0);
  }
  CHECK_EQ(dtype.bits & (dtype.bits - 1), 0);
}

inline size_t GetDataAlignment(const DLTensor& arr) {
  size_t align = (arr.dtype.bits / 8) * arr.dtype.lanes;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}

struct NDArray::Internal {
  // Default deleter for the container
  static void DefaultDeleter(Object* ptr_obj) {
    auto* ptr = static_cast<NDArray::Container*>(ptr_obj);
    if (ptr->manager_ctx != nullptr) {
      static_cast<NDArray::Container*>(ptr->manager_ctx)->DecRef();
    } else if (ptr->dl_tensor.data != nullptr) {
      tvm::runtime::DeviceAPI::Get(ptr->dl_tensor.ctx)->FreeDataSpace(
          ptr->dl_tensor.ctx, ptr->dl_tensor.data);
    }
    delete ptr;
  }
  // Deleter for NDArray converted from DLPack
  // This is used from data which is passed from external DLPack(DLManagedTensor)
  // that are not allocated inside of TVM.
  // This enables us to create NDArray from memory allocated by other
  // frameworks that are DLPack compatible
  static void DLPackDeleter(Object* ptr_obj) {
    auto* ptr = static_cast<NDArray::Container*>(ptr_obj);
    DLManagedTensor* tensor = static_cast<DLManagedTensor*>(ptr->manager_ctx);
    if (tensor->deleter != nullptr) {
      (*tensor->deleter)(tensor);
    }
    delete ptr;
  }
  // Local create function which allocates tensor metadata
  // but does not allocate space for the data.
  static NDArray Create(std::vector<int64_t> shape,
                        DLDataType dtype,
                        DLContext ctx) {
    VerifyDataType(dtype);

    // critical zone: construct header
    NDArray::Container* data = new NDArray::Container();
    data->SetDeleter(DefaultDeleter);

    // RAII now in effect
    NDArray ret(GetObjectPtr<Object>(data));
    // setup shape
    data->shape_ = std::move(shape);
    data->dl_tensor.shape = dmlc::BeginPtr(data->shape_);
    data->dl_tensor.ndim = static_cast<int>(data->shape_.size());
    // setup dtype
    data->dl_tensor.dtype = dtype;
    // setup ctx
    data->dl_tensor.ctx = ctx;
    return ret;
  }
  // Implementation of API function
  static DLTensor* MoveToFFIHandle(NDArray arr) {
    DLTensor* handle = NDArray::FFIGetHandle(arr);
    ObjectRef::FFIClearAfterMove(&arr);
    return handle;
  }
  static void FFIDecRef(TVMArrayHandle tensor) {
    NDArray::FFIDecRef(tensor);
  }
  // Container to DLManagedTensor
  static DLManagedTensor* ToDLPack(TVMArrayHandle handle) {
    auto* from = static_cast<NDArray::Container*>(
        reinterpret_cast<NDArray::ContainerBase*>(handle));
    return ToDLPack(from);
  }

  static DLManagedTensor* ToDLPack(NDArray::Container* from) {
    CHECK(from != nullptr);
    DLManagedTensor* ret = new DLManagedTensor();
    ret->dl_tensor = from->dl_tensor;
    ret->manager_ctx = from;
    from->IncRef();
    ret->deleter = TVMNDArrayDLPackDeleter;
    return ret;
  }
  // Delete dlpack object.
  static void NDArrayDLPackDeleter(DLManagedTensor* tensor) {
    static_cast<NDArray::Container*>(tensor->manager_ctx)->DecRef();
    delete tensor;
  }
};

NDArray NDArray::CreateView(std::vector<int64_t> shape, DLDataType dtype) {
  CHECK(data_ != nullptr);
  CHECK(get_mutable()->dl_tensor.strides == nullptr)
      << "Can only create view for compact tensor";
  NDArray ret = Internal::Create(shape, dtype, get_mutable()->dl_tensor.ctx);
  ret.get_mutable()->dl_tensor.byte_offset =
      this->get_mutable()->dl_tensor.byte_offset;
  size_t curr_size = GetDataSize(this->get_mutable()->dl_tensor);
  size_t view_size = GetDataSize(ret.get_mutable()->dl_tensor);
  CHECK_LE(view_size, curr_size)
      << "Tries to create a view that has bigger memory than current one";
  // increase ref count
  get_mutable()->IncRef();
  ret.get_mutable()->manager_ctx = get_mutable();
  ret.get_mutable()->dl_tensor.data = get_mutable()->dl_tensor.data;
  if(dtype.code == kCustomBegin) {
    void **ptr = static_cast<void**>(ret.get_mutable()->dl_tensor.data);
    ptr[0] = nullptr;   
  }  
  return ret;
}

DLManagedTensor* NDArray::ToDLPack() const {
  return Internal::ToDLPack(get_mutable());
}

NDArray NDArray::Empty(std::vector<int64_t> shape,
                       DLDataType dtype,
                       DLContext ctx) {
 
  NDArray ret = Internal::Create(shape, dtype, ctx);
  // setup memory content
  size_t size = GetDataSize(ret.get_mutable()->dl_tensor);
  size_t alignment = GetDataAlignment(ret.get_mutable()->dl_tensor);
  ret.get_mutable()->dl_tensor.data =
      DeviceAPI::Get(ret->ctx)->AllocDataSpace(
          ret->ctx, size, alignment, ret->dtype);
  return ret;
}

NDArray NDArray::FromDLPack(DLManagedTensor* tensor) {
  NDArray::Container* data = new NDArray::Container();
  // construct header
  data->SetDeleter(Internal::DLPackDeleter);
  // fill up content.
  data->manager_ctx = tensor;
  data->dl_tensor = tensor->dl_tensor;
  return NDArray(GetObjectPtr<Object>(data));
}

void NDArray::CopyFromTo(const DLTensor* from,
                         DLTensor* to,
                         TVMStreamHandle stream) {
  size_t from_size = GetDataSize(*from);
  size_t to_size = GetDataSize(*to);
  CHECK_EQ(from_size, to_size)
    << "TVMArrayCopyFromTo: The size must exactly match";

  CHECK(from->ctx.device_type == to->ctx.device_type
        || from->ctx.device_type == kDLCPU
        || to->ctx.device_type == kDLCPU)
    << "Can not copy across different ctx types directly";

  // Use the context that is *not* a cpu context to get the correct device
  // api manager.
  TVMContext ctx = from->ctx.device_type != kDLCPU ? from->ctx : to->ctx;

  DeviceAPI::Get(ctx)->CopyDataFromTo(
    from->data, static_cast<size_t>(from->byte_offset),
    to->data, static_cast<size_t>(to->byte_offset),
    from_size, from->ctx, to->ctx, from->dtype, stream);
}

std::vector<int64_t> NDArray::Shape() const {
  return get_mutable()->shape_;
}

TVM_REGISTER_OBJECT_TYPE(NDArray::Container);

}  // namespace runtime
}  // namespace tvm

using namespace tvm::runtime;

void TVMNDArrayDLPackDeleter(DLManagedTensor* tensor) {
  NDArray::Internal::NDArrayDLPackDeleter(tensor);
}

int TVMArrayGetTypeIndex(TVMArrayHandle handle, unsigned* out_tindex) {
  API_BEGIN();
  *out_tindex = TVMArrayHandleToObjectHandle(handle)->type_index();
  API_END();
}

int TVMArrayAlloc(const tvm_index_t* shape,
                  int ndim,
                  int dtype_code,
                  int dtype_bits,
                  int dtype_lanes,
                  int device_type,
                  int device_id,
                  TVMArrayHandle* out) {
  API_BEGIN();
  DLDataType dtype;
  dtype.code = static_cast<uint8_t>(dtype_code);
  dtype.bits = static_cast<uint8_t>(dtype_bits);
  dtype.lanes = static_cast<uint16_t>(dtype_lanes);
  DLContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id = device_id;
  *out = NDArray::Internal::MoveToFFIHandle(
      NDArray::Empty(std::vector<int64_t>(shape, shape + ndim), dtype, ctx));
  API_END();
}

int TVMArrayFree(TVMArrayHandle handle) {
  API_BEGIN();
  NDArray::Internal::FFIDecRef(handle);
  API_END();
}

int TVMArrayCopyFromTo(TVMArrayHandle from,
                       TVMArrayHandle to,
                       TVMStreamHandle stream) {
  API_BEGIN();
  NDArray::CopyFromTo(from, to, stream);
  API_END();
}

int TVMArrayFromDLPack(DLManagedTensor* from,
                       TVMArrayHandle* out) {
  API_BEGIN();
  *out = NDArray::Internal::MoveToFFIHandle(NDArray::FromDLPack(from));
  API_END();
}

int TVMArrayToDLPack(TVMArrayHandle from,
                     DLManagedTensor** out) {
  API_BEGIN();
  *out = NDArray::Internal::ToDLPack(from);
  API_END();
}

void TVMDLManagedTensorCallDeleter(DLManagedTensor* dltensor) {
  (*(dltensor->deleter))(dltensor);
}

int TVMArrayCopyFromBytes(TVMArrayHandle handle,
                          void* data,
                          size_t nbytes) {
  API_BEGIN();
  TVMContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  size_t arr_size = GetDataSize(*handle);



  CHECK_EQ(arr_size, nbytes)
      << "TVMArrayCopyFromBytes: size mismatch";
  DeviceAPI::Get(handle->ctx)->CopyDataFromTo(
      data, 0,
      handle->data, static_cast<size_t>(handle->byte_offset),
      nbytes, cpu_ctx, handle->ctx, handle->dtype, nullptr);
  API_END();
}


int TVMArrayCopyFromStrBytes(TVMArrayHandle handle,
                          void* data,
                          size_t nbytes,
                          size_t nlenth) {
  API_BEGIN();
  size_t arr_size = GetDataSize(*handle);

  std::string **str_ptr = static_cast<std::string**>(handle->data); 
  char *char_ptr = static_cast<char*>(data); 
  
  size_t ele_num = nbytes/nlenth;
  CHECK(ele_num*(handle->dtype.bits)/8 <= arr_size) << "String array size dismatch.";


  char * char_tmp = static_cast<char*> (malloc(sizeof(char)*(nlenth/4+1)));
  char * char_tmp_ref = char_tmp;
  for(uint32_t i = 0 ; i<ele_num; ++i) {
    for(uint32_t j = 0; j<nlenth/4; ++j) {  
     *char_tmp_ref = *char_ptr;
     char_tmp_ref++;
     char_ptr+=4;
    }
    *char_tmp_ref = '\0';
   std::string* str_tmp = new std::string(char_tmp); 
   char_tmp_ref = char_tmp;  
   str_ptr[i] = str_tmp;
  }
  
  //for(uint32_t i = 0; i<ele_num; ++i) {
   // printf("[%s]\n",str_ptr[i]->c_str());
  //}  

  API_END();
}



int TVMArrayCopyToBytes(TVMArrayHandle handle,
                        void* data,
                        size_t nbytes) {
  API_BEGIN();
  TVMContext cpu_ctx;
  cpu_ctx.device_type = kDLCPU;
  cpu_ctx.device_id = 0;
  size_t arr_size = GetDataSize(*handle);
  CHECK_EQ(arr_size, nbytes)
      << "TVMArrayCopyToBytes: size mismatch";
  DeviceAPI::Get(handle->ctx)->CopyDataFromTo(
      handle->data, static_cast<size_t>(handle->byte_offset),
      data, 0,
      nbytes, handle->ctx, cpu_ctx, handle->dtype, nullptr);
  API_END();
}



int TVMArrayCopyToStrBytes(TVMArrayHandle handle,
                        void* data,
                        size_t nbytes,
                        size_t nlenth) {
  API_BEGIN();
  size_t arr_size = GetDataSize(*handle);
    
  size_t ele_num = nbytes/nlenth;
  
  CHECK(ele_num*(handle->dtype.bits)/8 <= arr_size) << "String array size dismatch.";

  std::string **str_ptr = static_cast<std::string**>(handle->data); 
  char* char_tmp = static_cast<char*>(malloc(sizeof(char)*(nlenth/4+1)));
  char *char_ptr = static_cast<char*>(data); 

  for(uint32_t i = 0; i < ele_num; ++i) {
      strcpy(char_tmp, str_ptr[i]->c_str());
      char* char_tmp_ref = char_tmp;
      bool endofstr = false;
      for(uint32_t j = 0; j<nlenth; ++j) {
          if (endofstr) 
              *char_ptr = '\0';
          else if (j%4 == 0){
              if (*char_tmp_ref == '\0')
                  endofstr = true;  
              *char_ptr = *char_tmp_ref;
              char_tmp_ref++;   
             }
          else 
              *char_ptr = '\0';
          char_ptr++;
      }
  }
  API_END();
}


int TVMArrayStrArgsCalc(TVMArrayHandle handle,
                          void* data) {
  API_BEGIN();
  
  size_t arr_size = GetDataSize(*handle);

  
  std::string **str_ptr = static_cast<std::string**>(handle->data); 
  int32_t* data_ = static_cast<int32_t*>(data); 

  uint32_t max_len = 0;

  for(uint32_t i = 0; i< arr_size/8; i++) {
      if (str_ptr[i]->length() > max_len)
         max_len = str_ptr[i]->length(); 
  }
  data_[1] = max_len; 
  data_[0] = data_[1]*4*sizeof(char)*arr_size/8;

  API_END();
}
