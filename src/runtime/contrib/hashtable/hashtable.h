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
#pragma once

#include <tvm/runtime/registry.h>
#include <tvm/runtime/util.h>
#include <dlpack/dlpack.h>
#include <algorithm>
#include <vector>

namespace tvm{
namespace contrib{


class BaseTable{
 public:
  BaseTable(){}
  virtual size_t Size() = 0;
  virtual DLDataType KeyDtype() = 0;
  virtual DLDataType ValueDtype() = 0;
  virtual bool is_initialized() = 0;
  protected:
  virtual bool DoPrepare() = 0;
  public:
  virtual bool DoInsert(DLTensor* keys, DLTensor* values) = 0;

  virtual bool DoFind(DLTensor* keys, DLTensor* values,
                DLTensor* default_value) = 0;
};



template <typename KType, typename VType>
class HashTable : public BaseTable{
 public:
  HashTable(){}
  HashTable(DLDataType key_type, DLDataType value_type ) {
    is_prepared_ = false;
    DoPrepare();
    key_type_ = key_type;
    value_type_ = value_type;
  }
  inline size_t Size() override {return table_.get()->size();}
  inline DLDataType KeyDtype() override { return key_type_; }
  inline DLDataType ValueDtype() override { return value_type_; }
  inline bool is_initialized() override {
    if(!is_prepared_)
      DoPrepare();
    return Size()>0; 
  }

  protected:
  bool DoPrepare() override{
    if (is_prepared_) {
      LOG(FATAL) << "HashTable already prepared.";
    }
    if (!table_) {
      table_ = std::unique_ptr<std::unordered_map<KType, VType>>(
          new std::unordered_map<KType, VType>());
    }
    is_prepared_ = true;
    return true;
  };

  public:
  bool DoInsert(DLTensor* keys, DLTensor* values) override{
    if (!table_) {
      LOG(FATAL) << "HashTable is not prepared.";
    }

    KType* keys_ptr = static_cast<KType *>(keys->data);
    VType* values_ptr = static_cast<VType *>(values->data);
    int keys_size = 1;
    int values_size = 1;
    CHECK_EQ(keys->ndim, values->ndim) << "dimisions of keys and values not match";
    for (int i = 0; i<keys->ndim; ++i) {
      keys_size *= keys->shape[i];
      values_size *= values->shape[i];    
      }
    CHECK_EQ(keys_size, values_size) << "total size of keys and values not match";
    std::unordered_map<KType, VType>* table_ptr = table_.get();
    for (int i = 0; i < keys_size; ++i) {
     // printf("%d init to %d\n", keys_ptr[i], values_ptr[i]);
      table_ptr->insert(std::make_pair(keys_ptr[i], values_ptr[i]));
    }
    return true;
  }

  bool DoFind(DLTensor* keys, DLTensor* values,
                DLTensor* default_value) override {
    KType* keys_ptr = static_cast<KType *>(keys->data);
    VType* values_ptr = static_cast<VType *>(values->data);
    VType default_value_ = *(static_cast<VType *>(default_value->data));
    int keys_size = 1;
    int values_size = 1;
    CHECK_EQ(keys->ndim, values->ndim) << "dimisions of keys and values not match";
    for (int i=0; i<keys->ndim; ++i) {
      keys_size *= keys->shape[i];
      values_size *= values->shape[i];    
      }
    CHECK_EQ(keys_size, values_size) << "total size of keys and values not match";
    std::unordered_map<KType, VType>* table_ptr = table_.get();
    for (int i = 0; i < keys_size; ++i) {
      typename std::unordered_map<KType, VType>::const_iterator it;
      it  = table_ptr->find(keys_ptr[i]);
      if (it == table_ptr->end())
        values_ptr[i] = default_value_;
      else
        values_ptr[i] = it->second; 
     // printf("%d : %d\n", keys_ptr[i], values_ptr[i]); 
    }
    return true;
  }

 private:
  std::unique_ptr<std::unordered_map<KType, VType>> table_;
  bool is_prepared_;
  DLDataType key_type_;
  DLDataType value_type_;
};



template <typename VType>
class StrKHashTable : public BaseTable{
 public:
  StrKHashTable(){}
  StrKHashTable(DLDataType value_type ) {
    is_prepared_ = false;
    DoPrepare();
    key_type_ = runtime::String2TVMType("custom[string]64");
    value_type_ = value_type;
  }
  inline size_t Size() override {return table_.get()->size();}
  inline DLDataType KeyDtype() override { return key_type_; }
  inline DLDataType ValueDtype() override { return value_type_; }
  inline bool is_initialized() override {
    if(!is_prepared_)
      DoPrepare();
    return Size()>0; 
  }

  protected:
  bool DoPrepare() override{
    if (is_prepared_) {
      LOG(FATAL) << "HashTable already prepared.";
    }
    if (!table_) {
      table_ = std::unique_ptr<std::unordered_map<std::string, VType>>(
          new std::unordered_map<std::string, VType>());
    }
    is_prepared_ = true;
    return true;
  };

  public:
  bool DoInsert(DLTensor* keys, DLTensor* values) override{
    if (!table_) {
      LOG(FATAL) << "HashTable is not prepared.";
    }

    std::string** keys_ptr = static_cast<std::string**>(keys->data);
    VType* values_ptr = static_cast<VType *>(values->data);
    int keys_size = 1;
    int values_size = 1;
    CHECK_EQ(keys->ndim, values->ndim) << "dimisions of keys and values not match";
    for (int i = 0; i<keys->ndim; ++i) {
      keys_size *= keys->shape[i];
      values_size *= values->shape[i];    
      }
    CHECK_EQ(keys_size, values_size) << "total size of keys and values not match";
    std::unordered_map<std::string, VType>* table_ptr = table_.get();
    for (int i = 0; i < keys_size; ++i) {
     // printf("%d init to %d\n", keys_ptr[i], values_ptr[i]);
      table_ptr->insert(std::make_pair(*keys_ptr[i], values_ptr[i]));
    }
    return true;
  }

  bool DoFind(DLTensor* keys, DLTensor* values,
                DLTensor* default_value) override {
    std::string** keys_ptr = static_cast<std::string**>(keys->data);
    VType* values_ptr = static_cast<VType *>(values->data);
    VType default_value_ = *(static_cast<VType *>(default_value->data));
    int keys_size = 1;
    int values_size = 1;
    CHECK_EQ(keys->ndim, values->ndim) << "dimisions of keys and values not match";
    for (int i=0; i<keys->ndim; ++i) {
      keys_size *= keys->shape[i];
      values_size *= values->shape[i];    
      }
    CHECK_EQ(keys_size, values_size) << "total size of keys and values not match";
    std::unordered_map<std::string, VType>* table_ptr = table_.get();
    for (int i = 0; i < keys_size; ++i) {
      typename std::unordered_map<std::string, VType>::const_iterator it;
      it  = table_ptr->find(*keys_ptr[i]);
      if (it == table_ptr->end())
        values_ptr[i] = default_value_;
      else
        values_ptr[i] = it->second; 
     // printf("%d : %d\n", keys_ptr[i], values_ptr[i]); 
    }
    return true;
  }

 private:
  std::unique_ptr<std::unordered_map<std::string, VType>> table_;
  bool is_prepared_;
  DLDataType key_type_;
  DLDataType value_type_;
};



template <typename KType>
class StrVHashTable : public BaseTable{
 public:
  StrVHashTable(){}
  StrVHashTable(DLDataType key_type ) {
    is_prepared_ = false;
    DoPrepare();
    value_type_ = runtime::String2TVMType("custom[string]64");
    key_type_ = key_type;
  }
  inline size_t Size() override {return table_.get()->size();}
  inline DLDataType KeyDtype() override { return key_type_; }
  inline DLDataType ValueDtype() override { return value_type_; }
  inline bool is_initialized() override {
    if(!is_prepared_)
      DoPrepare();
    return Size()>0; 
  }

  protected:
  bool DoPrepare() override{
    if (is_prepared_) {
      LOG(FATAL) << "HashTable already prepared.";
    }
    if (!table_) {
      table_ = std::unique_ptr<std::unordered_map<KType, std::string>>(
          new std::unordered_map<KType, std::string>());
    }
    is_prepared_ = true;
    return true;
  };

  public:
  bool DoInsert(DLTensor* keys, DLTensor* values) override{
    if (!table_) {
      LOG(FATAL) << "HashTable is not prepared.";
    }

    std::string** values_ptr = static_cast<std::string**>(values->data);
    KType* keys_ptr = static_cast<KType *>(keys->data);
    int keys_size = 1;
    int values_size = 1;
    CHECK_EQ(keys->ndim, values->ndim) << "dimisions of keys and values not match";
    for (int i = 0; i<keys->ndim; ++i) {
      keys_size *= keys->shape[i];
      values_size *= values->shape[i];    
      }
    CHECK_EQ(keys_size, values_size) << "total size of keys and values not match";
    std::unordered_map<KType, std::string>* table_ptr = table_.get();
    for (int i = 0; i < keys_size; ++i) {
      table_ptr->insert(std::make_pair(keys_ptr[i], *values_ptr[i]));
    }
    return true;
  }

  bool DoFind(DLTensor* keys, DLTensor* values,
                DLTensor* default_value) override {
    std::string** values_ptr = static_cast<std::string**>(values->data);
    KType* keys_ptr = static_cast<KType *>(keys->data);
    std::string* default_value_ = *(static_cast<std::string**>(default_value->data));
    int keys_size = 1;
    int values_size = 1;
    CHECK_EQ(keys->ndim, values->ndim) << "dimisions of keys and values not match";
    for (int i=0; i<keys->ndim; ++i) {
      keys_size *= keys->shape[i];
      values_size *= values->shape[i];    
      }
    CHECK_EQ(keys_size, values_size) << "total size of keys and values not match";
    std::unordered_map<KType, std::string>* table_ptr = table_.get();
    for (int i = 0; i < keys_size; ++i) {
      typename std::unordered_map<KType, std::string>::const_iterator it;
      it  = table_ptr->find(keys_ptr[i]);
  
      if (it == table_ptr->end())
        values_ptr[i] = default_value_;
      else {
        std::string* find_res = new std::string(it->second);  
        values_ptr[i] = find_res;
        } 
    }
    return true;
  }

 private:
  std::unique_ptr<std::unordered_map<KType, std::string>> table_;
  bool is_prepared_;
  DLDataType key_type_;
  DLDataType value_type_;
};




class StrHashTable : public BaseTable{
 public:
  StrHashTable() {
    is_prepared_ = false;
    DoPrepare();
    key_type_ = runtime::String2TVMType("custom[string]64");
    value_type_ = key_type_;
  }
  inline size_t Size() override {return table_.get()->size();}
  inline DLDataType KeyDtype() override { return key_type_; }
  inline DLDataType ValueDtype() override { return value_type_; }
  inline bool is_initialized() override {
    if(!is_prepared_)
      DoPrepare();
    return Size()>0; 
  }

  protected:
  bool DoPrepare() override{
    if (is_prepared_) {
      LOG(FATAL) << "HashTable already prepared.";
    }
    if (!table_) {
      table_ = std::unique_ptr<std::unordered_map<std::string, std::string>>(
          new std::unordered_map<std::string, std::string>());
    }
    is_prepared_ = true;
    return true;
  };

  public:
  bool DoInsert(DLTensor* keys, DLTensor* values) override{
    if (!table_) {
      LOG(FATAL) << "HashTable is not prepared.";
    }

    std::string** keys_ptr = static_cast<std::string**>(keys->data);
    std::string** values_ptr = static_cast<std::string**>(values->data);
    int keys_size = 1;
    int values_size = 1;
    CHECK_EQ(keys->ndim, values->ndim) << "dimisions of keys and values not match";
    for (int i = 0; i<keys->ndim; ++i) {
      keys_size *= keys->shape[i];
      values_size *= values->shape[i];    
      }
    CHECK_EQ(keys_size, values_size) << "total size of keys and values not match";
    std::unordered_map<std::string, std::string>* table_ptr = table_.get();
    for (int i = 0; i < keys_size; ++i) {
     // printf("%d init to %d\n", keys_ptr[i], values_ptr[i]);
      table_ptr->insert(std::make_pair(*keys_ptr[i], *values_ptr[i]));
    }
    return true;
  }

  bool DoFind(DLTensor* keys, DLTensor* values,
                DLTensor* default_value) override {
    std::string** keys_ptr = static_cast<std::string**>(keys->data);
    std::string** values_ptr = static_cast<std::string**>(values->data);
    std::string* default_value_ = *(static_cast<std::string**>(default_value->data));
    int keys_size = 1;
    int values_size = 1;
    CHECK_EQ(keys->ndim, values->ndim) << "dimisions of keys and values not match";
    for (int i=0; i<keys->ndim; ++i) {
      keys_size *= keys->shape[i];
      values_size *= values->shape[i];    
      }
    CHECK_EQ(keys_size, values_size) << "total size of keys and values not match";
    std::unordered_map<std::string, std::string>* table_ptr = table_.get();
    for (int i = 0; i < keys_size; ++i) {
      typename std::unordered_map<std::string, std::string>::const_iterator it;
      it  = table_ptr->find(*keys_ptr[i]);
      if (it == table_ptr->end())
        values_ptr[i] = default_value_;
      else {
        std::string* find_res = new std::string(it->second);  
        values_ptr[i] = find_res;
        } 
     // printf("%d : %d\n", keys_ptr[i], values_ptr[i]); 
    }
    return true;
  }

 private:
  std::unique_ptr<std::unordered_map<std::string, std::string>> table_;
  bool is_prepared_;
  DLDataType key_type_;
  DLDataType value_type_;
};





}  // namespace contrib
}  // namespace tvm

