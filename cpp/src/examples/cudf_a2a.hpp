//
// Created by auyar on 3.02.2021.
//
#ifndef CYLON_CUDF_H
#define CYLON_CUDF_H

#include <unordered_map>

#include <net/ops/all_to_all.hpp>
#include <net/mpi/mpi_communicator.hpp>
#include <ctx/cylon_context.hpp>
#include <status.hpp>
#include <net/buffer.hpp>

#include <cudf/types.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/types.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <cuda_runtime.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

class CudfBuffer : public cylon::Buffer {
public:
    explicit CudfBuffer(std::shared_ptr<rmm::device_buffer> rmmBuf);
    int64_t GetLength() override;
    uint8_t * GetByteBuffer() override;
    std::shared_ptr<rmm::device_buffer> getBuf() const;
private:
    std::shared_ptr<rmm::device_buffer> rmmBuf;
};

class CudfAllocator : public cylon::Allocator {
public:
    cylon::Status Allocate(int64_t length, std::shared_ptr<cylon::Buffer> *buffer) override;
};

class CudfAllToAll : public cylon::ReceiveCallback {

public:
  CudfAllToAll();

  /**
   * This function is called when a data is received
   */
  bool onReceive(int source, std::shared_ptr<cylon::Buffer> buffer, int length) override;

  /**
   * Receive the header, this happens before we receive the actual data
   */
  bool onReceiveHeader(int source, int finished, int *buffer, int length) override;

  /**
   * This method is called after we successfully send a buffer
   * @return
   */
  bool onSendComplete(int target, const void *buffer, int length) override;

private:
    std::unordered_map<int, int> data_types;
};

#endif //CYLON_CUDF_H
