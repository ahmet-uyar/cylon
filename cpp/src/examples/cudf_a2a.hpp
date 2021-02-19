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

enum CudfHeader {
    CUDF_HEADER_INIT = 0,
    // wea are still sending the data about the column
    CUDF_HEADER_COLUMN_CONTINUE = 1,
    // this is the end of the column
    CUDF_HEADER_COLUMN_END = 2
};

/**
 * Keep track of the items to send for a target
 */
struct PendingSends {
    // pending tables to be sent with it's reference
    std::queue<std::pair<std::shared_ptr<cudf::table>, int32_t>> tableQueue{};

    // state of the send
    CudfHeader status = CUDF_HEADER_INIT;
};

struct PendingReceives {
    // number of columns in the table
    int numberOfColumns{};
    // the reference
    int reference{};

    // data type of the column
    int columnDataType{};
    // the current data column index
    int columnIndex{};
    // whether the current column has the null buffer
    bool hasNullBuffer{};
    // whether the current column has the offset buffer
    bool hasOffsetBuffer{};

    // number of data elements
    int dataSize{};


    // data buffer for the current column
    std::shared_ptr<rmm::device_buffer> dataBuffer;
    // null buffer for the current column
    std::shared_ptr<rmm::device_buffer> nullBuffer;
    // offsets buffer for the current column
    std::shared_ptr<rmm::device_buffer> offsetsBuffer;
    // currently received columns
    std::unordered_map<int, std::unique_ptr<cudf::column>> columns;
};

/**
 * This function is called when a table is fully received
 * @param source the source
 * @param table the table that is received
 * @param reference the reference number sent by the sender
 * @return true if we accept this buffer
 */
using CudfCallback = std::function<bool(int source, const std::shared_ptr<cudf::table> &table, int reference)>;

class CudfAllToAll : public cylon::ReceiveCallback {

public:
  CudfAllToAll(std::shared_ptr<cylon::CylonContext> &ctx,
               const std::vector<int> &sources,
               const std::vector<int> &targets,
               int edgeId,
               CudfCallback callback);

  /**
   * Insert a table to be sent, if the table is accepted return true
   *
   * @param table the table to send
   * @param target the target to send the table
   * @return true if the buffer is accepted
   */
  int insert(const std::shared_ptr<cudf::table> &table, int32_t target);

  /**
   * Insert a table to be sent, if the table is accepted return true
   *
   * @param table the table to send
   * @param target the target to send the table
   * @param reference a reference that can be sent in the header
   * @return true if the buffer is accepted
   */
  int insert(const std::shared_ptr<cudf::table> &arrow, int32_t target, int32_t reference);

  /**
   * Check weather the operation is complete, this method needs to be called until the operation is complete
   * @return true if the operation is complete
   */
  bool isComplete();

  /**
   * When this function is called, the operation finishes at both receivers and targets
   * @return
   */
  void finish();

  /**
   * Close the operation
   */
  void close();

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
    bool insertTableToA2A(std::shared_ptr<cudf::table> table, int target, int ref);
    bool insertColumnToA2A(cudf::column &clmn, int columnIndex, int target);
    void constructColumn(std::shared_ptr<PendingReceives> pr);
    std::shared_ptr<cudf::table> constructTable(std::shared_ptr<PendingReceives> pr);

    /**
     * The sources
     */
    std::vector<int> sources_;

    /**
     * The targets
     */
    std::vector<int> targets_;

    /**
     * The underlying alltoall communication
     */
    std::shared_ptr<cylon::AllToAll> all_;

    /**
     * Keep track of the inputs
     */
    std::unordered_map<int, std::shared_ptr<PendingSends>> inputs_;

    /**
     * Keep track of the receives
     */
    std::unordered_map<int, std::shared_ptr<PendingReceives>> receives_;

    /**
     * this is the allocator to create memory when receiving
     */
    CudfAllocator * allocator_;

    /**
     * inform the callback when a table received
     */
    CudfCallback recv_callback_;

    /**
     * We have received the finish
     */
    bool finished = false;

    bool completed_;
    bool finishCalled_;
};

#endif //CYLON_CUDF_H
