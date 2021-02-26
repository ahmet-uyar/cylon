//
// Created by auyar on 3.02.2021.
//
#ifndef CYLON_CUDF_H
#define CYLON_CUDF_H

#include <unordered_map>

#include <net/ops/all_to_all.hpp>
#include <ctx/cylon_context.hpp>
#include <status.hpp>
#include <net/buffer.hpp>

#include <cudf/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>

cudf::size_type dataLength(cudf::column_view const& cw);

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

class PendingBuffer {
public:
    explicit PendingBuffer(const uint8_t *buffer,
                           int bufferSize,
                           int target,
                           std::unique_ptr<int []> headers = nullptr,
                           int headersLength = -1);

    explicit PendingBuffer(int target,
                           std::unique_ptr<int []> headers,
                           int headersLength);

    bool sendBuffer(std::shared_ptr<cylon::AllToAll> all);

private:
    const uint8_t *buffer;
    int bufferSize;
    int target;
    std::unique_ptr<int []> headers;
    int headersLength;
};

/**
 * Keep track of the items to send for a target
 */
struct PendingSends {
    // pending tables to be sent with it's reference
    std::queue<std::pair<std::shared_ptr<cudf::table_view>, int32_t>> tableQueue{};

    // buffers to send from the current table
    std::queue<std::shared_ptr<PendingBuffer>> bufferQueue{};
};

struct PendingReceives {
    // table variables
    // currently received columns
    std::unordered_map<int, std::unique_ptr<cudf::column>> columns;
    // number of columns in the table
    int numberOfColumns{-1};
    // the reference
    int reference{-1};

    // column variables
    // data type of the column
    int columnDataType{-1};
    // the current data column index
    int columnIndex{-1};
    // whether the current column has the null buffer
    bool hasNullBuffer{false};
    // whether the current column has the offset buffer
    bool hasOffsetBuffer{false};
    // number of data elements
    int dataSize{0};
    // length of the data buffer
    int dataBufferLen{0};


    // data buffer for the current column
    std::shared_ptr<rmm::device_buffer> dataBuffer;
    // null buffer for the current column
    std::shared_ptr<rmm::device_buffer> nullBuffer;
    // offsets buffer for the current column
    std::shared_ptr<rmm::device_buffer> offsetsBuffer;
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
  int insert(const std::shared_ptr<cudf::table_view> &table, int32_t target);

  /**
   * Insert a table to be sent, if the table is accepted return true
   *
   * @param table the table to send
   * @param target the target to send the table
   * @param reference a reference that can be sent in the header
   * @return true if the buffer is accepted
   */
  int insert(const std::shared_ptr<cudf::table_view> &table, int32_t target, int32_t reference);

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
    bool insertBuffers(std::queue<std::shared_ptr<PendingBuffer>> &bufferQueue);

    void makeTableBuffers(std::shared_ptr<cudf::table_view> table,
                          int target,
                          int ref,
                          std::queue<std::shared_ptr<PendingBuffer>> &bufferQueue);

    void makeColumnBuffers(const cudf::column_view &cw,
                           int columnIndex,
                           int target,
                           std::queue<std::shared_ptr<PendingBuffer>> &bufferQueue);

    void constructColumn(std::shared_ptr<PendingReceives> pr);

    std::shared_ptr<cudf::table> constructTable(std::shared_ptr<PendingReceives> pr);

    /**
     * worker rank
     */
    int myrank;

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
