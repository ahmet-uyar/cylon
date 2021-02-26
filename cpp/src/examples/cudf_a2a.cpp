//
// Created by auyar on 3.02.2021.
//

#include "include/cudf_a2a.hpp"

#include <glog/logging.h>
#include <chrono>
#include <thread>

#include <net/mpi/mpi_communicator.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cudf/io/csv.hpp>
//#include <cuda_runtime.h>

int myrank = -1;

//////////////////////////////////////////////////////////////////////
// global types and fuctions
//////////////////////////////////////////////////////////////////////
// sizes of cudf types in tables
// ref: type_id in cudf/types.hpp
int type_bytes[] = {0, 1, 2, 4, 8, 1, 2, 4, 8, 4, 8, 1, 4, 8, 8, 8, 8, 4, 8, 8, 8, 8, -1, -1, -1, 4, 8, -1, -1};

/**
 * whether the data type is uniform in size such as int (4 bytes) or string(variable length)
 * @param input
 * @return
 */
int data_type_size(cudf::column_view const& cw){
    return type_bytes[static_cast<int>(cw.type().id())];
}

/**
 * whether the data type is uniform in size such as int (4 bytes) or string(variable length)
 * @param input
 * @return
 */
bool uniform_size_data(cudf::column_view const& cw){
    int dataTypeSize = data_type_size(cw);
    return dataTypeSize == -1 ? false : true;
}

/**
 * data buffer length of a column in bytes
 * @param input
 * @return
 */
cudf::size_type dataLength(cudf::column_view const& cw){
    int elementSize = type_bytes[static_cast<int>(cw.type().id())];
    if (elementSize == -1) {
        std::cout << "ERRORRRRRR unsupported type id: " << static_cast<int>(cw.type().id()) << std::endl;
        return -1;
    }

    // even null values exist in the buffer with unspecified values
    return elementSize * cw.size();
}

//////////////////////////////////////////////////////////////////////
// CudfBuffer implementations
//////////////////////////////////////////////////////////////////////
CudfBuffer::CudfBuffer(std::shared_ptr<rmm::device_buffer> rmmBuf) : rmmBuf(rmmBuf) {}

int64_t CudfBuffer::GetLength() {
  return rmmBuf->size();
}

uint8_t * CudfBuffer::GetByteBuffer() {
  return (uint8_t *)rmmBuf->data();
}

std::shared_ptr<rmm::device_buffer> CudfBuffer::getBuf() const {
  return rmmBuf;
}

//////////////////////////////////////////////////////////////////////
// CudfAllocator implementations
//////////////////////////////////////////////////////////////////////
cylon::Status CudfAllocator::Allocate(int64_t length, std::shared_ptr<cylon::Buffer> *buffer) {
  try {
    auto rmmBuf = std::make_shared<rmm::device_buffer>(length);
    *buffer = std::make_shared<CudfBuffer>(rmmBuf);
    return cylon::Status::OK();
  } catch (rmm::bad_alloc badAlloc) {
    LOG(ERROR) << "failed to allocate gpu memory with rmm: " << badAlloc.what();
    return cylon::Status(cylon::Code::GpuMemoryError);
  }
}

//////////////////////////////////////////////////////////////////////
// PendingBuffer implementations
//////////////////////////////////////////////////////////////////////
PendingBuffer::PendingBuffer(const uint8_t *buffer,
                             int bufferSize,
                             int target,
                             std::unique_ptr<int []> headers,
                             int headersLength):
        buffer(buffer),
        bufferSize(bufferSize),
        target(target),
        headers(std::move(headers)),
        headersLength(headersLength) {}

PendingBuffer::PendingBuffer(int target,
                             std::unique_ptr<int []> headers,
                             int headersLength):
        buffer(nullptr),
        bufferSize(-1),
        target(target),
        headers(std::move(headers)),
        headersLength(headersLength) {}

bool PendingBuffer::sendBuffer(std::shared_ptr<cylon::AllToAll> all) {
    // if there is no data buffer, only header buffer
    if (bufferSize < 0) {
        bool accepted = all->insert(nullptr, 0, target, headers.get(), headersLength);
        if (!accepted) {
            LOG(WARNING) << myrank << " header buffer not accepted to be sent";
        }
        return accepted;
    }

    // if there is no header buffer, only data buffer
    if (headersLength < 0) {
        bool accepted = all->insert(buffer, bufferSize, target);
        if (!accepted) {
            LOG(WARNING) << myrank << " data buffer not accepted to be sent";
        }
        return accepted;
    }

    bool accepted = all->insert(buffer, bufferSize, target, headers.get(), headersLength);
    if (!accepted) {
        LOG(WARNING) << myrank << " data buffer with header not accepted to be sent";
    }
    return accepted;
}
//////////////////////////////////////////////////////////////////////
// CudfAllToAll implementations
//////////////////////////////////////////////////////////////////////
CudfAllToAll::CudfAllToAll(std::shared_ptr<cylon::CylonContext> &ctx,
                           const std::vector<int> &sources,
                           const std::vector<int> &targets,
                           int edgeId,
                           CudfCallback callback) :
    sources_(sources),
    targets_(targets),
    recv_callback_(std::move(callback)),
    myrank(ctx->GetRank()),
    completed_(false),
    finishCalled_(false){

    allocator_ = new CudfAllocator();

    // we need to pass the correct arguments
    all_ = std::make_shared<cylon::AllToAll>(ctx, sources_, targets_, edgeId, this, allocator_);

    // add the trackers for sending
    for (auto t : targets_) {
        inputs_.insert(std::make_pair(t, std::make_shared<PendingSends>()));
    }

    for (auto t : sources_) {
        receives_.insert(std::make_pair(t, std::make_shared<PendingReceives>()));
    }
}

int CudfAllToAll::insert(const std::shared_ptr<cudf::table_view> &table, int32_t target) {
    return insert(table, target, -1);
}

int CudfAllToAll::insert(const std::shared_ptr<cudf::table_view> &table,
                          int32_t target,
                          int32_t reference) {
    // todo: check weather we have enough memory
    // lets save the table into pending and move on
    inputs_[target]->tableQueue.push(std::make_pair(table, reference));
    return 1;
}

bool CudfAllToAll::isComplete() {

    for (const auto &pendingSend : inputs_) {
        // if the buffer queue is not empty, first insert those buffers
        if (!pendingSend.second->bufferQueue.empty()) {
            bool inserted = insertBuffers(pendingSend.second->bufferQueue);
            if (!inserted) {
                return false;
            }
        }

        while (!pendingSend.second->tableQueue.empty()) {
          auto currentPair = pendingSend.second->tableQueue.front();
          pendingSend.second->tableQueue.pop();
          makeTableBuffers(currentPair.first, pendingSend.first, currentPair.second, pendingSend.second->bufferQueue);
          bool inserted = insertBuffers(pendingSend.second->bufferQueue);
          if (!inserted) {
              return false;
          }

//          insertTableToA2A(currentPair.first, t.first, currentPair.second);
          LOG(INFO) << myrank << ", inserted table for A2A. target: " << pendingSend.first << ", ref: " << currentPair.second;
        }
    }

    if (!finished)
        finish();

    return all_->isComplete();
}

bool CudfAllToAll::insertBuffers(std::queue<std::shared_ptr<PendingBuffer>> &bufferQueue) {
    while (!bufferQueue.empty()) {
        auto pb = bufferQueue.front();
        bool accepted = pb->sendBuffer(all_);
        if (accepted) {
            bufferQueue.pop();
        } else {
            return false;
        }
    }

    return true;
}

void CudfAllToAll::makeTableBuffers(std::shared_ptr<cudf::table_view> table,
                                    int target,
                                    int ref,
                                    std::queue<std::shared_ptr<PendingBuffer>> &bufferQueue) {
    // construct header message to send
    // for all columns, send data-type, whether it has null data and offsets buffer.
    // header data
    int columns = table->num_columns();
    int headersLength = 3;
    auto tableHeaders = std::make_unique<int []>(headersLength);
    tableHeaders[0] = 0; // shows it is a table header. todo: make it enum
    tableHeaders[1] = ref;
    tableHeaders[2] = columns;
    auto pb = std::make_shared<PendingBuffer>(target, std::move(tableHeaders), headersLength);
    bufferQueue.emplace(pb);

    for (int i = 0; i < columns; ++i) {
        makeColumnBuffers(table->column(i), i, target, bufferQueue);
    }
}

void CudfAllToAll::makeColumnBuffers(const cudf::column_view &cw,
                                     int columnIndex,
                                     int target,
                                     std::queue<std::shared_ptr<PendingBuffer>> &bufferQueue) {

    // we support uniform size data types and the string type
    if (!uniform_size_data(cw) && cw.type().id() != cudf::type_id::STRING) {
        throw "only uniform-size data-types and the string is supported.";
    }

    int headersLength = 6;
    auto columnHeaders = std::make_unique<int []>(headersLength);
    columnHeaders[0] = 1; // it is a column header.
    columnHeaders[1] = columnIndex;
    columnHeaders[2] = (int)(cw.type().id()); // data-type of the column
    // whether the column has null array, 1 has the null buffer, 0 means no null buffer
    columnHeaders[3] = cw.nullable() ? 1 : 0;
    //whether the column has offsets array
    columnHeaders[4] = (cw.num_children() > 0) ? 1 : 0;
    // number of elements in the column
    columnHeaders[5] = cw.size();

    // insert data buffer
    const uint8_t *dataBuffer;
    int bufferSize;

    // if it is a string column, get char buffer
    const uint8_t *offsetsBuffer;
    int offsetsSize = -1;
    if (cw.type().id() == cudf::type_id::STRING) {
        cudf::strings_column_view scv(cw);
        dataBuffer = scv.chars().data<uint8_t>();
        bufferSize = scv.chars_size();

        offsetsBuffer = scv.offsets().data<uint8_t>();
        offsetsSize = dataLength(scv.offsets());
        // get uniform size column data
    } else {
        dataBuffer = cw.data<uint8_t>();
        bufferSize = dataLength(cw);
//    LOG(INFO) << myrank << "******* inserting column buffer with length: " << dataLen;
    }
    // insert the data buffer
    if(bufferSize < 0) {
        throw "bufferSize is negative: " + std::to_string(bufferSize);
    }

    auto pb = std::make_shared<PendingBuffer>(dataBuffer, bufferSize, target, std::move(columnHeaders), headersLength);
    bufferQueue.emplace(pb);

    // insert null buffer if exists
    if (cw.nullable()) {
        uint8_t * nullBuffer = (uint8_t *)cw.null_mask();
        std::size_t nullBufSize = cudf::bitmask_allocation_size_bytes(cw.size());
        if(nullBufSize < 0) {
            throw "nullBufSize is negative: " + std::to_string(nullBufSize);
        }
        pb = std::make_shared<PendingBuffer>(nullBuffer, nullBufSize, target);
        bufferQueue.emplace(pb);
    }

    if (offsetsSize >= 0) {
        pb = std::make_shared<PendingBuffer>(offsetsBuffer, offsetsSize, target);
        bufferQueue.emplace(pb);
    }
}

void CudfAllToAll::finish() {
    finished = true;
    all_->finish();
}

void CudfAllToAll::close() {
    // clear the input map
    inputs_.clear();
    // call close on the underlying all-to-all
    all_->close();

    delete allocator_;
}

void CudfAllToAll::constructColumn(std::shared_ptr<PendingReceives> pr) {

    std::unique_ptr<cudf::column> column;

    cudf::data_type dt(static_cast<cudf::type_id>(pr->columnDataType));
    std::shared_ptr<rmm::device_buffer> dataBuffer = pr->dataBuffer;
    std::shared_ptr<rmm::device_buffer> nullBuffer = pr->nullBuffer;
    std::shared_ptr<rmm::device_buffer> offsetsBuffer = pr->offsetsBuffer;

    if (dt.id() != cudf::type_id::STRING)  {
        if(pr->hasNullBuffer) {
            column = std::make_unique<cudf::column>(dt, pr->dataSize, *dataBuffer, *nullBuffer);
        } else {
            column = std::make_unique<cudf::column>(dt, pr->dataSize, *dataBuffer);
        }

    // construct string column
    } else {
        // construct chars child column
        auto cdt = cudf::data_type{cudf::type_id::INT8};
        auto charsColumn = std::make_unique<cudf::column>(cdt, pr->dataBufferLen, *dataBuffer);
        auto odt = cudf::data_type{cudf::type_id::INT32};
        auto offsetsColumn = std::make_unique<cudf::column>(odt, pr->dataSize + 1, *offsetsBuffer);

        std::vector<std::unique_ptr<cudf::column>> children;
        children.emplace_back(std::move(offsetsColumn));
        children.emplace_back(std::move(charsColumn));

        if (pr->hasNullBuffer) {
            column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                                    pr->dataSize,
                                                    rmm::device_buffer{0},
                                                    *nullBuffer,
                                                    cudf::UNKNOWN_NULL_COUNT,
                                                    std::move(children));
        } else{
            column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::STRING},
                                                    pr->dataSize,
                                                    rmm::device_buffer{0},
                                                    rmm::device_buffer{0},
                                                    0,
                                                    std::move(children));
        }

    }

    // if the column is constructed, add it to the list
    if(column) {
        pr->columns.insert({pr->columnIndex, std::move(column)});

        // clear column related data from pr
        pr->columnIndex = -1;
        pr->columnDataType = -1;
        pr->dataSize = 0;
        pr->dataBuffer.reset();
        pr->nullBuffer.reset();
        pr->offsetsBuffer.reset();
        pr->hasNullBuffer = false;
        pr->hasOffsetBuffer = false;
        pr->dataBufferLen = 0;
    }
}

std::shared_ptr<cudf::table> CudfAllToAll::constructTable(std::shared_ptr<PendingReceives> pr) {

    auto columnVector = std::make_unique<std::vector<std::unique_ptr<cudf::column>>>();
    for (int i=0; i < pr->columns.size(); i++) {
        columnVector->push_back(std::move(pr->columns.at(i)));
    }

    std::shared_ptr<cudf::table> tbl = std::make_shared<cudf::table>(std::move(*columnVector));
    return tbl;
}

/**
 * This function is called when a data is received
 */
bool CudfAllToAll::onReceive(int source, std::shared_ptr<cylon::Buffer> buffer, int length) {
  LOG(INFO) << myrank << ",,,,, buffer received from the source: " << source << ", with length: " << length;
  if (length == 0) {
      return true;
  }
  std::shared_ptr<CudfBuffer> cb = std::dynamic_pointer_cast<CudfBuffer>(buffer);

  // if the data buffer is not received yet, get it
  std::shared_ptr<PendingReceives> pr = receives_.at(source);
  if (!pr->dataBuffer) {
    pr->dataBuffer = cb->getBuf();
    pr->dataBufferLen = length;
    LOG(INFO) << myrank <<  " ##### "<< pr->columnIndex << " assigned data buffer";

    // if there is no null buffer or offset buffer, create the column
    if(!pr->hasNullBuffer && !pr->hasOffsetBuffer) {
        constructColumn(pr);
        LOG(INFO) << myrank <<  " ##### "<< pr->columnIndex << " constructed column";
    }
  } else if(pr->hasNullBuffer && !pr->nullBuffer) {
      pr->nullBuffer = cb->getBuf();
      LOG(INFO) << myrank <<  " ##### "<< pr->columnIndex << " assigned null buffer";
      // if there is no offset buffer, create the column
      if (!pr->hasOffsetBuffer) {
          constructColumn(pr);
          LOG(INFO) << myrank <<  " ##### "<< pr->columnIndex << " constructed column with null buffer";
      }
  } else if(pr->hasOffsetBuffer && !pr->offsetsBuffer) {
      pr->offsetsBuffer = cb->getBuf();
      LOG(INFO) << myrank <<  " ##### "<< pr->columnIndex << " assigned offset buffer";
      constructColumn(pr);
      LOG(INFO) << myrank <<  " ##### "<< pr->columnIndex << " constructed column with offset buffer";

  } else {
      LOG(INFO) << myrank <<  " ##### "<< pr->columnIndex << " an unexpected buffer received from: " << source << ", buffer length: " << length;
  }

  // if all columns are created, create the table
  if (pr->columns.size() == pr->numberOfColumns) {
      LOG(INFO) << myrank << "***** all columns are created. create the table.";
      std::shared_ptr<cudf::table> tbl = constructTable(pr);
      recv_callback_(source, tbl, pr->reference);

      // clear table data from pr
      pr->columns.clear();
      pr->numberOfColumns = -1;
      pr->reference = -1;
  }

  return true;
}

/**
 * Receive the header, this happens before we receive the actual data
 */
bool CudfAllToAll::onReceiveHeader(int source, int finished, int *buffer, int length) {
  LOG(INFO) << myrank << "----received a header buffer with length: " << length;
  if (length > 0) {
      if (buffer[0] == 0) { // table header
          std::shared_ptr<PendingReceives> pr = receives_.at(source);
          pr->reference = buffer[1];
          pr->numberOfColumns = buffer[2];
          LOG(INFO) << myrank << "----received a table header from the source: " << source;
      } else if(buffer[0] == 1){ // column header
          std::shared_ptr<PendingReceives> pr = receives_.at(source);
          pr->columnIndex = buffer[1];
          pr->columnDataType = buffer[2];
          pr->hasNullBuffer = buffer[3] == 0 ? false: true;
          pr->hasOffsetBuffer = buffer[4] == 0 ? false: true;
          pr->dataSize = buffer[5];
          LOG(INFO) << myrank << "----received a column header from the source: " << source
                  << ", columnIndex: " << pr->columnIndex << std::endl
                  << ", columnDataType: " << pr->columnDataType << std::endl
                  << ", hasNullBuffer: " << pr->hasNullBuffer << std::endl
                  << ", hasOffsetBuffer: " << pr->hasOffsetBuffer << std::endl
                  << ", dataSize: " << pr->dataSize << std::endl;
      }
  }
  return true;
}

/**
 * This method is called after we successfully send a buffer
 * @return
 */
bool CudfAllToAll::onSendComplete(int target, const void *buffer, int length) {
  LOG(INFO) << myrank << ", SendComplete with length: " << length << " for the target: " << target;
  return true;
}

//////////////////////////////////////////////////////////////////////
// test functions
//////////////////////////////////////////////////////////////////////

void testAllocator() {
    CudfAllocator allocator{};
    std::shared_ptr<cylon::Buffer> buffer;
    cylon::Status stat = allocator.Allocate(20, &buffer);
    if (!stat.is_ok()) {
        LOG(FATAL) << "Failed to allocate buffer with length " << 20;
    }

    std::shared_ptr<CudfBuffer> cb = std::dynamic_pointer_cast<CudfBuffer>(buffer);
    std::cout << "buffer length: " << cb->GetLength() << std::endl;
    uint8_t *hostArray= new uint8_t[cb->GetLength()];
    cudaMemcpy(hostArray, cb->GetByteBuffer(), cb->GetLength(), cudaMemcpyDeviceToHost);
    std::cout << "copied from device to host" << std::endl;
}

void testColumnAccess(cudf::column_view const& input) {
    int dl = dataLength(input);
    LOG(INFO) << myrank << ": dataLength: " << dl;
    LOG(INFO) << myrank << ": column data type: " << static_cast<int>(input.type().id());

    uint8_t *hostArray= new uint8_t[dl];
    cudaMemcpy(hostArray, input.data<uint8_t>(), dl, cudaMemcpyDeviceToHost);
    int64_t * hdata = (int64_t *) hostArray;
    LOG(INFO) << myrank << "::::: first and last data: " << hdata[0] << ", " << hdata[input.size() -1];
}

void columnDataTypes(cudf::table * table) {
    for (int i = 0; i < table->num_columns(); ++i) {
        cudf::column_view cw = table->get_column(i).view();
        LOG(INFO) << myrank << ", column: " << i << ", size: " << cw.size() << ", data type: " << static_cast<int>(cw.type().id());
    }
}

void printFirstLastElements(cudf::table_view &tv) {
    for (int i = 0; i < tv.num_columns(); ++i) {
        cudf::column_view cw = tv.column(i);
        LOG(INFO) << myrank << ", column[" << i << "], size: " << cw.size() << ", data type: " << static_cast<int>(cw.type().id());
        if (cw.type().id() == cudf::type_id::STRING) {
            cudf::strings_column_view scv(cw);
            cudf::strings::print(scv, 0, 1);
            cudf::strings::print(scv, cw.size()-1, cw.size());
        } else {
            int dl = dataLength(cw);
            uint8_t *hostArray= new uint8_t[dl];
            cudaMemcpy(hostArray, cw.data<uint8_t>(), dl, cudaMemcpyDeviceToHost);
            if (cw.type().id() == cudf::type_id::INT32) {
                int32_t *hdata = (int32_t *) hostArray;
                LOG(INFO) << myrank << "::::: first and last data: " << hdata[0] << ", " << hdata[cw.size() - 1];
            } else if (cw.type().id() == cudf::type_id::INT64) {
                int64_t *hdata = (int64_t *) hostArray;
                LOG(INFO) << myrank << "::::: first and last data: " << hdata[0] << ", " << hdata[cw.size() - 1];
            } else if (cw.type().id() == cudf::type_id::FLOAT32) {
                float *hdata = (float *) hostArray;
                LOG(INFO) << myrank << "::::: first and last data: " << hdata[0] << ", " << hdata[cw.size() - 1];
            } else if (cw.type().id() == cudf::type_id::FLOAT64) {
                double *hdata = (double *) hostArray;
                LOG(INFO) << myrank << "::::: first and last data: " << hdata[0] << ", " << hdata[cw.size() - 1];
            }
        }
    }
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        std::cout << "You must specify a CSV input file.\n";
        return 1;
    }

    auto mpi_config = std::make_shared<cylon::net::MPIConfig>();
    auto ctx = cylon::CylonContext::InitDistributed(mpi_config);
    myrank = ctx->GetRank();

    LOG(INFO) << "myrank: "  << myrank << ", world size: " << ctx->GetWorldSize();

    std::vector<int> allWorkers{};
    for (int i = 0; i < ctx->GetWorldSize(); ++i) {
        allWorkers.push_back(i);
    }

    int numberOfGPUs;
    cudaGetDeviceCount(&numberOfGPUs);
    LOG(INFO) << "myrank: "  << myrank << ", number of GPUs: " << numberOfGPUs;

    // set the gpu
    cudaSetDevice(myrank % numberOfGPUs);

    // define call back to catch the receiving tables
    CudfCallback callback = [=](int source, const std::shared_ptr<cudf::table> &table, int reference) {
        LOG(INFO) << "received a table ...................)))))))))))))))))))))))))))";
        cudf::table_view tv = table->view();
        printFirstLastElements(tv);
        return true;
    };

    CudfAllToAll * cA2A = new CudfAllToAll(ctx, allWorkers, allWorkers, ctx->GetNextSequence(), callback);

    // construct table
    std::string input_csv_file = argv[1];
    cudf::io::source_info si(input_csv_file);
    cudf::io::csv_reader_options options = cudf::io::csv_reader_options::builder(si);
    cudf::io::table_with_metadata ctable = cudf::io::read_csv(options);
    LOG(INFO) << myrank << ", number of columns: " << ctable.tbl->num_columns();

//    std::shared_ptr<cudf::table> tbl = std::make_shared<cudf::table>(ctable.tbl);
    for (int wID: allWorkers) {
//        auto tbl = std::make_unique<cudf::table>(*(ctable.tbl));
        std::shared_ptr<cudf::table_view> tv = std::make_shared<cudf::table_view> (ctable.tbl->view());
        cA2A->insert(tv, wID);
    }

    LOG(INFO) << myrank << ", inserted tables: ";
//    cA2A->finish();

    int i = 1;
    while(!cA2A->isComplete()) {
        if (i % 1000 == 0) {
            LOG(INFO) << myrank << ", has not completed yet.";
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        i++;
    }


//    // column data
//    int columnIndex = myrank;
//    cudf::column_view cw = ctable.tbl->get_column(columnIndex).view();
////    testColumnAccess(cw);
//    LOG(INFO) << myrank << ", column[" << columnIndex << "] size: " << cw.size();
//    const uint8_t *sendBuffer = cw.data<uint8_t>();
//    int dataLen = dataLength(cw);
//
//    // header data
//    int headerLen = 2;
//    int * headers = new int[headerLen];
//    headers[0] = (int)(cw.type().id());
//    headers[1] = myrank;
//
//    for (int wID: allWorkers) {
//        all->insert(sendBuffer, dataLen, wID, headers, headerLen);
//    }

//    all->finish();
//
//    int i = 1;
//    while(!all->isComplete()) {
//        if (i % 100 == 0) {
//            LOG(INFO) << myrank << ", has not completed yet.";
//            std::this_thread::sleep_for(std::chrono::seconds(1));
//        }
//        i++;
//    }

    ctx->Finalize();
    return 0;
}
