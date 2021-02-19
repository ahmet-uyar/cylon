//
// Created by auyar on 3.02.2021.
//

#include <glog/logging.h>
#include <chrono>
#include <thread>
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

#include "cudf_a2a.hpp"

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
CudfBuffer::CudfBuffer(std::shared_ptr<rmm::device_buffer> rmmBuf) : rmmBuf(std::move(rmmBuf)) {}

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
  std::shared_ptr<rmm::device_buffer> rmmBuf;
  try {
    rmmBuf = std::make_shared<rmm::device_buffer>(length);
  } catch (rmm::bad_alloc	badAlloc) {
    LOG(ERROR) << "failed to allocate gpu memory with rmm: " << badAlloc.what();
    return cylon::Status(cylon::Code::GpuMemoryError);
  }
  *buffer = std::make_shared<CudfBuffer>(rmmBuf);
  return cylon::Status::OK();
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

int CudfAllToAll::insert(const std::shared_ptr<cudf::table> &table, int32_t target) {
    return insert(table, target, -1);
}

int CudfAllToAll::insert(const std::shared_ptr<cudf::table> &table,
                          int32_t target,
                          int32_t reference) {
    // todo: check weather we have enough memory
    // lets save the table into pending and move on
    inputs_[target]->tableQueue.push(std::make_pair(table, reference));
    return 1;
}

bool CudfAllToAll::isComplete() {

    for (const auto &t : inputs_) {
      std::pair<std::shared_ptr<cudf::table>, int32_t> currentPair = t.second->tableQueue.front();
      t.second->tableQueue.pop();
      insertTableToA2A(currentPair.first, t.first, currentPair.second);
    }

    return all_->isComplete();
}

bool CudfAllToAll::insertTableToA2A(std::shared_ptr<cudf::table> table, int target, int ref) {
    // construct header message to send
    // for all columns, send data-type, whether it has null data and offsets buffer.
    // header data
//    int columns = table->num_columns();
    int columns = 2;
    int headersLength = 3;
    int * tableHeaders = new int[headersLength];
    tableHeaders[0] = 0; // shows it is a table header. todo: make it enum
    tableHeaders[1] = ref;
    tableHeaders[2] = columns;
    all_->insert(nullptr, 0, target, tableHeaders, headersLength);

    for (int i = 0; i < columns; ++i) {
        insertColumnToA2A(table->get_column(i), i, target);
    }

    return true;
}

bool CudfAllToAll::insertColumnToA2A(cudf::column &clmn, int columnIndex, int target) {

    cudf::column_view cw = clmn.view();
    int headersLength = 6;
    int * columnHeaders = new int[headersLength];
    columnHeaders[0] = 1; // it is a column header. todo: make it enum
    columnHeaders[1] = columnIndex;
    columnHeaders[2] = (int)(cw.type().id()); // data-type of the column
    // whether the column has null array, 1 has the null buffer, 0 means no null buffer
    if (cw.nullable())
        columnHeaders[3] = 1;
    else
        columnHeaders[3] = 0;

    //whether the column has offsets array
    if (uniform_size_data(cw))
        columnHeaders[4] = 0;
    else
        columnHeaders[4] = 1;

    columnHeaders[5] = clmn.size();

    // send data
    const uint8_t *dataBuffer = cw.data<uint8_t>();
    int dataLen = dataLength(cw);
    all_->insert(dataBuffer, dataLen, target, columnHeaders, headersLength);

    // send null buffer
    if (cw.nullable()) {
        uint8_t * nullBuffer = (uint8_t *)cw.null_mask();
        double dataTypeSize = data_type_size(cw);
        int nullBufSize = ceil(cw.size() / dataTypeSize);
        all_->insert(nullBuffer, nullBufSize, target);
    }

    return true;
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

    // if there is no null buffer or offset buffer, create the column with data only
    if(!pr->hasNullBuffer) {
        column = std::make_unique<cudf::column>(dt, pr->dataSize, *dataBuffer);
    } else { //todo: handle offsets
        column = std::make_unique<cudf::column>(dt, pr->dataSize, *dataBuffer, *nullBuffer);
    }

    // if the column is constructed, add it to the list
    if(column) {
        pr->columns.insert({pr->columnIndex, std::move(column)});

        // clear column related data from pr
        pr->columnIndex = -1;
        pr->columnDataType = -1;
        pr->dataSize = -1;
        pr->dataBuffer.reset();
        pr->nullBuffer.reset();
        pr->hasNullBuffer = false;
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
  LOG(INFO) << "buffer received from the source: " << source;
  std::shared_ptr<CudfBuffer> cb = std::dynamic_pointer_cast<CudfBuffer>(buffer);

  // if the data buffer is not received yet, get it
  std::shared_ptr<PendingReceives> pr = receives_.at(source);
  if (!pr->dataBuffer) {
    pr->dataBuffer = cb->getBuf();

    // if there is no null buffer or offset buffer, create the column
    if(!pr->hasNullBuffer) {
        constructColumn(pr);
    }
  } else if(!pr->hasNullBuffer) {
    pr->nullBuffer = cb->getBuf();
    // if there is no offset buffer, create the column
    constructColumn(pr);
  } else {
      LOG(WARNING) << "although dataBuffer and nullBuffer have been received. a new buffer received from: " << source
        << ", buffer length: " << length;
  }

  // if all columns are created, create the table
  if (pr->columns.size() == pr->numberOfColumns) {
      std::shared_ptr<cudf::table> tbl = constructTable(pr);
      recv_callback_(source, tbl, pr->reference);

      // clear table data from pr
      pr->columns.clear();
      pr->numberOfColumns = -1;
      pr->reference = -1;
  }

//  uint8_t *hostArray= new uint8_t[length];
//  cudaMemcpy(hostArray, cb->GetByteBuffer(), length, cudaMemcpyDeviceToHost);
//
//  if (data_types.at(source) == 3) {
//    int32_t * hdata = (int32_t *) hostArray;
//    LOG(INFO) << "==== data[0]: " << hdata[0] << ", data[1]: " << hdata[1];
//  } else if (data_types.at(source) == 4) {
//    int64_t *hdata = (int64_t *) hostArray;
//    LOG(INFO) << "==== data[0]: " << hdata[0] << ", data[1]: " << hdata[1];
//  } else {
//    LOG(WARNING) << "Unrecognized data type ==== : " << data_types.at(source);
//  }

  return true;
}

/**
 * Receive the header, this happens before we receive the actual data
 */
bool CudfAllToAll::onReceiveHeader(int source, int finished, int *buffer, int length) {
  if (length > 0) {
      if (buffer[0] == 0) { // table header
          std::shared_ptr<PendingReceives> pr = receives_.at(source);
          pr->reference = buffer[1];
          pr->numberOfColumns = buffer[2];
          LOG(INFO) << "----received a table header from the source: " << source;
      } else if(buffer[0] == 1){ // column header
          std::shared_ptr<PendingReceives> pr = receives_.at(source);
          pr->columnIndex = buffer[1];
          pr->columnDataType = buffer[2];
          pr->hasNullBuffer = buffer[3] == 0 ? false: true;
          pr->hasOffsetBuffer = buffer[4] == 0 ? false: true;
          pr->dataSize = buffer[5];
      }
  }
  LOG(INFO) << "----received a header buffer with length: " << length;
  return true;
}

/**
 * This method is called after we successfully send a buffer
 * @return
 */
bool CudfAllToAll::onSendComplete(int target, const void *buffer, int length) {
//        LOG(INFO) << "called onSendComplete with length: " << length << " for the target: " << target;
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
    std::cout << "first data: " << hdata[0] << std::endl;
    LOG(INFO) << myrank << ": first 2 data: " << hdata[0] << ", " << hdata[1];
}

void columnDataTypes(cudf::table * table) {
    for (int i = 0; i < table->num_columns(); ++i) {
        cudf::column_view cw = table->get_column(i).view();
        LOG(INFO) << myrank << ", column: " << i << ", size: " << cw.size() << ", data type: " << static_cast<int>(cw.type().id());
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
    CudfCallback callback = [=](int source, const std::shared_ptr<cudf::table> &table_, int reference) {
        LOG(INFO) << "received a table ...................)))))))))))))))))))))))))))";
        return true;
    };

    CudfAllToAll * cA2A = new CudfAllToAll(ctx, allWorkers, allWorkers, 1, callback);
    CudfAllocator * allocator = new CudfAllocator();

    // construct table
    std::string input_csv_file = argv[1];
    cudf::io::source_info si(input_csv_file);
    cudf::io::csv_reader_options options = cudf::io::csv_reader_options::builder(si);
    cudf::io::table_with_metadata ctable = cudf::io::read_csv(options);
    LOG(INFO) << myrank << ", number of columns: " << ctable.tbl->num_columns();

//    std::shared_ptr<cudf::table> tbl = std::make_shared<cudf::table>(ctable.tbl);
    for (int wID: allWorkers) {
        cA2A->insert(std::move(ctable.tbl), wID);
    }

    cA2A->finish();

    int i = 1;
    while(!cA2A->isComplete()) {
        if (i % 100 == 0) {
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
