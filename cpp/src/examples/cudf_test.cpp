#include <iostream>
#include <cmath>

#include <cudf/types.hpp>
#include <cudf/copying.hpp>
#include <cudf/io/types.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cuda_runtime.h>

#include <rmm/cuda_stream_view.hpp>

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
 * data buffer length of a column in bytes
 * @param input
 * @return
 */
cudf::size_type dataLength(cudf::column_view const& input){
    int elementSize = type_bytes[static_cast<int>(input.type().id())];
    if (elementSize == -1) {
        std::cout << "ERRORRRRRR unsupported type id: " << static_cast<int>(input.type().id()) << std::endl;
        return -1;
    }

    // even null values exist in the buffer with unspecified values
    return elementSize * input.size();
}

std::unique_ptr<cudf::column> emptyLike(cudf::column_view const& input){
    int dl = dataLength(input);
    rmm::device_buffer dataBuf(dl);
    cudaMemcpy(dataBuf.data(), input.data<uint8_t>(), dl, cudaMemcpyDeviceToDevice);

    int nullBufSize = ceil(input.size() / 8.0);
    if (!input.nullable()) {
        nullBufSize = 0;
    }
    rmm::device_buffer nullBuf(nullBufSize);
    cudaMemcpy(nullBuf.data(), input.null_mask(), nullBufSize, cudaMemcpyDeviceToDevice);

    return std::make_unique<cudf::column>(
            input.type(), input.size(), dataBuf, nullBuf, input.null_count());
}

void testEmptyLike(cudf::column_view const& input) {
    std::unique_ptr<cudf::column> copyColumn = emptyLike(input);
    cudf::column_view cw = copyColumn->view();

    // copy column data to host memory and print
    uint8_t *hostArray= (uint8_t*)malloc(static_cast<int>(dataLength(cw)));
    cudaMemcpy(hostArray, cw.data<uint8_t>(), static_cast<int>(dataLength(cw)), cudaMemcpyDeviceToHost);
    int64_t * hdata = (int64_t *) hostArray;
    //char * hdata = (char *) hostArray;
    std::cout << "first data: " << hdata[0] << std::endl;
}

void printCharArray(cudf::column_view cv, int charSize) {
    char *hostArray= new char[charSize+1];
    cudaMemcpy(hostArray, cv.data<char>(), charSize, cudaMemcpyDeviceToHost);
    hostArray[charSize] = '\0';
    std::cout << "chars:: " << hostArray << std::endl;
}

void printOffsetsArray(cudf::column_view cv) {
    std::cout << "offsets column size: " << cv.size() << ", type: " << static_cast<int>(cv.type().id()) << std::endl;
    uint8_t *hostArray= new uint8_t[dataLength(cv)];
    cudaMemcpy(hostArray, cv.data<uint8_t>(), dataLength(cv), cudaMemcpyDeviceToHost);
    int32_t * hdata = (int32_t *) hostArray;
    for (int i = 0; i < cv.size(); ++i) {
        std::cout << hdata[i] << ", ";
    }
    std::cout << std::endl;
}

void printChildColumns(cudf::table_view tview) {

    for (int i = 0; i < tview.num_columns(); ++i) {
        cudf::column_view cw = tview.column(i);
        std::cout << "column[" << i << "] children: " << cw.num_children() << std::endl;
        if (cw.type().id() == cudf::type_id::STRING) {
            std::cout << "column type is STRING ------------------------------------------------ " << std::endl;
            cudf::strings_column_view scv(cw);
            std::cout << "number of strings: " << scv.size() << std::endl;
            std::cout << "number of nulls: " << scv.null_count() << std::endl;
            std::cout << "offsets_column_index: " << scv.offsets_column_index << std::endl;
            std::cout << "chars_column_index: " << scv.chars_column_index << std::endl;
            std::cout << "chars_size: " << scv.chars_size() << std::endl;
            printCharArray(scv.chars(), scv.chars_size());
            printOffsetsArray(scv.offsets(), scv.size());
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "CUDF Example\n";

    if (argc != 2) {
        std::cout << "You must specify a CSV input file.\n";
        return 1;
    }
    std::string input_csv_file = argv[1];
    cudf::io::source_info si(input_csv_file);
    cudf::io::csv_reader_options options = cudf::io::csv_reader_options::builder(si);
    cudf::io::table_with_metadata ctable = cudf::io::read_csv(options);
    std::cout << "number of columns: " << ctable.tbl->num_columns() << std::endl;

    cudf::table_view tview = ctable.tbl->view();
    printChildColumns(tview);
    return 0;

    cudf::column column1 = ctable.tbl->get_column(0);
    std::cout << "column size: " << column1.size() << std::endl;

    cudf::column_view cw = column1.view();
    std::cout << "column view size: " << cw.size() << std::endl;
    cudf::data_type dt = cw.type();
    if (dt.id() == cudf::type_id::STRING){
        std::unique_ptr<cudf::scalar> sp = cudf::get_element(cw, 0);
        cudf::string_scalar *ssp = static_cast<cudf::string_scalar*>(sp.get());
        //std::unique_ptr<cudf::string_scalar>  ssp(static_cast<cudf::string_scalar*>(sp.release()));
        std::cout << "element 0: " << ssp->to_string() << std::endl;
    }

    testEmptyLike(cw);

//    std::cout << "data type int value: " << static_cast<int>(dt.id()) << std::endl;
//    std::cout << "INT32 data type int value: " << static_cast<int>(cudf::type_id::INT32) << std::endl;
//    if (dt.scale() == cudf::type_id::UINT32) {
//       std::cout << "data type: UINT32" << std::endl;

//    int dataSize = cw.end<uint8_t>() - cw.begin<uint8_t>();
//    std::cout << "dataSize: " << dataSize << std::endl;
//    auto * data = cw.data<uint8_t>();


//    static_cast<cudf::type_id>(dt.id()) * data = cw.data();
//    std::cout << "first two data: " << cw.data()[0] << ", " << cw.data()[1] << std::endl;

//    cudf::column::contents ccontents = column1.release();
//    std::cout << "column data size: " << ccontents.data->size() << std::endl;
//    std::cout << "column null_mask size: " << ccontents.null_mask->size() << std::endl;

    return 0;
}