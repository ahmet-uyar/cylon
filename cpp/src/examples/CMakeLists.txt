add_definitions(-DEXAMPLE_CMAKE_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\")

#build a binary
macro(cylon_add_exe EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp)
    target_link_libraries(${EXENAME} ${MPI_LIBRARIES})
    target_link_libraries(${EXENAME} cylon)
    target_link_libraries(${EXENAME} ${ARROW_LIB})
    target_link_libraries(${EXENAME} ${GLOG_LIBRARIES})
    install(TARGETS ${EXENAME} RUNTIME DESTINATION examples)
endmacro(cylon_add_exe)

include_directories(../cylon)

cylon_add_exe(union_example)
cylon_add_exe(select_example)
cylon_add_exe(join_example)
cylon_add_exe(project_example)
cylon_add_exe(subtract_example)
cylon_add_exe(intersect_example)
cylon_add_exe(task_test)
cylon_add_exe(ra_test_inmem_datagen_example)
cylon_add_exe(table_from_vectors_example)
cylon_add_exe(compute_example)
cylon_add_exe(groupby_benchmark_example)
cylon_add_exe(groupby_pipeline_example)
cylon_add_exe(groupby_example)
cylon_add_exe(partition_example)
cylon_add_exe(unique_example)
cylon_add_exe(unique_bench_example)
cylon_add_exe(indexing_example)

if (CYLON_PARQUET)
    cylon_add_exe(parquet_test)
    cylon_add_exe(parquet_from_csv_test)
    cylon_add_exe(parquet_union_example)
    cylon_add_exe(parquet_join_example)
endif (CYLON_PARQUET)

macro(cylon_add_exe2 DIR EXENAME)
    add_executable(${EXENAME} ${DIR}/${EXENAME}.cpp)
    target_link_libraries(${EXENAME} ${MPI_LIBRARIES})
    target_link_libraries(${EXENAME} cylon)
    target_link_libraries(${EXENAME} ${ARROW_LIB})
    target_link_libraries(${EXENAME} ${GLOG_LIBRARIES})
    install(TARGETS ${EXENAME} RUNTIME DESTINATION examples)
endmacro(cylon_add_exe2)

cylon_add_exe2(ops com_comm_test)
cylon_add_exe2(ops join_op_example)

###################################################################
# add cuda example

# find CUDA
find_package(CUDAToolkit REQUIRED)

#build a binary with cuda
macro(cylon_add_exe3 EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp)
    target_link_libraries(${EXENAME} ${MPI_LIBRARIES})
    target_link_libraries(${EXENAME} cylon)
    target_link_libraries(${EXENAME} ${ARROW_LIB})
    target_link_libraries(${EXENAME} ${GLOG_LIBRARIES})
    target_link_libraries(${EXENAME} CUDA::cudart)
    install(TARGETS ${EXENAME} RUNTIME DESTINATION examples)
endmacro(cylon_add_exe3)

cylon_add_exe3(mpi_cuda_test)

###################################################################
# add cudf example
SET(CONDA_DIR /home/auyar/anaconda3/envs/cudf_dev/)
include_directories(
        ${CONDA_DIR}/include
        ${CONDA_DIR}/include/libcudf/libcudacxx/
)

#build a binary with cuda
macro(cylon_add_exe4 EXENAME)
    add_executable(${EXENAME} ${EXENAME}.cpp cudf/cudf_a2a.cpp cudf/gtable.cpp)
    target_link_libraries(${EXENAME} ${MPI_LIBRARIES})
    target_link_libraries(${EXENAME} cylon)
    target_link_libraries(${EXENAME} ${ARROW_LIB})
    target_link_libraries(${EXENAME} ${GLOG_LIBRARIES})
    target_link_libraries(${EXENAME} CUDA::cudart)
    target_link_libraries(${EXENAME}
            ${CONDA_DIR}/lib/libcudf_ast.so
            ${CONDA_DIR}/lib/libcudf_interop.so
            ${CONDA_DIR}/lib/libcudf_partitioning.so
            ${CONDA_DIR}/lib/libcudf_rolling.so
            ${CONDA_DIR}/lib/libcudf_base.so
            ${CONDA_DIR}/lib/libcudf_io.so
            ${CONDA_DIR}/lib/libcudf_reductions.so
            ${CONDA_DIR}/lib/libcudf.so
            ${CONDA_DIR}/lib/libcudf_comms.so
            ${CONDA_DIR}/lib/libcudf_join.so
            ${CONDA_DIR}/lib/libcudf_replace.so
            ${CONDA_DIR}/lib/libcudf_hash.so
            ${CONDA_DIR}/lib/libcudf_merge.so
            ${CONDA_DIR}/lib/libcudf_reshape.so
            ${CONDA_DIR}/lib/libcudf_transpose.so
            )
    install(TARGETS ${EXENAME} RUNTIME DESTINATION examples)
endmacro(cylon_add_exe4)

cylon_add_exe4(cudf_test)
#cylon_add_exe4(a2a_cudf_test)
cylon_add_exe4(gtable_test)
