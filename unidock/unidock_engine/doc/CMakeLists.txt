message("   [Section] Building DOC ...")

#sudo apt install doxygen graphviz
find_program(DOXYGEN_EXECUTABLE doxygen)
if (NOT DOXYGEN_EXECUTABLE)
    message(FATAL_ERROR "BUILD_DOC needs doxygen on PATH")
endif ()

# Assure the existence of working directory
if (EXISTS "${CMAKE_SOURCE_DIR}/doc")
    execute_process(
            COMMAND ${DOXYGEN_EXECUTABLE} Doxyfile
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/doc
            OUTPUT_VARIABLE _RES
            ERROR_VARIABLE _RES
    )
#    message("Doxygen output: ${_RES}")
else ()
    message(FATAL_ERROR "Directory ${CMAKE_SOURCE_DIR}/doc does not exist.")
endif ()


#message("Doxygen output: ${_RES}")
message("   [Section Done] DOC !")

