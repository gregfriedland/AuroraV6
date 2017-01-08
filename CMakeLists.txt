cmake_minimum_required (VERSION 3.0)
project(AuroraV6)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)

SET(SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Audio.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/AlienBlob.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Aurora.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Bzr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Camera.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Controller.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/FaceDetect.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/FindBeats.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GinzburgLandau.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/GrayScott.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/KeyboardAudio.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Off.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Palette.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/ReactionDiffusion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Serial.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Util.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/Video.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/VampPluginHost.cpp
)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm") 
  add_subdirectory(raspicam)

  file(GLOB RPI_RGB_MATRIX_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/rpi-rgb-led-matrix/lib/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rpi-rgb-led-matrix/lib/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/rpi-rgb-led-matrix/lib/*.c
  )
  add_executable(${PROJECT_NAME} ${SRCS} ${RPI_RGB_MATRIX_SRCS})
  target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${CMAKE_CURRENT_BINARY_DIR}/raspicam/src/libraspicam_cv.so
    vamp-hostsdk
  )
  target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/json/src
    ${CMAKE_CURRENT_SOURCE_DIR}/raspicam/src
    ${CMAKE_CURRENT_SOURCE_DIR}/rpi-rgb-led-matrix/include
    ${CMAKE_CURRENT_SOURCE_DIR}/aubio/src
    ${CMAKE_CURRENT_SOURCE_DIR}/vamp-plugin-sdk/src
  )
  target_compile_options(${PROJECT_NAME} PUBLIC -mfpu=neon -mfloat-abi=hard -mtune=cortex-a53)
else()
  add_executable(${PROJECT_NAME} ${SRCS})

  target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} portaudio aubio vamp-hostsdk)
  target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/json/src
    ${CMAKE_CURRENT_SOURCE_DIR}/aubio/src
    ${CMAKE_CURRENT_SOURCE_DIR}/vamp-plugin-sdk/src
  )
endif()


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/haarcascade_frontalface_alt2.xml
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_definitions(-DFACE_CASCADE_FILE="${CMAKE_CURRENT_BINARY_DIR}/haarcascade_frontalface_alt2.xml")

add_executable(detect_faces_test src/detect_faces_test.cpp)
target_link_libraries(detect_faces_test ${OpenCV_LIBS})
