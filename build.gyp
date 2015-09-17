{
  'includes': [ 'common.gypi' ],
  'targets': [
    {
      'target_name': 'aurora',
      'type': 'executable',
      'sources': [ '<!@(ls -1 src/*.cpp)', '<!@(ls -1 src/*.h)', 
                   './deps/lodepng/lodepng.cpp' ],
      'dependencies': [ './deps/libuv/uv.gyp:libuv',
                        './deps/http-parser/http_parser.gyp:http_parser' ],
      'include_dirs': [ './deps/lodepng', './deps/raspicam/src' ],
      'libraries': [ '../deps/raspicam/build/src/libraspicam.so',
                     '/usr/local/lib/libopencv_objdetect.so',
                     '/usr/local/lib/libopencv_core.so' ],
    },
  ],
}