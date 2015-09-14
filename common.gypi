{
  'variables': {
    'conditions': [
      ['OS == "mac"', {
        'target_arch%': 'x64'
      }, {
        'target_arch%': 'ia32'
      }]
    ]
  },
  'target_defaults': {
    'default_configuration': 'Release',
    'defines': [ 'HTTP_PARSER_STRICT=0' ],
    'conditions': [
      ['OS == "mac"', {
        'defines': [ 'MACOSX' ]
      }, {
        'defines': [ 'LINUX' ]
      }],
      ['OS == "mac" and target_arch == "x64"', {
        'xcode_settings': {
          'ARCHS': [ 'x86_64' ]
        },
      }]
    ],
    'configurations': {
      'Debug': {
        'cflags': [ '-g', '-O0' ],
        'defines': [ 'DEBUG' ],
        'xcode_settings': {
          'OTHER_CFLAGS': [ '-g', '-O0' ],
          'OTHER_CPLUSPLUSFLAGS' : [ '-g', '-O0', '-std=c++11' ],

        }
      },
      'Release': {
        'cflags': [ '-O3' ],
        'defines': [ 'NDEBUG' ],
        'xcode_settings': {
          'OTHER_CFLAGS': [ '-O3' ],
          'OTHER_CPLUSPLUSFLAGS' : [ '-g', '-O0', '-std=c++11' ],
        }
      }
    }
  }
}