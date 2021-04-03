project('cpzlib', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++17'])

eigen = dependency('eigen3', fallback: ['eigen', 'eigen3'])
robin_hood = dependency('robin_hood', fallback: ['robin_hood', 'robin_hood_dep'])
pdqsort = dependency('pdqsort', fallback: ['pdqsort', 'pdqsort_dep'])

test_exe = executable('cpzlib_test', 'test/cpzlib_test.cc',
  include_directories: include_directories('include'),
  dependencies: [eigen, robin_hood, pdqsort])
test('cpzlib', test_exe)

# Make this library usable as a Meson subproject.
cpzlib_dep = declare_dependency(
  include_directories: include_directories('include'),
  dependencies: [eigen, robin_hood, pdqsort])

# Make this library usable from the system's
# package manager.
install_headers('include/cpzlib.hh', subdir : 'cpzlib')

pkg_mod = import('pkgconfig')
pkg_mod.generate(
  name : 'cpzlib',
  filebase : 'cpzlib',
  description : 'Meson sample project.',
  subdirs : 'cpzlib',
  version : '0.1',
)
