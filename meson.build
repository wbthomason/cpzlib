project('cpzlib', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++20', 'buildtype=release', 'optimization=3',
  'debug=false'])

# Dependencies
eigen = dependency('eigen3', fallback: ['eigen', 'eigen3'])
robin_hood = dependency('robin_hood', fallback: ['robin_hood', 'robin_hood_dep'])
pdqsort = dependency('pdqsort', fallback: ['pdqsort', 'pdqsort_dep'])
dbg = dependency('dbg', fallback: ['dbg', 'dbg_dep'])

# Some hacks (per https://github.com/mesonbuild/meson/issues/7943) to handle CMake dependencies
# in the lazy way
cmake = import('cmake')

fastor = dependency('fastor', required: false)
if not fastor.found()
  fastor_cmake = cmake.subproject('fastor')
  fastor = fastor_cmake.dependency('Fastor')
endif

celero = dependency('celero', required: false)
if not celero.found()
  celero_cmake = cmake.subproject('celero')
  celero = celero_cmake.dependency('celero')
endif

rapidcheck = dependency('rapidcheck', required: false)
if not rapidcheck.found()
  rapidcheck_cmake = cmake.subproject('rapidcheck')
  rapidcheck = rapidcheck_cmake.dependency('rapidcheck')
endif

# Test executable
test_exe = executable('cpzlib_test', 'test/cpzlib_test.cc',
  include_directories: include_directories('include'),
  dependencies: [eigen, robin_hood, pdqsort], cpp_args: '-march=native')
test('cpzlib', test_exe)

# Benchmark executable
benchmark_exe = executable('cpzlib_benchmarks', 'test/cpzlib_benchmarks.cc', 
  include_directories: include_directories('include'),
  dependencies: [eigen, robin_hood, pdqsort, celero], cpp_args: '-march=native')

# Make this library usable as a Meson subproject.
cpzlib_dep = declare_dependency(
  include_directories: include_directories('include'),
  dependencies: [eigen, robin_hood, pdqsort])

# Make this library usable from the system's package manager.
install_headers('include/cpzlib.hh', subdir : 'cpzlib')

pkg_mod = import('pkgconfig')
pkg_mod.generate(
  name : 'cpzlib',
  filebase : 'cpzlib',
  description : 'Meson sample project.',
  subdirs : 'cpzlib',
  version : '0.1',
)
