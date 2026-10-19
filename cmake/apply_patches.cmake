# Applies mlip.cpp's patches to the fetched ggml source; the ggml package's
# PATCH_COMMAND (CMakeLists.txt), run in the source directory:
#
#   cmake -P apply_patches.cmake -- a.patch b.patch ...
#
# CMake re-runs the patch step whenever it re-configures, and on a source that
# already carries the patches a plain `patch -p1` fails. So the step records
# the SHA-256 of the patch set it applied in .mlipcpp-patches and does nothing
# when the set is unchanged. When it changed (a patch added, edited or
# removed), or the source has no record but local changes (patched by an older
# build), a git checkout is reset to its commit and the whole set re-applied.

set(patches "")
set(after_dashes OFF)
foreach(i RANGE 1 ${CMAKE_ARGC})
  if(i EQUAL CMAKE_ARGC)
    break()
  endif()
  if(after_dashes)
    list(APPEND patches "${CMAKE_ARGV${i}}")
  elseif("${CMAKE_ARGV${i}}" STREQUAL "--")
    set(after_dashes ON)
  endif()
endforeach()

set(want "")
foreach(p IN LISTS patches)
  if(NOT EXISTS "${p}")
    message(FATAL_ERROR "ggml patch not found: ${p}")
  endif()
  file(SHA256 "${p}" h)
  get_filename_component(name "${p}" NAME)
  string(APPEND want "${name} ${h}\n")
endforeach()

set(src "${CMAKE_CURRENT_SOURCE_DIR}")
set(stamp "${src}/.mlipcpp-patches")
if(EXISTS "${stamp}")
  file(READ "${stamp}" have)
  if(have STREQUAL want)
    list(LENGTH patches n)
    message(STATUS "ggml: the ${n} patches are already applied")
    return()
  endif()
endif()

find_program(PATCH_EXECUTABLE patch REQUIRED)
find_program(GIT_EXECUTABLE git)
set(reset OFF)
if(EXISTS "${stamp}")
  set(reset ON) # a different patch set was applied
elseif(EXISTS "${src}/.git" AND GIT_EXECUTABLE)
  execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=no
                  WORKING_DIRECTORY "${src}" OUTPUT_VARIABLE dirty RESULT_VARIABLE rc)
  if(rc EQUAL 0 AND NOT dirty STREQUAL "")
    set(reset ON) # patched by a build that kept no record
  endif()
endif()
if(reset)
  if(NOT EXISTS "${src}/.git" OR NOT GIT_EXECUTABLE)
    message(FATAL_ERROR "ggml source ${src} carries other patches and is not a git checkout "
                        "(or git is missing): delete it so that CMake fetches it again")
  endif()
  message(STATUS "ggml: patch set changed, resetting ${src}")
  execute_process(COMMAND "${GIT_EXECUTABLE}" checkout -q -- . WORKING_DIRECTORY "${src}" COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND "${GIT_EXECUTABLE}" clean -qfd WORKING_DIRECTORY "${src}" COMMAND_ERROR_IS_FATAL ANY)
endif()

foreach(p IN LISTS patches)
  execute_process(COMMAND "${PATCH_EXECUTABLE}" -p1 --forward --batch -i "${p}"
                  WORKING_DIRECTORY "${src}" RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
  if(NOT rc EQUAL 0)
    message(FATAL_ERROR "ggml: ${p} does not apply:\n${out}${err}")
  endif()
endforeach()
file(WRITE "${stamp}" "${want}")
list(LENGTH patches n)
message(STATUS "ggml: applied ${n} patches")
