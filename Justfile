build_dir := "build"
artifacts_dir := "bin"
exec_dir := "exe"

exec := "nlshade_exec"
exec_test := "nlshade_exec"

cpu_cores := "28"
build_mode := "Debug"

alias i := init
alias b := build
alias r := run
alias c := clean



init:
  mkdir {{build_dir}}
  cmake -B {{ build_dir }} -S . -DCMAKE_BUILD_TYPE={{build_mode}}

build:
  cmake --build {{ build_dir }} -j {{ cpu_cores }}

clean: 
  rm -rf {{ build_dir }}
  rm -rf {{ artifacts_dir }}

run: 
  ./{{ artifacts_dir }}/{{ exec_dir }}/{{ exec }}

