#! /bin/sh
#
# cscope.sh
# Copyright (C) 2018 lucas <lucas@lucas>
#
# Distributed under terms of the MIT license.
#


find ./   -name "*.cpp" -o -name "*.c" -o -name "*.h" -o -name "*.cpp" > cscope.files
cscope -Rbqk
ctags -R
